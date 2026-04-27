// ============================================================================
// blur_cuda.cu — GPU separable box blur (prefix-sum + sliding-window)
//
// v3 kernels:
//   Horizontal: per-row prefix sum in shared memory → O(1) per output pixel.
//   Vertical:   per-column sliding accumulator      → O(1) per output pixel.
//
// Both replace the naive O(radius) kernels, dropping blur time from ~19 ms
// to ~3-5 ms for typical radius-28 / 2-pass configurations.
// ============================================================================

#include "blur_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Legacy naive kernels (kept for fallback if shared memory is too small)
// ---------------------------------------------------------------------------

__global__ void box_blur_h_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int w, int h, int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int lo = max(0,     x - radius);
    const int hi = min(w - 1, x + radius);
    const float* row = in + (size_t)y * w;

    float sum = 0.0f;
    for (int i = lo; i <= hi; ++i)
        sum += row[i];

    out[(size_t)y * w + x] = sum / (float)(hi - lo + 1);
}

__global__ void box_blur_v_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int w, int h, int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int lo = max(0,     y - radius);
    const int hi = min(h - 1, y + radius);

    float sum = 0.0f;
    for (int i = lo; i <= hi; ++i)
        sum += in[(size_t)i * w + x];

    out[(size_t)y * w + x] = sum / (float)(hi - lo + 1);
}

// ---------------------------------------------------------------------------
// v3: prefix-sum horizontal blur
//
// One block per row.  The row is loaded into shared memory, scanned into a
// prefix sum by thread 0 (< 6 K additions for 4K — < 3 µs), then every
// thread computes its output in O(1) via  (prefix[hi+1] − prefix[lo]).
//
// Shared memory: (w + 1) × sizeof(float) bytes.
// Max supported w with 48 KB default shared mem: 12287.
// ---------------------------------------------------------------------------

__global__ void box_blur_h_prefix_kernel(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int w, int h, int radius)
{
    extern __shared__ float prefix[];  // (w+1) floats

    const int y = blockIdx.x;
    if (y >= h) return;

    const size_t row_off = (size_t)y * w;

    // Load row values into prefix[1..w]; set prefix[0] = 0.
    prefix[0] = 0.0f;
    for (int i = (int)threadIdx.x; i < w; i += (int)blockDim.x)
        prefix[i + 1] = in[row_off + i];
    __syncthreads();

    // Inclusive prefix sum (thread 0 — sequential, L1-fast in shared memory).
    if (threadIdx.x == 0)
    {
        for (int i = 1; i <= w; ++i)
            prefix[i] += prefix[i - 1];
    }
    __syncthreads();

    // Each thread computes blurred output in O(1).
    for (int x = (int)threadIdx.x; x < w; x += (int)blockDim.x)
    {
        int lo = max(0,     x - radius);
        int hi = min(w - 1, x + radius);
        out[row_off + x] = (prefix[hi + 1] - prefix[lo]) / (float)(hi - lo + 1);
    }
}

// ---------------------------------------------------------------------------
// v3: sliding-window vertical blur
//
// One thread per column.  Maintains a running sum and slides it downward,
// adding the new bottom element and removing the old top element each step.
// O(1) amortised per output pixel, O(radius) initial seeding.
// ---------------------------------------------------------------------------

__global__ void box_blur_v_sliding_kernel(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          int w, int h, int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;

    // Seed: sum of elements in the window for y = 0.
    const int init_hi = min(radius, h - 1);
    float sum = 0.0f;
    for (int i = 0; i <= init_hi; ++i)
        sum += in[(size_t)i * w + x];

    // Slide window downward.
    for (int y = 0; y < h; ++y)
    {
        // If window expanded downward (new element entered from bottom):
        if (y > 0)
        {
            int new_hi = y + radius;
            if (new_hi < h)
                sum += in[(size_t)new_hi * w + x];
            // If window contracted from top (old element left):
            int old_lo = y - radius - 1;
            if (old_lo >= 0)
                sum -= in[(size_t)old_lo * w + x];
        }

        int lo = max(0,     y - radius);
        int hi = min(h - 1, y + radius);
        out[(size_t)y * w + x] = sum / (float)(hi - lo + 1);
    }
}

// ---------------------------------------------------------------------------
// Stream-aware wrapper functions (callable from other .cu files)
// ---------------------------------------------------------------------------

// Max row width for prefix-sum kernel (48 KB shared memory / 4 bytes - 1).
static constexpr int MAX_PREFIX_W = 12287;

void launch_blur_h_on_stream(const float* in, float* out,
                             int w, int h, int radius, void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    if (w <= MAX_PREFIX_W)
    {
        // Prefix-sum: 1 block per row, 256 threads, shared mem = (w+1)*4
        size_t smem = ((size_t)w + 1) * sizeof(float);
        box_blur_h_prefix_kernel<<<h, 256, smem, s>>>(in, out, w, h, radius);
    }
    else
    {
        // Fallback: naive per-pixel kernel.
        const dim3 block(32, 8);
        const dim3 grid((w + block.x - 1) / block.x,
                        (h + block.y - 1) / block.y);
        box_blur_h_kernel<<<grid, block, 0, s>>>(in, out, w, h, radius);
    }
}

void launch_blur_v_on_stream(const float* in, float* out,
                             int w, int h, int radius, void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int blocks = (w + 255) / 256;
    box_blur_v_sliding_kernel<<<blocks, 256, 0, s>>>(in, out, w, h, radius);
}

// ---------------------------------------------------------------------------
// Legacy launchers (kept for backward compatibility)
// ---------------------------------------------------------------------------

void launch_box_blur_cuda(
    float* cpu_r, float* cpu_g, float* cpu_b,
    int w, int h, int radius, int passes,
    GpuBufferCache& cache, std::string* out_error)
{
    if (radius < 1 || passes < 1 || w <= 0 || h <= 0) return;

    const size_t n_px = (size_t)w * h;

    auto report = [&](cudaError_t e, const char* site) {
        fprintf(stderr, "FlareSim blur CUDA error at %s -- %s\n",
                site, cudaGetErrorString(e));
        if (out_error && out_error->empty()) {
            char buf[256];
            snprintf(buf, sizeof(buf), "blur CUDA error at %s -- %s",
                     site, cudaGetErrorString(e));
            *out_error = buf;
        }
    };

    if (n_px > cache.blur_floats) {
        cudaFree(cache.blur_a);  cache.blur_a = nullptr;
        cudaFree(cache.blur_b);  cache.blur_b = nullptr;
        cache.blur_floats = 0;
        cudaError_t e;
        e = cudaMalloc(&cache.blur_a, n_px * sizeof(float));
        if (e != cudaSuccess) { report(e, "blur_a"); return; }
        e = cudaMalloc(&cache.blur_b, n_px * sizeof(float));
        if (e != cudaSuccess) { report(e, "blur_b"); return; }
        cache.blur_floats = n_px;
    }

    float* cpu_ch[3] = { cpu_r, cpu_g, cpu_b };

    for (int ch = 0; ch < 3; ++ch) {
        cudaError_t e;
        e = cudaMemcpy(cache.blur_a, cpu_ch[ch],
                       n_px * sizeof(float), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) { report(e, "blur upload"); return; }

        for (int p = 0; p < passes; ++p) {
            launch_blur_h_on_stream(cache.blur_a, cache.blur_b, w, h, radius, nullptr);
            launch_blur_v_on_stream(cache.blur_b, cache.blur_a, w, h, radius, nullptr);
        }

        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { report(e, "blur sync"); return; }
        e = cudaMemcpy(cpu_ch[ch], cache.blur_a,
                       n_px * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) { report(e, "blur download"); return; }
    }
}

void launch_box_blur_gpu_resident(
    int w, int h, int radius, int passes,
    GpuBufferCache& cache, std::string* out_error)
{
    if (radius < 1 || passes < 1 || w <= 0 || h <= 0) return;

    const size_t n_px = (size_t)w * h;

    auto report = [&](cudaError_t e, const char* site) {
        fprintf(stderr, "FlareSim blur_gpu CUDA error at %s -- %s\n",
                site, cudaGetErrorString(e));
        if (out_error && out_error->empty()) {
            char buf[256];
            snprintf(buf, sizeof(buf), "blur_gpu CUDA error at %s -- %s",
                     site, cudaGetErrorString(e));
            *out_error = buf;
        }
    };

    if (n_px > cache.blur_floats) {
        cudaFree(cache.blur_a);  cache.blur_a = nullptr;
        cudaFree(cache.blur_b);  cache.blur_b = nullptr;
        cache.blur_floats = 0;
        cudaError_t e;
        e = cudaMalloc(&cache.blur_a, n_px * sizeof(float));
        if (e != cudaSuccess) { report(e, "blur_a"); return; }
        e = cudaMalloc(&cache.blur_b, n_px * sizeof(float));
        if (e != cudaSuccess) { report(e, "blur_b"); return; }
        cache.blur_floats = n_px;
    }

    float* d_ch[3] = { cache.d_out_r, cache.d_out_g, cache.d_out_b };

    for (int ch = 0; ch < 3; ++ch) {
        cudaMemcpy(cache.blur_a, d_ch[ch],
                   n_px * sizeof(float), cudaMemcpyDeviceToDevice);
        for (int p = 0; p < passes; ++p) {
            launch_blur_h_on_stream(cache.blur_a, cache.blur_b, w, h, radius, nullptr);
            launch_blur_v_on_stream(cache.blur_b, cache.blur_a, w, h, radius, nullptr);
        }
        cudaMemcpy(d_ch[ch], cache.blur_a,
                   n_px * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { report(e, "blur sync"); return; }
}
