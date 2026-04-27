// ============================================================================
// ghost_cuda.h — CUDA ghost rendering launcher declaration
// ============================================================================
#pragma once

#include "ghost.h"
#include "lens.h"

#include <cstddef>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// GpuBufferCache — persistent device-memory allocations across frames.
//
// The caller (typically the FlareSim Iop node instance) owns one of these.
// Pass it to launch_ghost_cuda() every call.  The launcher grows each buffer
// only when the new data is larger than the current allocation, avoiding a
// cudaMalloc + cudaFree pair on every frame.
//
// Internal GPU POD types (GPUPair, GPUSource, etc.) are not exposed in this
// header; the buffers are stored as void* and cast inside ghost_cuda.cu.
// ---------------------------------------------------------------------------

struct GpuBufferCache
{
    // Input parameter buffers (void* hides internal GPU POD types).
    void*  d_surfs = nullptr;  std::size_t surfs_bytes = 0;
    void*  d_pairs = nullptr;  std::size_t pairs_bytes = 0;
    void*  d_src   = nullptr;  std::size_t src_bytes   = 0;
    void*  d_grid  = nullptr;  std::size_t grid_bytes  = 0;
    void*  d_spec  = nullptr;  std::size_t spec_bytes  = 0;

    // Output accumulation buffers — float* (FP32) for precision during scatter.
    // All four channels are always grown together (same pixel count).
    float* d_out_r    = nullptr;
    float* d_out_g    = nullptr;
    float* d_out_b    = nullptr;
    float* d_out_a    = nullptr;   // GPU-side alpha (computed by launch_alpha_cuda)
    std::size_t out_floats = 0;   // capacity of each channel in floats

    // FP16 device staging — FP32 is converted here before DMA to host.
    // 4 channels contiguous: [R|G|B|A], each n_px × sizeof(uint16_t).
    void*       d_out_fp16       = nullptr;
    std::size_t d_out_fp16_elems = 0;   // capacity per channel in elements

    // --- Box blur GPU ping-pong buffers ---
    float*      blur_a      = nullptr;
    float*      blur_b      = nullptr;
    std::size_t blur_floats = 0;

    // --- CUDA streams for async pipeline ---
    // Opaque: stored as void* to avoid pulling cuda_runtime.h into the header.
    void*       compute_stream = nullptr;   // cudaStream_t
    void*       copy_stream    = nullptr;   // cudaStream_t

    // --- Persistent pinned host staging for async D2H readback (FP16) ---
    // Allocated once via cudaHostAlloc, reused every frame.
    // 4 channels (RGBA) × n_px × sizeof(uint16_t) contiguous.
    void*       h_pinned       = nullptr;
    std::size_t h_pinned_elems = 0;   // capacity per channel in elements

    // --- Cached pupil grid (rebuilt only when config changes) ---
    void*       d_cached_grid = nullptr;
    std::size_t cached_grid_bytes = 0;
    int         cached_grid_count = 0;       // number of valid samples
    // Config key — if any of these change, the grid is rebuilt.
    int         cached_ray_grid       = -1;
    int         cached_aperture_blades = -1;
    float       cached_aperture_rot   = -999.0f;
    int         cached_pupil_jitter   = -1;
    int         cached_jitter_seed    = -1;

    // Free all device allocations and reset capacities to zero.
    // Safe to call multiple times and on a partially-constructed cache.
    void release();

    ~GpuBufferCache() { release(); }

    GpuBufferCache()  = default;
    GpuBufferCache(const GpuBufferCache&)            = delete;
    GpuBufferCache& operator=(const GpuBufferCache&) = delete;
};

// Ensure the pinned host staging buffer in cache is at least n_px × 4 channels
// of FP16 (uint16_t) data.  Returns a pointer to the pinned buffer.
// On failure, falls back to malloc.
// Call from FlareSim.cpp (pure C++) — wraps CUDA API calls.
void* ensure_pinned_output_fp16(GpuBufferCache& cache, std::size_t n_px);

// Ensure the pinned host staging buffer in cache is at least n_px × 4 channels
// of FP32 (float) data.  Returns a pointer to the pinned buffer.
// Use this instead of ensure_pinned_output_fp16 for banding-free output.
void* ensure_pinned_output_fp32(GpuBufferCache& cache, std::size_t n_px);

// Convert one scanline of FP16 data to FP32.
// src: pointer to n uint16_t values (IEEE 754 half-precision).
// dst: pointer to n float values (output).
// Portable implementation — no CUDA headers required by the caller.
void convert_fp16_scanline(const void* src, float* dst, int n);

// One wavelength sample with pre-computed RGB colour weights.
// rw/gw/bw are normalised so the sum across all samples for each channel
// equals 1.0 (energy-preserving across different spectral_samples counts).
struct GPUSpectralSample
{
    float lambda; // wavelength in nm
    float rw, gw, bw; // linear RGB contribution weights
};

// Launch the CUDA ghost rendering kernel.
//
// active_pairs / pair_area_boosts must already be pre-filtered (below-threshold
// pairs removed, area-normalisation boosts computed) — see filter_ghost_pairs()
// in ghost.cpp.
//
// sensor_half_w/h: half-dimensions of the sensor in mm, computed from focal
// length and field of view (used for pixel-to-sensor coordinate mapping).
//
// out_r/g/b: CPU-side output buffers, width×height (bounding-box sized), must
// be zeroed by caller.  After return they contain the accumulated ghost
// contribution in linear light.
//
// fmt_w/h: pixel dimensions of the output format (may be smaller than the
// bounding box when off-frame content is present).
// fmt_x0_in_buf / fmt_y0_in_buf: pixel offset of the format origin within
// the output buffer, i.e. (format.x() - bbox_x0) and (format.y() - bbox_y0).
// Ghost positions are always computed relative to the format centre so that
// the optical axis stays fixed even when upstream nodes push the bounding box
// beyond the frame edges.
//
// out_error: if non-null and a CUDA error occurs, receives an error message.
void launch_ghost_cuda(
    const LensSystem&               lens,
    const std::vector<GhostPair>&   active_pairs,
    const std::vector<float>&       pair_area_boosts,
    const std::vector<BrightPixel>& sources,
    float                           sensor_half_w,
    float                           sensor_half_h,
    float*                          out_r,
    float*                          out_g,
    float*                          out_b,
    int                             width,
    int                             height,
    int                             fmt_w,
    int                             fmt_h,
    int                             fmt_x0_in_buf,
    int                             fmt_y0_in_buf,
    const GhostConfig&              config,
    GpuBufferCache&                 cache,
    bool                            skip_readback = false,
    std::string*                    out_error = nullptr,
    const std::vector<float>*       pair_colors  = nullptr,  // 3 floats per pair (RGB), null = all white
    const std::vector<float>*       pair_offsets = nullptr,   // 2 floats per pair (XY px), null = no offset
    const std::vector<float>*       pair_scales  = nullptr    // 1 float per pair, null = 1.0
);

// Compute alpha from GPU-resident RGB buffers:
//   alpha = clamp(0.2126*R + 0.7152*G + 0.0722*B, 0, 1)
// Operates on cache.d_out_r/g/b → cache.d_out_a entirely on device.
void launch_alpha_cuda(
    int             width,
    int             height,
    GpuBufferCache& cache,
    std::string*    out_error = nullptr
);

// Read back GPU-resident RGBA to CPU buffers (single batched D2H copy).
// cpu_r/g/b/a must each have width × height floats pre-allocated.
void readback_gpu_output(
    float*          cpu_r,
    float*          cpu_g,
    float*          cpu_b,
    float*          cpu_a,
    int             width,
    int             height,
    GpuBufferCache& cache,
    std::string*    out_error = nullptr
);

// Combined blur → alpha → readback with async CUDA streams.
//
// Runs the separable box blur on all 3 channels using prefix-sum + sliding-
// window kernels, computes alpha, and reads back all 4 channels — pipelining
// readback of finished channels with blur of remaining ones.
//
// Replaces the three separate calls: blur, alpha, readback.
// If blur_radius < 1 or blur_passes < 1, the blur step is skipped.
// If highlight_clip > 0, a soft-clip is applied after blur (before alpha).
// highlight_knee controls transition sharpness (0=hard, 1=very soft).
void launch_blur_alpha_readback_async(
    float*          cpu_r,
    float*          cpu_g,
    float*          cpu_b,
    float*          cpu_a,
    int             w,
    int             h,
    int             blur_radius,
    int             blur_passes,
    GpuBufferCache& cache,
    std::string*    out_error = nullptr,
    float           highlight_clip = 0.0f,
    float           highlight_knee = 0.5f,
    int             highlight_metric = 1
);

// FP32 variant — same pipeline but skips FP16 conversion.
// DMA transfers FP32 directly to pinned host memory.
// Eliminates banding artifacts from FP16 quantisation.
// cpu_r/g/b/a are float* into the FP32 pinned buffer.
void launch_blur_alpha_readback_fp32(
    float*          cpu_r,
    float*          cpu_g,
    float*          cpu_b,
    float*          cpu_a,
    int             w,
    int             h,
    int             blur_radius,
    int             blur_passes,
    GpuBufferCache& cache,
    std::string*    out_error = nullptr
);
