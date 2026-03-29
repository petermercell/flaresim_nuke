// ============================================================================
// ghost_metal.h — Metal GPU buffer cache + ghost rendering declarations
//
// Replaces ghost_cuda.h for the macOS / Apple Silicon Metal backend.
//
// On Apple Silicon, all buffers use MTLResourceStorageModeShared (unified
// memory) — the same pointer is valid for both CPU and GPU, eliminating
// all H2D / D2H memcpy overhead.
// ============================================================================
#pragma once

#include "ghost.h"
#include "lens.h"

#include <cstddef>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// MetalBufferCache — persistent Metal buffer allocations across frames.
//
// Wraps MTLBuffer objects as void* to keep Metal headers out of pure C++
// compilation units.  The .mm files cast to id<MTLBuffer>.
// ---------------------------------------------------------------------------

struct MetalBufferCache
{
    // Metal device + command queue (id<MTLDevice>, id<MTLCommandQueue>)
    void* mtl_device  = nullptr;   // retained
    void* mtl_queue   = nullptr;   // retained

    // Compiled shader library (id<MTLLibrary>)
    void* mtl_library = nullptr;   // retained

    // Compute pipeline states (id<MTLComputePipelineState>)
    void* pso_ghost       = nullptr;
    void* pso_alpha       = nullptr;
    void* pso_fp32_to_fp16 = nullptr;
    void* pso_clear       = nullptr;
    void* pso_blur_h      = nullptr;
    void* pso_blur_v      = nullptr;
    void* pso_blur_h_naive = nullptr;

    // Input parameter buffers (id<MTLBuffer>)
    void*  buf_surfs = nullptr;  std::size_t surfs_bytes = 0;
    void*  buf_pairs = nullptr;  std::size_t pairs_bytes = 0;
    void*  buf_src   = nullptr;  std::size_t src_bytes   = 0;
    void*  buf_grid  = nullptr;  std::size_t grid_bytes  = 0;
    void*  buf_spec  = nullptr;  std::size_t spec_bytes  = 0;

    // Ghost params uniform buffer
    void*  buf_params = nullptr;

    // Output accumulation buffers — FP32 (id<MTLBuffer>)
    // On Apple Silicon these are shared memory: CPU and GPU access same address.
    void*       buf_out_r  = nullptr;
    void*       buf_out_g  = nullptr;
    void*       buf_out_b  = nullptr;
    void*       buf_out_a  = nullptr;
    std::size_t out_floats = 0;    // capacity per channel in floats

    // FP16 output staging buffer (id<MTLBuffer>)
    // 4 channels contiguous: [R|G|B|A], each n_px × sizeof(uint16_t)
    void*       buf_out_fp16       = nullptr;
    std::size_t buf_out_fp16_elems = 0;

    // Blur scratch buffers (id<MTLBuffer>, FP32)
    void*       blur_a      = nullptr;
    void*       blur_b      = nullptr;
    std::size_t blur_floats = 0;

    // Cached pupil grid (rebuilt only when config changes)
    int   cached_grid_count        = 0;
    int   cached_ray_grid          = -1;
    int   cached_aperture_blades   = -1;
    float cached_aperture_rot      = -999.0f;
    int   cached_pupil_jitter      = -1;
    int   cached_jitter_seed       = -1;

    // Free all allocations.
    void release();

    ~MetalBufferCache() { release(); }

    MetalBufferCache()  = default;
    MetalBufferCache(const MetalBufferCache&)            = delete;
    MetalBufferCache& operator=(const MetalBufferCache&) = delete;
};

// One wavelength sample with pre-computed RGB colour weights.
struct GPUSpectralSample
{
    float lambda;
    float rw, gw, bw;
};

// Initialise the Metal device, command queue, and shader library.
// Call once at plugin construction or first use.
// Returns false on failure (no Metal GPU, shader compilation error, etc.).
bool metal_init(MetalBufferCache& cache, std::string* out_error = nullptr);

// Get a CPU pointer to the FP16 output staging buffer, growing if needed.
// On Apple Silicon, this is the same pointer the GPU writes to (unified memory).
void* ensure_fp16_output(MetalBufferCache& cache, std::size_t n_px);

// Convert one scanline of FP16 data to FP32 (portable, no Metal headers needed).
void convert_fp16_scanline(const void* src, float* dst, int n);

// Launch the Metal ghost rendering kernel.
// Same interface as the CUDA version (minus CUDA-specific types).
void launch_ghost_metal(
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
    MetalBufferCache&               cache,
    bool                            skip_readback = false,
    std::string*                    out_error = nullptr,
    const std::vector<float>*       pair_colors  = nullptr,
    const std::vector<float>*       pair_offsets = nullptr,
    const std::vector<float>*       pair_scales  = nullptr
);

// Compute alpha from GPU-resident RGB, convert all to FP16, and make
// available at the CPU pointer returned by ensure_fp16_output().
// On Apple Silicon: no DMA needed — unified memory makes GPU writes
// immediately visible to the CPU after command buffer completion.
void launch_blur_alpha_readback_metal(
    float*             cpu_r,    // reinterpreted as uint16_t* into FP16 staging
    float*             cpu_g,
    float*             cpu_b,
    float*             cpu_a,
    int                w,
    int                h,
    int                blur_radius,
    int                blur_passes,
    MetalBufferCache&  cache,
    std::string*       out_error = nullptr
);
