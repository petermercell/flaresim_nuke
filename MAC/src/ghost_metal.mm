// ============================================================================
// ghost_metal.mm — Metal GPU ghost rendering launcher + buffer management
//
// Port of ghost_cuda.cu host-side code to Metal / Objective-C++.
//
// Key architectural differences from the CUDA version:
//   • Apple Silicon unified memory — no H2D/D2H copies; CPU and GPU share
//     the same physical memory via MTLResourceStorageModeShared.
//   • Command buffers replace CUDA streams — work is encoded into a
//     MTLCommandBuffer and committed; the GPU schedules it.
//   • Metal shader library compiled at build time (.metallib), embedded in
//     the plugin binary and loaded at init.
//   • Atomic floats use atomic_fetch_add_explicit (Metal 3.0+, Apple Silicon).
//
// ============================================================================

#include "ghost_metal.h"
#include "lens.h"
#include "ghost.h"
#include "trace.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Embedded metallib — linked via assembly (.incbin) at build time.
// See CMakeLists.txt for the build steps that produce embed_metallib.s.
// ---------------------------------------------------------------------------
extern "C" {
    extern const char _binary_flare_shaders_metallib_start[];
    extern const char _binary_flare_shaders_metallib_end[];
}

// ===========================================================================
// GPU-side POD structs — must match the structs in flare_shaders.metal
// ===========================================================================

struct GPUPair
{
    int   surf_a, surf_b;
    float area_boost;
    int   tile_cx, tile_cy;
    float color_r, color_g, color_b;
    float offset_x, offset_y;
    float scale;
};

struct GPUSource  { float angle_x, angle_y, r, g, b; };
struct GPUSample  { float u, v; };
struct GPUSpectralSampleDev { float lambda, rw, gw, bw; };

// Must match the Metal shader struct exactly.
struct GhostParams
{
    int   n_surfs;
    float sensor_z;
    int   n_sources;
    int   n_grid;
    float front_R;
    float start_z;
    float sensor_half_w;
    float sensor_half_h;
    int   width;
    int   height;
    float gain;
    float ray_weight;
    int   n_spec;
    float fmt_w;
    float fmt_h;
    float fmt_x0_in_buf;
    float fmt_y0_in_buf;
    int   spectral_jitter;
    float spec_bin_width;
    float cie_norm_r;
    float cie_norm_g;
    float cie_norm_b;
    uint32_t spec_jitter_seed;
};

// GPU-side Surface POD — must match the Metal shader struct.
// We need is_stop as int (not bool) for Metal struct alignment.
struct GPUSurface
{
    float radius;
    float thickness;
    float ior;
    float abbe_v;
    float semi_aperture;
    int   coating;
    int   is_stop;
    float z;
};

// ===========================================================================
// Helper: ARC bridge macros for managing ObjC objects inside void* fields
// ===========================================================================

static inline void retainObj(void*& field, id obj)
{
    if (field) CFRelease(field);
    field = obj ? (void*)CFBridgingRetain(obj) : nullptr;
}

static inline void releaseObj(void*& field)
{
    if (field) { CFRelease(field); field = nullptr; }
}

template<typename T>
static inline T bridgeGet(void* field)
{
    return (__bridge T)field;
}

// ===========================================================================
// MetalBufferCache implementation
// ===========================================================================

void MetalBufferCache::release()
{
    releaseObj(buf_surfs);   surfs_bytes = 0;
    releaseObj(buf_pairs);   pairs_bytes = 0;
    releaseObj(buf_src);     src_bytes   = 0;
    releaseObj(buf_grid);    grid_bytes  = 0;
    releaseObj(buf_spec);    spec_bytes  = 0;
    releaseObj(buf_params);

    releaseObj(buf_out_r);
    releaseObj(buf_out_g);
    releaseObj(buf_out_b);
    releaseObj(buf_out_a);
    out_floats = 0;

    releaseObj(buf_out_fp16);
    buf_out_fp16_elems = 0;

    releaseObj(blur_a);
    releaseObj(blur_b);
    blur_floats = 0;

    releaseObj(pso_ghost);
    releaseObj(pso_alpha);
    releaseObj(pso_fp32_to_fp16);
    releaseObj(pso_clear);
    releaseObj(pso_blur_h);
    releaseObj(pso_blur_v);
    releaseObj(pso_blur_h_naive);
    releaseObj(pso_soft_clip);

    releaseObj(mtl_library);
    releaseObj(mtl_queue);
    releaseObj(mtl_device);

    cached_grid_count = 0;
    cached_ray_grid = -1;
}

// ---------------------------------------------------------------------------
// Grow-or-reuse helper for shared-mode Metal buffers.
// Returns the raw CPU pointer (valid for both CPU and GPU on Apple Silicon).
// ---------------------------------------------------------------------------

static bool ensure_buffer(id<MTLDevice> dev, void*& field, std::size_t& cap,
                           std::size_t need, const char* tag)
{
    if (need <= cap && field)
        return true;

    releaseObj(field);
    cap = 0;

    id<MTLBuffer> buf = [dev newBufferWithLength:need
                                         options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "FlareSim Metal: failed to allocate %zu bytes for %s\n", need, tag);
        return false;
    }
    retainObj(field, buf);
    cap = need;
    return true;
}

// Typed variant for output channel buffers.
static bool ensure_output(id<MTLDevice> dev, void*& field, std::size_t n_floats,
                            const char* tag)
{
    std::size_t dummy = 0;
    std::size_t need  = n_floats * sizeof(float);
    // We don't track capacity separately for outputs — just use dummy.
    releaseObj(field);
    id<MTLBuffer> buf = [dev newBufferWithLength:need
                                         options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "FlareSim Metal: failed to allocate output %s\n", tag);
        return false;
    }
    retainObj(field, buf);
    return true;
}

// ===========================================================================
// metal_init — create device, queue, compile shaders
// ===========================================================================

static id<MTLComputePipelineState> makePSO(id<MTLDevice> dev,
                                            id<MTLLibrary> lib,
                                            const char* name,
                                            std::string* out_error)
{
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
    if (!fn) {
        NSString* msg = [NSString stringWithFormat:@"FlareSim: shader function '%s' not found", name];
        fprintf(stderr, "%s\n", msg.UTF8String);
        if (out_error) *out_error = msg.UTF8String;
        return nil;
    }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn
                                                                        error:&err];
    if (!pso) {
        NSString* msg = [NSString stringWithFormat:@"FlareSim: PSO creation failed for '%s': %@",
                         name, err.localizedDescription];
        fprintf(stderr, "%s\n", msg.UTF8String);
        if (out_error) *out_error = msg.UTF8String;
        return nil;
    }
    return pso;
}

bool metal_init(MetalBufferCache& cache, std::string* out_error)
{
    if (cache.mtl_device)
        return true;  // already initialised

    @autoreleasepool {
        // Get the default Metal device (Apple Silicon GPU)
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            const char* msg = "FlareSim: no Metal GPU found. "
                              "FlareSim requires Apple Silicon (M1 or later).";
            fprintf(stderr, "%s\n", msg);
            if (out_error) *out_error = msg;
            return false;
        }


        // Load embedded metallib
        const char* start = _binary_flare_shaders_metallib_start;
        const char* end   = _binary_flare_shaders_metallib_end;
        std::size_t lib_size = (std::size_t)(end - start);

        if (lib_size == 0) {
            const char* msg = "FlareSim: embedded metallib is empty";
            fprintf(stderr, "%s\n", msg);
            if (out_error) *out_error = msg;
            return false;
        }

        NSData* lib_data = [NSData dataWithBytesNoCopy:(void*)start
                                                length:lib_size
                                          freeWhenDone:NO];
        NSError* err = nil;
        dispatch_data_t dd = dispatch_data_create(lib_data.bytes, lib_data.length,
                                                   nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> library = [dev newLibraryWithData:dd error:&err];
        if (!library) {
            NSString* msg = [NSString stringWithFormat:
                @"FlareSim: failed to load metallib: %@", err.localizedDescription];
            fprintf(stderr, "%s\n", msg.UTF8String);
            if (out_error) *out_error = msg.UTF8String;
            return false;
        }

        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) {
            const char* msg = "FlareSim: failed to create Metal command queue";
            fprintf(stderr, "%s\n", msg);
            if (out_error) *out_error = msg;
            return false;
        }

        // Create compute pipeline states
        id<MTLComputePipelineState> pso_ghost  = makePSO(dev, library, "ghost_kernel", out_error);
        id<MTLComputePipelineState> pso_alpha  = makePSO(dev, library, "alpha_kernel", out_error);
        id<MTLComputePipelineState> pso_fp16   = makePSO(dev, library, "fp32_to_fp16_kernel", out_error);
        id<MTLComputePipelineState> pso_clear  = makePSO(dev, library, "clear_buffer_kernel", out_error);
        id<MTLComputePipelineState> pso_blur_h = makePSO(dev, library, "box_blur_h_prefix_kernel", out_error);
        id<MTLComputePipelineState> pso_blur_v = makePSO(dev, library, "box_blur_v_sliding_kernel", out_error);
        id<MTLComputePipelineState> pso_blur_hn = makePSO(dev, library, "box_blur_h_naive_kernel", out_error);
        id<MTLComputePipelineState> pso_sc     = makePSO(dev, library, "soft_clip_kernel", out_error);

        if (!pso_ghost || !pso_alpha || !pso_fp16 || !pso_clear ||
            !pso_blur_h || !pso_blur_v || !pso_blur_hn || !pso_sc)
            return false;

        // Store everything (retain via CFBridgingRetain)
        retainObj(cache.mtl_device,  dev);
        retainObj(cache.mtl_queue,   queue);
        retainObj(cache.mtl_library, library);
        retainObj(cache.pso_ghost,         pso_ghost);
        retainObj(cache.pso_alpha,         pso_alpha);
        retainObj(cache.pso_fp32_to_fp16,  pso_fp16);
        retainObj(cache.pso_clear,         pso_clear);
        retainObj(cache.pso_blur_h,        pso_blur_h);
        retainObj(cache.pso_blur_v,        pso_blur_v);
        retainObj(cache.pso_blur_h_naive,  pso_blur_hn);
        retainObj(cache.pso_soft_clip,     pso_sc);

        return true;
    }
}

// ===========================================================================
// ensure_fp16_output — FP16 staging in unified memory
// ===========================================================================

void* ensure_fp16_output(MetalBufferCache& cache, std::size_t n_px)
{
    if (n_px <= cache.buf_out_fp16_elems && cache.buf_out_fp16)
    {
        id<MTLBuffer> buf = bridgeGet<id<MTLBuffer>>(cache.buf_out_fp16);
        return buf.contents;
    }

    releaseObj(cache.buf_out_fp16);
    cache.buf_out_fp16_elems = 0;

    id<MTLDevice> dev = bridgeGet<id<MTLDevice>>(cache.mtl_device);
    std::size_t bytes = 4 * n_px * sizeof(uint16_t);
    id<MTLBuffer> buf = [dev newBufferWithLength:bytes
                                         options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "FlareSim Metal: failed to allocate FP16 staging (%zu bytes)\n", bytes);
        // Fallback to plain malloc
        void* ptr = malloc(bytes);
        cache.buf_out_fp16_elems = n_px;
        return ptr;
    }
    retainObj(cache.buf_out_fp16, buf);
    cache.buf_out_fp16_elems = n_px;
    return buf.contents;
}

// ===========================================================================
// convert_fp16_scanline — portable FP16 → FP32
// ===========================================================================

// IEEE 754 half → float (portable, no Metal headers needed by caller)
static inline float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t r = sign;
            float f; memcpy(&f, &r, 4); return f;
        }
        // Denormal
        float f2 = (float)mant / 1024.0f;
        f2 *= (1.0f / 16384.0f);  // 2^-14
        return (sign ? -f2 : f2);
    }
    if (exp == 31) {
        uint32_t r = sign | 0x7f800000u | ((uint32_t)mant << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    uint32_t r = sign | ((exp + 112) << 23) | ((uint32_t)mant << 13);
    float f; memcpy(&f, &r, 4); return f;
}

__attribute__((visibility("default")))
void convert_fp16_scanline(const void* src, float* dst, int n)
{
    const uint16_t* h = static_cast<const uint16_t*>(src);
    for (int i = 0; i < n; ++i)
        dst[i] = half_to_float(h[i]);
}

// ===========================================================================
// launch_ghost_metal — main GPU ghost rendering entry point
// ===========================================================================

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
    bool                            skip_readback,
    std::string*                    out_error,
    const std::vector<float>*       pair_colors,
    const std::vector<float>*       pair_offsets,
    const std::vector<float>*       pair_scales)
{
    @autoreleasepool {

    if (active_pairs.empty() || sources.empty()) return;


    const int n_surfs = lens.num_surfaces();
    if (n_surfs <= 0) return;

    constexpr int MAX_SURFACES = 64;
    if (n_surfs > MAX_SURFACES) {
        fprintf(stderr, "FlareSim WARNING: lens has %d surfaces, max %d supported\n",
                n_surfs, MAX_SURFACES);
    }

    // Ensure Metal is initialised
    if (!cache.mtl_device) {
        if (!metal_init(cache, out_error))
            return;
    }

    id<MTLDevice>       dev   = bridgeGet<id<MTLDevice>>(cache.mtl_device);
    id<MTLCommandQueue> queue = bridgeGet<id<MTLCommandQueue>>(cache.mtl_queue);

    const int n_pairs = (int)active_pairs.size();
    const int n_src   = (int)sources.size();

    // ---- Build entrance-pupil grid ----

    auto wang_hash = [](uint32_t s) -> uint32_t {
        s = (s ^ 61u) ^ (s >> 16u);
        s *= 9u;
        s ^= s >> 4u;
        s *= 0x27d4eb2du;
        s ^= s >> 15u;
        return s;
    };
    auto halton2 = [](uint32_t n) -> float {
        n = (n << 16u) | (n >> 16u);
        n = ((n & 0x00ff00ffu) << 8u) | ((n & 0xff00ff00u) >> 8u);
        n = ((n & 0x0f0f0f0fu) << 4u) | ((n & 0xf0f0f0f0u) >> 4u);
        n = ((n & 0x33333333u) << 2u) | ((n & 0xccccccccu) >> 2u);
        n = ((n & 0x55555555u) << 1u) | ((n & 0xaaaaaaaau) >> 1u);
        return (float)n * (1.0f / 4294967296.0f);
    };
    auto halton3 = [](uint32_t n) -> float {
        float r = 0.0f, f = 1.0f / 3.0f;
        while (n > 0) { r += (n % 3u) * f; n /= 3u; f /= 3.0f; }
        return r;
    };

    const bool seed_matches = (config.pupil_jitter != 1) ||
                              (cache.cached_jitter_seed == config.pupil_jitter_seed);
    const bool grid_cached =
        (cache.cached_ray_grid        == config.ray_grid) &&
        (cache.cached_aperture_blades == config.aperture_blades) &&
        (cache.cached_aperture_rot    == config.aperture_rotation_deg) &&
        (cache.cached_pupil_jitter    == config.pupil_jitter) &&
        seed_matches &&
        (cache.cached_grid_count      > 0);

    int n_grid = 0;
    std::vector<GPUSample> grid_samples;

    if (grid_cached) {
        n_grid = cache.cached_grid_count;
    } else {
        const int   N         = config.ray_grid;
        const int   n_blades  = config.aperture_blades;
        const float rot_rad   = config.aperture_rotation_deg * ((float)M_PI / 180.0f);
        const bool  polygonal = (n_blades >= 3);
        const float apothem   = polygonal ? std::cos((float)M_PI / n_blades) : 1.0f;
        const float sector_ang = polygonal ? (2.0f * (float)M_PI / n_blades) : 1.0f;
        const int      jitter      = config.pupil_jitter;
        const uint32_t seed_offset = (uint32_t)config.pupil_jitter_seed * 1000003u;

        grid_samples.reserve((size_t)N * N);
        for (int k = 0; k < N * N; ++k) {
            const int gx = k % N;
            const int gy = k / N;

            float u, v;
            if (jitter == 2) {
                u = halton2((uint32_t)k) * 2.0f - 1.0f;
                v = halton3((uint32_t)k) * 2.0f - 1.0f;
            } else {
                float ju = (jitter == 1) ? wang_hash((uint32_t)k + seed_offset)
                                               / 4294967296.0f : 0.5f;
                float jv = (jitter == 1) ? wang_hash((uint32_t)k + (uint32_t)(N*N) + seed_offset)
                                               / 4294967296.0f : 0.5f;
                u = ((gx + ju) / N) * 2.0f - 1.0f;
                v = ((gy + jv) / N) * 2.0f - 1.0f;
            }
            float r2 = u*u + v*v;
            if (r2 > 1.0f) continue;
            if (polygonal) {
                float angle  = std::atan2(v, u) - rot_rad;
                float sector = std::fmod(angle, sector_ang);
                if (sector < 0.0f) sector += sector_ang;
                if (std::sqrt(r2) * std::cos(sector - sector_ang * 0.5f) > apothem)
                    continue;
            }
            grid_samples.push_back({u, v});
        }
        n_grid = (int)grid_samples.size();
    }
    if (n_grid == 0) return;

    const float ray_weight = 1.0f / n_grid;
    const float front_R    = lens.surfaces[0].semi_aperture;
    const float start_z    = lens.surfaces[0].z - 20.0f;

    // ---- Pack GPU-side arrays ----

    std::vector<GPUPair> gpu_pairs(n_pairs);
    for (int i = 0; i < n_pairs; ++i) {
        float cr = 1.0f, cg = 1.0f, cb = 1.0f;
        float ox = 0.0f, oy = 0.0f;
        if (pair_colors && (int)pair_colors->size() >= (i + 1) * 3) {
            cr = (*pair_colors)[i*3]; cg = (*pair_colors)[i*3+1]; cb = (*pair_colors)[i*3+2];
        }
        if (pair_offsets && (int)pair_offsets->size() >= (i + 1) * 2) {
            ox = (*pair_offsets)[i*2]; oy = (*pair_offsets)[i*2+1];
        }
        float sc = 1.0f;
        if (pair_scales && (int)pair_scales->size() > i) sc = (*pair_scales)[i];
        gpu_pairs[i] = { active_pairs[i].surf_a, active_pairs[i].surf_b,
                         pair_area_boosts[i], -99999, -99999,
                         cr, cg, cb, ox, oy, sc };
    }

    // Sort by descending cost
    std::sort(gpu_pairs.begin(), gpu_pairs.end(),
              [](const GPUPair& a, const GPUPair& b) {
                  return (a.surf_b - a.surf_a) > (b.surf_b - b.surf_a);
              });

    // Tile centroids
    for (int i = 0; i < n_pairs; ++i) {
        Ray probe;
        probe.origin = Vec3f(0, 0, start_z);
        probe.dir    = Vec3f(0, 0, 1);
        TraceResult tr = trace_ghost_ray(probe, lens, gpu_pairs[i].surf_a,
                                          gpu_pairs[i].surf_b, 550.0f);
        if (tr.valid && std::isfinite(tr.position.x) && std::isfinite(tr.position.y)) {
            float px = (tr.position.x / (2.0f * sensor_half_w) + 0.5f)
                       * (float)fmt_w + (float)fmt_x0_in_buf;
            float py = (tr.position.y / (2.0f * sensor_half_h) + 0.5f)
                       * (float)fmt_h + (float)fmt_y0_in_buf;
            gpu_pairs[i].tile_cx = (int)std::round(px);
            gpu_pairs[i].tile_cy = (int)std::round(py);
        }
    }

    std::vector<GPUSource> gpu_src(n_src);
    for (int i = 0; i < n_src; ++i)
        gpu_src[i] = { sources[i].angle_x, sources[i].angle_y,
                       sources[i].r, sources[i].g, sources[i].b };

    // Spectral samples
    std::vector<GPUSpectralSampleDev> spectral_cpu;
    float cie_sum_r = 1.0f, cie_sum_g = 1.0f, cie_sum_b = 1.0f;
    float spec_bin_width = 0.0f;
    {
        const int ns = std::max(3, config.spectral_samples);
        spectral_cpu.resize(ns);

        auto cie_r = [](float l) {
            float a = (l - 600.0f) / 70.0f, b = (l - 450.0f) / 30.0f;
            return std::max(0.0f, 0.63f * std::exp(-0.5f*a*a) + 0.22f * std::exp(-0.5f*b*b));
        };
        auto cie_g = [](float l) {
            float a = (l - 545.0f) / 55.0f;
            return std::max(0.0f, std::exp(-0.5f*a*a));
        };
        auto cie_b = [](float l) {
            float a = (l - 445.0f) / 45.0f;
            return std::max(0.0f, std::exp(-0.5f*a*a));
        };

        if (ns == 3) {
            spectral_cpu[0] = { config.wavelengths[0], 1.0f, 0.0f, 0.0f };
            spectral_cpu[1] = { config.wavelengths[1], 0.0f, 1.0f, 0.0f };
            spectral_cpu[2] = { config.wavelengths[2], 0.0f, 0.0f, 1.0f };
            spec_bin_width = 100.0f;
            cie_sum_r = cie_r(450.0f) + cie_r(550.0f) + cie_r(650.0f);
            cie_sum_g = cie_g(450.0f) + cie_g(550.0f) + cie_g(650.0f);
            cie_sum_b = cie_b(450.0f) + cie_b(550.0f) + cie_b(650.0f);
        } else {
            spec_bin_width = 300.0f / (ns - 1);
            float sum_r = 0, sum_g = 0, sum_b = 0;
            for (int i = 0; i < ns; ++i) {
                float lam = 400.0f + (300.0f * i) / (ns - 1);
                spectral_cpu[i].lambda = lam;
                spectral_cpu[i].rw = cie_r(lam);
                spectral_cpu[i].gw = cie_g(lam);
                spectral_cpu[i].bw = cie_b(lam);
                sum_r += spectral_cpu[i].rw;
                sum_g += spectral_cpu[i].gw;
                sum_b += spectral_cpu[i].bw;
            }
            cie_sum_r = sum_r;
            cie_sum_g = sum_g;
            cie_sum_b = sum_b;
            for (int i = 0; i < ns; ++i) {
                if (sum_r > 1e-9f) spectral_cpu[i].rw /= sum_r;
                if (sum_g > 1e-9f) spectral_cpu[i].gw /= sum_g;
                if (sum_b > 1e-9f) spectral_cpu[i].bw /= sum_b;
            }
        }
    }
    const int n_spec = (int)spectral_cpu.size();

    // Convert Surface → GPUSurface
    std::vector<GPUSurface> gpu_surfs(n_surfs);
    for (int i = 0; i < n_surfs; ++i) {
        const Surface& s = lens.surfaces[i];
        gpu_surfs[i] = { s.radius, s.thickness, s.ior, s.abbe_v,
                         s.semi_aperture, s.coating, s.is_stop ? 1 : 0, s.z };
    }


    // ---- Allocate / grow Metal buffers ----

    const size_t n_px = (size_t)width * height;

    if (!ensure_buffer(dev, cache.buf_surfs, cache.surfs_bytes,
                        n_surfs * sizeof(GPUSurface), "surfs")) return;
    if (!ensure_buffer(dev, cache.buf_pairs, cache.pairs_bytes,
                        n_pairs * sizeof(GPUPair), "pairs")) return;
    if (!ensure_buffer(dev, cache.buf_src, cache.src_bytes,
                        n_src * sizeof(GPUSource), "src")) return;
    if (!grid_cached) {
        if (!ensure_buffer(dev, cache.buf_grid, cache.grid_bytes,
                            n_grid * sizeof(GPUSample), "grid")) return;
    }
    if (!ensure_buffer(dev, cache.buf_spec, cache.spec_bytes,
                        n_spec * sizeof(GPUSpectralSampleDev), "spec")) return;

    // Params uniform
    if (!cache.buf_params) {
        std::size_t dummy = 0;
        if (!ensure_buffer(dev, cache.buf_params, dummy, sizeof(GhostParams), "params"))
            return;
    }

    // Output buffers
    if (n_px > cache.out_floats) {
        releaseObj(cache.buf_out_r);
        releaseObj(cache.buf_out_g);
        releaseObj(cache.buf_out_b);
        releaseObj(cache.buf_out_a);
        cache.out_floats = 0;

        if (!ensure_output(dev, cache.buf_out_r, n_px, "out_r")) return;
        if (!ensure_output(dev, cache.buf_out_g, n_px, "out_g")) return;
        if (!ensure_output(dev, cache.buf_out_b, n_px, "out_b")) return;
        if (!ensure_output(dev, cache.buf_out_a, n_px, "out_a")) return;
        cache.out_floats = n_px;
    }


    // ---- Upload: copy data into unified-memory buffers ----
    // On Apple Silicon this is a plain memcpy (same physical memory).

    auto copyTo = [](void* buf_field, const void* data, size_t bytes) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buf_field;
        memcpy(buf.contents, data, bytes);
    };

    copyTo(cache.buf_surfs, gpu_surfs.data(), n_surfs * sizeof(GPUSurface));
    copyTo(cache.buf_pairs, gpu_pairs.data(), n_pairs * sizeof(GPUPair));
    copyTo(cache.buf_src,   gpu_src.data(),   n_src * sizeof(GPUSource));
    if (!grid_cached) {
        copyTo(cache.buf_grid, grid_samples.data(), n_grid * sizeof(GPUSample));
        cache.cached_grid_count      = n_grid;
        cache.cached_ray_grid        = config.ray_grid;
        cache.cached_aperture_blades = config.aperture_blades;
        cache.cached_aperture_rot    = config.aperture_rotation_deg;
        cache.cached_pupil_jitter    = config.pupil_jitter;
        cache.cached_jitter_seed     = config.pupil_jitter_seed;
    }
    copyTo(cache.buf_spec, spectral_cpu.data(), n_spec * sizeof(GPUSpectralSampleDev));

    // Fill params struct
    {
        GhostParams p = {};
        p.n_surfs       = std::min(n_surfs, MAX_SURFACES);
        p.sensor_z      = lens.sensor_z;
        p.n_sources     = n_src;
        p.n_grid        = n_grid;
        p.front_R       = front_R;
        p.start_z       = start_z;
        p.sensor_half_w = sensor_half_w;
        p.sensor_half_h = sensor_half_h;
        p.width         = width;
        p.height        = height;
        p.gain          = config.gain;
        p.ray_weight    = ray_weight;
        p.n_spec        = n_spec;
        p.fmt_w         = (float)fmt_w;
        p.fmt_h         = (float)fmt_h;
        p.fmt_x0_in_buf = (float)fmt_x0_in_buf;
        p.fmt_y0_in_buf = (float)fmt_y0_in_buf;
        p.spectral_jitter = config.spectral_jitter ? 1 : 0;
        p.spec_bin_width  = spec_bin_width * std::max(0.0f, config.spectral_jitter_scale);
        p.cie_norm_r      = cie_sum_r;
        p.cie_norm_g      = cie_sum_g;
        p.cie_norm_b      = cie_sum_b;
        p.spec_jitter_seed = (uint32_t)config.spectral_jitter_seed;
        copyTo(cache.buf_params, &p, sizeof(GhostParams));
    }

    // Zero output buffers
    memset(bridgeGet<id<MTLBuffer>>(cache.buf_out_r).contents, 0, n_px * sizeof(float));
    memset(bridgeGet<id<MTLBuffer>>(cache.buf_out_g).contents, 0, n_px * sizeof(float));
    memset(bridgeGet<id<MTLBuffer>>(cache.buf_out_b).contents, 0, n_px * sizeof(float));


    // ---- Dispatch ghost kernel ----
    constexpr int BLOCK_SIZE = 512;
    int grid_blocks = (n_grid + BLOCK_SIZE - 1) / BLOCK_SIZE;

    {
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        id<MTLComputePipelineState> pso = bridgeGet<id<MTLComputePipelineState>>(cache.pso_ghost);
        [enc setComputePipelineState:pso];

        [enc setBuffer:bridgeGet<id<MTLBuffer>>(cache.buf_surfs) offset:0 atIndex:0];
        [enc setBuffer:bridgeGet<id<MTLBuffer>>(cache.buf_pairs) offset:0 atIndex:1];
        [enc setBuffer:bridgeGet<id<MTLBuffer>>(cache.buf_src)   offset:0 atIndex:2];
        [enc setBuffer:bridgeGet<id<MTLBuffer>>(cache.buf_grid)  offset:0 atIndex:3];
        [enc setBuffer:bridgeGet<id<MTLBuffer>>(cache.buf_spec)  offset:0 atIndex:4];
        [enc setBuffer:bridgeGet<id<MTLBuffer>>(cache.buf_out_r) offset:0 atIndex:5];
        [enc setBuffer:bridgeGet<id<MTLBuffer>>(cache.buf_out_g) offset:0 atIndex:6];
        [enc setBuffer:bridgeGet<id<MTLBuffer>>(cache.buf_out_b) offset:0 atIndex:7];
        [enc setBuffer:bridgeGet<id<MTLBuffer>>(cache.buf_params) offset:0 atIndex:8];

        MTLSize threadgroups = MTLSizeMake(n_pairs * n_src, grid_blocks, 1);
        MTLSize threadsPerGroup = MTLSizeMake(BLOCK_SIZE, 1, 1);

        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.error) {
            NSString* msg = [NSString stringWithFormat:
                @"FlareSim Metal: ghost kernel error: %@", cmdBuf.error.localizedDescription];
            fprintf(stderr, "%s\n", msg.UTF8String);
            if (out_error) *out_error = msg.UTF8String;
        }
    }

    // ---- Readback: on Apple Silicon, data is already in CPU-visible memory ----
    if (!skip_readback && out_r && out_g && out_b) {
        memcpy(out_r, bridgeGet<id<MTLBuffer>>(cache.buf_out_r).contents, n_px * sizeof(float));
        memcpy(out_g, bridgeGet<id<MTLBuffer>>(cache.buf_out_g).contents, n_px * sizeof(float));
        memcpy(out_b, bridgeGet<id<MTLBuffer>>(cache.buf_out_b).contents, n_px * sizeof(float));
    }


    } // @autoreleasepool
}

// ===========================================================================
// launch_blur_alpha_readback_metal
//
// Combined blur → alpha → FP16 conversion, all on GPU.
// On Apple Silicon: unified memory means the FP16 output is immediately
// visible to the CPU once the command buffer completes — zero DMA overhead.
// ===========================================================================

void launch_blur_alpha_readback_metal(
    float*            cpu_r,
    float*            cpu_g,
    float*            cpu_b,
    float*            cpu_a,
    int               w,
    int               h,
    int               blur_radius,
    int               blur_passes,
    MetalBufferCache& cache,
    std::string*      out_error,
    float             highlight_clip,
    float             highlight_knee,
    int               highlight_metric)
{
    @autoreleasepool {

    const size_t n_px = (size_t)w * h;
    if (n_px == 0) return;

    id<MTLDevice>       dev   = bridgeGet<id<MTLDevice>>(cache.mtl_device);
    id<MTLCommandQueue> queue = bridgeGet<id<MTLCommandQueue>>(cache.mtl_queue);

    // Ensure blur scratch buffers
    if (n_px > cache.blur_floats) {
        releaseObj(cache.blur_a);
        releaseObj(cache.blur_b);
        cache.blur_floats = 0;
        if (!ensure_output(dev, cache.blur_a, n_px, "blur_a")) return;
        if (!ensure_output(dev, cache.blur_b, n_px, "blur_b")) return;
        cache.blur_floats = n_px;
    }

    // Ensure alpha buffer
    if (n_px > cache.out_floats || !cache.buf_out_a) {
        releaseObj(cache.buf_out_a);
        if (!ensure_output(dev, cache.buf_out_a, n_px, "out_a")) return;
    }

    // Ensure FP16 staging
    if (n_px > cache.buf_out_fp16_elems) {
        releaseObj(cache.buf_out_fp16);
        cache.buf_out_fp16_elems = 0;
        std::size_t bytes = 4 * n_px * sizeof(uint16_t);
        id<MTLBuffer> fp16buf = [dev newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
        if (!fp16buf) return;
        retainObj(cache.buf_out_fp16, fp16buf);
        cache.buf_out_fp16_elems = n_px;
    }

    const bool do_blur = (blur_radius >= 1 && blur_passes >= 1);
    const bool do_clip = (highlight_clip > 0.0f);

    id<MTLBuffer> d_ch[3] = {
        bridgeGet<id<MTLBuffer>>(cache.buf_out_r),
        bridgeGet<id<MTLBuffer>>(cache.buf_out_g),
        bridgeGet<id<MTLBuffer>>(cache.buf_out_b)
    };
    id<MTLBuffer> d_alpha = bridgeGet<id<MTLBuffer>>(cache.buf_out_a);
    id<MTLBuffer> d_fp16  = bridgeGet<id<MTLBuffer>>(cache.buf_out_fp16);

    id<MTLBuffer> blurA = bridgeGet<id<MTLBuffer>>(cache.blur_a);
    id<MTLBuffer> blurB = bridgeGet<id<MTLBuffer>>(cache.blur_b);

    // Uniform buffers for blur/alpha params
    id<MTLBuffer> paramW   = [dev newBufferWithBytes:&w      length:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> paramH   = [dev newBufferWithBytes:&h      length:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> paramRad = [dev newBufferWithBytes:&blur_radius length:sizeof(int) options:MTLResourceStorageModeShared];
    int n_px_int = (int)n_px;
    id<MTLBuffer> paramN   = [dev newBufferWithBytes:&n_px_int length:sizeof(int) options:MTLResourceStorageModeShared];

    // Soft-clip params
    struct { float clip; float knee; int metric; int n_px; } sc_params = {
        highlight_clip,
        fmaxf(0.001f, fminf(highlight_knee, 0.999f)),
        highlight_metric,
        n_px_int
    };
    id<MTLBuffer> paramSC = do_clip
        ? [dev newBufferWithBytes:&sc_params length:sizeof(sc_params) options:MTLResourceStorageModeShared]
        : nil;

    id<MTLComputePipelineState> pso_blur_h = bridgeGet<id<MTLComputePipelineState>>(cache.pso_blur_h);
    id<MTLComputePipelineState> pso_blur_v = bridgeGet<id<MTLComputePipelineState>>(cache.pso_blur_v);
    id<MTLComputePipelineState> pso_alpha  = bridgeGet<id<MTLComputePipelineState>>(cache.pso_alpha);
    id<MTLComputePipelineState> pso_fp16   = bridgeGet<id<MTLComputePipelineState>>(cache.pso_fp32_to_fp16);

    // Max prefix-sum row width (threadgroup memory limit)
    constexpr int MAX_PREFIX_W = 12287;

    // Single command buffer for the entire pipeline
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

    // ---- Per-channel: blur(FP32) ----
    for (int ch = 0; ch < 3; ++ch)
    {
        if (do_blur)
        {
            // Copy channel to blur_a
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit copyFromBuffer:d_ch[ch] sourceOffset:0
                        toBuffer:blurA destinationOffset:0
                            size:n_px * sizeof(float)];
            [blit endEncoding];

            for (int p = 0; p < blur_passes; ++p)
            {
                // Horizontal blur: blurA → blurB
                {
                    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                    if (w <= MAX_PREFIX_W) {
                        [enc setComputePipelineState:pso_blur_h];
                        [enc setBuffer:blurA  offset:0 atIndex:0];
                        [enc setBuffer:blurB  offset:0 atIndex:1];
                        [enc setBuffer:paramW offset:0 atIndex:2];
                        [enc setBuffer:paramH offset:0 atIndex:3];
                        [enc setBuffer:paramRad offset:0 atIndex:4];
                        NSUInteger smem = ((NSUInteger)w + 1) * sizeof(float);
                        [enc setThreadgroupMemoryLength:smem atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(h, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    } else {
                        id<MTLComputePipelineState> pso_naive =
                            bridgeGet<id<MTLComputePipelineState>>(cache.pso_blur_h_naive);
                        [enc setComputePipelineState:pso_naive];
                        [enc setBuffer:blurA  offset:0 atIndex:0];
                        [enc setBuffer:blurB  offset:0 atIndex:1];
                        [enc setBuffer:paramW offset:0 atIndex:2];
                        [enc setBuffer:paramH offset:0 atIndex:3];
                        [enc setBuffer:paramRad offset:0 atIndex:4];
                        MTLSize tg = MTLSizeMake(32, 8, 1);
                        MTLSize grid = MTLSizeMake((w + 31)/32, (h + 7)/8, 1);
                        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                    }
                    [enc endEncoding];
                }
                // Vertical blur: blurB → blurA
                {
                    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso_blur_v];
                    [enc setBuffer:blurB  offset:0 atIndex:0];
                    [enc setBuffer:blurA  offset:0 atIndex:1];
                    [enc setBuffer:paramW offset:0 atIndex:2];
                    [enc setBuffer:paramH offset:0 atIndex:3];
                    [enc setBuffer:paramRad offset:0 atIndex:4];
                    int blocks = (w + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
            }

            // Copy blurred result back to channel buffer
            id<MTLBlitCommandEncoder> blit2 = [cmdBuf blitCommandEncoder];
            [blit2 copyFromBuffer:blurA sourceOffset:0
                         toBuffer:d_ch[ch] destinationOffset:0
                             size:n_px * sizeof(float)];
            [blit2 endEncoding];
        }
    }

    // ---- Soft highlight compression (after blur, before alpha) ----
    if (do_clip)
    {
        id<MTLComputePipelineState> pso_sc = bridgeGet<id<MTLComputePipelineState>>(cache.pso_soft_clip);
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso_sc];
        [enc setBuffer:d_ch[0]  offset:0 atIndex:0];
        [enc setBuffer:d_ch[1]  offset:0 atIndex:1];
        [enc setBuffer:d_ch[2]  offset:0 atIndex:2];
        [enc setBuffer:paramSC  offset:0 atIndex:3];
        int blocks = ((int)n_px + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // ---- Alpha: compute from RGB ----
    {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso_alpha];
        [enc setBuffer:d_ch[0] offset:0 atIndex:0];
        [enc setBuffer:d_ch[1] offset:0 atIndex:1];
        [enc setBuffer:d_ch[2] offset:0 atIndex:2];
        [enc setBuffer:d_alpha offset:0 atIndex:3];
        [enc setBuffer:paramN  offset:0 atIndex:4];
        int blocks = ((int)n_px + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // ---- FP32 → FP16 conversion for all 4 channels ----
    {
        id<MTLBuffer> d_all[4] = { d_ch[0], d_ch[1], d_ch[2], d_alpha };
        for (int ch = 0; ch < 4; ++ch) {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pso_fp16];
            [enc setBuffer:d_all[ch] offset:0 atIndex:0];
            [enc setBuffer:d_fp16    offset:(ch * n_px * sizeof(uint16_t)) atIndex:1];
            [enc setBuffer:paramN    offset:0 atIndex:2];
            int blocks = ((int)n_px + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (cmdBuf.error) {
        NSString* msg = [NSString stringWithFormat:
            @"FlareSim Metal: blur/alpha error: %@", cmdBuf.error.localizedDescription];
        fprintf(stderr, "%s\n", msg.UTF8String);
        if (out_error) *out_error = msg.UTF8String;
    }

    // ---- "Readback": on Apple Silicon, unified memory means the FP16 data
    //       is already at d_fp16.contents. The cpu_r/g/b/a pointers point
    //       into the same staging buffer returned by ensure_fp16_output(). ----
    // If the caller's pointers differ from the buffer contents (shouldn't
    // happen, but be safe), do a memcpy.
    uint16_t* fp16_ptr = (uint16_t*)d_fp16.contents;
    uint16_t* dst_r = reinterpret_cast<uint16_t*>(cpu_r);
    uint16_t* dst_g = reinterpret_cast<uint16_t*>(cpu_g);
    uint16_t* dst_b = reinterpret_cast<uint16_t*>(cpu_b);
    uint16_t* dst_a = reinterpret_cast<uint16_t*>(cpu_a);

    if (dst_r != fp16_ptr)
        memcpy(dst_r, fp16_ptr, n_px * sizeof(uint16_t));
    if (dst_g != fp16_ptr + n_px)
        memcpy(dst_g, fp16_ptr + n_px, n_px * sizeof(uint16_t));
    if (dst_b != fp16_ptr + 2 * n_px)
        memcpy(dst_b, fp16_ptr + 2 * n_px, n_px * sizeof(uint16_t));
    if (dst_a && dst_a != fp16_ptr + 3 * n_px)
        memcpy(dst_a, fp16_ptr + 3 * n_px, n_px * sizeof(uint16_t));

    } // @autoreleasepool
}
