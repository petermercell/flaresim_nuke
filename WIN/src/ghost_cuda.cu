// ============================================================================
// ghost_cuda.cu — GPU ghost rendering: device math, scatter kernel, launcher
//
// Design notes
// ─────────────
// • The device-side math is self-contained (no STL) to avoid portability
//   issues with std::clamp / std::sqrt in device code across toolchain
//   versions.  It mirrors the logic in fresnel.h, vec3.h and trace.cpp
//   exactly — keep them in sync if those files change.
//
// • Surface (from lens.h) is plain POD and is uploaded to device as-is via
//   cudaMemcpy.  Only the flat array is uploaded; LensSystem (which owns a
//   std::vector) stays on the CPU.
//
// • Thread layout:
//     gridDim  = (n_pairs * n_sources, grid_blocks, 1)
//     blockDim = (BLOCK_SIZE, 1, 1)
//   Each thread handles one entrance-pupil sample for one (pair, source) and
//   traces 3 wavelengths (R, G, B) sequentially.  AtomicAdd scatters results.
// ============================================================================

#include "ghost_cuda.h"
#include "blur_cuda.h"  // launch_blur_h_on_stream, launch_blur_v_on_stream
#include "lens.h"       // Surface (POD) — uploaded to device
#include "ghost.h"      // BrightPixel, GhostPair, GhostConfig (host side)
#include "trace.h"      // trace_ghost_ray — used for tile centroid computation

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Cross-platform symbol export for functions callable from pure C++ (.cpp).
//
// On Linux/macOS we need __attribute__((visibility("default"))) because
// FlareSim builds with -fvisibility=hidden by default.  On Windows MSVC
// nvcc rejects __attribute__ syntax outright (it parses it as junk and
// the parser then desyncs for the rest of the file — every other "error"
// in the build log is a cascade from this one), so we expand to nothing
// and let the standard CMake .def / dllexport mechanism handle exports.
// ---------------------------------------------------------------------------
#ifdef _WIN32
#  define FLARESIM_EXPORT
#else
#  define FLARESIM_EXPORT __attribute__((visibility("default")))
#endif

// ===========================================================================
// GpuBufferCache — implementation of the persistent buffer cache.
// ===========================================================================

void GpuBufferCache::release()
{
    // Ghost resources
    cudaFree(d_surfs);  d_surfs = nullptr;  surfs_bytes = 0;
    cudaFree(d_pairs);  d_pairs = nullptr;  pairs_bytes = 0;
    cudaFree(d_src);    d_src   = nullptr;  src_bytes   = 0;
    cudaFree(d_grid);   d_grid  = nullptr;  grid_bytes  = 0;
    cudaFree(d_spec);   d_spec  = nullptr;  spec_bytes  = 0;
    cudaFree(d_out_r);  d_out_r = nullptr;
    cudaFree(d_out_g);  d_out_g = nullptr;
    cudaFree(d_out_b);  d_out_b = nullptr;
    cudaFree(d_out_a);  d_out_a = nullptr;
    out_floats = 0;
    cudaFree(d_out_fp16);  d_out_fp16 = nullptr;
    d_out_fp16_elems = 0;

    // Blur resources
    cudaFree(blur_a);  blur_a = nullptr;
    cudaFree(blur_b);  blur_b = nullptr;
    blur_floats = 0;

    // CUDA streams
    if (compute_stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(compute_stream));
        compute_stream = nullptr;
    }
    if (copy_stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(copy_stream));
        copy_stream = nullptr;
    }

    // Pinned host staging
    if (h_pinned) { cudaFreeHost(h_pinned); h_pinned = nullptr; }
    h_pinned_elems = 0;

    // Cached pupil grid
    cudaFree(d_cached_grid);  d_cached_grid = nullptr;
    cached_grid_bytes = 0;  cached_grid_count = 0;
    cached_ray_grid = -1;
}

// ---------------------------------------------------------------------------
// ensure_pinned_output_fp16 — callable from pure C++ (FlareSim.cpp)
// Explicit visibility ensures the symbol is exported in the shared library.
// ---------------------------------------------------------------------------

FLARESIM_EXPORT
void* ensure_pinned_output_fp16(GpuBufferCache& cache, std::size_t n_px)
{
    if (n_px <= cache.h_pinned_elems && cache.h_pinned)
        return cache.h_pinned;

    if (cache.h_pinned) {
        cudaFreeHost(cache.h_pinned);
        cache.h_pinned = nullptr;
    }
    cache.h_pinned_elems = 0;

    // 4 channels × n_px × sizeof(uint16_t)
    std::size_t bytes = 4 * n_px * sizeof(uint16_t);
    cudaError_t e = cudaHostAlloc(&cache.h_pinned, bytes, cudaHostAllocDefault);
    if (e != cudaSuccess) {
        fprintf(stderr, "FlareSim: cudaHostAlloc FP16 failed (%s) — falling back to malloc\n",
                cudaGetErrorString(e));
        cache.h_pinned = malloc(bytes);
    }
    cache.h_pinned_elems = n_px;
    return cache.h_pinned;
}

// ---------------------------------------------------------------------------
// ensure_pinned_output_fp32 — FP32 variant for banding-free output.
// Same pinned buffer slot, but 4× the size (float vs uint16_t).
// ---------------------------------------------------------------------------

FLARESIM_EXPORT
void* ensure_pinned_output_fp32(GpuBufferCache& cache, std::size_t n_px)
{
    // FP32 needs 4 channels × n_px × sizeof(float) = 4× the FP16 size.
    // We reuse h_pinned / h_pinned_elems but encode the FP32 capacity
    // by storing a sentinel: h_pinned_elems is set to n_px but the
    // allocation is 4× larger.  To distinguish, we just always reallocate
    // if the caller asks for FP32 and the current capacity doesn't match.
    const std::size_t bytes = 4 * n_px * sizeof(float);

    // Check if current allocation is large enough for FP32
    // (h_pinned_elems tracks FP16 capacity, so FP32 needs 2× that count)
    const bool enough = cache.h_pinned &&
                        (cache.h_pinned_elems >= n_px * 2);
    if (enough) return cache.h_pinned;

    if (cache.h_pinned) {
        cudaFreeHost(cache.h_pinned);
        cache.h_pinned = nullptr;
    }
    cache.h_pinned_elems = 0;

    cudaError_t e = cudaHostAlloc(&cache.h_pinned, bytes, cudaHostAllocDefault);
    if (e != cudaSuccess) {
        fprintf(stderr, "FlareSim: cudaHostAlloc FP32 failed (%s) — falling back to malloc\n",
                cudaGetErrorString(e));
        cache.h_pinned = malloc(bytes);
    }
    // Store 2× n_px so FP16 callers won't accidentally reuse this oversized buffer
    // as if it were FP16-capacity.
    cache.h_pinned_elems = n_px * 2;
    return cache.h_pinned;
}

// ---------------------------------------------------------------------------
// convert_fp16_scanline — portable FP16 → FP32 conversion for host code.
// Uses CUDA's __half2float which nvcc compiles to efficient host code.
// ---------------------------------------------------------------------------

FLARESIM_EXPORT
void convert_fp16_scanline(const void* src, float* dst, int n)
{
    const __half* h = static_cast<const __half*>(src);
    for (int i = 0; i < n; ++i)
        dst[i] = __half2float(h[i]);
}

// ===========================================================================
// Device-side math — no STL, no std:: calls
// ===========================================================================

// ---- Minimal 3-vector ----

struct DVec3 { float x, y, z; };

__device__ __forceinline__ DVec3 dv(float x, float y, float z)
    { return {x, y, z}; }
__device__ __forceinline__ DVec3 dv_add(DVec3 a, DVec3 b)
    { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ __forceinline__ DVec3 dv_sub(DVec3 a, DVec3 b)
    { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ __forceinline__ DVec3 dv_scale(DVec3 a, float s)
    { return {a.x*s, a.y*s, a.z*s}; }
__device__ __forceinline__ DVec3 dv_neg(DVec3 a)
    { return {-a.x, -a.y, -a.z}; }
__device__ __forceinline__ float dv_dot(DVec3 a, DVec3 b)
    { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ DVec3 dv_normalize(DVec3 v)
{
    float inv = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return {v.x*inv, v.y*inv, v.z*inv};
}

struct DRay { DVec3 origin; DVec3 dir; };

// ---- Device-side Wang hash (fast 32-bit → uniform float) ----

__device__ __forceinline__
uint32_t d_wang_hash(uint32_t s)
{
    s = (s ^ 61u) ^ (s >> 16u);
    s *= 9u;
    s ^= s >> 4u;
    s *= 0x27d4eb2du;
    s ^= s >> 15u;
    return s;
}

__device__ __forceinline__
float d_hash_float(uint32_t s)  // → [0, 1)
{
    return (float)d_wang_hash(s) / 4294967296.0f;
}

// ---- Device-side CIE colour matching (Gaussian approximation) ----
// Must match the host-side lambdas in launch_ghost_cuda().

__device__ __forceinline__ float d_cie_r(float l)
{
    float a = (l - 600.0f) / 70.0f, b = (l - 450.0f) / 30.0f;
    return fmaxf(0.0f, 0.63f * expf(-0.5f*a*a) + 0.22f * expf(-0.5f*b*b));
}
__device__ __forceinline__ float d_cie_g(float l)
{
    float a = (l - 545.0f) / 55.0f;
    return fmaxf(0.0f, expf(-0.5f*a*a));
}
__device__ __forceinline__ float d_cie_b(float l)
{
    float a = (l - 445.0f) / 45.0f;
    return fmaxf(0.0f, expf(-0.5f*a*a));
}

// ---- Dispersion (Cauchy via Abbe number) ----

__device__ __forceinline__
float d_dispersion_ior(float n_d, float V_d, float lambda_nm)
{
    if (V_d < 0.1f || n_d <= 1.0001f) return n_d;

    // Precomputed constants — avoids 3 divisions per call.
    constexpr float lF = 486.13f, lC = 656.27f, ld = 587.56f;
    constexpr float inv_lF2 = 1.0f / (lF * lF);   // 4.228e-6
    constexpr float inv_lC2 = 1.0f / (lC * lC);   // 2.321e-6
    constexpr float inv_ld2 = 1.0f / (ld * ld);   // 2.898e-6
    constexpr float inv_diff = 1.0f / (inv_lF2 - inv_lC2);

    float dn = __fdividef(n_d - 1.0f, V_d);
    float B  = dn * inv_diff;
    float A  = n_d - B * inv_ld2;
    return A + __fdividef(B, lambda_nm * lambda_nm);
}

__device__ __forceinline__
float d_ior_at(const Surface& s, float lambda_nm)
{
    return d_dispersion_ior(s.ior, s.abbe_v, lambda_nm);
}

__device__ __forceinline__
float d_ior_before(const Surface* surfs, int idx, float lambda_nm)
{
    return (idx <= 0) ? 1.0f : d_ior_at(surfs[idx - 1], lambda_nm);
}

// ---- Fresnel reflectance ----

__device__ __forceinline__
float d_fresnel_reflectance(float cos_i, float n1, float n2)
{
    cos_i        = fabsf(cos_i);
    float eta    = __fdividef(n1, n2);
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f) return 1.0f;  // TIR
    float cos_t  = __fsqrt_rn(1.0f - sin2_t);
    float rs     = __fdividef(n1*cos_i - n2*cos_t, n1*cos_i + n2*cos_t);
    float rp     = __fdividef(n2*cos_i - n1*cos_t, n2*cos_i + n1*cos_t);
    return 0.5f * (rs*rs + rp*rp);
}

// Single-layer MgF2 AR coating (mirrors coating_reflectance in fresnel.h)
__device__ __forceinline__
float d_coating_reflectance(float cos_i, float n1, float n2,
                             float coating_n, float d_nm, float lambda_nm)
{
    float ratio1 = __fdividef(n1, coating_n);
    float sin2_c = ratio1 * ratio1 * (1.0f - cos_i*cos_i);
    if (sin2_c >= 1.0f) return d_fresnel_reflectance(cos_i, n1, n2);
    float cos_c  = __fsqrt_rn(1.0f - sin2_c);

    constexpr float TWO_PI = 2.0f * 3.14159265358979323846f;
    float delta  = TWO_PI * coating_n * d_nm * __fdividef(cos_c, lambda_nm);
    float r01    = __fdividef(n1*cos_i - coating_n*cos_c,
                              n1*cos_i + coating_n*cos_c);

    float ratio2 = __fdividef(coating_n, n2);
    float sin2_2 = ratio2 * ratio2 * (1.0f - cos_c*cos_c);
    if (sin2_2 >= 1.0f) return d_fresnel_reflectance(cos_i, n1, n2);
    float cos_2  = __fsqrt_rn(1.0f - sin2_2);
    float r12    = __fdividef(coating_n*cos_c - n2*cos_2,
                              coating_n*cos_c + n2*cos_2);

    float cos_2d = __cosf(2.0f * delta);
    float r01r12 = r01 * r12;
    float num    = r01*r01 + r12*r12 + 2.0f*r01r12*cos_2d;
    float den    = 1.0f   + r01*r01*r12*r12 + 2.0f*r01r12*cos_2d;
    float R      = __fdividef(num, den);
    return fminf(fmaxf(R, 0.0f), 1.0f);
}

__device__ __forceinline__
float d_surface_reflectance(float cos_i, float n1, float n2,
                             int coating_layers, float lambda_nm)
{
    if (coating_layers <= 0)
        return d_fresnel_reflectance(cos_i, n1, n2);

    // MgF2 quarter-wave at 550 nm — precomputed constant.
    constexpr float mgf2_n        = 1.38f;
    constexpr float design_lambda = 550.0f;
    constexpr float qw_thick      = design_lambda / (4.0f * mgf2_n);  // 99.637 nm
    float R = d_coating_reflectance(cos_i, n1, n2, mgf2_n, qw_thick, lambda_nm);
    for (int i = 1; i < coating_layers; ++i) R *= 0.25f;
    return fminf(fmaxf(R, 0.0f), 1.0f);
}

// ---- Ray–surface intersection ----
//
// Mirrors the CPU intersect_surface() in trace.cpp: flat shortcut at the top
// (applies regardless of surface_type), then a switch on surface_type.
// TORIC dispatches to a Newton-Raphson solver (M5 -- mirror of CPU M4).
//
// Warp-divergence note: every thread in a warp traces through the SAME
// surface at each step of the trace loop, so all threads land in the same
// switch arm together — no intra-warp divergence.  Cost vs. M2's pure-sphere
// kernel is just slightly larger SASS / I-cache footprint.

__device__ __forceinline__
bool d_intersect_spherical(const DRay& ray, const Surface& surf,
                            DVec3& hit, DVec3& norm)
{
    float R   = surf.radius;
    DVec3 ctr = dv(0.0f, 0.0f, surf.z + R);
    DVec3 oc  = dv_sub(ray.origin, ctr);

    float a    = dv_dot(ray.dir, ray.dir);
    float b    = 2.0f * dv_dot(oc, ray.dir);
    float c    = dv_dot(oc, oc) - R*R;
    float disc = b*b - 4.0f*a*c;
    if (disc < 0.0f) return false;

    float sd    = __fsqrt_rn(disc);
    float inv2a = __fdividef(0.5f, a);
    float t1    = (-b - sd) * inv2a;
    float t2    = (-b + sd) * inv2a;

    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1*ray.dir.z;
        float z2 = ray.origin.z + t2*ray.dir.z;
        t = (fabsf(z1 - surf.z) < fabsf(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f) t = t1;
    else if (t2 > 1e-6f) t = t2;
    else return false;

    hit = dv(ray.origin.x + ray.dir.x*t,
             ray.origin.y + ray.dir.y*t,
             ray.origin.z + ray.dir.z*t);

    if (!(hit.x*hit.x + hit.y*hit.y <= surf.semi_aperture*surf.semi_aperture))
        return false;

    float invR = __frcp_rn(fabsf(R));
    norm = dv((hit.x - ctr.x)*invR,
              (hit.y - ctr.y)*invR,
              (hit.z - ctr.z)*invR);
    if (dv_dot(norm, ray.dir) > 0.0f) norm = dv_neg(norm);
    return true;
}

// Cylinder, axis along X — curves in YZ.  surf.radius = Ry.
// Locus: y² + (z - cz)² = R²,  cz = surf.z + R.   X is invariant.
__device__ __forceinline__
bool d_intersect_cylinder_x(const DRay& ray, const Surface& surf,
                             DVec3& hit, DVec3& norm)
{
    float R  = surf.radius;
    float cz = surf.z + R;

    float oy = ray.origin.y;
    float oz = ray.origin.z - cz;
    float dy = ray.dir.y;
    float dz = ray.dir.z;

    float a = dy*dy + dz*dz;
    if (a < 1e-18f) return false;          // ray parallel to cylinder axis

    float b    = 2.0f * (oy*dy + oz*dz);
    float c    = oy*oy + oz*oz - R*R;
    float disc = b*b - 4.0f*a*c;
    if (disc < 0.0f) return false;

    float sd    = __fsqrt_rn(disc);
    float inv2a = __fdividef(0.5f, a);
    float t1    = (-b - sd) * inv2a;
    float t2    = (-b + sd) * inv2a;

    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1*ray.dir.z;
        float z2 = ray.origin.z + t2*ray.dir.z;
        t = (fabsf(z1 - surf.z) < fabsf(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f) t = t1;
    else if (t2 > 1e-6f) t = t2;
    else return false;

    hit = dv(ray.origin.x + ray.dir.x*t,
             ray.origin.y + ray.dir.y*t,
             ray.origin.z + ray.dir.z*t);

    if (!(hit.x*hit.x + hit.y*hit.y <= surf.semi_aperture*surf.semi_aperture))
        return false;

    // ∇F = (0, 2y, 2(z-cz));  X-component pinned to zero is what produces
    // horizontal streaks on reflection.
    float invR = __frcp_rn(fabsf(R));
    norm = dv(0.0f, hit.y*invR, (hit.z - cz)*invR);
    if (dv_dot(norm, ray.dir) > 0.0f) norm = dv_neg(norm);
    return true;
}

// Cylinder, axis along Y — curves in XZ.  surf.radius = Rx.   X↔Y mirror.
__device__ __forceinline__
bool d_intersect_cylinder_y(const DRay& ray, const Surface& surf,
                             DVec3& hit, DVec3& norm)
{
    float R  = surf.radius;
    float cz = surf.z + R;

    float ox = ray.origin.x;
    float oz = ray.origin.z - cz;
    float dx = ray.dir.x;
    float dz = ray.dir.z;

    float a = dx*dx + dz*dz;
    if (a < 1e-18f) return false;

    float b    = 2.0f * (ox*dx + oz*dz);
    float c    = ox*ox + oz*oz - R*R;
    float disc = b*b - 4.0f*a*c;
    if (disc < 0.0f) return false;

    float sd    = __fsqrt_rn(disc);
    float inv2a = __fdividef(0.5f, a);
    float t1    = (-b - sd) * inv2a;
    float t2    = (-b + sd) * inv2a;

    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1*ray.dir.z;
        float z2 = ray.origin.z + t2*ray.dir.z;
        t = (fabsf(z1 - surf.z) < fabsf(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f) t = t1;
    else if (t2 > 1e-6f) t = t2;
    else return false;

    hit = dv(ray.origin.x + ray.dir.x*t,
             ray.origin.y + ray.dir.y*t,
             ray.origin.z + ray.dir.z*t);

    if (!(hit.x*hit.x + hit.y*hit.y <= surf.semi_aperture*surf.semi_aperture))
        return false;

    float invR = __frcp_rn(fabsf(R));
    norm = dv(hit.x*invR, 0.0f, (hit.z - cz)*invR);
    if (dv_dot(norm, ray.dir) > 0.0f) norm = dv_neg(norm);
    return true;
}

// Toric: two-radii surface (M5 — GPU port of CPU intersect_toric).
//
// See trace.cpp for the full derivation.  Implicit form, apex-centered:
//   F(x, y, z') = (x² + (Rx - z')² + W² + y² - Ry²)²
//                 - 4 W² (x² + (Rx - z')²) = 0,    W = Rx - Ry.
//
// Newton-Raphson from the apex-plane intersection.  Falls back to spherical
// when |W| ≈ 0 (doubled-root quartic ⇒ ∇F vanishes on the surface).
//
// Register-pressure note: this function is heavier than sphere/cylinder
// (~30 floats live at peak inside the Newton loop), but every thread in
// a warp traces the same surface so loop-iteration counts stay coherent.
__device__ __forceinline__
bool d_intersect_toric(const DRay& ray, const Surface& surf,
                        DVec3& hit, DVec3& norm)
{
    const float Rx = surf.radius;
    const float Ry = surf.radius_y;
    const float W  = Rx - Ry;

    // Degenerate-W guard: doubled-root quartic ⇒ use spherical equivalent.
    if (fabsf(W) < 1e-3f)
    {
        Surface tmp = surf;
        tmp.radius = (Rx + Ry) * 0.5f;
        return d_intersect_spherical(ray, tmp, hit, norm);
    }

    const float ox = ray.origin.x;
    const float oy = ray.origin.y;
    const float oz = ray.origin.z - surf.z;       // apex-centered z
    const float dx = ray.dir.x;
    const float dy = ray.dir.y;
    const float dz = ray.dir.z;

    if (fabsf(dz) < 1e-12f) return false;

    // Initial guess: ray-plane intersection at the apex (z' = 0).
    float t = __fdividef(-oz, dz);
    if (!(t > 1e-6f)) return false;

    // Tolerance: F has units of length⁴ (Q ~ R², F ~ R⁴).
    const float scale  = fmaxf(fabsf(Rx), fabsf(Ry));
    const float scale2 = scale * scale;
    const float TOL_F  = scale2 * scale2 * 1e-8f;
    const float STEP_CAP = fabsf(t) + scale;
    const float W2  = W  * W;
    const float Ry2 = Ry * Ry;

    // Newton-Raphson loop.
    //
    // MAX_ITER tuning (M7): mirrors the CPU.  Newton converges in 3-5 iters
    // near axis; saddle / strongly off-axis pushes to ~8.  Instrumented
    // sweep on the CPU path (which uses identical IEEE math) gave
    // avg=7.89 max=12 on adversarial inputs; 16 gives 2x headroom.
    // Profile via the CPU path before re-tuning.
    constexpr int MAX_ITER = 16;
    bool converged = false;
    #pragma unroll 1
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        const float px = ox + t*dx;
        const float py = oy + t*dy;
        const float pz = oz + t*dz;
        const float Az = Rx - pz;
        const float Q  = px*px + Az*Az;
        const float dQ = 2.0f * (px*dx - Az*dz);
        const float G  = Q + W2 + py*py - Ry2;
        const float dG = dQ + 2.0f * py*dy;

        const float F  = G*G - 4.0f*W2*Q;
        if (fabsf(F) < TOL_F) { converged = true; break; }

        const float dF = 2.0f*G*dG - 4.0f*W2*dQ;
        if (fabsf(dF) < 1e-30f) return false;

        float step = __fdividef(F, dF);
        if (step >  STEP_CAP) step =  STEP_CAP;
        if (step < -STEP_CAP) step = -STEP_CAP;
        t -= step;
        if (!(t > 1e-6f)) return false;
        if (fabsf(step) < 1e-7f) { converged = true; break; }
    }
    if (!converged)
    {
        // Final residual sanity check — Newton may have stalled near a
        // non-root if the ray genuinely missed the surface.
        const float px = ox + t*dx;
        const float py = oy + t*dy;
        const float pz = oz + t*dz;
        const float Az = Rx - pz;
        const float Q  = px*px + Az*Az;
        const float G  = Q + W2 + py*py - Ry2;
        if (fabsf(G*G - 4.0f*W2*Q) > TOL_F * 1e3f) return false;
    }

    // Hit position (back to world coords).
    hit = dv(ox + t*dx, oy + t*dy, ray.origin.z + t*dz);

    // Aperture clip (NaN-safe form, matches the other helpers).
    if (!(hit.x*hit.x + hit.y*hit.y <= surf.semi_aperture*surf.semi_aperture))
        return false;

    // Normal as gradient ∇F at the hit, oriented to oppose the ray.
    //   ∂F/∂x = 4 x (Q + y² - Ry² - W²)
    //   ∂F/∂y = 4 y (Q + W² + y² - Ry²)
    //   ∂F/∂z = -4 (Rx - z') (Q + y² - Ry² - W²)
    {
        const float px = hit.x;
        const float py = hit.y;
        const float pz = hit.z - surf.z;
        const float Az = Rx - pz;
        const float Q  = px*px + Az*Az;
        const float K1 = Q + py*py - Ry2 - W2;
        const float G  = Q + py*py - Ry2 + W2;
        norm = dv_normalize(dv( 4.0f * px * K1,
                                4.0f * py * G,
                               -4.0f * Az * K1));
        if (dv_dot(norm, ray.dir) > 0.0f) norm = dv_neg(norm);
    }
    return true;
}

__device__ __forceinline__
bool d_intersect_surface(const DRay& ray, const Surface& surf,
                          DVec3& hit, DVec3& norm)
{
    // Flat shortcut — applies regardless of surface_type.
    if (fabsf(surf.radius) < 1e-6f)
    {
        if (fabsf(ray.dir.z) < 1e-12f) return false;
        float t = __fdividef(surf.z - ray.origin.z, ray.dir.z);
        if (!(t > 1e-6f)) return false;  // catches t<=0, NaN, and Inf

        hit = dv(ray.origin.x + ray.dir.x*t,
                 ray.origin.y + ray.dir.y*t,
                 ray.origin.z + ray.dir.z*t);

        if (!(hit.x*hit.x + hit.y*hit.y <= surf.semi_aperture*surf.semi_aperture))
            return false;

        norm = dv(0.0f, 0.0f, (ray.dir.z > 0) ? -1.0f : 1.0f);
        return true;
    }

    switch (surf.surface_type)
    {
        case SURF_SPHERICAL:   return d_intersect_spherical (ray, surf, hit, norm);
        case SURF_CYLINDER_X:  return d_intersect_cylinder_x(ray, surf, hit, norm);
        case SURF_CYLINDER_Y:  return d_intersect_cylinder_y(ray, surf, hit, norm);
        case SURF_TORIC:       return d_intersect_toric    (ray, surf, hit, norm);
        default:
            return false;
    }
}

// ---- Refraction / reflection ----

__device__ __forceinline__
bool d_refract(const DVec3& dir, const DVec3& norm,
               float n_ratio, DVec3& out)
{
    float cos_i  = -dv_dot(norm, dir);
    float sin2_t = n_ratio*n_ratio*(1.0f - cos_i*cos_i);
    if (sin2_t >= 1.0f) return false;
    float cos_t  = __fsqrt_rn(1.0f - sin2_t);
    float k      = n_ratio*cos_i - cos_t;
    float ox = dir.x*n_ratio + norm.x*k;
    float oy = dir.y*n_ratio + norm.y*k;
    float oz = dir.z*n_ratio + norm.z*k;
    float sq = ox*ox + oy*oy + oz*oz;
    if (sq < 1e-18f || !isfinite(sq)) return false;   // degenerate: discard
    float inv = rsqrtf(sq);
    out = dv(ox*inv, oy*inv, oz*inv);
    return true;
}

__device__ __forceinline__
bool d_reflect(const DVec3& dir, const DVec3& norm, DVec3& out)
{
    float d2 = 2.0f * dv_dot(dir, norm);
    float ox = dir.x - norm.x*d2;
    float oy = dir.y - norm.y*d2;
    float oz = dir.z - norm.z*d2;
    float sq = ox*ox + oy*oy + oz*oz;
    if (sq < 1e-18f || !isfinite(sq)) return false;
    float inv = rsqrtf(sq);
    out = dv(ox*inv, oy*inv, oz*inv);
    return true;
}

// ===========================================================================
// Device-side ghost ray trace
// Mirrors trace_ghost_ray() in trace.cpp exactly.
// ===========================================================================

struct DTraceResult { DVec3 position; float weight; bool valid; };

__device__ __forceinline__
DTraceResult d_trace_ghost_ray(const DRay& ray_in,
                                const Surface* surfs, int n_surfs,
                                float sensor_z,
                                int bounce_a, int bounce_b,
                                float lambda_nm,
                                float min_weight = 0.0f)
{
    DRay  ray          = ray_in;
    float current_ior  = 1.0f;
    float weight       = 1.0f;

    // ---- Phase 1: forward 0..bounce_b (reflect at bounce_b) ----
    for (int s = 0; s <= bounce_b; ++s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfs[s], hit, norm))
            return {dv(0,0,0), 0.0f, false};

        ray.origin = hit;
        float n1   = current_ior;
        float n2   = d_ior_at(surfs[s], lambda_nm);
        float cos_i = fabsf(dv_dot(norm, ray.dir));
        float R     = d_surface_reflectance(cos_i, n1, n2,
                                             surfs[s].coating, lambda_nm);

        if (s == bounce_b)
        {
            DVec3 rdir;
            if (!d_reflect(ray.dir, norm, rdir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir = rdir;
            weight *= R;
        }
        else
        {
            DVec3 new_dir;
            if (!d_refract(ray.dir, norm, __fdividef(n1, n2), new_dir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir     = new_dir;
            weight     *= (1.0f - R);
            current_ior = n2;
        }

        // Early termination: weight too small to contribute
        if (weight < min_weight)
            return {dv(0,0,0), 0.0f, false};
    }

    // ---- Phase 2: backward bounce_b-1..bounce_a (reflect at bounce_a) ----
    for (int s = bounce_b - 1; s >= bounce_a; --s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfs[s], hit, norm))
            return {dv(0,0,0), 0.0f, false};

        ray.origin = hit;
        float n1   = current_ior;
        float n2   = d_ior_before(surfs, s, lambda_nm);
        float cos_i = fabsf(dv_dot(norm, ray.dir));
        float R     = d_surface_reflectance(cos_i, n1, n2,
                                             surfs[s].coating, lambda_nm);

        if (s == bounce_a)
        {
            DVec3 rdir;
            if (!d_reflect(ray.dir, norm, rdir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir     = rdir;
            weight     *= R;
            current_ior = d_ior_at(surfs[bounce_a], lambda_nm);
        }
        else
        {
            DVec3 new_dir;
            if (!d_refract(ray.dir, norm, __fdividef(n1, n2), new_dir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir     = new_dir;
            weight     *= (1.0f - R);
            current_ior = n2;
        }

        // Early termination
        if (weight < min_weight)
            return {dv(0,0,0), 0.0f, false};
    }

    // ---- Phase 3: forward bounce_a+1..n_surfs-1 ----
    for (int s = bounce_a + 1; s < n_surfs; ++s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfs[s], hit, norm))
            return {dv(0,0,0), 0.0f, false};

        ray.origin = hit;
        float n1   = current_ior;
        float n2   = d_ior_at(surfs[s], lambda_nm);
        float cos_i = fabsf(dv_dot(norm, ray.dir));
        float R     = d_surface_reflectance(cos_i, n1, n2,
                                             surfs[s].coating, lambda_nm);

        DVec3 new_dir;
        if (!d_refract(ray.dir, norm, __fdividef(n1, n2), new_dir))
            return {dv(0,0,0), 0.0f, false};
        ray.dir     = new_dir;
        weight     *= (1.0f - R);
        current_ior = n2;

        // Early termination
        if (weight < min_weight)
            return {dv(0,0,0), 0.0f, false};
    }

    // ---- Propagate to sensor plane ----
    if (fabsf(ray.dir.z) < 1e-12f) return {dv(0,0,0), 0.0f, false};
    float t = __fdividef(sensor_z - ray.origin.z, ray.dir.z);
    // Use !(t > 0) rather than (t < 0) so that NaN is caught as invalid.
    if (!(t > 0.0f)) return {dv(0,0,0), 0.0f, false};

    DVec3 pos = dv(ray.origin.x + ray.dir.x*t,
                   ray.origin.y + ray.dir.y*t,
                   ray.origin.z + ray.dir.z*t);
    return {pos, weight, true};
}

// ===========================================================================
// GPU-side data structures (plain POD, no STL)
// ===========================================================================

struct GPUPair   { int surf_a, surf_b; float area_boost; int tile_cx, tile_cy;
                   float color_r, color_g, color_b; float offset_x, offset_y; float scale; };
struct GPUSource { float angle_x, angle_y, r, g, b; };
struct GPUSample { float u, v; };  // entrance-pupil grid point
// One wavelength sample with pre-computed RGB colour weights.
// Matches GPUSpectralSample in ghost_cuda.h.
struct GPUSpectralSampleDev { float lambda, rw, gw, bw; };

// ===========================================================================
// Warp-cooperative atomic reduction
//
// When multiple threads in a warp splat to the same pixel (common for focused
// ghost images), the per-lane atomicAdd calls serialise on the same L2 cache
// sector.  warp_splat() detects same-pixel peers via __match_any_sync, sums
// their contributions with warp shuffles, and issues a single atomicAdd per
// unique pixel — reducing atomic traffic by up to 32× in the best case.
//
// Requires compute capability >= 7.0 (Volta / sm_70).
// ===========================================================================

__device__ __forceinline__
void warp_splat(float* d_buf, int pixel_idx, float value,
                int buf_size)
{
    unsigned mask  = __activemask();
    unsigned peers = __match_any_sync(mask, pixel_idx);
    int leader     = __ffs(peers) - 1;
    int lane       = (int)threadIdx.x & 31;

    float sum = 0.0f;
    unsigned remaining = peers;
    while (remaining)
    {
        int src = __ffs(remaining) - 1;
        sum += __shfl_sync(peers, value, src);
        remaining &= remaining - 1;
    }

    if (lane == leader)
        atomicAdd(&d_buf[pixel_idx], sum);
}

// ===========================================================================
// Scatter kernel
// ===========================================================================

static constexpr int BLOCK_SIZE    = 512;
static constexpr int MAX_SURFACES  = 64;   // shared-memory surface cache capacity
static constexpr int TILE_DIM      = 32;   // shared-memory tile accumulator: 32×32 px

__global__ void ghost_kernel(
    const Surface*  d_surfs,
    int             n_surfs,
    float           sensor_z,
    const GPUPair*  d_pairs,
    int             n_sources,          // needed for index decode
    const GPUSource* d_sources,
    const GPUSample* d_grid,
    int             n_grid,
    float           front_R,
    float           start_z,
    float           sensor_half_w,
    float           sensor_half_h,
    int             width,              // bounding-box buffer width
    int             height,             // bounding-box buffer height
    float*          d_out_r,
    float*          d_out_g,
    float*          d_out_b,
    float           gain,
    float           ray_weight,
    const GPUSpectralSampleDev* d_spec,
    int             n_spec,
    float           fmt_w,              // format width in pixels
    float           fmt_h,              // format height in pixels
    float           fmt_x0_in_buf,      // format origin x within buffer
    float           fmt_y0_in_buf,      // format origin y within buffer
    int             spectral_jitter,    // 0=off, 1=on
    float           spec_bin_width,     // nm spacing between spectral bins
    float           cie_norm_r,         // CIE normalisation sums (for jittered weights)
    float           cie_norm_g,
    float           cie_norm_b,
    uint32_t        spec_jitter_seed)   // per-frame randomisation seed
{
    // =================================================================
    // Shared memory: surface cache + tile accumulator + beam direction
    // =================================================================
    __shared__ Surface s_surfs[MAX_SURFACES];
    __shared__ float   s_tile_r[TILE_DIM * TILE_DIM];
    __shared__ float   s_tile_g[TILE_DIM * TILE_DIM];
    __shared__ float   s_tile_b[TILE_DIM * TILE_DIM];
    __shared__ float   s_beam[3];   // normalised beam direction (computed once per block)

    // ---- Decode indices (all threads, before cooperative loads) ------
    const int ps_idx   = (int)blockIdx.x;
    const int pair_idx = ps_idx / n_sources;
    const int src_idx  = ps_idx % n_sources;
    const int grid_idx = (int)blockIdx.y * BLOCK_SIZE + (int)threadIdx.x;

    // Cooperative load: surfaces + tile zeroing + beam direction.
    const int clamped_n = (n_surfs <= MAX_SURFACES) ? n_surfs : MAX_SURFACES;
    for (int i = (int)threadIdx.x; i < clamped_n; i += BLOCK_SIZE)
        s_surfs[i] = d_surfs[i];
    for (int i = (int)threadIdx.x; i < TILE_DIM * TILE_DIM; i += BLOCK_SIZE)
    {
        s_tile_r[i] = 0.0f;
        s_tile_g[i] = 0.0f;
        s_tile_b[i] = 0.0f;
    }

    // Thread 0 computes the beam direction once for the entire block.
    // All 512 threads share the same source, so this replaces 512
    // redundant tanf() + normalize calls with one.
    if (threadIdx.x == 0)
    {
        const GPUSource& src = d_sources[src_idx];
        float bx = __tanf(src.angle_x);
        float by = __tanf(src.angle_y);
        float inv = rsqrtf(bx*bx + by*by + 1.0f);
        s_beam[0] = bx * inv;
        s_beam[1] = by * inv;
        s_beam[2] = inv;
    }
    __syncthreads();

    const GPUPair& pair = d_pairs[pair_idx];

    // Tile origin: on-axis centroid (precomputed on CPU) minus half-tile.
    const int  tile_x0     = pair.tile_cx - TILE_DIM / 2;
    const int  tile_y0     = pair.tile_cy - TILE_DIM / 2;
    const bool tile_usable = (pair.tile_cx > -90000);

    // Active flag replaces early returns — every thread must reach the
    // tile-flush __syncthreads at the bottom of the kernel.
    const bool active = (grid_idx < n_grid) &&
                        (pair.surf_a >= 0) &&
                        (pair.surf_b > pair.surf_a) &&
                        (pair.surf_b < clamped_n);

    if (active)
    {
        const GPUSource& src = d_sources[src_idx];
        const GPUSample& gs  = d_grid[grid_idx];

        // Read the pre-computed beam direction from shared memory
        DVec3 beam_dir = dv(s_beam[0], s_beam[1], s_beam[2]);

        DRay ray;
        ray.origin = dv(gs.u * front_R, gs.v * front_R, start_z);
        ray.dir    = beam_dir;

        const int buf_size = width * height;

        // Adaptive early-termination threshold.
        // A ray's final pixel contribution is:
        //   weight × ray_weight × gain × area_boost × src_color
        // We kill the trace when the contribution is guaranteed below 1e-5
        // (invisible in any compositing scenario).  Dividing by the known
        // scale factors gives the weight threshold below which the trace
        // is wasted work.  This adapts to per-pair boost and per-source
        // brightness, killing far more rays than a fixed 1e-10.
        const float max_src = fmaxf(src.r, fmaxf(src.g, src.b));
        const float scale   = ray_weight * gain * pair.area_boost * fmaxf(max_src, 1e-10f);
        const float min_weight = fminf(1e-5f / fmaxf(scale, 1e-20f), 0.01f);

        for (int s = 0; s < n_spec; ++s)
        {
            const GPUSpectralSampleDev& spec = d_spec[s];

            // ---- Spectral jitter: randomise wavelength within this bin ----
            float trace_lambda = spec.lambda;
            float use_rw = spec.rw, use_gw = spec.gw, use_bw = spec.bw;

            if (spectral_jitter && spec_bin_width > 0.1f)
            {
                // Unique seed per (ray, spectral sample, frame)
                uint32_t ray_id = (uint32_t)(ps_idx * n_grid + grid_idx);
                uint32_t seed = d_wang_hash(ray_id * 31u + (uint32_t)s * 7919u + spec_jitter_seed);
                float offset = d_hash_float(seed) - 0.5f;  // [-0.5, +0.5)

                trace_lambda = spec.lambda + offset * spec_bin_width;
                trace_lambda = fmaxf(380.0f, fminf(720.0f, trace_lambda));

                // Recompute CIE weights for the jittered wavelength
                use_rw = d_cie_r(trace_lambda) / fmaxf(cie_norm_r, 1e-9f);
                use_gw = d_cie_g(trace_lambda) / fmaxf(cie_norm_g, 1e-9f);
                use_bw = d_cie_b(trace_lambda) / fmaxf(cie_norm_b, 1e-9f);
            }

            // Skip this wavelength if the source has no energy in any
            // channel that this spectral sample contributes to.
            float src_contrib = src.r * use_rw + src.g * use_gw + src.b * use_bw;
            if (src_contrib < 1e-14f) continue;

            DTraceResult res = d_trace_ghost_ray(
                ray, s_surfs, clamped_n, sensor_z,
                pair.surf_a, pair.surf_b, trace_lambda,
                min_weight);

            if (!res.valid) continue;

            if (!isfinite(res.position.x) || !isfinite(res.position.y)) continue;

            float base = res.weight * ray_weight * gain * pair.area_boost;
            if (base < 1e-14f) continue;

            // Map sensor position (mm) → pixel coordinate + per-pair transform.
            float px = (res.position.x / (2.0f * sensor_half_w) + 0.5f) * fmt_w + fmt_x0_in_buf;
            float py = (res.position.y / (2.0f * sensor_half_h) + 0.5f) * fmt_h + fmt_y0_in_buf;

            // Scale relative to format center, then apply offset.
            float fcx = fmt_x0_in_buf + fmt_w * 0.5f;
            float fcy = fmt_y0_in_buf + fmt_h * 0.5f;
            px = fcx + (px - fcx) * pair.scale + pair.offset_x;
            py = fcy + (py - fcy) * pair.scale + pair.offset_y;

            if (!isfinite(px) || !isfinite(py)) continue;

            px = fmaxf(-2.0e9f, fminf(2.0e9f, px));
            py = fmaxf(-2.0e9f, fminf(2.0e9f, py));

            int   x0 = (int)floorf(px - 0.5f);
            int   y0 = (int)floorf(py - 0.5f);
            float fx = (px - 0.5f) - (float)x0;
            float fy = (py - 0.5f) - (float)y0;

            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx           * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx           * fy;

            float cr = src.r * use_rw * base * pair.color_r;
            float cg = src.g * use_gw * base * pair.color_g;
            float cb = src.b * use_bw * base * pair.color_b;

            // Bilinear corners
            const int   cx[4] = { x0, x0+1, x0,   x0+1 };
            const int   cy[4] = { y0, y0,   y0+1, y0+1 };
            const float cw[4] = { w00, w10,  w01,  w11  };

            for (int corner = 0; corner < 4; ++corner)
            {
                int xi = cx[corner];
                int yi = cy[corner];
                if (xi < 0 || xi >= width || yi < 0 || yi >= height) continue;

                float wt = cw[corner];
                float vr = cr * wt, vg = cg * wt, vb = cb * wt;

                // Local tile coordinates
                int lx = xi - tile_x0;
                int ly = yi - tile_y0;

                if (tile_usable &&
                    lx >= 0 && lx < TILE_DIM &&
                    ly >= 0 && ly < TILE_DIM)
                {
                    // Within tile: shared-memory atomics (Ampere+ executes
                    // these at L1 speed, ~10× faster than global atomics).
                    int tidx = ly * TILE_DIM + lx;
                    if (vr > 1e-14f) atomicAdd(&s_tile_r[tidx], vr);
                    if (vg > 1e-14f) atomicAdd(&s_tile_g[tidx], vg);
                    if (vb > 1e-14f) atomicAdd(&s_tile_b[tidx], vb);
                }
                else
                {
                    // Outside tile: warp-cooperative global splat
                    int pix = yi * width + xi;
                    if (vr > 1e-14f) warp_splat(d_out_r, pix, vr, buf_size);
                    if (vg > 1e-14f) warp_splat(d_out_g, pix, vg, buf_size);
                    if (vb > 1e-14f) warp_splat(d_out_b, pix, vb, buf_size);
                }
            }
        }
    } // end if (active)

    // =================================================================
    // Tile flush: cooperatively write the shared-memory tile to global
    // output.  ALL threads participate (the __syncthreads is mandatory).
    // =================================================================
    __syncthreads();

    if (tile_usable)
    {
        for (int i = (int)threadIdx.x; i < TILE_DIM * TILE_DIM; i += BLOCK_SIZE)
        {
            float rv = s_tile_r[i];
            float gv = s_tile_g[i];
            float bv = s_tile_b[i];
            if (rv > 0.0f || gv > 0.0f || bv > 0.0f)
            {
                int gx = tile_x0 + (i % TILE_DIM);
                int gy = tile_y0 + (i / TILE_DIM);
                if (gx >= 0 && gx < width && gy >= 0 && gy < height)
                {
                    int gidx = gy * width + gx;
                    if (rv > 0.0f) atomicAdd(&d_out_r[gidx], rv);
                    if (gv > 0.0f) atomicAdd(&d_out_g[gidx], gv);
                    if (bv > 0.0f) atomicAdd(&d_out_b[gidx], bv);
                }
            }
        }
    }
}

// ===========================================================================
// CPU launcher
// ===========================================================================

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
    bool                            skip_readback,
    std::string*                    out_error,
    const std::vector<float>*       pair_colors,
    const std::vector<float>*       pair_offsets,
    const std::vector<float>*       pair_scales)
{
    if (active_pairs.empty() || sources.empty()) return;


    const int n_surfs  = lens.num_surfaces();
    if (n_surfs <= 0) return;   // no surfaces — nothing to render

    if (n_surfs > MAX_SURFACES)
    {
        fprintf(stderr,
                "FlareSim WARNING: lens has %d surfaces, but the shared-memory "
                "cache supports %d.  Surfaces beyond %d will be ignored.\n",
                n_surfs, MAX_SURFACES, MAX_SURFACES);
    }

    // Friendly GPU availability check — diagnose common "no CUDA" situations
    // before we hit a cryptic allocator error.
    {
        int device_count = 0;
        cudaError_t ce = cudaGetDeviceCount(&device_count);
        if (ce == cudaErrorInsufficientDriver) {
            if (out_error) {
                int drv = 0, rt = 0;
                cudaDriverGetVersion(&drv);
                cudaRuntimeGetVersion(&rt);
                *out_error = "FlareSim: CUDA driver/runtime mismatch. "
                             "Driver reports CUDA "
                             + std::to_string(drv / 1000) + "."
                             + std::to_string((drv % 1000) / 10)
                             + ", plugin was built with CUDA "
                             + std::to_string(rt / 1000) + "."
                             + std::to_string((rt % 1000) / 10)
                             + ". Please update your NVIDIA driver or "
                               "rebuild the plugin with an older CUDA toolkit.";
            }
            return;
        }
        if (ce == cudaErrorNoDevice || device_count == 0) {
            if (out_error)
                *out_error = "FlareSim requires an NVIDIA CUDA GPU — no compatible GPU "
                             "was detected on this system. FlareSim will produce black output.";
            return;
        }
        if (ce != cudaSuccess) {
            if (out_error)
                *out_error = std::string("FlareSim: CUDA initialisation failed (")
                             + cudaGetErrorString(ce)
                             + "). Check that your NVIDIA driver is installed correctly.";
            return;
        }
    }

    // Clear any sticky error left by a previous frame's kernel.
    // Note: on Windows/WDDM, once the CUDA context is lost (e.g. from a GPU
    // fault) there is no in-process recovery — cudaDeviceReset() would also
    // destroy Nuke's own CUDA contexts.  We simply clear the error flag here
    // and let the subsequent cudaMalloc report a fresh failure if the context
    // is genuinely gone.
    {
        cudaError_t prev_err = cudaGetLastError();
        if (prev_err != cudaSuccess) {
            fprintf(stderr,
                    "FlareSim: clearing CUDA sticky error: %s\n",
                    cudaGetErrorString(prev_err));
        }
    }

    const int n_pairs  = (int)active_pairs.size();

    // No internal source cap — the caller (Max Sources knob) is responsible
    // for limiting source count.  gridDim.x supports up to 2^31-1 on all
    // CUDA compute capability >= 3.0 devices (i.e. any GPU since 2012).
    const std::vector<BrightPixel>& capped_sources = sources;
    const int n_src    = (int)capped_sources.size();

    // ---- Build entrance-pupil grid ----
    // Three sampling modes controlled by config.pupil_jitter:
    //   0 = regular NxN grid (cell centres)
    //   1 = stratified jitter (one random sample per cell, stays within cell)
    //   2 = Halton quasi-random low-discrepancy sequence (base-2 / base-3)
    //
    // All modes generate up to N*N candidate positions, apply the circular /
    // polygonal aperture mask, and pack survivors into grid_samples[].
    //
    // Wang hash — fast 32-bit scramble, uniform in [0, 2^32).
    auto wang_hash = [](uint32_t s) -> uint32_t {
        s = (s ^ 61u) ^ (s >> 16u);
        s *= 9u;
        s ^= s >> 4u;
        s *= 0x27d4eb2du;
        s ^= s >> 15u;
        return s;
    };
    // Halton base-2 — bit reversal (O(1) via parallel bit ops).
    auto halton2 = [](uint32_t n) -> float {
        n = (n << 16u) | (n >> 16u);
        n = ((n & 0x00ff00ffu) << 8u) | ((n & 0xff00ff00u) >> 8u);
        n = ((n & 0x0f0f0f0fu) << 4u) | ((n & 0xf0f0f0f0u) >> 4u);
        n = ((n & 0x33333333u) << 2u) | ((n & 0xccccccccu) >> 2u);
        n = ((n & 0x55555555u) << 1u) | ((n & 0xaaaaaaaau) >> 1u);
        return (float)n * (1.0f / 4294967296.0f);
    };
    // Halton base-3 — iterative radical inverse.
    auto halton3 = [](uint32_t n) -> float {
        float r = 0.0f, f = 1.0f / 3.0f;
        while (n > 0) { r += (n % 3u) * f; n /= 3u; f /= 3.0f; }
        return r;
    };


    // Check if we can reuse the cached pupil grid.
    // The seed only affects the grid in stratified mode (jitter=1).
    // For regular (0) and Halton (2), the seed is irrelevant.
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

    if (grid_cached)
    {
        // Reuse previous frame's grid — already on device in cache.d_grid.
        n_grid = cache.cached_grid_count;
    }
    else
    {
        const int   N          = config.ray_grid;
        const int   n_blades   = config.aperture_blades;
        const float rot_rad    = config.aperture_rotation_deg * ((float)M_PI / 180.0f);
        const bool  polygonal  = (n_blades >= 3);
        const float apothem    = polygonal ? std::cos((float)M_PI / n_blades) : 1.0f;
        const float sector_ang = polygonal ? (2.0f * (float)M_PI / n_blades)  : 1.0f;
        const int      jitter      = config.pupil_jitter;
        const uint32_t seed_offset = (uint32_t)config.pupil_jitter_seed * 1000003u;

        grid_samples.reserve((size_t)N * N);
        for (int k = 0; k < N * N; ++k)
        {
            const int gx = k % N;
            const int gy = k / N;

            float u, v;
            if (jitter == 2)
            {
                u = halton2((uint32_t)k) * 2.0f - 1.0f;
                v = halton3((uint32_t)k) * 2.0f - 1.0f;
            }
            else
            {
                float ju = (jitter == 1) ? wang_hash((uint32_t)k + seed_offset)
                                               / 4294967296.0f : 0.5f;
                float jv = (jitter == 1) ? wang_hash((uint32_t)k + (uint32_t)(N * N) + seed_offset)
                                               / 4294967296.0f : 0.5f;
                u = ((gx + ju) / N) * 2.0f - 1.0f;
                v = ((gy + jv) / N) * 2.0f - 1.0f;
            }

            float r2 = u*u + v*v;
            if (r2 > 1.0f) continue;
            if (polygonal)
            {
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
    if (n_grid == 0) return;  // ray_grid knob set to 0
    const float ray_weight = 1.0f / n_grid;
    const float front_R    = lens.surfaces[0].semi_aperture;
    const float start_z    = lens.surfaces[0].z - 20.0f;

    // ---- Pack GPU-side pair / source arrays ----
    std::vector<GPUPair> gpu_pairs(n_pairs);
    for (int i = 0; i < n_pairs; ++i) {
        float cr = 1.0f, cg = 1.0f, cb = 1.0f;
        float ox = 0.0f, oy = 0.0f;
        if (pair_colors && (int)pair_colors->size() >= (i + 1) * 3) {
            cr = (*pair_colors)[i * 3 + 0];
            cg = (*pair_colors)[i * 3 + 1];
            cb = (*pair_colors)[i * 3 + 2];
        }
        if (pair_offsets && (int)pair_offsets->size() >= (i + 1) * 2) {
            ox = (*pair_offsets)[i * 2 + 0];
            oy = (*pair_offsets)[i * 2 + 1];
        }
        float sc = 1.0f;
        if (pair_scales && (int)pair_scales->size() > i) {
            sc = (*pair_scales)[i];
        }
        gpu_pairs[i] = { active_pairs[i].surf_a,
                         active_pairs[i].surf_b,
                         pair_area_boosts[i],
                         -99999, -99999,
                         cr, cg, cb, ox, oy, sc };
    }

    // Sort pairs by descending trace cost so expensive pairs launch first.
    // The three-phase trace visits roughly 3×(surf_b - surf_a) surfaces;
    // putting heavy pairs at the front avoids a long tail where the last
    // SMs are stuck with the most expensive work while everyone else idles.
    std::sort(gpu_pairs.begin(), gpu_pairs.end(),
              [](const GPUPair& a, const GPUPair& b) {
                  int cost_a = (a.surf_b - a.surf_a) * 3;
                  int cost_b = (b.surf_b - b.surf_a) * 3;
                  return cost_a > cost_b; // descending: expensive first
              });

    // ---- Compute tile centroids ----
    // Trace an on-axis probe ray per pair (green wavelength) and convert
    // the ghost hit position to pixel coordinates.  The 32×32 shared-memory
    // tile in the kernel is centred on this position.  Off-axis sources or
    // diffuse ghosts that fall outside the tile use the warp_splat fallback.
    for (int i = 0; i < n_pairs; ++i)
    {
        Ray probe;
        probe.origin = Vec3f(0, 0, start_z);
        probe.dir    = Vec3f(0, 0, 1);
        TraceResult tr = trace_ghost_ray(probe, lens,
                                         gpu_pairs[i].surf_a,
                                         gpu_pairs[i].surf_b,
                                         550.0f);
        if (tr.valid &&
            std::isfinite(tr.position.x) &&
            std::isfinite(tr.position.y))
        {
            float px = (tr.position.x / (2.0f * sensor_half_w) + 0.5f)
                       * (float)fmt_w + (float)fmt_x0_in_buf;
            float py = (tr.position.y / (2.0f * sensor_half_h) + 0.5f)
                       * (float)fmt_h + (float)fmt_y0_in_buf;
            gpu_pairs[i].tile_cx = (int)std::round(px);
            gpu_pairs[i].tile_cy = (int)std::round(py);
        }
        // else: sentinel -99999 remains → tile_usable = false in kernel
    }

    std::vector<GPUSource> gpu_src(n_src);
    for (int i = 0; i < n_src; ++i)
        gpu_src[i] = { capped_sources[i].angle_x, capped_sources[i].angle_y,
                       capped_sources[i].r, capped_sources[i].g, capped_sources[i].b };

    // ---- Build spectral sample table ----
    // Each entry has a wavelength and pre-normalised RGB colour weights.
    // With spectral_samples=3 we use the config wavelengths directly and
    // assign one channel per sample (exact backward compat).
    // With more samples we distribute evenly 400–700 nm and use CIE-approximate
    // Gaussian colour matching, normalised so the sum per channel equals 1.
    std::vector<GPUSpectralSampleDev> spectral_cpu;
    float cie_sum_r = 1.0f, cie_sum_g = 1.0f, cie_sum_b = 1.0f;
    float spec_bin_width = 0.0f;
    {
        const int ns = std::max(3, config.spectral_samples);
        spectral_cpu.resize(ns);

        // Gaussian approximation of CIE 1931 xyz (→ sRGB).
        auto cie_r = [](float l) {
            float a = (l - 600.0f) / 70.0f, b = (l - 450.0f) / 30.0f;
            return std::max(0.0f, 0.63f * std::exp(-0.5f*a*a)
                                + 0.22f * std::exp(-0.5f*b*b));
        };
        auto cie_g = [](float l) {
            float a = (l - 545.0f) / 55.0f;
            return std::max(0.0f, std::exp(-0.5f*a*a));
        };
        auto cie_b = [](float l) {
            float a = (l - 445.0f) / 45.0f;
            return std::max(0.0f, std::exp(-0.5f*a*a));
        };

        if (ns == 3)
        {
            // Exact backward compat: one sample per channel, full weight.
            spectral_cpu[0] = { config.wavelengths[0], 1.0f, 0.0f, 0.0f };
            spectral_cpu[1] = { config.wavelengths[1], 0.0f, 1.0f, 0.0f };
            spectral_cpu[2] = { config.wavelengths[2], 0.0f, 0.0f, 1.0f };

            // For spectral jitter in 3-sample mode: bin width spans
            // the full visible range / 3 ≈ 100 nm per bin.
            spec_bin_width = 100.0f;
            // CIE sums: when jitter is on, the kernel recomputes weights
            // from CIE curves.  Precompute the normalization sums by
            // integrating over the same range the jittered samples will hit.
            cie_sum_r = cie_r(450.0f) + cie_r(550.0f) + cie_r(650.0f);
            cie_sum_g = cie_g(450.0f) + cie_g(550.0f) + cie_g(650.0f);
            cie_sum_b = cie_b(450.0f) + cie_b(550.0f) + cie_b(650.0f);
        }
        else
        {
            spec_bin_width = 300.0f / (ns - 1);

            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            for (int i = 0; i < ns; ++i)
            {
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
            // Normalise so each channel's weights sum to 1.
            for (int i = 0; i < ns; ++i)
            {
                if (sum_r > 1e-9f) spectral_cpu[i].rw /= sum_r;
                if (sum_g > 1e-9f) spectral_cpu[i].gw /= sum_g;
                if (sum_b > 1e-9f) spectral_cpu[i].bw /= sum_b;
            }
        }
    }
    const int n_spec = (int)spectral_cpu.size();

    const size_t n_px = (size_t)width * height;

    int  grid_blocks = (n_grid + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_dim((unsigned)(n_pairs * n_src), (unsigned)grid_blocks, 1);

    // ---- Grow device buffers if the current allocation is too small ----
    // On success the allocation is reused; on failure we report and return.
    // The cache retains all valid buffers — the next frame will reuse them.

    // Error reporter (used inline below — not a macro to avoid goto issues).
    auto report = [&](cudaError_t e, const char* site) {
        fprintf(stderr, "FlareSim CUDA error at %s -- %s\n", site, cudaGetErrorString(e));
        if (out_error && out_error->empty())
        {
            char buf[256];
            snprintf(buf, sizeof(buf), "CUDA error at %s -- %s", site, cudaGetErrorString(e));
            *out_error = buf;
        }
    };

    // Generic void* buffer grow.
    auto ensure = [&](void*& ptr, size_t& cap, size_t need, const char* tag) -> bool {
        if (need <= cap) return true;
        cudaFree(ptr);  ptr = nullptr;  cap = 0;
        cudaError_t e = cudaMalloc(&ptr, need);
        if (e != cudaSuccess) { report(e, tag); return false; }
        cap = need;
        return true;
    };

    if (!ensure(cache.d_surfs, cache.surfs_bytes, n_surfs * sizeof(Surface),    "d_surfs")) return;
    if (!ensure(cache.d_pairs, cache.pairs_bytes, n_pairs * sizeof(GPUPair),     "d_pairs")) return;
    if (!ensure(cache.d_src,   cache.src_bytes,   n_src   * sizeof(GPUSource),   "d_src"  )) return;
    if (!grid_cached) {
        if (!ensure(cache.d_grid, cache.grid_bytes, n_grid * sizeof(GPUSample), "d_grid")) return;
    }
    if (!ensure(cache.d_spec,  cache.spec_bytes,  n_spec  * sizeof(GPUSpectralSampleDev), "d_spec")) return;

    // Output channels are always the same size; grow all four together.
    if (n_px > cache.out_floats)
    {
        cudaFree(cache.d_out_r);  cache.d_out_r = nullptr;
        cudaFree(cache.d_out_g);  cache.d_out_g = nullptr;
        cudaFree(cache.d_out_b);  cache.d_out_b = nullptr;
        cudaFree(cache.d_out_a);  cache.d_out_a = nullptr;
        cache.out_floats = 0;

        cudaError_t er = cudaMalloc(&cache.d_out_r, n_px * sizeof(float));
        if (er != cudaSuccess) { report(er, "d_out_r"); return; }
        cudaError_t eg = cudaMalloc(&cache.d_out_g, n_px * sizeof(float));
        if (eg != cudaSuccess) { report(eg, "d_out_g"); return; }
        cudaError_t eb = cudaMalloc(&cache.d_out_b, n_px * sizeof(float));
        if (eb != cudaSuccess) { report(eb, "d_out_b"); return; }
        cudaError_t ea = cudaMalloc(&cache.d_out_a, n_px * sizeof(float));
        if (ea != cudaSuccess) { report(ea, "d_out_a"); return; }
        cache.out_floats = n_px;
    }

    // ---- Upload frame data ----


#define GPU_CHK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { report(_e, #call); return; } \
    } while(0)

    GPU_CHK(cudaMemcpy(cache.d_surfs, lens.surfaces.data(),
                       n_surfs * sizeof(Surface), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(cache.d_pairs, gpu_pairs.data(),
                       n_pairs * sizeof(GPUPair), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(cache.d_src,   gpu_src.data(),
                       n_src   * sizeof(GPUSource), cudaMemcpyHostToDevice));
    if (!grid_cached) {
        GPU_CHK(cudaMemcpy(cache.d_grid, grid_samples.data(),
                           n_grid * sizeof(GPUSample), cudaMemcpyHostToDevice));
        // Update cache key
        cache.cached_grid_count      = n_grid;
        cache.cached_ray_grid        = config.ray_grid;
        cache.cached_aperture_blades = config.aperture_blades;
        cache.cached_aperture_rot    = config.aperture_rotation_deg;
        cache.cached_pupil_jitter    = config.pupil_jitter;
        cache.cached_jitter_seed     = config.pupil_jitter_seed;
    }
    GPU_CHK(cudaMemcpy(cache.d_spec,  spectral_cpu.data(),
                       n_spec  * sizeof(GPUSpectralSampleDev), cudaMemcpyHostToDevice));

    // Zero the output buffers each frame (kernel scatters by addition).
    GPU_CHK(cudaMemset(cache.d_out_r, 0, n_px * sizeof(float)));
    GPU_CHK(cudaMemset(cache.d_out_g, 0, n_px * sizeof(float)));
    GPU_CHK(cudaMemset(cache.d_out_b, 0, n_px * sizeof(float)));

    // ---- Launch ----
    ghost_kernel<<<grid_dim, block>>>(
        static_cast<Surface*>(cache.d_surfs), n_surfs, lens.sensor_z,
        static_cast<GPUPair*>(cache.d_pairs), n_src,
        static_cast<GPUSource*>(cache.d_src),
        static_cast<GPUSample*>(cache.d_grid), n_grid,
        front_R, start_z,
        sensor_half_w, sensor_half_h,
        width, height,
        cache.d_out_r, cache.d_out_g, cache.d_out_b,
        config.gain, ray_weight,
        static_cast<GPUSpectralSampleDev*>(cache.d_spec), n_spec,
        (float)fmt_w, (float)fmt_h,
        (float)fmt_x0_in_buf, (float)fmt_y0_in_buf,
        config.spectral_jitter ? 1 : 0,
        spec_bin_width * std::max(0.0f, config.spectral_jitter_scale),
        cie_sum_r, cie_sum_g, cie_sum_b,
        (uint32_t)config.spectral_jitter_seed);

    GPU_CHK(cudaDeviceSynchronize());

    // ---- Copy results back to CPU (skipped when pipeline keeps data on GPU) ----
    if (!skip_readback)
    {
        GPU_CHK(cudaMemcpy(out_r, cache.d_out_r, n_px * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_CHK(cudaMemcpy(out_g, cache.d_out_g, n_px * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_CHK(cudaMemcpy(out_b, cache.d_out_b, n_px * sizeof(float), cudaMemcpyDeviceToHost));
    }

#undef GPU_CHK

    // Device buffers remain alive in cache for the next frame.
}

// ===========================================================================
// Alpha kernel — Rec.709 luminance, clamped [0, 1]
// ===========================================================================

__global__ void alpha_kernel(const float* __restrict__ d_r,
                             const float* __restrict__ d_g,
                             const float* __restrict__ d_b,
                             float* __restrict__ d_a,
                             int n_px)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_px) return;
    float lum = 0.2126f * d_r[i] + 0.7152f * d_g[i] + 0.0722f * d_b[i];
    d_a[i] = fminf(lum, 1.0f);
}

// ===========================================================================
// Luminance-preserving soft-clip kernel with selectable metric
//
// Matches AFXToneMap convention:
//   metric 0 = Value     (max of R, G, B)
//   metric 1 = Luminance (Rec.709: 0.2126R + 0.7152G + 0.0722B)
//   metric 2 = Lightness (cube root of Rec.709 luminance)
//
// clip_val : maximum output (asymptotic ceiling)
// knee     : transition sharpness (0 = very soft, 1 = hard clip)
//            — matches AFXToneMap's knee convention
// ===========================================================================

__global__ void soft_clip_kernel(float* __restrict__ d_r,
                                 float* __restrict__ d_g,
                                 float* __restrict__ d_b,
                                 int n_px,
                                 float clip_val,
                                 float knee,
                                 int metric)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_px) return;

    float r = d_r[i], g = d_g[i], b = d_b[i];

    // Compute metric
    float m = 0.0f;
    switch (metric) {
        case 0:  // Value (max RGB)
            m = fmaxf(r, fmaxf(g, b));
            break;
        case 1:  // Luminance (Rec.709)
            m = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            break;
        case 2:  // Lightness (cube root of luminance)
            m = cbrtf(fmaxf(0.0f, 0.2126f * r + 0.7152f * g + 0.0722f * b));
            break;
        default:
            m = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            break;
    }

    if (m <= 0.0f) return;

    // Soft-clip curve (AFXToneMap convention: knee=1 is hard, knee=0 is soft)
    // softness = 1 - knee  (internal param, 0=hard, 1=very soft)
    const float softness  = 1.0f - knee;
    const float threshold = clip_val * (1.0f - softness);
    const float knee_range = clip_val * softness;

    float compressed;
    if (m <= threshold) {
        compressed = m;
    } else {
        float excess = m - threshold;
        compressed = threshold + knee_range * excess / (excess + knee_range);
    }

    float scale = compressed / m;
    d_r[i] = r * scale;
    d_g[i] = g * scale;
    d_b[i] = b * scale;
}

// ===========================================================================
// FP32 → FP16 conversion kernel
// ===========================================================================

__global__ void fp32_to_fp16_kernel(const float* __restrict__ in,
                                    __half* __restrict__ out,
                                    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

void launch_alpha_cuda(int width, int height,
                       GpuBufferCache& cache,
                       std::string* out_error)
{
    const size_t n_px = (size_t)width * height;
    if (n_px == 0 || !cache.d_out_r || !cache.d_out_g || !cache.d_out_b) return;

    // Ensure alpha buffer is allocated
    if (n_px > cache.out_floats || !cache.d_out_a)
    {
        cudaFree(cache.d_out_a);  cache.d_out_a = nullptr;
        cudaError_t e = cudaMalloc(&cache.d_out_a, n_px * sizeof(float));
        if (e != cudaSuccess)
        {
            fprintf(stderr, "FlareSim CUDA error at alpha alloc -- %s\n",
                    cudaGetErrorString(e));
            if (out_error && out_error->empty())
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "CUDA error at alpha alloc -- %s",
                         cudaGetErrorString(e));
                *out_error = buf;
            }
            return;
        }
    }

    constexpr int BLK = 256;
    int grid = ((int)n_px + BLK - 1) / BLK;
    alpha_kernel<<<grid, BLK>>>(cache.d_out_r, cache.d_out_g, cache.d_out_b,
                                cache.d_out_a, (int)n_px);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "FlareSim CUDA error at alpha kernel -- %s\n",
                cudaGetErrorString(e));
        if (out_error && out_error->empty())
        {
            char buf[256];
            snprintf(buf, sizeof(buf), "CUDA error at alpha kernel -- %s",
                     cudaGetErrorString(e));
            *out_error = buf;
        }
    }
}

// ===========================================================================
// Batched GPU → CPU readback (RGBA)
// ===========================================================================

void readback_gpu_output(float* cpu_r, float* cpu_g, float* cpu_b, float* cpu_a,
                         int width, int height,
                         GpuBufferCache& cache,
                         std::string* out_error)
{
    const size_t n_px    = (size_t)width * height;
    const size_t n_bytes = n_px * sizeof(float);
    if (n_bytes == 0) return;

    auto report = [&](cudaError_t e, const char* site) {
        fprintf(stderr, "FlareSim readback error at %s -- %s\n",
                site, cudaGetErrorString(e));
        if (out_error && out_error->empty()) {
            char buf[256];
            snprintf(buf, sizeof(buf), "readback error at %s -- %s",
                     site, cudaGetErrorString(e));
            *out_error = buf;
        }
    };

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { report(e, "sync"); return; }

    // Simple synchronous FP32 readback (legacy fallback).
    e = cudaMemcpy(cpu_r, cache.d_out_r, n_bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { report(e, "d2h_r"); return; }
    e = cudaMemcpy(cpu_g, cache.d_out_g, n_bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { report(e, "d2h_g"); return; }
    e = cudaMemcpy(cpu_b, cache.d_out_b, n_bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { report(e, "d2h_b"); return; }
    if (cpu_a && cache.d_out_a) {
        e = cudaMemcpy(cpu_a, cache.d_out_a, n_bytes, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) { report(e, "d2h_a"); return; }
    }
}

// ===========================================================================
// Combined blur → alpha → readback with async CUDA streams
//
// Pipeline layout (time flows →):
//
//   compute_stream: [blur_R] ──ev0──→ [blur_G] ──ev1──→ [blur_B] ──ev2──→ [alpha] ──ev3──→
//   copy_stream:              wait0 → [D2H R]          wait1 → [D2H G]          wait2+3 → [D2H B][D2H A]
//
// Readback of channel R overlaps with blur of G and B.
// Readback of channel G overlaps with blur of B and alpha.
// ===========================================================================

void launch_blur_alpha_readback_async(
    float* cpu_r, float* cpu_g, float* cpu_b, float* cpu_a,
    int w, int h, int blur_radius, int blur_passes,
    GpuBufferCache& cache, std::string* out_error,
    float highlight_clip, float highlight_knee, int highlight_metric)
{
    const size_t n_px       = (size_t)w * h;
    const size_t n_bytes_f  = n_px * sizeof(float);
    const size_t n_bytes_h  = n_px * sizeof(__half);
    if (n_px == 0) return;

    auto report = [&](cudaError_t e, const char* site) {
        fprintf(stderr, "FlareSim async error at %s -- %s\n",
                site, cudaGetErrorString(e));
        if (out_error && out_error->empty()) {
            char buf[256];
            snprintf(buf, sizeof(buf), "async error at %s -- %s",
                     site, cudaGetErrorString(e));
            *out_error = buf;
        }
    };

    // ---- Lazily create persistent streams ----
    if (!cache.compute_stream) {
        cudaStream_t cs;
        cudaError_t e = cudaStreamCreate(&cs);
        if (e != cudaSuccess) { report(e, "create compute_stream"); return; }
        cache.compute_stream = static_cast<void*>(cs);
    }
    if (!cache.copy_stream) {
        cudaStream_t cs;
        cudaError_t e = cudaStreamCreate(&cs);
        if (e != cudaSuccess) { report(e, "create copy_stream"); return; }
        cache.copy_stream = static_cast<void*>(cs);
    }

    cudaStream_t comp = static_cast<cudaStream_t>(cache.compute_stream);
    cudaStream_t copy = static_cast<cudaStream_t>(cache.copy_stream);

    // ---- Ensure blur scratch buffers (FP32) ----
    if (n_px > cache.blur_floats) {
        cudaFree(cache.blur_a);  cache.blur_a = nullptr;
        cudaFree(cache.blur_b);  cache.blur_b = nullptr;
        cache.blur_floats = 0;
        cudaError_t e;
        e = cudaMalloc(&cache.blur_a, n_bytes_f);
        if (e != cudaSuccess) { report(e, "blur_a"); return; }
        e = cudaMalloc(&cache.blur_b, n_bytes_f);
        if (e != cudaSuccess) { report(e, "blur_b"); return; }
        cache.blur_floats = n_px;
    }

    // ---- Ensure alpha buffer (FP32) ----
    if (n_px > cache.out_floats || !cache.d_out_a) {
        cudaFree(cache.d_out_a);  cache.d_out_a = nullptr;
        cudaError_t e = cudaMalloc(&cache.d_out_a, n_bytes_f);
        if (e != cudaSuccess) { report(e, "d_out_a"); return; }
    }

    // ---- Ensure FP16 device staging (4 channels contiguous) ----
    if (n_px > cache.d_out_fp16_elems) {
        cudaFree(cache.d_out_fp16);  cache.d_out_fp16 = nullptr;
        cache.d_out_fp16_elems = 0;
        cudaError_t e = cudaMalloc(&cache.d_out_fp16, 4 * n_bytes_h);
        if (e != cudaSuccess) { report(e, "d_out_fp16"); return; }
        cache.d_out_fp16_elems = n_px;
    }

    __half* d_fp16_base = static_cast<__half*>(cache.d_out_fp16);
    __half* d_fp16[4] = {
        d_fp16_base,
        d_fp16_base + n_px,
        d_fp16_base + 2 * n_px,
        d_fp16_base + 3 * n_px
    };

    __half* h_dst[4] = {
        reinterpret_cast<__half*>(cpu_r),
        reinterpret_cast<__half*>(cpu_g),
        reinterpret_cast<__half*>(cpu_b),
        reinterpret_cast<__half*>(cpu_a)
    };

    const bool do_blur = (blur_radius >= 1 && blur_passes >= 1);
    const bool do_clip = (highlight_clip > 0.0f);
    float* d_ch[3] = { cache.d_out_r, cache.d_out_g, cache.d_out_b };
    constexpr int CVT_BLK = 256;
    int cvt_grid = ((int)n_px + CVT_BLK - 1) / CVT_BLK;

    cudaEvent_t ev[4];
    for (int i = 0; i < 4; ++i)
        cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);

    // ---- Blur all 3 channels first (needed before soft clip) ----
    if (do_blur)
    {
        for (int ch = 0; ch < 3; ++ch)
        {
            cudaMemcpyAsync(cache.blur_a, d_ch[ch], n_bytes_f,
                            cudaMemcpyDeviceToDevice, comp);
            for (int p = 0; p < blur_passes; ++p)
            {
                launch_blur_h_on_stream(cache.blur_a, cache.blur_b, w, h,
                                        blur_radius, static_cast<void*>(comp));
                launch_blur_v_on_stream(cache.blur_b, cache.blur_a, w, h,
                                        blur_radius, static_cast<void*>(comp));
            }
            cudaMemcpyAsync(d_ch[ch], cache.blur_a, n_bytes_f,
                            cudaMemcpyDeviceToDevice, comp);
        }
    }

    // ---- Soft highlight compression (after blur, before alpha) ----
    if (do_clip)
    {
        float k = fmaxf(0.001f, fminf(highlight_knee, 0.999f));
        soft_clip_kernel<<<cvt_grid, CVT_BLK, 0, comp>>>(
            cache.d_out_r, cache.d_out_g, cache.d_out_b,
            (int)n_px, highlight_clip, k, highlight_metric);
    }

    // ---- Alpha ----
    alpha_kernel<<<cvt_grid, CVT_BLK, 0, comp>>>(
        cache.d_out_r, cache.d_out_g, cache.d_out_b,
        cache.d_out_a, (int)n_px);

    // ---- Convert all 4 channels FP32 → FP16 and DMA ----
    float* d_all[4] = { cache.d_out_r, cache.d_out_g, cache.d_out_b, cache.d_out_a };
    for (int ch = 0; ch < 4; ++ch)
    {
        fp32_to_fp16_kernel<<<cvt_grid, CVT_BLK, 0, comp>>>(
            d_all[ch], d_fp16[ch], (int)n_px);
        cudaEventRecord(ev[ch], comp);

        if (h_dst[ch]) {
            cudaStreamWaitEvent(copy, ev[ch], 0);
            cudaMemcpyAsync(h_dst[ch], d_fp16[ch], n_bytes_h,
                            cudaMemcpyDeviceToHost, copy);
        }
    }

    // ---- Wait for everything ----
    cudaStreamSynchronize(copy);
    cudaStreamSynchronize(comp);

    for (int i = 0; i < 4; ++i)
        cudaEventDestroy(ev[i]);
}

// ===========================================================================
// FP32 variant — blur → alpha → FP32 DMA (no FP16 conversion)
//
// Same async pipeline as the FP16 version, but transfers full FP32 data
// directly to pinned host memory.  Eliminates FP16 quantisation banding.
// ===========================================================================

void launch_blur_alpha_readback_fp32(
    float* cpu_r, float* cpu_g, float* cpu_b, float* cpu_a,
    int w, int h, int blur_radius, int blur_passes,
    GpuBufferCache& cache, std::string* out_error)
{
    const size_t n_px      = (size_t)w * h;
    const size_t n_bytes_f = n_px * sizeof(float);
    if (n_px == 0) return;

    auto report = [&](cudaError_t e, const char* site) {
        fprintf(stderr, "FlareSim async-fp32 error at %s -- %s\n",
                site, cudaGetErrorString(e));
        if (out_error && out_error->empty()) {
            char buf[256];
            snprintf(buf, sizeof(buf), "async-fp32 error at %s -- %s",
                     site, cudaGetErrorString(e));
            *out_error = buf;
        }
    };

    // ---- Lazily create persistent streams ----
    if (!cache.compute_stream) {
        cudaStream_t cs;
        cudaError_t e = cudaStreamCreate(&cs);
        if (e != cudaSuccess) { report(e, "create compute_stream"); return; }
        cache.compute_stream = static_cast<void*>(cs);
    }
    if (!cache.copy_stream) {
        cudaStream_t cs;
        cudaError_t e = cudaStreamCreate(&cs);
        if (e != cudaSuccess) { report(e, "create copy_stream"); return; }
        cache.copy_stream = static_cast<void*>(cs);
    }

    cudaStream_t comp = static_cast<cudaStream_t>(cache.compute_stream);
    cudaStream_t copy = static_cast<cudaStream_t>(cache.copy_stream);

    // ---- Ensure blur scratch buffers (FP32) ----
    if (n_px > cache.blur_floats) {
        cudaFree(cache.blur_a);  cache.blur_a = nullptr;
        cudaFree(cache.blur_b);  cache.blur_b = nullptr;
        cache.blur_floats = 0;
        cudaError_t e;
        e = cudaMalloc(&cache.blur_a, n_bytes_f);
        if (e != cudaSuccess) { report(e, "blur_a"); return; }
        e = cudaMalloc(&cache.blur_b, n_bytes_f);
        if (e != cudaSuccess) { report(e, "blur_b"); return; }
        cache.blur_floats = n_px;
    }

    // ---- Ensure alpha buffer (FP32) ----
    if (n_px > cache.out_floats || !cache.d_out_a) {
        cudaFree(cache.d_out_a);  cache.d_out_a = nullptr;
        cudaError_t e = cudaMalloc(&cache.d_out_a, n_bytes_f);
        if (e != cudaSuccess) { report(e, "d_out_a"); return; }
    }

    const bool do_blur = (blur_radius >= 1 && blur_passes >= 1);
    float* d_ch[3] = { cache.d_out_r, cache.d_out_g, cache.d_out_b };

    cudaEvent_t ev[4];
    for (int i = 0; i < 4; ++i)
        cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);

    float* h_dst[4] = { cpu_r, cpu_g, cpu_b, cpu_a };

    // ---- Per-channel: blur(FP32) → DMA(FP32) ----
    for (int ch = 0; ch < 3; ++ch)
    {
        if (do_blur)
        {
            cudaMemcpyAsync(cache.blur_a, d_ch[ch], n_bytes_f,
                            cudaMemcpyDeviceToDevice, comp);
            for (int p = 0; p < blur_passes; ++p)
            {
                launch_blur_h_on_stream(cache.blur_a, cache.blur_b, w, h,
                                        blur_radius, static_cast<void*>(comp));
                launch_blur_v_on_stream(cache.blur_b, cache.blur_a, w, h,
                                        blur_radius, static_cast<void*>(comp));
            }
            cudaMemcpyAsync(d_ch[ch], cache.blur_a, n_bytes_f,
                            cudaMemcpyDeviceToDevice, comp);
        }

        // Signal: this channel is ready.
        cudaEventRecord(ev[ch], comp);

        // Async DMA: FP32 device → FP32 pinned host.
        cudaStreamWaitEvent(copy, ev[ch], 0);
        cudaMemcpyAsync(h_dst[ch], d_ch[ch], n_bytes_f,
                        cudaMemcpyDeviceToHost, copy);
    }

    // ---- Alpha: compute → DMA(FP32) ----
    {
        constexpr int BLK = 256;
        int grid = ((int)n_px + BLK - 1) / BLK;
        alpha_kernel<<<grid, BLK, 0, comp>>>(
            cache.d_out_r, cache.d_out_g, cache.d_out_b,
            cache.d_out_a, (int)n_px);
        cudaEventRecord(ev[3], comp);
    }

    if (h_dst[3])
    {
        cudaStreamWaitEvent(copy, ev[3], 0);
        cudaMemcpyAsync(h_dst[3], cache.d_out_a, n_bytes_f,
                        cudaMemcpyDeviceToHost, copy);
    }

    // ---- Wait for everything ----
    cudaStreamSynchronize(copy);
    cudaStreamSynchronize(comp);

    for (int i = 0; i < 4; ++i)
        cudaEventDestroy(ev[i]);
}
