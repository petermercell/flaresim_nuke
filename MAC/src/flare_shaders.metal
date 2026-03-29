// ============================================================================
// flare_shaders.metal — Metal compute shaders for FlareSim
//
// Port of ghost_cuda.cu + blur_cuda.cu to Metal Shading Language (MSL).
//
// Apple Silicon unified memory eliminates all H2D / D2H copies.
// SIMD-group intrinsics replace CUDA warp primitives for cooperative scatter.
//
// Minimum: Apple Silicon (M1+), macOS 13+, Metal 3.0
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// Shared data structures — must match the C++ host structs exactly
// ===========================================================================

// Lens surface (POD, matches Surface in lens.h minus unused fields)
struct Surface
{
    float radius;
    float thickness;
    float ior;
    float abbe_v;
    float semi_aperture;
    int   coating;
    int   is_stop;    // bool as int for Metal alignment
    float z;
};

struct GPUPair
{
    int   surf_a;
    int   surf_b;
    float area_boost;
    int   tile_cx;
    int   tile_cy;
    float color_r;
    float color_g;
    float color_b;
    float offset_x;
    float offset_y;
    float scale;
};

struct GPUSource
{
    float angle_x;
    float angle_y;
    float r;
    float g;
    float b;
};

struct GPUSample
{
    float u;
    float v;
};

struct GPUSpectralSample
{
    float lambda;
    float rw;
    float gw;
    float bw;
};

// Uniform parameters for the ghost kernel (avoids excessive buffer bindings)
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
};

// ===========================================================================
// Device-side math — mirrors vec3.h / fresnel.h / trace.cpp
// ===========================================================================

struct DVec3 { float x, y, z; };

inline DVec3 dv(float x, float y, float z) { return {x, y, z}; }
inline DVec3 dv_add(DVec3 a, DVec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline DVec3 dv_sub(DVec3 a, DVec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline DVec3 dv_scale(DVec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
inline DVec3 dv_neg(DVec3 a) { return {-a.x, -a.y, -a.z}; }
inline float dv_dot(DVec3 a, DVec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline DVec3 dv_normalize(DVec3 v)
{
    float inv = rsqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return {v.x*inv, v.y*inv, v.z*inv};
}

struct DRay { DVec3 origin; DVec3 dir; };

// ---- Dispersion (Cauchy via Abbe number) ----

inline float d_dispersion_ior(float n_d, float V_d, float lambda_nm)
{
    if (V_d < 0.1f || n_d <= 1.0001f) return n_d;

    constexpr float lF = 486.13f, lC = 656.27f, ld = 587.56f;
    constexpr float inv_lF2 = 1.0f / (lF * lF);
    constexpr float inv_lC2 = 1.0f / (lC * lC);
    constexpr float inv_ld2 = 1.0f / (ld * ld);
    constexpr float inv_diff = 1.0f / (inv_lF2 - inv_lC2);

    float dn = (n_d - 1.0f) / V_d;
    float B  = dn * inv_diff;
    float A  = n_d - B * inv_ld2;
    return A + B / (lambda_nm * lambda_nm);
}

inline float d_ior_at(const device Surface& s, float lambda_nm)
{
    return d_dispersion_ior(s.ior, s.abbe_v, lambda_nm);
}

// Overload for threadgroup surfaces
inline float d_ior_at(threadgroup const Surface& s, float lambda_nm)
{
    return d_dispersion_ior(s.ior, s.abbe_v, lambda_nm);
}

inline float d_ior_before(const threadgroup Surface* surfs, int idx, float lambda_nm)
{
    return (idx <= 0) ? 1.0f : d_ior_at(surfs[idx - 1], lambda_nm);
}

// ---- Fresnel reflectance ----

inline float d_fresnel_reflectance(float cos_i, float n1, float n2)
{
    cos_i = abs(cos_i);
    float eta    = n1 / n2;
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f) return 1.0f;  // TIR
    float cos_t  = sqrt(1.0f - sin2_t);
    float rs     = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t);
    float rp     = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t);
    return 0.5f * (rs*rs + rp*rp);
}

// Single-layer MgF2 AR coating
inline float d_coating_reflectance(float cos_i, float n1, float n2,
                                    float coating_n, float d_nm, float lambda_nm)
{
    float ratio1 = n1 / coating_n;
    float sin2_c = ratio1 * ratio1 * (1.0f - cos_i*cos_i);
    if (sin2_c >= 1.0f) return d_fresnel_reflectance(cos_i, n1, n2);
    float cos_c  = sqrt(1.0f - sin2_c);

    constexpr float TWO_PI = 2.0f * 3.14159265358979323846f;
    float delta  = TWO_PI * coating_n * d_nm * cos_c / lambda_nm;
    float r01    = (n1*cos_i - coating_n*cos_c) / (n1*cos_i + coating_n*cos_c);

    float ratio2 = coating_n / n2;
    float sin2_2 = ratio2 * ratio2 * (1.0f - cos_c*cos_c);
    if (sin2_2 >= 1.0f) return d_fresnel_reflectance(cos_i, n1, n2);
    float cos_2  = sqrt(1.0f - sin2_2);
    float r12    = (coating_n*cos_c - n2*cos_2) / (coating_n*cos_c + n2*cos_2);

    float cos_2d = cos(2.0f * delta);
    float r01r12 = r01 * r12;
    float num    = r01*r01 + r12*r12 + 2.0f*r01r12*cos_2d;
    float den    = 1.0f   + r01*r01*r12*r12 + 2.0f*r01r12*cos_2d;
    float R      = num / den;
    return clamp(R, 0.0f, 1.0f);
}

inline float d_surface_reflectance(float cos_i, float n1, float n2,
                                    int coating_layers, float lambda_nm)
{
    if (coating_layers <= 0)
        return d_fresnel_reflectance(cos_i, n1, n2);

    constexpr float mgf2_n        = 1.38f;
    constexpr float design_lambda = 550.0f;
    constexpr float qw_thick      = design_lambda / (4.0f * mgf2_n);
    float R = d_coating_reflectance(cos_i, n1, n2, mgf2_n, qw_thick, lambda_nm);
    for (int i = 1; i < coating_layers; ++i) R *= 0.25f;
    return clamp(R, 0.0f, 1.0f);
}

// ---- Ray–surface intersection ----

inline bool d_intersect_surface(DRay ray, threadgroup const Surface& surf,
                                 thread DVec3& hit, thread DVec3& norm)
{
    if (abs(surf.radius) < 1e-6f)
    {
        if (abs(ray.dir.z) < 1e-12f) return false;
        float t = (surf.z - ray.origin.z) / ray.dir.z;
        if (!(t > 1e-6f)) return false;

        hit = dv(ray.origin.x + ray.dir.x*t,
                 ray.origin.y + ray.dir.y*t,
                 ray.origin.z + ray.dir.z*t);

        if (!(hit.x*hit.x + hit.y*hit.y <= surf.semi_aperture*surf.semi_aperture))
            return false;

        norm = dv(0.0f, 0.0f, (ray.dir.z > 0) ? -1.0f : 1.0f);
        return true;
    }

    float R   = surf.radius;
    DVec3 ctr = dv(0.0f, 0.0f, surf.z + R);
    DVec3 oc  = dv_sub(ray.origin, ctr);

    float a    = dv_dot(ray.dir, ray.dir);
    float b    = 2.0f * dv_dot(oc, ray.dir);
    float c    = dv_dot(oc, oc) - R*R;
    float disc = b*b - 4.0f*a*c;
    if (disc < 0.0f) return false;

    float sd    = sqrt(disc);
    float inv2a = 0.5f / a;
    float t1    = (-b - sd) * inv2a;
    float t2    = (-b + sd) * inv2a;

    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1*ray.dir.z;
        float z2 = ray.origin.z + t2*ray.dir.z;
        t = (abs(z1 - surf.z) < abs(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f) t = t1;
    else if (t2 > 1e-6f) t = t2;
    else return false;

    hit = dv(ray.origin.x + ray.dir.x*t,
             ray.origin.y + ray.dir.y*t,
             ray.origin.z + ray.dir.z*t);

    if (!(hit.x*hit.x + hit.y*hit.y <= surf.semi_aperture*surf.semi_aperture))
        return false;

    float invR = 1.0f / abs(R);
    norm = dv((hit.x - ctr.x)*invR,
              (hit.y - ctr.y)*invR,
              (hit.z - ctr.z)*invR);
    if (dv_dot(norm, ray.dir) > 0.0f) norm = dv_neg(norm);
    return true;
}

// ---- Refraction / reflection ----

inline bool d_refract(DVec3 dir, DVec3 norm, float n_ratio, thread DVec3& out)
{
    float cos_i  = -dv_dot(norm, dir);
    float sin2_t = n_ratio*n_ratio*(1.0f - cos_i*cos_i);
    if (sin2_t >= 1.0f) return false;
    float cos_t  = sqrt(1.0f - sin2_t);
    float k      = n_ratio*cos_i - cos_t;
    float ox = dir.x*n_ratio + norm.x*k;
    float oy = dir.y*n_ratio + norm.y*k;
    float oz = dir.z*n_ratio + norm.z*k;
    float sq = ox*ox + oy*oy + oz*oz;
    if (sq < 1e-18f || !isfinite(sq)) return false;
    float inv = rsqrt(sq);
    out = dv(ox*inv, oy*inv, oz*inv);
    return true;
}

inline bool d_reflect(DVec3 dir, DVec3 norm, thread DVec3& out)
{
    float d2 = 2.0f * dv_dot(dir, norm);
    float ox = dir.x - norm.x*d2;
    float oy = dir.y - norm.y*d2;
    float oz = dir.z - norm.z*d2;
    float sq = ox*ox + oy*oy + oz*oz;
    if (sq < 1e-18f || !isfinite(sq)) return false;
    float inv = rsqrt(sq);
    out = dv(ox*inv, oy*inv, oz*inv);
    return true;
}

// ===========================================================================
// Device-side ghost ray trace — mirrors trace_ghost_ray() in trace.cpp
// ===========================================================================

struct DTraceResult { DVec3 position; float weight; bool valid; };

inline DTraceResult d_trace_ghost_ray(DRay ray_in,
                                       threadgroup const Surface* surfs,
                                       int n_surfs,
                                       float sensor_z,
                                       int bounce_a, int bounce_b,
                                       float lambda_nm,
                                       float min_weight)
{
    DRay  ray         = ray_in;
    float current_ior = 1.0f;
    float weight      = 1.0f;

    // Phase 1: forward 0..bounce_b (reflect at bounce_b)
    for (int s = 0; s <= bounce_b; ++s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfs[s], hit, norm))
            return {dv(0,0,0), 0.0f, false};

        ray.origin = hit;
        float n1    = current_ior;
        float n2    = d_ior_at(surfs[s], lambda_nm);
        float cos_i = abs(dv_dot(norm, ray.dir));
        float R     = d_surface_reflectance(cos_i, n1, n2, surfs[s].coating, lambda_nm);

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
            if (!d_refract(ray.dir, norm, n1 / n2, new_dir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir     = new_dir;
            weight     *= (1.0f - R);
            current_ior = n2;
        }
        if (weight < min_weight)
            return {dv(0,0,0), 0.0f, false};
    }

    // Phase 2: backward bounce_b-1..bounce_a (reflect at bounce_a)
    for (int s = bounce_b - 1; s >= bounce_a; --s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfs[s], hit, norm))
            return {dv(0,0,0), 0.0f, false};

        ray.origin = hit;
        float n1    = current_ior;
        float n2    = d_ior_before(surfs, s, lambda_nm);
        float cos_i = abs(dv_dot(norm, ray.dir));
        float R     = d_surface_reflectance(cos_i, n1, n2, surfs[s].coating, lambda_nm);

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
            if (!d_refract(ray.dir, norm, n1 / n2, new_dir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir     = new_dir;
            weight     *= (1.0f - R);
            current_ior = n2;
        }
        if (weight < min_weight)
            return {dv(0,0,0), 0.0f, false};
    }

    // Phase 3: forward bounce_a+1..n_surfs-1
    for (int s = bounce_a + 1; s < n_surfs; ++s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfs[s], hit, norm))
            return {dv(0,0,0), 0.0f, false};

        ray.origin = hit;
        float n1    = current_ior;
        float n2    = d_ior_at(surfs[s], lambda_nm);
        float cos_i = abs(dv_dot(norm, ray.dir));
        float R     = d_surface_reflectance(cos_i, n1, n2, surfs[s].coating, lambda_nm);

        DVec3 new_dir;
        if (!d_refract(ray.dir, norm, n1 / n2, new_dir))
            return {dv(0,0,0), 0.0f, false};
        ray.dir     = new_dir;
        weight     *= (1.0f - R);
        current_ior = n2;

        if (weight < min_weight)
            return {dv(0,0,0), 0.0f, false};
    }

    // Propagate to sensor plane
    if (abs(ray.dir.z) < 1e-12f)
        return {dv(0,0,0), 0.0f, false};
    float t = (sensor_z - ray.origin.z) / ray.dir.z;
    if (!(t > 0.0f))
        return {dv(0,0,0), 0.0f, false};

    DVec3 pos = dv(ray.origin.x + ray.dir.x*t,
                   ray.origin.y + ray.dir.y*t,
                   ray.origin.z + ray.dir.z*t);
    return {pos, weight, true};
}

// ===========================================================================
// Ghost scatter kernel
// ===========================================================================

kernel void ghost_kernel(
    const device Surface*            d_surfs       [[buffer(0)]],
    const device GPUPair*            d_pairs       [[buffer(1)]],
    const device GPUSource*          d_sources     [[buffer(2)]],
    const device GPUSample*          d_grid        [[buffer(3)]],
    const device GPUSpectralSample*  d_spec        [[buffer(4)]],
    device atomic_float*             d_out_r       [[buffer(5)]],
    device atomic_float*             d_out_g       [[buffer(6)]],
    device atomic_float*             d_out_b       [[buffer(7)]],
    constant GhostParams&            params        [[buffer(8)]],
    uint3                            tid           [[thread_position_in_threadgroup]],
    uint3                            gid           [[threadgroup_position_in_grid]],
    uint3                            tg_size       [[threads_per_threadgroup]])
{
    // Threadgroup memory: surface cache + shared beam direction.
    //
    // NOTE: the CUDA version has a 32×32 threadgroup tile accumulator that
    // batches atomics via shared-memory atomicAdd, then flushes to global.
    // Metal does NOT support threadgroup atomic_float, so we skip the tile
    // and always use global atomics.  Apple Silicon's L2 cache has very low
    // atomic latency (~4 ns vs NVIDIA's ~30 ns for global), so the
    // performance difference is small.  If profiling shows this is a
    // bottleneck, a CAS-loop on threadgroup atomic_uint can be added.
    threadgroup Surface s_surfs[64];   // MAX_SURFACES
    threadgroup float   s_beam[3];

    const int local_id = (int)tid.x;
    const int ps_idx   = (int)gid.x;
    const int pair_idx = ps_idx / params.n_sources;
    const int src_idx  = ps_idx % params.n_sources;
    const int grid_idx = (int)gid.y * (int)tg_size.x + local_id;

    const int clamped_n = min(params.n_surfs, 64);

    // Cooperative load: surfaces into threadgroup memory
    for (int i = local_id; i < clamped_n; i += (int)tg_size.x)
        s_surfs[i] = d_surfs[i];

    // Thread 0: compute shared beam direction
    if (local_id == 0)
    {
        const device GPUSource& src = d_sources[src_idx];
        float bx = tan(src.angle_x);
        float by = tan(src.angle_y);
        float inv = rsqrt(bx*bx + by*by + 1.0f);
        s_beam[0] = bx * inv;
        s_beam[1] = by * inv;
        s_beam[2] = inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device GPUPair& pair = d_pairs[pair_idx];

    const bool active = (grid_idx < params.n_grid) &&
                        (pair.surf_a >= 0) &&
                        (pair.surf_b > pair.surf_a) &&
                        (pair.surf_b < clamped_n);

    if (!active) return;

    const device GPUSource& src = d_sources[src_idx];
    const device GPUSample& gs  = d_grid[grid_idx];

    DVec3 beam_dir = dv(s_beam[0], s_beam[1], s_beam[2]);

    DRay ray;
    ray.origin = dv(gs.u * params.front_R, gs.v * params.front_R, params.start_z);
    ray.dir    = beam_dir;

    // Adaptive early-termination threshold
    const float max_src   = max(src.r, max(src.g, src.b));
    const float scale_val = params.ray_weight * params.gain * pair.area_boost * max(max_src, 1e-10f);
    const float min_weight = min(1e-5f / max(scale_val, 1e-20f), 0.01f);

    for (int s = 0; s < params.n_spec; ++s)
    {
        const device GPUSpectralSample& spec = d_spec[s];

        float src_contrib = src.r * spec.rw + src.g * spec.gw + src.b * spec.bw;
        if (src_contrib < 1e-14f) continue;

        DTraceResult res = d_trace_ghost_ray(
            ray, s_surfs, clamped_n, params.sensor_z,
            pair.surf_a, pair.surf_b, spec.lambda,
            min_weight);

        if (!res.valid) continue;
        if (!isfinite(res.position.x) || !isfinite(res.position.y)) continue;

        float base = res.weight * params.ray_weight * params.gain * pair.area_boost;
        if (base < 1e-14f) continue;

        // Sensor → pixel + per-pair transform
        float px = (res.position.x / (2.0f * params.sensor_half_w) + 0.5f)
                   * params.fmt_w + params.fmt_x0_in_buf;
        float py = (res.position.y / (2.0f * params.sensor_half_h) + 0.5f)
                   * params.fmt_h + params.fmt_y0_in_buf;

        float fcx = params.fmt_x0_in_buf + params.fmt_w * 0.5f;
        float fcy = params.fmt_y0_in_buf + params.fmt_h * 0.5f;
        px = fcx + (px - fcx) * pair.scale + pair.offset_x;
        py = fcy + (py - fcy) * pair.scale + pair.offset_y;

        if (!isfinite(px) || !isfinite(py)) continue;
        px = clamp(px, -2.0e9f, 2.0e9f);
        py = clamp(py, -2.0e9f, 2.0e9f);

        int   x0 = (int)floor(px - 0.5f);
        int   y0 = (int)floor(py - 0.5f);
        float fx = (px - 0.5f) - (float)x0;
        float fy = (py - 0.5f) - (float)y0;

        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx           * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx           * fy;

        float cr = src.r * spec.rw * base * pair.color_r;
        float cg = src.g * spec.gw * base * pair.color_g;
        float cb = src.b * spec.bw * base * pair.color_b;

        // Bilinear splat — 4 corners, global atomics
        const int   cx[4] = { x0, x0+1, x0,   x0+1 };
        const int   cy[4] = { y0, y0,   y0+1, y0+1 };
        const float cw[4] = { w00, w10,  w01,  w11  };

        for (int corner = 0; corner < 4; ++corner)
        {
            int xi = cx[corner];
            int yi = cy[corner];
            if (xi < 0 || xi >= params.width || yi < 0 || yi >= params.height)
                continue;

            float wt = cw[corner];
            int pix = yi * params.width + xi;

            float vr = cr * wt;
            float vg = cg * wt;
            float vb = cb * wt;

            if (vr > 1e-14f)
                atomic_fetch_add_explicit(&d_out_r[pix], vr, memory_order_relaxed);
            if (vg > 1e-14f)
                atomic_fetch_add_explicit(&d_out_g[pix], vg, memory_order_relaxed);
            if (vb > 1e-14f)
                atomic_fetch_add_explicit(&d_out_b[pix], vb, memory_order_relaxed);
        }
    }
}

// ===========================================================================
// Alpha kernel — Rec.709 luminance, clamped [0, 1]
// ===========================================================================

kernel void alpha_kernel(
    const device float* d_r   [[buffer(0)]],
    const device float* d_g   [[buffer(1)]],
    const device float* d_b   [[buffer(2)]],
    device       float* d_a   [[buffer(3)]],
    constant     int&   n_px  [[buffer(4)]],
    uint                gid   [[thread_position_in_grid]])
{
    if ((int)gid >= n_px) return;
    float lum = 0.2126f * d_r[gid] + 0.7152f * d_g[gid] + 0.0722f * d_b[gid];
    d_a[gid] = min(lum, 1.0f);
}

// ===========================================================================
// FP32 → FP16 conversion kernel
// ===========================================================================

kernel void fp32_to_fp16_kernel(
    const device float* in    [[buffer(0)]],
    device       half*  out   [[buffer(1)]],
    constant     int&   n     [[buffer(2)]],
    uint                gid   [[thread_position_in_grid]])
{
    if ((int)gid < n)
        out[gid] = half(in[gid]);
}

// ===========================================================================
// Clear buffer kernel (memset to zero)
// ===========================================================================

kernel void clear_buffer_kernel(
    device float* buf    [[buffer(0)]],
    constant int& n      [[buffer(1)]],
    uint          gid    [[thread_position_in_grid]])
{
    if ((int)gid < n)
        buf[gid] = 0.0f;
}

// ===========================================================================
// Box blur — prefix-sum horizontal
//
// One threadgroup per row.  Cooperative load + sequential prefix sum by
// thread 0 + O(1) output per thread.
// ===========================================================================

kernel void box_blur_h_prefix_kernel(
    const device float* in      [[buffer(0)]],
    device       float* out     [[buffer(1)]],
    constant     int&   w       [[buffer(2)]],
    constant     int&   h       [[buffer(3)]],
    constant     int&   radius  [[buffer(4)]],
    threadgroup  float* prefix  [[threadgroup(0)]],  // (w+1) floats
    uint                row_idx [[threadgroup_position_in_grid]],
    uint                lid     [[thread_position_in_threadgroup]],
    uint                tg_sz   [[threads_per_threadgroup]])
{
    const int y = (int)row_idx;
    if (y >= h) return;

    const int row_off = y * w;

    // Load row into prefix[1..w], set prefix[0] = 0
    prefix[0] = 0.0f;
    for (int i = (int)lid; i < w; i += (int)tg_sz)
        prefix[i + 1] = in[row_off + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sequential prefix sum (thread 0)
    if (lid == 0)
    {
        for (int i = 1; i <= w; ++i)
            prefix[i] += prefix[i - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes blurred output in O(1)
    for (int x = (int)lid; x < w; x += (int)tg_sz)
    {
        int lo = max(0,     x - radius);
        int hi = min(w - 1, x + radius);
        out[row_off + x] = (prefix[hi + 1] - prefix[lo]) / (float)(hi - lo + 1);
    }
}

// ===========================================================================
// Box blur — sliding-window vertical
//
// One thread per column.  O(1) amortised per output pixel.
// ===========================================================================

kernel void box_blur_v_sliding_kernel(
    const device float* in      [[buffer(0)]],
    device       float* out     [[buffer(1)]],
    constant     int&   w       [[buffer(2)]],
    constant     int&   h       [[buffer(3)]],
    constant     int&   radius  [[buffer(4)]],
    uint                gid     [[thread_position_in_grid]])
{
    const int x = (int)gid;
    if (x >= w) return;

    const int init_hi = min(radius, h - 1);
    float sum = 0.0f;
    for (int i = 0; i <= init_hi; ++i)
        sum += in[i * w + x];

    for (int y = 0; y < h; ++y)
    {
        if (y > 0)
        {
            int new_hi = y + radius;
            if (new_hi < h)
                sum += in[new_hi * w + x];
            int old_lo = y - radius - 1;
            if (old_lo >= 0)
                sum -= in[old_lo * w + x];
        }

        int lo = max(0,     y - radius);
        int hi = min(h - 1, y + radius);
        out[y * w + x] = sum / (float)(hi - lo + 1);
    }
}

// ===========================================================================
// Naive fallback blur kernels (for rows wider than threadgroup memory limit)
// ===========================================================================

kernel void box_blur_h_naive_kernel(
    const device float* in      [[buffer(0)]],
    device       float* out     [[buffer(1)]],
    constant     int&   w       [[buffer(2)]],
    constant     int&   h       [[buffer(3)]],
    constant     int&   radius  [[buffer(4)]],
    uint2               gid     [[thread_position_in_grid]])
{
    const int x = (int)gid.x;
    const int y = (int)gid.y;
    if (x >= w || y >= h) return;

    const int lo = max(0,     x - radius);
    const int hi = min(w - 1, x + radius);
    const int row_off = y * w;

    float sum = 0.0f;
    for (int i = lo; i <= hi; ++i)
        sum += in[row_off + i];

    out[row_off + x] = sum / (float)(hi - lo + 1);
}
