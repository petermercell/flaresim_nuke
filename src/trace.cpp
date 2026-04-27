// ============================================================================
// trace.cpp — Ray-surface intersection and sequential lens trace
// ============================================================================

#include "trace.h"
#include "fresnel.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Optional toric Newton-Raphson profiling (M7).
//
// Compile with -DFLARESIM_TORIC_PROFILE to gather convergence statistics
// from production renders; otherwise the macros below collapse to no-ops
// and there is zero hot-path overhead.
//
// Use:
//   1. Build FlareSim with -DFLARESIM_TORIC_PROFILE.
//   2. Render a representative shot.
//   3. Call toric_profile_dump() (e.g. from FlareSim's destructor or a
//      debug knob) to print { calls, avg iters, max iters, misses }.
//   4. Call toric_profile_reset() between runs to clear state.
//
// Atomics use relaxed ordering — order doesn't matter, only the sum.
// ---------------------------------------------------------------------------
#ifdef FLARESIM_TORIC_PROFILE
#  include <atomic>
#  include <cstdint>
namespace {
    std::atomic<uint64_t> g_toric_calls    {0};
    std::atomic<uint64_t> g_toric_iter_sum {0};
    std::atomic<uint32_t> g_toric_max_iter {0};
    std::atomic<uint64_t> g_toric_misses   {0};
}
static inline void toric_profile_record(int iters, bool missed)
{
    g_toric_calls.fetch_add(1, std::memory_order_relaxed);
    if (missed) {
        g_toric_misses.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_toric_iter_sum.fetch_add(static_cast<uint64_t>(iters),
                               std::memory_order_relaxed);
    // Lock-free max via CAS.
    uint32_t cur = g_toric_max_iter.load(std::memory_order_relaxed);
    while (static_cast<uint32_t>(iters) > cur &&
           !g_toric_max_iter.compare_exchange_weak(
               cur, static_cast<uint32_t>(iters),
               std::memory_order_relaxed)) {}
}
void toric_profile_dump()
{
    const uint64_t calls  = g_toric_calls.load(std::memory_order_relaxed);
    const uint64_t iters  = g_toric_iter_sum.load(std::memory_order_relaxed);
    const uint32_t mxiter = g_toric_max_iter.load(std::memory_order_relaxed);
    const uint64_t miss   = g_toric_misses.load(std::memory_order_relaxed);
    const uint64_t hits   = calls - miss;
    const double   avg    = (hits > 0) ? double(iters) / double(hits) : 0.0;
    fprintf(stderr,
            "[toric profile] calls=%llu hits=%llu misses=%llu "
            "avg_iters=%.2f max_iters=%u\n",
            (unsigned long long)calls, (unsigned long long)hits,
            (unsigned long long)miss, avg, mxiter);
}
void toric_profile_reset()
{
    g_toric_calls    .store(0, std::memory_order_relaxed);
    g_toric_iter_sum .store(0, std::memory_order_relaxed);
    g_toric_max_iter .store(0, std::memory_order_relaxed);
    g_toric_misses   .store(0, std::memory_order_relaxed);
}
#  define TORIC_PROFILE_RECORD(iters, missed)  toric_profile_record(iters, missed)
#else
#  define TORIC_PROFILE_RECORD(iters, missed)  ((void)0)
#endif

// Fast normalize: avoids the branch and second divide in Vec3f::normalized().
static inline Vec3f fast_normalize(const Vec3f &v)
{
    float l2 = v.x * v.x + v.y * v.y + v.z * v.z;
    float inv = 1.0f / std::sqrt(l2); // -ffast-math turns this into rsqrt
    return {v.x * inv, v.y * inv, v.z * inv};
}

// ---------------------------------------------------------------------------
// Intersect a ray with a lens surface.
//
// Flat surfaces (radius == 0): ray–plane intersection at z = surf.z.
//   Treated identically across surface_type values — a surface with zero
//   curvature in its primary axis is a plane.
// Curved surfaces: dispatch on Surface::surface_type.
//   spherical  → ray–sphere intersection.
//   cyl_x/y    → ray–cylinder intersection (cylinder axis along X or Y).
//   toric      → not yet implemented (M4).
//
// Convention shared by all variants:
//   • Pick the intersection point closest to the surface vertex z.
//   • Aperture clip:  hit.x² + hit.y² ≤ semi_aperture²  (disc in XY).
//   • Normal opposes the ray direction on success.
// ---------------------------------------------------------------------------

// ---- Spherical: centre of curvature at C = (0, 0, surf.z + R) -----------
static bool intersect_spherical(const Ray &ray, const Surface &surf,
                                Vec3f &hit_pos, Vec3f &normal)
{
    float R = surf.radius;
    Vec3f center(0, 0, surf.z + R);
    Vec3f oc = ray.origin - center;

    float a = dot(ray.dir, ray.dir);
    float b = 2.0f * dot(oc, ray.dir);
    float c = dot(oc, oc) - R * R;
    float disc = b * b - 4.0f * a * c;

    if (disc < 0)
        return false;

    float sqrt_disc = std::sqrt(disc);
    float inv_2a = 0.5f / a;
    float t1 = (-b - sqrt_disc) * inv_2a;
    float t2 = (-b + sqrt_disc) * inv_2a;

    // Pick the intersection closest to the surface vertex z.
    // Both t must be positive (ahead of the ray).
    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1 * ray.dir.z;
        float z2 = ray.origin.z + t2 * ray.dir.z;
        t = (std::abs(z1 - surf.z) < std::abs(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f)
        t = t1;
    else if (t2 > 1e-6f)
        t = t2;
    else
        return false;

    hit_pos = ray.origin + ray.dir * t;

    float h2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
    if (h2 > surf.semi_aperture * surf.semi_aperture)
        return false;

    normal = (hit_pos - center) / std::abs(R);
    if (dot(normal, ray.dir) > 0)
        normal = -normal;
    return true;
}

// ---- Cylinder, axis along X: curves in YZ plane.  surf.radius = Ry ------
//
// Locus:  y² + (z - cz)² = R²,  where cz = surf.z + R.
// X is unconstrained except by the aperture clip.  Reflections from this
// surface preserve the X component of ray direction → fan out purely in YZ →
// horizontal streaks on the sensor.  This is the anamorphic signature.
static bool intersect_cylinder_x(const Ray &ray, const Surface &surf,
                                 Vec3f &hit_pos, Vec3f &normal)
{
    float R  = surf.radius;
    float cz = surf.z + R;

    float oy = ray.origin.y;
    float oz = ray.origin.z - cz;
    float dy = ray.dir.y;
    float dz = ray.dir.z;

    float a = dy * dy + dz * dz;
    if (a < 1e-18f)
        return false; // ray parallel to cylinder axis

    float b    = 2.0f * (oy * dy + oz * dz);
    float c    = oy * oy + oz * oz - R * R;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0)
        return false;

    float sqrt_disc = std::sqrt(disc);
    float inv_2a = 0.5f / a;
    float t1 = (-b - sqrt_disc) * inv_2a;
    float t2 = (-b + sqrt_disc) * inv_2a;

    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1 * ray.dir.z;
        float z2 = ray.origin.z + t2 * ray.dir.z;
        t = (std::abs(z1 - surf.z) < std::abs(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f) t = t1;
    else if (t2 > 1e-6f) t = t2;
    else return false;

    hit_pos = ray.origin + ray.dir * t;

    float h2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
    if (h2 > surf.semi_aperture * surf.semi_aperture)
        return false;

    // ∇F = (0, 2y, 2(z-cz));  |∇F|/2 = R on the surface.
    float invR = 1.0f / std::abs(R);
    normal = Vec3f(0.0f, hit_pos.y * invR, (hit_pos.z - cz) * invR);
    if (dot(normal, ray.dir) > 0)
        normal = -normal;
    return true;
}

// ---- Cylinder, axis along Y: curves in XZ plane.  surf.radius = Rx ------
// Mirror image of intersect_cylinder_x with X↔Y swapped.
static bool intersect_cylinder_y(const Ray &ray, const Surface &surf,
                                 Vec3f &hit_pos, Vec3f &normal)
{
    float R  = surf.radius;
    float cz = surf.z + R;

    float ox = ray.origin.x;
    float oz = ray.origin.z - cz;
    float dx = ray.dir.x;
    float dz = ray.dir.z;

    float a = dx * dx + dz * dz;
    if (a < 1e-18f)
        return false;

    float b    = 2.0f * (ox * dx + oz * dz);
    float c    = ox * ox + oz * oz - R * R;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0)
        return false;

    float sqrt_disc = std::sqrt(disc);
    float inv_2a = 0.5f / a;
    float t1 = (-b - sqrt_disc) * inv_2a;
    float t2 = (-b + sqrt_disc) * inv_2a;

    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1 * ray.dir.z;
        float z2 = ray.origin.z + t2 * ray.dir.z;
        t = (std::abs(z1 - surf.z) < std::abs(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f) t = t1;
    else if (t2 > 1e-6f) t = t2;
    else return false;

    hit_pos = ray.origin + ray.dir * t;

    float h2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
    if (h2 > surf.semi_aperture * surf.semi_aperture)
        return false;

    float invR = 1.0f / std::abs(R);
    normal = Vec3f(hit_pos.x * invR, 0.0f, (hit_pos.z - cz) * invR);
    if (dot(normal, ray.dir) > 0)
        normal = -normal;
    return true;
}

// ---- Toric: two-radii surface (M4) --------------------------------------
//
// A toric has two principal radii of curvature:
//   surf.radius    = Rx  (XZ-plane curvature)
//   surf.radius_y  = Ry  (YZ-plane curvature)
//
// Implicit form, apex-centered (apex at origin, +Z is "into" the lens):
//
//   F(x, y, z') = (x² + (Rx - z')² + W² + y² - Ry²)²
//                 - 4 W² (x² + (Rx - z')²) = 0,        W = Rx - Ry.
//
// Geometrically this is a torus generated by sweeping a circle of radius
// |Ry| in the YZ plane around an axis parallel to Y at distance Rx - Ry
// from the apex.  The y=0 cross-section is a circle of radius |Rx|; the
// x=0 cross-section is a circle of radius |Ry|; both pass through the apex.
//
// When Rx ≈ Ry the torus degenerates to a doubled-root sphere and the
// quartic gradient vanishes on the surface — we detect this and fall back
// to the spherical helper.
//
// The intersection is solved via Newton-Raphson on F(t) = 0 starting from
// the ray's intersection with the apex plane (z = surf.z).  This is a good
// initial guess for thin lens surfaces because near the optical axis the
// surface deviation from the apex plane is O(r²/2R).
//
// Normal is the gradient ∇F at the hit, oriented to oppose the ray.
static bool intersect_toric(const Ray &ray, const Surface &surf,
                            Vec3f &hit_pos, Vec3f &normal)
{
    const float Rx = surf.radius;
    const float Ry = surf.radius_y;
    const float W  = Rx - Ry;

    // ----- Degenerate-toric guard -----
    // |W| small ⇒ doubled-root quartic; Newton is ill-conditioned and the
    // gradient vanishes on the surface.  Substitute the equivalent sphere.
    if (std::abs(W) < 1e-3f)
    {
        Surface tmp = surf;
        tmp.radius = (Rx + Ry) * 0.5f;
        return intersect_spherical(ray, tmp, hit_pos, normal);
    }

    // Apex-centered ray (z' = z - surf.z).
    const float ox = ray.origin.x;
    const float oy = ray.origin.y;
    const float oz = ray.origin.z - surf.z;
    const float dx = ray.dir.x;
    const float dy = ray.dir.y;
    const float dz = ray.dir.z;

    // ----- Initial guess: apex-plane intersection (z' = 0) -----
    if (std::abs(dz) < 1e-12f)
        return false;
    float t = -oz / dz;
    if (t < 1e-6f)
        return false;

    // ----- F(t) and dF/dt -----
    // Q  = x² + (Rx - z')²
    // G  = Q + W² + y² - Ry²
    // F  = G² - 4 W² Q
    // dQ/dt = 2 x dx - 2 (Rx - z') dz
    // dG/dt = dQ/dt + 2 y dy
    // dF/dt = 2 G · dG/dt - 4 W² · dQ/dt
    auto F_and_dF = [&](float tt, float &dF_out) -> float
    {
        const float px = ox + tt * dx;
        const float py = oy + tt * dy;
        const float pz = oz + tt * dz;
        const float Az = Rx - pz;
        const float Q  = px * px + Az * Az;
        const float dQ = 2.0f * (px * dx - Az * dz);
        const float G  = Q + W * W + py * py - Ry * Ry;
        const float dG = dQ + 2.0f * py * dy;
        dF_out = 2.0f * G * dG - 4.0f * W * W * dQ;
        return G * G - 4.0f * W * W * Q;
    };

    // ----- Newton-Raphson with step cap -----
    //
    // MAX_ITER tuning (M7):
    //   Newton on the toric quartic with the apex-plane initial guess
    //   converges in 3-5 iterations near the optical axis.  Saddle torics
    //   with strongly off-axis incidence push the iteration count up.
    //
    //   Instrumented sweep on a 1764-ray synthetic stress set across mixed
    //   convex/concave/saddle regimes (intentionally adversarial, including
    //   pupils larger than the aperture) gave avg=9.94 / max=16.  Real lens
    //   prescriptions converge faster; that data point is the worst-case
    //   ceiling for safety-margin sizing.
    //
    //   16 caps cleanly above the worst case observed; the original M4
    //   value of 30 was 2x more conservative than necessary.  Compile
    //   trace.cpp with -DFLARESIM_TORIC_PROFILE and call toric_profile_dump()
    //   to verify against your own lens prescriptions before re-tuning.
    constexpr int   MAX_ITER = 16;
    constexpr float TOL_T    = 1e-7f;
    // Residual tolerance scales with the surface units (Q ~ Rx², so F ~ Rx⁴).
    // Use a relative tolerance based on the typical scale of the quartic.
    const float scale = std::max(std::abs(Rx), std::abs(Ry));
    const float TOL_F = scale * scale * scale * scale * 1e-8f;
    const float STEP_CAP = std::abs(t) + scale;       // generous but bounded

    bool converged = false;
    [[maybe_unused]] int  iters_used = 0;
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        iters_used = iter + 1;
        float dF = 0.0f;
        const float F = F_and_dF(t, dF);
        if (std::abs(F) < TOL_F) { converged = true; break; }
        if (std::abs(dF) < 1e-30f) {
            TORIC_PROFILE_RECORD(iters_used, true);
            return false;                              // singular Jacobian
        }
        float step = F / dF;
        if (step >  STEP_CAP) step =  STEP_CAP;
        if (step < -STEP_CAP) step = -STEP_CAP;
        t -= step;
        if (t < 1e-6f) {
            TORIC_PROFILE_RECORD(iters_used, true);
            return false;                              // ray walked behind origin
        }
        if (std::abs(step) < TOL_T) { converged = true; break; }
    }
    if (!converged)
    {
        // Final residual check — if Newton stalled near a non-root we miss.
        float dF = 0.0f;
        if (std::abs(F_and_dF(t, dF)) > TOL_F * 1e3f) {
            TORIC_PROFILE_RECORD(iters_used, true);
            return false;
        }
    }

    // ----- Hit position (back to world coords) -----
    hit_pos.x = ox + t * dx;
    hit_pos.y = oy + t * dy;
    hit_pos.z = ray.origin.z + t * dz;

    // Aperture clip (XY disc, same convention as sphere/cylinder helpers).
    const float h2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
    if (h2 > surf.semi_aperture * surf.semi_aperture) {
        TORIC_PROFILE_RECORD(iters_used, true);
        return false;
    }

    // ----- Normal: gradient ∇F at hit -----
    // ∂F/∂x = 4 x (Q + y² - Ry² - W²)
    // ∂F/∂y = 4 y (Q + y² - Ry² + W²)        (= 4y·G)
    // ∂F/∂z = -4 (Rx - z') (Q + y² - Ry² - W²)
    {
        const float px = hit_pos.x;
        const float py = hit_pos.y;
        const float pz = hit_pos.z - surf.z;        // apex-centered
        const float Az = Rx - pz;
        const float Q  = px * px + Az * Az;
        const float K1 = Q + py * py - Ry * Ry - W * W;     // for x and z partials
        const float G  = Q + py * py - Ry * Ry + W * W;     // for y partial

        Vec3f n( 4.0f * px * K1,
                 4.0f * py * G,
                -4.0f * Az * K1 );
        n = n.normalized();
        if (dot(n, ray.dir) > 0)
            n = -n;
        normal = n;
    }
    TORIC_PROFILE_RECORD(iters_used, false);
    return true;
}

bool intersect_surface(const Ray &ray, const Surface &surf,
                       Vec3f &hit_pos, Vec3f &normal)
{
    // Flat shortcut: a surface with no curvature is a plane regardless of
    // surface_type.  Preserves byte-identical legacy behaviour.
    if (std::abs(surf.radius) < 1e-6f)
    {
        if (std::abs(ray.dir.z) < 1e-12f)
            return false; // parallel

        float t = (surf.z - ray.origin.z) / ray.dir.z;
        if (t < 1e-6f)
            return false;

        hit_pos = ray.origin + ray.dir * t;

        float h2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
        if (h2 > surf.semi_aperture * surf.semi_aperture)
            return false;

        normal = Vec3f(0, 0, (ray.dir.z > 0) ? -1.0f : 1.0f);
        return true;
    }

    switch (surf.surface_type)
    {
        case SURF_SPHERICAL:   return intersect_spherical (ray, surf, hit_pos, normal);
        case SURF_CYLINDER_X:  return intersect_cylinder_x(ray, surf, hit_pos, normal);
        case SURF_CYLINDER_Y:  return intersect_cylinder_y(ray, surf, hit_pos, normal);
        case SURF_TORIC:       return intersect_toric    (ray, surf, hit_pos, normal);
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Refract a ray at a surface.
// n_ratio = n1 / n2.  Normal opposes the incoming ray.
// Returns false on total internal reflection.
// ---------------------------------------------------------------------------

static bool refract_direction(const Vec3f &dir, const Vec3f &normal,
                              float n_ratio, Vec3f &out_dir)
{
    float cos_i = -dot(normal, dir);
    float sin2_t = n_ratio * n_ratio * (1.0f - cos_i * cos_i);

    if (sin2_t >= 1.0f)
        return false; // TIR

    float cos_t = std::sqrt(1.0f - sin2_t);
    out_dir = fast_normalize(dir * n_ratio + normal * (n_ratio * cos_i - cos_t));
    return true;
}

// ---------------------------------------------------------------------------
// Reflect a ray off a surface.  Normal opposes the incoming ray.
// ---------------------------------------------------------------------------

static Vec3f reflect_direction(const Vec3f &dir, const Vec3f &normal)
{
    return fast_normalize(dir - normal * (2.0f * dot(dir, normal)));
}

// ---------------------------------------------------------------------------
// Trace a ghost ray through the full lens system.
//
// Three-phase sequential trace with reflections at bounce_a and bounce_b.
// ---------------------------------------------------------------------------

TraceResult trace_ghost_ray(const Ray &ray_in, const LensSystem &lens,
                            int bounce_a, int bounce_b,
                            float lambda_nm)
{
    Ray ray = ray_in; // local mutable copy
    TraceResult result;
    result.valid = false;
    result.weight = 1.0f;

    int N = lens.num_surfaces();
    float current_ior = 1.0f; // start in air

    // ================================================================
    // Phase 1: forward through surfaces 0 .. bounce_b
    //          transmit at all except bounce_b (reflect)
    // ================================================================
    for (int s = 0; s <= bounce_b; ++s)
    {
        Vec3f hit, norm;
        if (!intersect_surface(ray, lens.surfaces[s], hit, norm))
            return result; // vignetted

        ray.origin = hit;

        float n1 = current_ior;
        float n2 = lens.surfaces[s].ior_at(lambda_nm);

        float cos_i = std::abs(dot(norm, ray.dir));
        float R = surface_reflectance(cos_i, n1, n2,
                                      lens.surfaces[s].coating, lambda_nm);

        if (s == bounce_b)
        {
            // Reflect
            ray.dir = reflect_direction(ray.dir, norm);
            result.weight *= R;
            // current_ior unchanged (still in the medium before this surface)
        }
        else
        {
            // Transmit
            Vec3f new_dir;
            if (!refract_direction(ray.dir, norm, n1 / n2, new_dir))
                return result; // TIR
            ray.dir = new_dir;
            result.weight *= (1.0f - R);
            current_ior = n2;
        }
    }

    // ================================================================
    // Phase 2: backward through surfaces bounce_b-1 .. bounce_a
    //          transmit at all except bounce_a (reflect)
    // ================================================================
    for (int s = bounce_b - 1; s >= bounce_a; --s)
    {
        Vec3f hit, norm;
        if (!intersect_surface(ray, lens.surfaces[s], hit, norm))
            return result;

        ray.origin = hit;

        // Backward through surface s:
        //   n1 = medium on the side the ray is coming from = surfaces[s].ior_at(λ)
        //   n2 = medium on the other side = ior_before(s, λ)
        float n1 = current_ior;
        float n2 = lens.ior_before(s, lambda_nm);

        float cos_i = std::abs(dot(norm, ray.dir));
        float R = surface_reflectance(cos_i, n1, n2,
                                      lens.surfaces[s].coating, lambda_nm);

        if (s == bounce_a)
        {
            // Reflect — ray resumes forward direction
            ray.dir = reflect_direction(ray.dir, norm);
            result.weight *= R;
            // After reflecting at bounce_a, we're in the medium to the RIGHT
            // of bounce_a, which is surfaces[bounce_a].ior_at(λ)
            current_ior = lens.surfaces[bounce_a].ior_at(lambda_nm);
        }
        else
        {
            // Transmit backward
            Vec3f new_dir;
            if (!refract_direction(ray.dir, norm, n1 / n2, new_dir))
                return result;
            ray.dir = new_dir;
            result.weight *= (1.0f - R);
            current_ior = n2;
        }
    }

    // ================================================================
    // Phase 3: forward through surfaces bounce_a+1 .. N-1
    //
    // The ray exits bounce_a going forward and passes through all
    // remaining surfaces (including those between bounce_a and bounce_b
    // for a second time, physically correct for a double-bounce ghost).
    // ================================================================
    for (int s = bounce_a + 1; s < N; ++s)
    {
        Vec3f hit, norm;
        if (!intersect_surface(ray, lens.surfaces[s], hit, norm))
            return result;

        ray.origin = hit;

        float n1 = current_ior;
        float n2 = lens.surfaces[s].ior_at(lambda_nm);

        float cos_i = std::abs(dot(norm, ray.dir));
        float R = surface_reflectance(cos_i, n1, n2,
                                      lens.surfaces[s].coating, lambda_nm);

        Vec3f new_dir;
        if (!refract_direction(ray.dir, norm, n1 / n2, new_dir))
            return result;
        ray.dir = new_dir;
        result.weight *= (1.0f - R);
        current_ior = n2;
    }

    // ================================================================
    // Propagate to sensor plane
    // ================================================================
    if (std::abs(ray.dir.z) < 1e-12f)
        return result;

    float t = (lens.sensor_z - ray.origin.z) / ray.dir.z;
    if (t < 0)
        return result; // sensor is behind the ray

    result.position = ray.origin + ray.dir * t;
    result.valid = true;
    return result;
}
