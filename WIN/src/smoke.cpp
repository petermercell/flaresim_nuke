// Smoke test for trace.cpp surface intersections.
//
// Covers the M2 cylinder dispatcher (CYL_X/CYL_Y, regression on SPH/flat)
// and the M4 toric Newton-Raphson solver (TORIC).
//
// Acceptance per plan: hit position < 1 µm error, normal < 0.001 rad error.
// We test convex (R>0) and concave (R<0) variants, the cylinder's
// invariant axis, and for the toric: reduction to sphere when Rx==Ry,
// the principal cross-sections, off-axis interior hits, mixed-sign
// (saddle) configs, and aperture clipping.

#include "trace.h"
#include "lens.h"

#include <cassert>
#include <cmath>
#include <cstdio>

static bool approx(float a, float b, float eps = 1e-4f)
{
    return std::abs(a - b) < eps;
}

// Angle (radians) between two unit vectors.
static float angle_between(const Vec3f& a, const Vec3f& b)
{
    float d = std::max(-1.0f, std::min(1.0f, dot(a, b)));
    return std::acos(d);
}

static Surface make_surface(int type, float R, float vertex_z,
                            float semi_ap = 50.0f)
{
    Surface s{};
    s.radius        = R;
    s.thickness     = 0;
    s.ior           = 1.5f;
    s.abbe_v        = 0;
    s.semi_aperture = semi_ap;
    s.coating       = 0;
    s.is_stop       = false;
    s.z             = vertex_z;
    s.surface_type  = type;
    s.radius_y      = 0;
    return s;
}

static Ray ray_from(float ox, float oy, float oz, float dx, float dy, float dz)
{
    Ray r;
    r.origin = Vec3f(ox, oy, oz);
    r.dir    = Vec3f(dx, dy, dz).normalized();
    return r;
}

int main()
{
    constexpr float R = 100.0f;
    constexpr float Z = 10.0f;

    // ===================================================================
    // 1. CYL_Y (axis ‖ Y, curves in XZ).  surf.radius = Rx.
    //    Locus: x² + (z - cz)² = R²,  cz = Z + R = 110.
    // ===================================================================
    Surface cy = make_surface(SURF_CYLINDER_Y, R, Z);

    // 1a. On-axis ray (0,0,0) heading +Z → must hit vertex at (0,0,Z).
    {
        Ray r = ray_from(0, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, cy, hit, nrm));
        assert(approx(hit.x, 0.0f) && approx(hit.y, 0.0f) && approx(hit.z, Z));
        // Normal at vertex = (0, 0, -1) (opposing +Z ray)
        assert(angle_between(nrm, Vec3f(0,0,-1)) < 0.001f);
        printf("PASS: CYL_Y on-axis hit (0,0,%.3f), normal (0,0,-1)\n", Z);
    }

    // 1b. Off-axis along curved direction: ray at (5,0,0) → +Z.
    //     Analytic: 25 + (z-110)² = 10000 → z = 110 - sqrt(9975) ≈ 10.12516.
    //     Normal (in YZ-zero, XZ direction) = (5, 0, z-110)/100 = (0.05, 0, -0.99875).
    {
        Ray r = ray_from(5, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, cy, hit, nrm));
        float expected_z = 110.0f - std::sqrt(R*R - 25.0f);
        assert(approx(hit.x, 5.0f));
        assert(approx(hit.y, 0.0f));
        assert(approx(hit.z, expected_z, 1e-3f));   // ~10.125
        Vec3f n_exp(0.05f, 0.0f, (expected_z - 110.0f) / R);
        n_exp = n_exp.normalized();
        assert(angle_between(nrm, n_exp) < 0.001f);
        printf("PASS: CYL_Y off-axis x=5 → z=%.5f (expected %.5f)\n",
               hit.z, expected_z);
    }

    // 1c. Invariant-axis property: ray at (0,5,0) → +Z must hit at z=Z exactly,
    //     because Y is the cylinder's axis — the surface profile is
    //     translation-invariant along Y.  Normal must lie in the XZ plane
    //     (i.e. normal.y == 0).
    {
        Ray r = ray_from(0, 5, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, cy, hit, nrm));
        assert(approx(hit.x, 0.0f));
        assert(approx(hit.y, 5.0f));
        assert(approx(hit.z, Z, 1e-4f));
        assert(approx(nrm.y, 0.0f, 1e-6f));
        printf("PASS: CYL_Y invariant along Y — y-translation does not "
               "shift z (hit z=%.5f), normal.y==0\n", hit.z);
    }

    // 1d. Aperture clip: ray at (60, 0, 0) → +Z misses (semi_ap = 50).
    {
        Ray r = ray_from(60, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(!intersect_surface(r, cy, hit, nrm));
        printf("PASS: CYL_Y aperture clip rejects ray outside semi_aperture\n");
    }

    // ===================================================================
    // 2. CYL_X (axis ‖ X, curves in YZ).  surf.radius = Ry.
    //    Locus: y² + (z - cz)² = R²,  cz = Z + R = 110.
    //    By symmetry: swap X↔Y in everything above.
    // ===================================================================
    Surface cx = make_surface(SURF_CYLINDER_X, R, Z);

    // 2a. Off-axis along curved direction: ray at (0,5,0) → +Z.
    {
        Ray r = ray_from(0, 5, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, cx, hit, nrm));
        float expected_z = 110.0f - std::sqrt(R*R - 25.0f);
        assert(approx(hit.x, 0.0f));
        assert(approx(hit.y, 5.0f));
        assert(approx(hit.z, expected_z, 1e-3f));
        Vec3f n_exp(0.0f, 0.05f, (expected_z - 110.0f) / R);
        n_exp = n_exp.normalized();
        assert(angle_between(nrm, n_exp) < 0.001f);
        printf("PASS: CYL_X off-axis y=5 → z=%.5f (expected %.5f)\n",
               hit.z, expected_z);
    }

    // 2b. Invariant-axis property: ray at (5,0,0) → +Z hits at z=Z.
    {
        Ray r = ray_from(5, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, cx, hit, nrm));
        assert(approx(hit.z, Z, 1e-4f));
        assert(approx(nrm.x, 0.0f, 1e-6f));
        printf("PASS: CYL_X invariant along X\n");
    }

    // ===================================================================
    // 3. Concave cylinder (R<0).  Centre of curvature at z = Z + R = -90;
    //    ray going +Z hits the FAR side, which is at z = -90 + R = -190?
    //    Actually for a concave (R<0) front surface, the ray approaches it
    //    from -Z and the chosen root is the one with z closest to the
    //    vertex Z=10.  Let's verify on-axis still hits at the vertex.
    // ===================================================================
    Surface cy_neg = make_surface(SURF_CYLINDER_Y, -R, Z);
    {
        Ray r = ray_from(0, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, cy_neg, hit, nrm));
        assert(approx(hit.z, Z, 1e-4f));
        printf("PASS: CYL_Y concave (R<0) on-axis hits vertex\n");
    }

    // ===================================================================
    // 4. Spherical regression — the M2 refactor must NOT change spherical
    //    behaviour.  Same R, same vertex Z, same on-axis ray: hit (0,0,Z).
    // ===================================================================
    Surface sph = make_surface(SURF_SPHERICAL, R, Z);
    {
        Ray r = ray_from(0, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, sph, hit, nrm));
        assert(approx(hit.z, Z, 1e-4f));
        printf("PASS: SPH on-axis still works after refactor\n");
    }
    {
        // Off-axis spherical hit:  x²+y² + (z-cz)² = R²
        // Ray (3, 4, 0) → +Z.  r² = 25, expected z = 110 - sqrt(10000 - 25) ~10.12516.
        Ray r = ray_from(3, 4, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, sph, hit, nrm));
        float expected_z = 110.0f - std::sqrt(R*R - 25.0f);
        assert(approx(hit.z, expected_z, 1e-3f));
        printf("PASS: SPH off-axis (3,4) hits at z=%.5f\n", hit.z);
    }

    // ===================================================================
    // 5. SURF_TORIC (M4): full Newton-Raphson on the toric quartic.
    //    The toric has two principal radii — Rx in XZ, Ry in YZ — so
    //    the y=0 cross-section must match a sphere with R=Rx, and the
    //    x=0 cross-section must match a sphere with R=Ry.  At Rx=Ry it
    //    must reduce exactly to the spherical case (regression).
    // ===================================================================
    constexpr float Rx = 100.0f, Ry =  50.0f;

    // 5a. Toric reducing to sphere (Rx == Ry).
    {
        Surface tor = make_surface(SURF_TORIC, R, Z);
        tor.radius_y = R;                  // Rx = Ry = 100  →  pure sphere
        Ray r = ray_from(0, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, tor, hit, nrm));
        assert(approx(hit.z, Z, 1e-4f));
        assert(angle_between(nrm, Vec3f(0,0,-1)) < 0.001f);
        printf("PASS: TORIC reduces to sphere when Rx == Ry\n");
    }

    // 5b. On-axis hit must land at the apex with normal (0,0,-1).
    {
        Surface tor = make_surface(SURF_TORIC, Rx, Z);
        tor.radius_y = Ry;
        Ray r = ray_from(0, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, tor, hit, nrm));
        assert(approx(hit.x, 0.0f, 1e-4f));
        assert(approx(hit.y, 0.0f, 1e-4f));
        assert(approx(hit.z, Z, 1e-4f));
        assert(angle_between(nrm, Vec3f(0,0,-1)) < 0.001f);
        printf("PASS: TORIC on-axis hits apex (0,0,%.2f)\n", Z);
    }

    // 5c. y=0 cross-section is a sphere with R=Rx.
    //     At x=5: z = vertex + Rx - sqrt(Rx² - 25).
    {
        Surface tor = make_surface(SURF_TORIC, Rx, Z);
        tor.radius_y = Ry;
        Ray r = ray_from(5, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, tor, hit, nrm));
        const float expected_z = Z + Rx - std::sqrt(Rx*Rx - 25.0f);
        assert(approx(hit.x, 5.0f, 1e-4f));
        assert(approx(hit.y, 0.0f, 1e-4f));
        assert(approx(hit.z, expected_z, 1e-3f));
        // Normal in XZ plane → ny == 0.
        assert(std::abs(nrm.y) < 1e-5f);
        printf("PASS: TORIC y=0 section matches sphere R=Rx (z=%.5f, expected %.5f)\n",
               hit.z, expected_z);
    }

    // 5d. x=0 cross-section is a sphere with R=Ry.
    //     At y=5: z = vertex + Ry - sqrt(Ry² - 25).
    {
        Surface tor = make_surface(SURF_TORIC, Rx, Z);
        tor.radius_y = Ry;
        Ray r = ray_from(0, 5, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, tor, hit, nrm));
        const float expected_z = Z + Ry - std::sqrt(Ry*Ry - 25.0f);
        assert(approx(hit.x, 0.0f, 1e-4f));
        assert(approx(hit.y, 5.0f, 1e-4f));
        assert(approx(hit.z, expected_z, 1e-3f));
        // Normal in YZ plane → nx == 0.
        assert(std::abs(nrm.x) < 1e-5f);
        printf("PASS: TORIC x=0 section matches sphere R=Ry (z=%.5f, expected %.5f)\n",
               hit.z, expected_z);
    }

    // 5e. Off-axis at (3, 4): hit z must lie strictly between the two
    //     extreme cross-sections (Ry curves more strongly in YZ).
    //     Also normal must have non-zero components in both X and Y.
    {
        Surface tor = make_surface(SURF_TORIC, Rx, Z);
        tor.radius_y = Ry;
        Ray r = ray_from(3, 4, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, tor, hit, nrm));
        assert(approx(hit.x, 3.0f, 1e-3f));
        assert(approx(hit.y, 4.0f, 1e-3f));
        // Both spherical bounds: with Rx=100, hit_z @ r²=25 = 10.125;
        // with Ry=50,  hit_z @ r²=25 = 10.251.  Toric should sit between.
        const float z_xz = Z + Rx - std::sqrt(Rx*Rx - 25.0f);  // ~10.125
        const float z_yz = Z + Ry - std::sqrt(Ry*Ry - 25.0f);  // ~10.251
        assert(hit.z > z_xz - 1e-3f && hit.z < z_yz + 1e-3f);
        assert(std::abs(nrm.x) > 1e-3f && std::abs(nrm.y) > 1e-3f);
        printf("PASS: TORIC off-axis (3,4) z=%.5f in [%.5f, %.5f]\n",
               hit.z, z_xz, z_yz);
    }

    // 5f. Concave toric (both radii negative).
    {
        Surface tor = make_surface(SURF_TORIC, -Rx, Z);
        tor.radius_y = -Ry;
        Ray r = ray_from(0, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, tor, hit, nrm));
        assert(approx(hit.z, Z, 1e-3f));
        printf("PASS: TORIC concave (Rx<0, Ry<0) on-axis hits vertex\n");
    }

    // 5g. Mixed-sign toric (saddle): Rx>0, Ry<0.  Apex still at vertex.
    {
        Surface tor = make_surface(SURF_TORIC, Rx, Z);
        tor.radius_y = -Ry;
        Ray r = ray_from(0, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, tor, hit, nrm));
        assert(approx(hit.z, Z, 1e-3f));
        printf("PASS: TORIC saddle (Rx>0, Ry<0) on-axis hits vertex\n");
    }

    // 5h. Aperture clip rejects rays outside the disc.
    {
        Surface tor = make_surface(SURF_TORIC, Rx, Z, /*semi_ap*/ 5.0f);
        tor.radius_y = Ry;
        Ray r = ray_from(10, 0, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(!intersect_surface(r, tor, hit, nrm));
        printf("PASS: TORIC aperture clip rejects rays outside semi_aperture\n");
    }

    // ===================================================================
    // 6. Flat shortcut still works for any surface_type.
    // ===================================================================
    {
        Surface flat_cyl = make_surface(SURF_CYLINDER_Y, 0.0f, Z);
        Ray r = ray_from(2, 3, 0, 0, 0, 1);
        Vec3f hit, nrm;
        assert(intersect_surface(r, flat_cyl, hit, nrm));
        assert(approx(hit.z, Z, 1e-6f));
        assert(approx(nrm.z, -1.0f, 1e-6f));
        printf("PASS: flat (radius=0) shortcut works for non-spherical type\n");
    }

    printf("\nAll M2 + M4 smoke tests passed.\n");
    return 0;
}
