// ============================================================================
// lens.h — Lens system data structures and file parser
// ============================================================================
#pragma once

#include <string>
#include <vector>
#include "fresnel.h"

// ---------------------------------------------------------------------------
// Surface geometry type.
//
// SURF_SPHERICAL is the legacy default and matches every existing .lens file.
// Cylinder X/Y and toric are the anamorphic extensions; their intersection
// logic lives in trace.cpp (CPU) and ghost_cuda.cu (GPU).
//
// Underlying values are fixed (0..3) so the enum is wire-compatible across
// host and device — Surface is uploaded to the GPU as raw bytes via
// cudaMemcpy in ghost_cuda.cu.  Do not renumber.
// ---------------------------------------------------------------------------
enum SurfaceType : int
{
    SURF_SPHERICAL  = 0,   // radius = R; current behaviour.
    SURF_CYLINDER_X = 1,   // axis ‖ X, curves in YZ plane (`radius` = Ry).
    SURF_CYLINDER_Y = 2,   // axis ‖ Y, curves in XZ plane (`radius` = Rx).
    SURF_TORIC      = 3    // two orthogonal radii: `radius` (=Rx) + `radius_y`.
};

// One optical surface in the lens system.
//
// Standard-layout / trivially-copyable: uploaded to the GPU as raw bytes
// alongside the rest of the surface array.  Do not add virtual functions
// or non-trivial members.
struct Surface
{
    float radius;        // signed radius of curvature in mm (0 = flat).
                         //   spherical : R
                         //   cyl_x     : Ry  (curvature in YZ plane)
                         //   cyl_y     : Rx  (curvature in XZ plane)
                         //   toric     : Rx
    float thickness;     // axial distance to next surface (mm)
    float ior;           // refractive index of medium AFTER this surface (d-line)
    float abbe_v;        // Abbe number (0 = air / non-dispersive)
    float semi_aperture; // clear semi-diameter (mm)
    int coating;         // AR coating layers (0 = uncoated)
    bool is_stop;        // is this the aperture stop?

    // Computed by LensSystem::compute_geometry()
    float z; // axial position of surface vertex (mm)

    // ---- Anamorphic extension (M1) ----
    // When loading a legacy spherical .lens file the parser writes
    // SURF_SPHERICAL / 0.0f here, so the in-memory representation is
    // identical to pre-M1 binaries.  The GPU side picks these fields up
    // automatically (Surface is cudaMemcpy'd verbatim) but does not yet
    // dispatch on them — that lands in M3.
    int   surface_type = SURF_SPHERICAL;
    float radius_y     = 0.0f;   // toric only; ignored otherwise.

    // Wavelength-dependent IOR (Cauchy via Abbe number)
    float ior_at(float lambda_nm) const
    {
        return dispersion_ior(ior, abbe_v, lambda_nm);
    }
};

// Complete lens system: ordered sequence of surfaces + sensor plane.
struct LensSystem
{
    std::string name;
    float focal_length = 0; // nominal focal length (mm), from file

    std::vector<Surface> surfaces;
    float sensor_z = 0; // axial position of sensor plane (mm), computed

    // Load a lens prescription from a .lens file.
    bool load(const char *filename);

    // Compute surface z positions and sensor_z from thicknesses.
    void compute_geometry();

    // IOR of the medium BEFORE surface idx (air for the first surface).
    float ior_before(int idx) const
    {
        return (idx <= 0) ? 1.0f : surfaces[idx - 1].ior;
    }

    // Wavelength-dependent version.
    float ior_before(int idx, float lambda_nm) const
    {
        if (idx <= 0)
            return 1.0f;
        return surfaces[idx - 1].ior_at(lambda_nm);
    }

    int num_surfaces() const { return (int)surfaces.size(); }

    void print_summary() const;
};
