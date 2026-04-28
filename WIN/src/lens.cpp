// ============================================================================
// lens.cpp — Lens file parser and geometry computation
// ============================================================================

#include "lens.h"

#include <cstdio>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// .lens file format:
//
//   # comment
//   name: My Lens 50mm f/1.4
//   focal_length: 50.0
//
//   surfaces:
//   # radius   thickness   ior    abbe   semi_ap   coating  [type]   [radius_y]
//     58.95    7.52        1.670  47.2   25.0      1
//     169.66   0.24        1.000  0.0    25.0      0
//    -140.0    3.00        1.750  35.0   18.0      1        cyl_y                # Y-cylinder, Rx = -140
//    -200.0    1.50        1.000  0.0    18.0      0        toric    -95.0       # toric, Rx=-200, Ry=-95
//     stop     2.00        1.000  0.0    12.5      0
//     ...
//
//   The last surface's thickness is the back focal distance to the sensor.
//   A radius of "stop" or "STOP" marks the aperture stop (flat surface).
//   A radius of 0 or "inf" means a flat surface.
//
//   Optional 7th token (surface type, default = sph):
//     sph   / spherical    spherical (legacy; value of `radius` is R)
//     cyl_x / cylinder_x   cylinder, axis along X (`radius` = Ry)
//     cyl_y / cylinder_y   cylinder, axis along Y (`radius` = Rx)
//     toric                two-axis curvature; requires an 8th token (radius_y).
//                          `radius` is Rx, the 8th token is Ry.
// ---------------------------------------------------------------------------

bool LensSystem::load(const char *filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        fprintf(stderr, "ERROR: cannot open lens file: %s\n", filename);
        return false;
    }

    surfaces.clear();
    name.clear();
    focal_length = 0;

    bool in_surfaces = false;
    std::string line;

    while (std::getline(file, line))
    {
        // Strip leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos)
            continue;
        line = line.substr(start);

        // Skip comments
        if (line[0] == '#')
            continue;

        // Check for metadata
        if (!in_surfaces)
        {
            if (line.substr(0, 5) == "name:")
            {
                name = line.substr(5);
                size_t ns = name.find_first_not_of(" \t");
                if (ns != std::string::npos)
                    name = name.substr(ns);
                continue;
            }
            if (line.substr(0, 13) == "focal_length:")
            {
                // Imbue with classic locale so '.' is always the decimal separator.
                std::istringstream fl_ss(line.substr(13));
                fl_ss.imbue(std::locale::classic());
                fl_ss >> focal_length;
                continue;
            }
            if (line.substr(0, 9) == "surfaces:")
            {
                in_surfaces = true;
                continue;
            }
            continue;
        }

        // Parse surface line
        // Expected: radius thickness ior abbe semi_aperture coating
        // Imbue with classic locale so '.' is always the decimal separator.
        std::istringstream iss(line);
        iss.imbue(std::locale::classic());
        std::string radius_str;
        iss >> radius_str;
        if (radius_str.empty() || radius_str[0] == '#')
            continue;

        Surface s{};
        s.is_stop      = false;
        s.surface_type = SURF_SPHERICAL;
        s.radius_y     = 0.0f;

        if (radius_str == "stop" || radius_str == "STOP")
        {
            s.radius = 0.0f;
            s.is_stop = true;
        }
        else if (radius_str == "inf" || radius_str == "INF")
        {
            s.radius = 0.0f;
        }
        else
        {
            std::istringstream rad_ss(radius_str);
            rad_ss.imbue(std::locale::classic());
            rad_ss >> s.radius;
        }

        iss >> s.thickness >> s.ior >> s.abbe_v >> s.semi_aperture >> s.coating;

        // ---- Optional surface-type token (M1) -----------------------------
        // After the six legacy tokens we accept an optional 7th token naming
        // the surface geometry.  Missing => SURF_SPHERICAL, so all existing
        // .lens files continue to parse identically.
        std::string type_str;
        if (iss >> type_str && !type_str.empty() && type_str[0] != '#')
        {
            if (type_str == "sph" || type_str == "spherical")
            {
                s.surface_type = SURF_SPHERICAL;
            }
            else if (type_str == "cyl_x" || type_str == "cylinder_x")
            {
                s.surface_type = SURF_CYLINDER_X;
            }
            else if (type_str == "cyl_y" || type_str == "cylinder_y")
            {
                s.surface_type = SURF_CYLINDER_Y;
            }
            else if (type_str == "toric")
            {
                s.surface_type = SURF_TORIC;
                if (!(iss >> s.radius_y))
                {
                    fprintf(stderr,
                            "WARNING: surface %zu marked 'toric' but no "
                            "radius_y token; treating as spherical.\n",
                            surfaces.size());
                    s.surface_type = SURF_SPHERICAL;
                    s.radius_y     = 0.0f;
                }
            }
            else
            {
                fprintf(stderr,
                        "WARNING: surface %zu has unknown type '%s'; "
                        "treating as spherical.\n",
                        surfaces.size(), type_str.c_str());
            }
        }

        if (s.semi_aperture <= 0)
        {
            fprintf(stderr, "WARNING: surface %zu has semi_aperture <= 0, skipping\n",
                    surfaces.size());
            continue;
        }

        surfaces.push_back(s);
    }

    if (surfaces.empty())
    {
        fprintf(stderr, "ERROR: no surfaces found in lens file: %s\n", filename);
        return false;
    }

    compute_geometry();
    return true;
}

void LensSystem::compute_geometry()
{
    float z = 0;
    for (size_t i = 0; i < surfaces.size(); ++i)
    {
        surfaces[i].z = z;
        z += surfaces[i].thickness;
    }
    sensor_z = z;
}

static const char* surface_type_label(int t)
{
    switch (t)
    {
        case SURF_SPHERICAL:  return "sph";
        case SURF_CYLINDER_X: return "cyl_x";
        case SURF_CYLINDER_Y: return "cyl_y";
        case SURF_TORIC:      return "toric";
        default:              return "?";
    }
}

void LensSystem::print_summary() const
{
    printf("Lens: %s\n", name.c_str());
    printf("  focal_length: %.1f mm\n", focal_length);
    printf("  surfaces: %d\n", num_surfaces());
    printf("  sensor_z: %.2f mm\n", sensor_z);
    printf("  %-4s  %10s  %8s  %6s  %6s  %8s  %4s  %-5s  %s\n",
           "Idx", "Radius", "Thick", "IOR", "Abbe", "SemiAp", "Coat", "Type", "");

    for (int i = 0; i < num_surfaces(); ++i)
    {
        const Surface &s = surfaces[i];
        const char* tlabel = surface_type_label(s.surface_type);

        // Trailing comment: stop marker takes precedence; otherwise Ry for
        // toric surfaces, blank for everything else.
        char trail[64] = "";
        if (s.is_stop)
            snprintf(trail, sizeof(trail), "<-- aperture stop");
        else if (s.surface_type == SURF_TORIC)
            snprintf(trail, sizeof(trail), "Ry=%.3f", s.radius_y);

        if (s.radius == 0.0f)
        {
            printf("  %-4d  %10s  %8.3f  %6.3f  %6.1f  %8.2f  %4d  %-5s  %s\n",
                   i, s.is_stop ? "STOP" : "flat",
                   s.thickness, s.ior, s.abbe_v, s.semi_aperture, s.coating,
                   tlabel, trail);
        }
        else
        {
            printf("  %-4d  %10.3f  %8.3f  %6.3f  %6.1f  %8.2f  %4d  %-5s  %s\n",
                   i, s.radius, s.thickness, s.ior, s.abbe_v,
                   s.semi_aperture, s.coating, tlabel, trail);
        }
    }
}
