// ============================================================================
// FlareSim3D.cpp — 3D-aware lens flare simulation for Nuke (CUDA backend)
//
// Variant of FlareSim that takes Camera and Axis node inputs instead of a
// manual Source XY knob.  The Axis position is projected through the Camera
// to derive screen position and source distance automatically.
//
// Inputs:
//   0 — plate   (Iop)       : image to sample source colour from
//   1 — cam     (CameraOp)  : scene camera (provides FOV + world transform)
//   2 — light   (AxisOp)    : light source position in world space
//   3 — mask    (Iop)       : optional mask gating which pixels drive flare
//
// The source's Z distance from the camera enables:
//   • Distance-based inverse-square intensity falloff (optional)
//   • Future: convergent ray geometry for close light sources
//
// When the projected source is off-screen, the Outside Source knobs take
// over seamlessly (same behaviour as FlareSim).  When the source is behind
// the camera, no flare is produced.
// ============================================================================

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Channel.h"
#include "DDImage/ChannelSet.h"
#include "DDImage/CameraOp.h"
#include "DDImage/AxisOp.h"
#include "DDImage/Matrix4.h"
#include "DDImage/Vector4.h"

#include "lens.h"
#include "ghost.h"
#include "ghost_cuda.h"
#include "blur_cuda.h"

// Set to 1 to enable per-frame profiler output to stderr.
#define FLARESIM_PROFILE 0

#if FLARESIM_PROFILE
#include "profiler.h"
#define PROF_DECL          FlareProfiler _prof
#define PROF_BEGIN(name)   _prof.begin(name)
#define PROF_END()         _prof.end()
#define PROF_SUMMARY()     _prof.print_summary()
#else
#define PROF_DECL          ((void)0)
#define PROF_BEGIN(name)   ((void)0)
#define PROF_END()         ((void)0)
#define PROF_SUMMARY()     ((void)0)
#endif

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>    // still needed for std::lock_guard
#include <string>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// See FlareSim.cpp for rationale — avoids CRT version mismatch crash.
struct SpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
    void lock()   { while (flag_.test_and_set(std::memory_order_acquire)) { /* spin */ } }
    void unlock() { flag_.clear(std::memory_order_release); }
};

using namespace DD::Image;

// ---------------------------------------------------------------------------
// Separable box blur (prefix-sum, O(w*h) per pass) — CPU fallback
// ---------------------------------------------------------------------------

static void box_blur(float* buf, int w, int h, int radius, int passes,
                     std::vector<float>& tmp)
{
    if (radius < 1 || passes < 1) return;
    const size_t npx = (size_t)w * h;
    tmp.resize(npx);

    const int maxdim = std::max(w, h);
    std::vector<double> ps(maxdim + 1);

    for (int p = 0; p < passes; ++p)
    {
        for (int y = 0; y < h; ++y)
        {
            const float* row = buf + (size_t)y * w;
            float*       out = tmp.data() + (size_t)y * w;
            ps[0] = 0.0;
            for (int x = 0; x < w; ++x)
                ps[x + 1] = ps[x] + row[x];
            for (int x = 0; x < w; ++x)
            {
                int lo = std::max(0,     x - radius);
                int hi = std::min(w - 1, x + radius);
                out[x] = (float)((ps[hi + 1] - ps[lo]) / (hi - lo + 1));
            }
        }
        std::memcpy(buf, tmp.data(), npx * sizeof(float));

        for (int x = 0; x < w; ++x)
        {
            ps[0] = 0.0;
            for (int y = 0; y < h; ++y)
                ps[y + 1] = ps[y] + buf[(size_t)y * w + x];
            for (int y = 0; y < h; ++y)
            {
                int lo = std::max(0,     y - radius);
                int hi = std::min(h - 1, y + radius);
                tmp[(size_t)y * w + x] = (float)((ps[hi + 1] - ps[lo]) / (hi - lo + 1));
            }
        }
        std::memcpy(buf, tmp.data(), npx * sizeof(float));
    }
}

// ===========================================================================
// Per-surface toggle + gain knob name storage
// ===========================================================================

static constexpr int MAX_SURFS_UI = 50;

struct SurfKnobNames
{
    char toggle_names[MAX_SURFS_UI][16];
    char gain_names[MAX_SURFS_UI][24];
    char color_names[MAX_SURFS_UI][24];
    char offx_names[MAX_SURFS_UI][24];
    char offy_names[MAX_SURFS_UI][24];
    char scale_names[MAX_SURFS_UI][24];
    SurfKnobNames()
    {
        for (int i = 0; i < MAX_SURFS_UI; ++i) {
            snprintf(toggle_names[i], sizeof(toggle_names[i]), "surf_%d", i);
            snprintf(gain_names[i],   sizeof(gain_names[i]),   "surf_gain_%d", i);
            snprintf(color_names[i],  sizeof(color_names[i]),  "surf_color_%d", i);
            snprintf(offx_names[i],   sizeof(offx_names[i]),   "surf_offx_%d", i);
            snprintf(offy_names[i],   sizeof(offy_names[i]),   "surf_offy_%d", i);
            snprintf(scale_names[i],  sizeof(scale_names[i]),  "surf_scale_%d", i);
        }
    }
};
static SurfKnobNames s_surf_knob_names;

// ---------------------------------------------------------------------------
// FlareSim3D
// ---------------------------------------------------------------------------

class FlareSim3D : public Iop
{
public:
    // ========================================================================
    // SpinLock instead of std::mutex — avoids CRT version mismatch crash.
    // ========================================================================
    SpinLock  compute_mutex_;

    // ---- Knob storage ----
    const char* lens_file_;
    float       source_intensity_;
    float       flare_gain_;
    int         ray_grid_;
    float       threshold_;

    int         sample_radius_;

    // Distance-based intensity falloff
    bool        intensity_falloff_;
    float       reference_distance_;   // scene units — intensity = source_intensity at this distance

    float       ghost_blur_;
    int         ghost_blur_passes_;

    int         aperture_blades_;
    float       aperture_rotation_;

    int         pupil_jitter_;
    int         jitter_seed_;
    bool        jitter_auto_seed_;

    int         spectral_idx_;
    bool        spectral_jitter_;
    int         spectral_jitter_seed_;
    bool        spectral_jitter_auto_seed_;
    float       spectral_jitter_scale_;

    bool        highlight_compress_;
    float       highlight_clip_;
    float       highlight_knee_;
    int         highlight_metric_;

    // Off-screen source (same as FlareSim)
    bool        outside_source_enable_;
    float       outside_source_color_[3];
    float       outside_source_intensity_;
    float       outside_source_falloff_;

    // Per-surface art direction
    bool        surf_enabled_[MAX_SURFS_UI];
    float       surf_gain_[MAX_SURFS_UI];
    float       surf_color_[MAX_SURFS_UI][3];
    float       surf_offx_[MAX_SURFS_UI];
    float       surf_offy_[MAX_SURFS_UI];
    float       surf_scale_[MAX_SURFS_UI];
    char        surf_labels_[MAX_SURFS_UI][64];

    // ---- Runtime state ----
    LensSystem  lens_;
    std::string last_lens_file_;

    std::vector<GhostPair> all_lens_pairs_;

    std::atomic<int> last_src_count_  {0};
    std::atomic<int> last_pair_count_ {0};

    std::vector<float> pair_colors_;
    std::vector<float> pair_offsets_;
    std::vector<float> pair_scales_;

    uint16_t* ghost_r_  = nullptr;
    uint16_t* ghost_g_  = nullptr;
    uint16_t* ghost_b_  = nullptr;
    uint16_t* alpha_    = nullptr;
    size_t cache_npx_ = 0;

    int cache_width_  = 0;
    int cache_height_ = 0;

    GpuBufferCache gpu_cache_;

    // Cached ghost_filter results
    std::vector<GhostPair> cached_filter_pairs_;
    std::vector<float>     cached_filter_boosts_;
    float                  cached_filter_shw_ = -1.0f;
    float                  cached_filter_shh_ = -1.0f;
    std::string            cached_filter_lens_;

    std::string pending_cuda_error_;

    bool        needs_compute_ = false;
    int         pending_frame_ = -1;
    int         cache_frame_   = -1;
    int         pending_x0_ = 0, pending_y0_ = 0;
    int         pending_x1_ = 0, pending_y1_ = 0;
    int         pending_w_  = 0, pending_h_  = 0;
    int         pending_fmt_x0_ = 0, pending_fmt_y0_ = 0;
    int         pending_fmt_w_  = 0, pending_fmt_h_  = 0;
    int         pending_num_surfs_ = 0;

    // ---- Constructor ----
    explicit FlareSim3D(Node* node)
        : Iop(node)
        , lens_file_("")
        , source_intensity_(8.0f)
        , flare_gain_(10.0f)
        , ray_grid_(64)
        , threshold_(0.0f)
        , sample_radius_(4)
        , intensity_falloff_(false)
        , reference_distance_(1000.0f)
        , ghost_blur_(0.003f)
        , ghost_blur_passes_(3)
        , aperture_blades_(0)
        , aperture_rotation_(0.0f)
        , pupil_jitter_(0)
        , jitter_seed_(0)
        , jitter_auto_seed_(true)
        , spectral_idx_(0)
        , spectral_jitter_(true)
        , spectral_jitter_seed_(0)
        , spectral_jitter_auto_seed_(true)
        , spectral_jitter_scale_(1.0f)
        , highlight_compress_(false)
        , highlight_clip_(2.0f)
        , highlight_knee_(0.5f)
        , highlight_metric_(1)
        , outside_source_enable_(true)
        , outside_source_intensity_(8.0f)
        , outside_source_falloff_(0.0f)
    {
        outside_source_color_[0] = 1.0f;
        outside_source_color_[1] = 1.0f;
        outside_source_color_[2] = 1.0f;
        for (int k = 0; k < MAX_SURFS_UI; ++k) surf_enabled_[k] = true;
        for (int k = 0; k < MAX_SURFS_UI; ++k) surf_gain_[k] = 1.0f;
        for (int k = 0; k < MAX_SURFS_UI; ++k) {
            surf_color_[k][0] = surf_color_[k][1] = surf_color_[k][2] = 1.0f;
        }
        for (int k = 0; k < MAX_SURFS_UI; ++k) {
            surf_offx_[k] = 0.0f; surf_offy_[k] = 0.0f;
        }
        for (int k = 0; k < MAX_SURFS_UI; ++k) surf_scale_[k] = 1.0f;
        for (int k = 0; k < MAX_SURFS_UI; ++k)
            snprintf(surf_labels_[k], sizeof(surf_labels_[k]), "Surf %d", k);
    }

    const char* Class()     const override { return "FlareSim3D"; }
    const char* node_help() const override
    {
        return "Physically-based lens flare simulation — 3D source mode.\n\n"
               "Same ghost ray tracing engine as FlareSim, but the flare source\n"
               "is defined by an Axis node in 3D space, projected through a\n"
               "Camera node.  This eliminates manual Source XY tracking and\n"
               "provides Z distance for intensity falloff.\n\n"
               "Inputs:\n"
               "  plate — image to sample source colour from\n"
               "  cam   — scene camera (provides FOV and projection)\n"
               "  light — Axis at the light source position in world space\n"
               "  mask  — optional alpha mask gating which regions drive flare\n\n"
               "When the source is behind the camera: no flare is produced.\n"
               "When the source is off-screen: Outside Source knobs take over.\n\n"
               "Merge the flare over the beauty with a Merge (plus) node.";
    }

    static const Iop::Description d;

    // ---- Input handling ----
    // 0: plate (Iop), 1: cam (CameraOp), 2: light (AxisOp), 3: mask (Iop)

    int maximum_inputs() const override { return 4; }
    int minimum_inputs() const override { return 3; }

    bool test_input(int idx, Op* op) const override
    {
        switch (idx) {
            case 0: return dynamic_cast<Iop*>(op) != nullptr;
            case 1: return dynamic_cast<CameraOp*>(op) != nullptr;
            case 2: return dynamic_cast<AxisOp*>(op) != nullptr;
            case 3: return dynamic_cast<Iop*>(op) != nullptr;
            default: return false;
        }
    }

    Op* default_input(int idx) const override
    {
        if (idx == 0) return Iop::default_input(idx);
        return nullptr;   // no sensible default for camera, axis, or mask
    }

    const char* input_label(int idx, char*) const override
    {
        switch (idx) {
            case 1: return "cam";
            case 2: return "light";
            case 3: return "mask";
            default: return "";
        }
    }

    // ---- rebuild_surf_ui ----
    void rebuild_surf_ui()
    {
        cached_filter_pairs_.clear();
        cached_filter_boosts_.clear();
        cached_filter_lens_.clear();

        if (lens_.surfaces.empty()) {
            all_lens_pairs_.clear();
            for (int k = 0; k < MAX_SURFS_UI; ++k) {
                Knob* kt = knob(s_surf_knob_names.toggle_names[k]);
                if (kt) kt->set_flag(Knob::HIDDEN);
                Knob* kg = knob(s_surf_knob_names.gain_names[k]);
                if (kg) kg->set_flag(Knob::HIDDEN);
                Knob* kc = knob(s_surf_knob_names.color_names[k]);
                if (kc) kc->set_flag(Knob::HIDDEN);
                Knob* kox = knob(s_surf_knob_names.offx_names[k]);
                if (kox) kox->set_flag(Knob::HIDDEN);
                Knob* koy = knob(s_surf_knob_names.offy_names[k]);
                if (koy) koy->set_flag(Knob::HIDDEN);
                Knob* ks = knob(s_surf_knob_names.scale_names[k]);
                if (ks) ks->set_flag(Knob::HIDDEN);
            }
            return;
        }

        all_lens_pairs_ = enumerate_ghost_pairs(lens_);
        if ((int)all_lens_pairs_.size() > 500)
            all_lens_pairs_.resize(500);

        const int nsurf = std::min(lens_.num_surfaces(), MAX_SURFS_UI);
        for (int k = 0; k < nsurf; ++k)
        {
            const Surface& s = lens_.surfaces[k];
            if (s.is_stop)
                snprintf(surf_labels_[k], sizeof(surf_labels_[k]),
                         "Surf %d  (STOP, semi_ap=%.1f)", k, s.semi_aperture);
            else if (std::abs(s.radius) < 1e-6f)
                snprintf(surf_labels_[k], sizeof(surf_labels_[k]),
                         "Surf %d  (flat, IOR=%.3f)", k, s.ior);
            else
                snprintf(surf_labels_[k], sizeof(surf_labels_[k]),
                         "Surf %d  (R=%.1f, IOR=%.3f)", k, s.radius, s.ior);

            Knob* kt = knob(s_surf_knob_names.toggle_names[k]);
            if (kt) { kt->label(surf_labels_[k]); kt->clear_flag(Knob::HIDDEN); }
            Knob* kg = knob(s_surf_knob_names.gain_names[k]);
            if (kg) kg->clear_flag(Knob::HIDDEN);
            Knob* kc = knob(s_surf_knob_names.color_names[k]);
            if (kc) kc->clear_flag(Knob::HIDDEN);
            Knob* kox = knob(s_surf_knob_names.offx_names[k]);
            if (kox) kox->clear_flag(Knob::HIDDEN);
            Knob* koy = knob(s_surf_knob_names.offy_names[k]);
            if (koy) koy->clear_flag(Knob::HIDDEN);
            Knob* ks = knob(s_surf_knob_names.scale_names[k]);
            if (ks) ks->clear_flag(Knob::HIDDEN);
        }
        for (int k = nsurf; k < MAX_SURFS_UI; ++k) {
            Knob* kt = knob(s_surf_knob_names.toggle_names[k]);
            if (kt) kt->set_flag(Knob::HIDDEN);
            Knob* kg = knob(s_surf_knob_names.gain_names[k]);
            if (kg) kg->set_flag(Knob::HIDDEN);
            Knob* kc = knob(s_surf_knob_names.color_names[k]);
            if (kc) kc->set_flag(Knob::HIDDEN);
            Knob* kox = knob(s_surf_knob_names.offx_names[k]);
            if (kox) kox->set_flag(Knob::HIDDEN);
            Knob* koy = knob(s_surf_knob_names.offy_names[k]);
            if (koy) koy->set_flag(Knob::HIDDEN);
            Knob* ks = knob(s_surf_knob_names.scale_names[k]);
            if (ks) ks->set_flag(Knob::HIDDEN);
        }
        updateUI(outputContext());
    }

    // ---- Knobs ----
    void knobs(Knob_Callback f) override
    {
        File_knob(f, &lens_file_, "lens_file", "Lens File");
        Tooltip(f, "Path to a .lens prescription file.");

        Divider(f, "Ghost");
        Int_knob(f, &ray_grid_, "ray_grid", "Ray Grid (NxN)");
        Tooltip(f, "NxN entrance-pupil samples per source. "
                   "Higher = smoother ghosts, longer render time.");
        Float_knob(f, &flare_gain_, "flare_gain", "Flare Gain");
        SetRange(f, 0.0, 50.0);
        Tooltip(f, "Ghost intensity multiplier.  Default 10.");

        static const char* const kJitterModes[] = {
            "Off", "Stratified", "Halton", nullptr
        };
        Enumeration_knob(f, &pupil_jitter_, kJitterModes, "pupil_jitter", "Pupil Jitter");
        Int_knob(f, &jitter_seed_, "jitter_seed", "Jitter Seed");
        Bool_knob(f, &jitter_auto_seed_, "jitter_auto_seed", "Auto Seed");

        Divider(f, "Source");
        Int_knob(f, &sample_radius_, "sample_radius", "Sample Radius");
        Tooltip(f, "Radius (in pixels) of the area averaged around the projected "
                   "source position to determine source colour and brightness.");
        Float_knob(f, &source_intensity_, "source_intensity", "Source Intensity");
        SetRange(f, 1.0, 50.0);
        Tooltip(f, "HDR brightness boost for the sampled plate colour.  Default 8.");
        Float_knob(f, &threshold_, "threshold", "Threshold");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Minimum luminance at the projected position for a flare to appear.");

        Divider(f, "Distance");
        Bool_knob(f, &intensity_falloff_, "intensity_falloff", "Intensity Falloff");
        Tooltip(f, "Apply inverse-square-law intensity scaling based on the distance "
                   "from the camera to the light source Axis.\n\n"
                   "Off by default — enable for physically correct falloff.");
        Float_knob(f, &reference_distance_, "reference_distance", "Reference Distance");
        SetRange(f, 0.1, 100000.0);
        Tooltip(f, "Distance (in scene units) at which the flare has its nominal "
                   "Source Intensity.  The intensity scales as (ref / distance)^2.\n\n"
                   "Example: if your scene is in centimetres and the source is a "
                   "street lamp 500 cm away, set this to 500.");

        Divider(f, "Outside Source");
        Bool_knob(f, &outside_source_enable_, "outside_source_enable", "Enable Outside Source");
        Tooltip(f, "When the projected light position is outside the frame, use the "
                   "colour and intensity below instead of sampling the image.");
        Color_knob(f, outside_source_color_, "outside_source_color", "Outside Color");
        SetFlags(f, Knob::STARTLINE);
        Float_knob(f, &outside_source_intensity_, "outside_source_intensity", "Outside Intensity");
        SetRange(f, 0.0, 50.0);
        Float_knob(f, &outside_source_falloff_, "outside_source_falloff", "Edge Falloff (px)");
        SetRange(f, 0.0, 200.0);
        Tooltip(f, "Blend zone in pixels at the frame edge.  0 = hard switch.");

        Divider(f, "Aperture");
        Int_knob(f, &aperture_blades_, "aperture_blades", "Aperture Blades");
        SetRange(f, 0.0, 16.0);
        Float_knob(f, &aperture_rotation_, "aperture_rotation", "Aperture Rotation");
        SetRange(f, -180.0, 180.0);

        Divider(f, "Spectral");
        static const char* const kSpecNames[] = { "3 (R/G/B)", "5", "7", "9", "11", "15", "21", "31", nullptr };
        Enumeration_knob(f, &spectral_idx_, kSpecNames, "spectral_samples", "Spectral Samples");
        Bool_knob(f, &spectral_jitter_, "spectral_jitter", "Spectral Jitter");
        Tooltip(f, "Randomise each ray's wavelength within its spectral bin.\n"
                   "Smooths the hard colour boundaries between discrete wavelength\n"
                   "samples at no extra ray-trace cost.");
        Int_knob(f, &spectral_jitter_seed_, "spectral_jitter_seed", "Spectral Jitter Seed");
        Tooltip(f, "Fixed seed for spectral jitter noise pattern.\n"
                   "Only used when Spectral Jitter Auto Seed is off.");
        Bool_knob(f, &spectral_jitter_auto_seed_, "spectral_jitter_auto_seed", "Auto Seed");
        Tooltip(f, "Derive spectral jitter seed from the current frame number.");
        Float_knob(f, &spectral_jitter_scale_, "spectral_jitter_scale", "Jitter Scale");
        SetRange(f, 0.0, 3.0);
        Tooltip(f, "Multiplier on spectral jitter range.\n"
                   "1.0 = one spectral bin width.  2.0 = aggressive cross-bin blending.");

        Divider(f, "Highlight");
        Bool_knob(f, &highlight_compress_, "highlight_compress", "Highlight Compression");
        Tooltip(f, "Apply soft-clip to ghost highlights.");
        static const char* const kMetricNames[] = { "Value", "Luminance", "Lightness", nullptr };
        Enumeration_knob(f, &highlight_metric_, kMetricNames, "highlight_metric", "Metric");
        Tooltip(f, "Value = max(R,G,B).  Luminance = Rec.709.  Lightness = cube root.");
        Float_knob(f, &highlight_clip_, "highlight_clip", "Clip");
        SetRange(f, 0.1, 10.0);
        Tooltip(f, "Maximum output value.  Default 2.0.");
        Float_knob(f, &highlight_knee_, "highlight_knee", "Knee");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "0 = soft rolloff, 1 = hard clip.  Matches AFXToneMap convention.");

        Divider(f, "Post-process");
        Float_knob(f, &ghost_blur_, "ghost_blur", "Ghost Blur");
        Int_knob(f, &ghost_blur_passes_, "ghost_blur_passes", "Ghost Blur Passes");

        // ---- Surfaces tab ----
        Tab_knob(f, "Surfaces");
        Button(f, "surf_refresh",      "Refresh Surfaces");
        Button(f, "surf_select_all",   "Select All");
        Button(f, "surf_deselect_all", "Deselect All");
        Button(f, "surf_reset_gains",  "Reset Gains");
        for (int k = 0; k < MAX_SURFS_UI; ++k)
        {
            Bool_knob(f, &surf_enabled_[k], s_surf_knob_names.toggle_names[k], surf_labels_[k]);
            SetFlags(f, Knob::STARTLINE | Knob::HIDDEN);
            Float_knob(f, &surf_gain_[k], s_surf_knob_names.gain_names[k], "gain");
            SetRange(f, 0.0, 10.0);
            SetFlags(f, Knob::HIDDEN | Knob::LOG_SLIDER | Knob::STARTLINE);
            Color_knob(f, surf_color_[k], s_surf_knob_names.color_names[k], "color");
            SetFlags(f, Knob::HIDDEN | Knob::STARTLINE);
            Float_knob(f, &surf_offx_[k], s_surf_knob_names.offx_names[k], "offset x");
            SetRange(f, -500.0, 500.0);
            SetFlags(f, Knob::HIDDEN | Knob::STARTLINE);
            Float_knob(f, &surf_offy_[k], s_surf_knob_names.offy_names[k], "offset y");
            SetRange(f, -500.0, 500.0);
            SetFlags(f, Knob::HIDDEN);
            ClearFlags(f, Knob::STARTLINE);
            Float_knob(f, &surf_scale_[k], s_surf_knob_names.scale_names[k], "scale");
            SetRange(f, 0.0, 5.0);
            SetFlags(f, Knob::HIDDEN | Knob::STARTLINE);
        }

        // ---- About tab ----
        Tab_knob(f, "About");
        Text_knob(f, "FlareSim3D\n"
                     "\n"
                     "Physically-based lens flare simulation for Nuke — 3D source mode.\n"
                     "\n"
                     "Based on the original work by Eamonn Nugent (space55/blackhole-rt)\n"
                     "\n"
                     "Copyright \xC2\xA9 2026 Steve Watts Kennedy (LocalStarlight/flaresim_nuke)\n"
                     "Copyright \xC2\xA9 2026 Peter Mercell — GPU optimisation\n");
    }

    // ---- knob_changed ----
    int knob_changed(Knob* k) override
    {
        if (k->is("surf_refresh")) { rebuild_surf_ui(); return 1; }
        if (k->is("surf_select_all") || k->is("surf_deselect_all")) {
            const bool val = k->is("surf_select_all");
            const int n = std::min(lens_.num_surfaces(), MAX_SURFS_UI);
            for (int i = 0; i < n; ++i) {
                surf_enabled_[i] = val;
                Knob* kn = knob(s_surf_knob_names.toggle_names[i]);
                if (kn) kn->set_value(val ? 1.0 : 0.0);
            }
            return 1;
        }
        if (k->is("surf_reset_gains")) {
            const int n = std::min(lens_.num_surfaces(), MAX_SURFS_UI);
            for (int i = 0; i < n; ++i) {
                surf_gain_[i] = 1.0f;
                knob(s_surf_knob_names.gain_names[i])->set_value(1.0);
                surf_color_[i][0] = surf_color_[i][1] = surf_color_[i][2] = 1.0f;
                knob(s_surf_knob_names.color_names[i])->set_value(1.0);
                surf_offx_[i] = 0.0f; surf_offy_[i] = 0.0f;
                knob(s_surf_knob_names.offx_names[i])->set_value(0.0);
                knob(s_surf_knob_names.offy_names[i])->set_value(0.0);
                surf_scale_[i] = 1.0f;
                knob(s_surf_knob_names.scale_names[i])->set_value(1.0);
            }
            return 1;
        }
        if (k->is("lens_file")) {
            const std::string path(lens_file_ ? lens_file_ : "");
            if (!path.empty() && path != last_lens_file_) {
                if (lens_.load(path.c_str())) {
                    last_lens_file_ = path;
                    rebuild_surf_ui();
                } else {
                    fprintf(stderr, "FlareSim3D: failed to load lens: %s\n", path.c_str());
                }
            }
            return 1;
        }
        return Iop::knob_changed(k);
    }

    void append(Hash& hash) override
    {
        Iop::append(hash);
        hash.append((int)outputContext().frame());
    }

    // ---- _validate ----
    void _validate(bool for_real) override
    {
        copy_info();
        info_.turn_on(Chan_Red);
        info_.turn_on(Chan_Green);
        info_.turn_on(Chan_Blue);
        info_.turn_on(Chan_Alpha);

        if (!for_real) return;

        input0().validate(for_real);

        // Validate 3D inputs (camera, axis)
        if (Op* op = Op::input(1)) op->validate(for_real);
        if (Op* op = Op::input(2)) op->validate(for_real);

        // Validate mask if connected
        if (Op* op = Op::input(3)) op->validate(for_real);

        const int x0 = info_.x(), y0 = info_.y();
        const int x1 = info_.r(), y1 = info_.t();
        const int w = x1 - x0, h = y1 - y0;
        if (w <= 0 || h <= 0) return;

        const std::string cur_lens(lens_file_ ? lens_file_ : "");
        if (cur_lens != last_lens_file_) {
            if (!cur_lens.empty()) {
                if (!lens_.load(cur_lens.c_str()))
                    fprintf(stderr, "FlareSim3D: failed to load lens file: %s\n",
                            cur_lens.c_str());
            }
            last_lens_file_ = cur_lens;
            rebuild_surf_ui();
        }

        {
            std::lock_guard<SpinLock> lock(compute_mutex_);
            if (!pending_cuda_error_.empty()) {
                std::string msg; msg.swap(pending_cuda_error_);
                error("%s", msg.c_str());
                return;
            }
            pending_x0_ = x0; pending_y0_ = y0;
            pending_x1_ = x1; pending_y1_ = y1;
            pending_w_ = w;   pending_h_ = h;
            const Format& fmt = format();
            pending_fmt_x0_ = fmt.x(); pending_fmt_y0_ = fmt.y();
            pending_fmt_w_  = fmt.width(); pending_fmt_h_ = fmt.height();
            pending_num_surfs_ = lens_.num_surfaces();
            pending_frame_ = (int)outputContext().frame();
            needs_compute_ = true;
        }
    }

    // ---- do_compute — 3D source projection ----
    void do_compute()
    {
        PROF_DECL;

        const int w = pending_w_;
        const int h = pending_h_;
        const size_t npx = (size_t)w * h;

        PROF_BEGIN("alloc_pinned");
        void* pinned = ensure_pinned_output_fp16(gpu_cache_, npx);
        uint16_t* pin16 = static_cast<uint16_t*>(pinned);
        ghost_r_ = pin16;
        ghost_g_ = pin16 + npx;
        ghost_b_ = pin16 + 2 * npx;
        alpha_   = pin16 + 3 * npx;
        cache_npx_ = npx;

        cache_width_  = w;
        cache_height_ = h;
        PROF_END();

        auto zero_buffers = [&]() {
            if (ghost_r_) std::memset(ghost_r_, 0, npx * sizeof(uint16_t));
            if (ghost_g_) std::memset(ghost_g_, 0, npx * sizeof(uint16_t));
            if (ghost_b_) std::memset(ghost_b_, 0, npx * sizeof(uint16_t));
            if (alpha_)   std::memset(alpha_,   0, npx * sizeof(uint16_t));
        };

        if (lens_.surfaces.empty()) { zero_buffers(); return; }

        // ================================================================
        // Get camera and axis transforms
        // ================================================================
        CameraOp* cam = nullptr;
        AxisOp* axis  = nullptr;

        if (Op* op1 = Op::input(1)) cam  = dynamic_cast<CameraOp*>(op1);
        if (Op* op2 = Op::input(2)) axis = dynamic_cast<AxisOp*>(op2);

        if (!cam || !axis) { zero_buffers(); return; }

        const float cam_focal = cam->focal_length();   // mm
        const float cam_hap   = cam->film_width();      // mm
        const float cam_vap   = cam->film_height();     // mm

        if (cam_focal < 0.1f || cam_hap < 0.1f || cam_vap < 0.1f) {
            zero_buffers();
            pending_cuda_error_ = "FlareSim3D: invalid camera parameters "
                                  "(check focal length and film back)";
            return;
        }

        const float fov_h = 2.0f * std::atan(cam_hap / (2.0f * cam_focal));
        const float fov_v = 2.0f * std::atan(cam_vap / (2.0f * cam_focal));

        // ---- Extract transforms ----
        const Matrix4 cam_world  = cam->matrix();
        const Matrix4 axis_world = axis->matrix();
        Matrix4 world_to_cam = cam_world.inverse();

        // Axis world position — try column-vector then row-vector convention.
        Vector4 axis_origin_cv = axis_world * Vector4(0, 0, 0, 1);
        float rv_x = axis_world[3][0];
        float rv_y = axis_world[3][1];
        float rv_z = axis_world[3][2];

        float ax_x, ax_y, ax_z;
        {
            const float cv_dist = std::abs(axis_origin_cv.x) + std::abs(axis_origin_cv.y) + std::abs(axis_origin_cv.z);
            const float rv_dist = std::abs(rv_x) + std::abs(rv_y) + std::abs(rv_z);
            if (cv_dist > 0.001f) {
                ax_x = axis_origin_cv.x;
                ax_y = axis_origin_cv.y;
                ax_z = axis_origin_cv.z;
            } else if (rv_dist > 0.001f) {
                ax_x = rv_x; ax_y = rv_y; ax_z = rv_z;
            } else {
                zero_buffers(); return;
            }
        }

        // Camera-space position — try both conventions.
        Vector4 light_cam_cv = world_to_cam * Vector4(ax_x, ax_y, ax_z, 1);
        float lc_x = ax_x * world_to_cam[0][0] + ax_y * world_to_cam[1][0] + ax_z * world_to_cam[2][0] + world_to_cam[3][0];
        float lc_y = ax_x * world_to_cam[0][1] + ax_y * world_to_cam[1][1] + ax_z * world_to_cam[2][1] + world_to_cam[3][1];
        float lc_z = ax_x * world_to_cam[0][2] + ax_y * world_to_cam[1][2] + ax_z * world_to_cam[2][2] + world_to_cam[3][2];

        float cam_lx, cam_ly, cam_lz;
        if (light_cam_cv.z < -0.001f) {
            cam_lx = light_cam_cv.x; cam_ly = light_cam_cv.y; cam_lz = light_cam_cv.z;
        } else if (lc_z < -0.001f) {
            cam_lx = lc_x; cam_ly = lc_y; cam_lz = lc_z;
        } else {
            zero_buffers(); return;  // source behind camera
        }

        const float source_z = -cam_lz;
        const float source_dist = std::sqrt(cam_lx * cam_lx + cam_ly * cam_ly + cam_lz * cam_lz);

        // ================================================================
        // Project to screen coordinates
        // ================================================================
        const int x0 = pending_x0_;
        const int y0 = pending_y0_;
        const int x1 = pending_x1_;
        const int y1 = pending_y1_;

        const float fmt_cx = pending_fmt_x0_ + pending_fmt_w_ * 0.5f;
        const float fmt_cy = pending_fmt_y0_ + pending_fmt_h_ * 0.5f;

        const float film_x = cam_lx * cam_focal / source_z;
        const float film_y = cam_ly * cam_focal / source_z;
        const float norm_x = film_x / cam_hap;
        const float norm_y = film_y / cam_vap;

        const int mx = (int)std::round(norm_x * pending_fmt_w_ + fmt_cx);
        const int my = (int)std::round(norm_y * pending_fmt_h_ + fmt_cy);

        const float source_angle_x = std::atan2(cam_lx, source_z);
        const float source_angle_y = std::atan2(cam_ly, source_z);

        // ================================================================
        // Determine on-screen / off-screen status
        // ================================================================
        const float fmt_left   = (float)pending_fmt_x0_;
        const float fmt_right  = (float)(pending_fmt_x0_ + pending_fmt_w_);
        const float fmt_bottom = (float)pending_fmt_y0_;
        const float fmt_top    = (float)(pending_fmt_y0_ + pending_fmt_h_);
        const float fmx = (float)mx;
        const float fmy = (float)my;

        const float dist_inside = std::min({fmx - fmt_left,
                                            fmt_right - fmx,
                                            fmy - fmt_bottom,
                                            fmt_top - fmy});

        const bool source_on_screen = (dist_inside > 0.0f);

        // ================================================================
        // Sample plate colour (when source is on screen)
        // ================================================================
        const int sr  = std::max(sample_radius_, 1);
        const int sy0 = std::max(y0, my - sr);
        const int sy1 = std::min(y1, my + sr + 1);
        const int sx0 = std::max(x0, mx - sr);
        const int sx1 = std::min(x1, mx + sr + 1);

        float img_r = 0, img_g = 0, img_b = 0;
        bool  have_image_sample = false;

        if (source_on_screen && sx0 < sx1 && sy0 < sy1)
        {
            float sum_r = 0, sum_g = 0, sum_b = 0;
            int   cnt   = 0;
            Row sample_row(x0, x1);
            for (int iy = sy0; iy < sy1; ++iy)
            {
                input0().get(iy, x0, x1, Mask_RGB, sample_row);
                if (Op::aborted()) { zero_buffers(); return; }
                const float* rp = sample_row[Chan_Red];
                const float* gp = sample_row[Chan_Green];
                const float* bp = sample_row[Chan_Blue];
                for (int ix = sx0; ix < sx1; ++ix)
                {
                    sum_r += rp[ix]; sum_g += gp[ix]; sum_b += bp[ix]; ++cnt;
                }
            }
            if (cnt > 0) {
                img_r = sum_r / cnt;
                img_g = sum_g / cnt;
                img_b = sum_b / cnt;
                float luma = 0.2126f * img_r + 0.7152f * img_g + 0.0722f * img_b;
                have_image_sample = (luma >= threshold_);
            }
        }

        // ================================================================
        // Build the source, blending with outside colour if needed
        // ================================================================
        std::vector<BrightPixel> sources;

        {
            const float out_si = outside_source_intensity_ * 1000.0f;
            const float out_r = outside_source_color_[0] * out_si;
            const float out_g = outside_source_color_[1] * out_si;
            const float out_b = outside_source_color_[2] * out_si;

            float final_r, final_g, final_b;
            bool  emit_source = false;

            if (have_image_sample && !outside_source_enable_) {
                const float si = source_intensity_ * 1000.0f;
                final_r = img_r * si;
                final_g = img_g * si;
                final_b = img_b * si;
                emit_source = true;
            }
            else if (have_image_sample && outside_source_enable_) {
                const float falloff = std::max(outside_source_falloff_, 0.0f);
                if (falloff > 0.0f && dist_inside < falloff) {
                    const float t = std::clamp(dist_inside / falloff, 0.0f, 1.0f);
                    const float si = source_intensity_ * 1000.0f;
                    final_r = t * (img_r * si) + (1.0f - t) * out_r;
                    final_g = t * (img_g * si) + (1.0f - t) * out_g;
                    final_b = t * (img_b * si) + (1.0f - t) * out_b;
                } else {
                    const float si = source_intensity_ * 1000.0f;
                    final_r = img_r * si;
                    final_g = img_g * si;
                    final_b = img_b * si;
                }
                emit_source = true;
            }
            else if (!source_on_screen && outside_source_enable_) {
                final_r = out_r;
                final_g = out_g;
                final_b = out_b;
                emit_source = true;
            }
            // else: off-screen and outside_source disabled → no flare

            // Apply distance-based inverse-square intensity falloff
            if (emit_source && intensity_falloff_ && source_dist > 0.001f) {
                const float ref = std::max(reference_distance_, 0.001f);
                float scale = (ref * ref) / (source_dist * source_dist);
                scale = std::min(scale, 10000.0f);  // clamp to prevent explosion
                final_r *= scale;
                final_g *= scale;
                final_b *= scale;
            }

            if (emit_source) {
                BrightPixel bp_out;
                bp_out.angle_x = source_angle_x;
                bp_out.angle_y = source_angle_y;
                bp_out.r = final_r;
                bp_out.g = final_g;
                bp_out.b = final_b;
                sources.push_back(bp_out);
            }
        }
        last_src_count_.store((int)sources.size());

        if (Op::aborted()) { zero_buffers(); return; }
        if (sources.empty()) { zero_buffers(); return; }

        // ================================================================
        // Ghost rendering — same as FlareSim from here on
        // ================================================================

        // sensor_half uses the lens model's focal length for coordinate mapping,
        // but FOV from the camera.  This is intentional: the .lens file describes
        // the optical elements (ghost shapes), while the camera defines the FOV
        // (where ghosts land on screen).
        const float lens_fl = (lens_.focal_length > 0.1f)
                                ? lens_.focal_length : cam_focal;
        const float sensor_half_w = lens_fl * std::tan(fov_h * 0.5f);
        const float sensor_half_h = lens_fl * std::tan(fov_v * 0.5f);

        GhostConfig cfg;
        cfg.ray_grid              = ray_grid_;
        cfg.gain                  = flare_gain_ * 1000.0f;
        cfg.aperture_blades       = aperture_blades_;
        cfg.aperture_rotation_deg = aperture_rotation_;
        static const int kSpecCounts[] = {3, 5, 7, 9, 11, 15, 21, 31};
        cfg.spectral_samples      = kSpecCounts[std::max(0, std::min(spectral_idx_, 7))];
        cfg.spectral_jitter       = spectral_jitter_ ? 1 : 0;
        cfg.spectral_jitter_seed  = spectral_jitter_auto_seed_
                                      ? (int)outputContext().frame()
                                      : spectral_jitter_seed_;
        cfg.spectral_jitter_scale = spectral_jitter_scale_;
        cfg.highlight_clip        = highlight_compress_ ? highlight_clip_ : 0.0f;
        cfg.highlight_knee        = highlight_knee_;
        cfg.highlight_metric      = highlight_metric_;
        cfg.pupil_jitter          = pupil_jitter_;
        cfg.pupil_jitter_seed     = jitter_auto_seed_
                                      ? (int)outputContext().frame()
                                      : jitter_seed_;

        std::vector<GhostPair> active_pairs;
        std::vector<float>     area_boosts;

        PROF_BEGIN("filter_pairs");
        const bool filter_cached =
            !cached_filter_pairs_.empty() &&
            cached_filter_lens_ == lens_.name &&
            cached_filter_shw_  == sensor_half_w &&
            cached_filter_shh_  == sensor_half_h;

        if (filter_cached) {
            active_pairs = cached_filter_pairs_;
            area_boosts  = cached_filter_boosts_;
        } else {
            filter_ghost_pairs(lens_, sensor_half_w, sensor_half_h, cfg,
                               active_pairs, area_boosts);
            cached_filter_pairs_  = active_pairs;
            cached_filter_boosts_ = area_boosts;
            cached_filter_shw_    = sensor_half_w;
            cached_filter_shh_    = sensor_half_h;
            cached_filter_lens_   = lens_.name;
        }
        PROF_END();

        // Apply per-surface toggles and gains
        {
            std::vector<GhostPair> ep; std::vector<float> eb, ec, eo, es;
            ep.reserve(active_pairs.size());
            eb.reserve(active_pairs.size());
            ec.reserve(active_pairs.size() * 3);
            eo.reserve(active_pairs.size() * 2);
            es.reserve(active_pairs.size());

            for (int i = 0; i < (int)active_pairs.size(); ++i)
            {
                const int sa = active_pairs[i].surf_a;
                const int sb = active_pairs[i].surf_b;
                const bool a_on = (sa < 0 || sa >= MAX_SURFS_UI) || surf_enabled_[sa];
                const bool b_on = (sb < 0 || sb >= MAX_SURFS_UI) || surf_enabled_[sb];
                if (!a_on || !b_on) continue;

                float ga = (sa >= 0 && sa < MAX_SURFS_UI) ? surf_gain_[sa] : 1.0f;
                float gb = (sb >= 0 && sb < MAX_SURFS_UI) ? surf_gain_[sb] : 1.0f;
                ep.push_back(active_pairs[i]);
                eb.push_back(area_boosts[i] * ga * gb);

                float cr = 1, cg = 1, cb = 1;
                if (sa >= 0 && sa < MAX_SURFS_UI) {
                    cr *= surf_color_[sa][0]; cg *= surf_color_[sa][1]; cb *= surf_color_[sa][2];
                }
                if (sb >= 0 && sb < MAX_SURFS_UI) {
                    cr *= surf_color_[sb][0]; cg *= surf_color_[sb][1]; cb *= surf_color_[sb][2];
                }
                ec.push_back(cr); ec.push_back(cg); ec.push_back(cb);

                float ox = 0, oy = 0;
                if (sa >= 0 && sa < MAX_SURFS_UI) { ox += surf_offx_[sa]; oy += surf_offy_[sa]; }
                if (sb >= 0 && sb < MAX_SURFS_UI) { ox += surf_offx_[sb]; oy += surf_offy_[sb]; }
                eo.push_back(ox); eo.push_back(oy);

                float sc = 1;
                if (sa >= 0 && sa < MAX_SURFS_UI) sc *= surf_scale_[sa];
                if (sb >= 0 && sb < MAX_SURFS_UI) sc *= surf_scale_[sb];
                es.push_back(sc);
            }
            active_pairs = std::move(ep);
            area_boosts  = std::move(eb);
            pair_colors_  = std::move(ec);
            pair_offsets_ = std::move(eo);
            pair_scales_  = std::move(es);
        }

        last_pair_count_.store((int)active_pairs.size());
        if (active_pairs.empty()) { zero_buffers(); return; }
        if (Op::aborted()) { zero_buffers(); return; }

        pending_cuda_error_.clear();
        const int fmt_x0_in_buf = pending_fmt_x0_ - x0;
        const int fmt_y0_in_buf = pending_fmt_y0_ - y0;

        PROF_BEGIN("gpu_ghost_render");
        launch_ghost_cuda(lens_, active_pairs, area_boosts, sources,
                          sensor_half_w, sensor_half_h,
                          nullptr, nullptr, nullptr,
                          w, h,
                          pending_fmt_w_, pending_fmt_h_,
                          fmt_x0_in_buf, fmt_y0_in_buf,
                          cfg, gpu_cache_,
                          /*skip_readback=*/true,
                          &pending_cuda_error_,
                          &pair_colors_, &pair_offsets_, &pair_scales_);
        PROF_END();

        PROF_BEGIN("gpu_blur_alpha_readback");
        {
            int radius = 0;
            if (ghost_blur_ > 0.0f && ghost_blur_passes_ > 0) {
                float diag = std::sqrt((float)w * w + (float)h * h);
                radius = std::max(1, (int)std::round(ghost_blur_ * diag));
            }
            launch_blur_alpha_readback_async(
                reinterpret_cast<float*>(ghost_r_),
                reinterpret_cast<float*>(ghost_g_),
                reinterpret_cast<float*>(ghost_b_),
                reinterpret_cast<float*>(alpha_),
                w, h, radius, ghost_blur_passes_,
                gpu_cache_, &pending_cuda_error_,
                cfg.highlight_clip, cfg.highlight_knee, cfg.highlight_metric);
        }
        PROF_END();

        PROF_SUMMARY();
    }

    // ---- engine ----
    void engine(int y, int x, int r,
                ChannelMask channels, Row& row) override
    {
        const int cur_frame = (int)outputContext().frame();
        {
            std::lock_guard<SpinLock> lock(compute_mutex_);
            if (needs_compute_ && pending_frame_ == cur_frame) {
                needs_compute_ = false;
                do_compute();
                cache_frame_ = cur_frame;
            }
        }

        const bool frame_ok = (cache_frame_ == cur_frame);

        {
            ChannelSet pass_through(channels);
            pass_through -= Chan_Red; pass_through -= Chan_Green;
            pass_through -= Chan_Blue; pass_through -= Chan_Alpha;
            if (pass_through) input0().get(y, x, r, pass_through, row);
        }

        const int  cache_y     = y - info_.y();
        const bool valid_cache = frame_ok && ghost_r_ && cache_y >= 0 && cache_y < cache_height_;
        const size_t row_off   = valid_cache ? (size_t)cache_y * cache_width_ : 0;
        const int    x_off     = info_.x();

        auto write_buf = [&](Channel ch, const uint16_t* buf) {
            if (!channels.contains(ch)) return;
            float* dst = row.writable(ch);
            if (!valid_cache || !buf) {
                for (int i = x; i < r; ++i) dst[i] = 0.0f;
                return;
            }
            const int xi_first = std::max(0, x - x_off);
            const int xi_last  = std::min(cache_width_, r - x_off);
            for (int i = x; i < std::min(r, x_off + xi_first); ++i) dst[i] = 0.0f;
            if (xi_last > xi_first)
                convert_fp16_scanline(buf + row_off + xi_first,
                                      dst + x_off + xi_first,
                                      xi_last - xi_first);
            for (int i = std::max(x, x_off + xi_last); i < r; ++i) dst[i] = 0.0f;
        };

        write_buf(Chan_Red,   ghost_r_);
        write_buf(Chan_Green, ghost_g_);
        write_buf(Chan_Blue,  ghost_b_);
        write_buf(Chan_Alpha, alpha_);
    }

    // ---- _request ----
    void _request(int x, int y, int r, int t,
                  ChannelMask channels, int count) override
    {
        // Request full frame RGB from plate for source colour sampling.
        input0().request(info_.x(), info_.y(), info_.r(), info_.t(),
                         Mask_RGB, count);

        // Mask (input 3) — optional Iop input.
        if (Op* op = Op::input(3)) {
            if (Iop* mask = dynamic_cast<Iop*>(op))
                mask->request(info_.x(), info_.y(), info_.r(), info_.t(),
                              Mask_Alpha, count);
        }

        // Don't request from camera (input 1) or axis (input 2) —
        // they provide transforms, not pixels.

        ChannelSet passthru(channels);
        passthru -= Mask_RGBA;
        if (passthru) input0().request(x, y, r, t, passthru, count);
    }
};

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

static Iop* build(Node* node) { return new FlareSim3D(node); }

const Iop::Description FlareSim3D::d(
    "FlareSim3D",
    "Filter/FlareSim3D",
    build);
