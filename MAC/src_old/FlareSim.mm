// ============================================================================
// FlareSim.mm — Nuke Iop plugin for physically-based lens flare simulation
// macOS / Apple Silicon Metal compute backend.
//
// Architecture (lazy-compute-in-engine):
//   _validate()  — sets up channels/format, loads lens, pre-clears buffers,
//                  sets needs_compute_ = true.
//   engine()     — on the first scanline call, acquires compute_mutex_ and
//                  runs the full tile read + CUDA simulation (do_compute()).
//                  Subsequent scanline calls read directly from the cache.
//
// This design is required because Nuke's background playback cacher calls
// _validate() before input pixels are available. By deferring pixel reads to
// engine() we guarantee the input is fully rendered before we touch it.
// ============================================================================

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Channel.h"
#include "DDImage/ChannelSet.h"

#include "lens.h"
#include "ghost.h"
#include "ghost_metal.h"

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
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace DD::Image;

// ---------------------------------------------------------------------------
// Named output channels
// ---------------------------------------------------------------------------

// (AOV channels removed — output is RGBA only)

// ---------------------------------------------------------------------------
// Sensor size presets
// ---------------------------------------------------------------------------

struct SensorPreset { const char* name; float w, h; };
static const SensorPreset kSensorPresets[] = {
    { "Custom",            0.0f,   0.0f  },
    { "Full Frame",       36.0f,  24.0f  },
    { "Super 35",         24.89f, 18.66f },
    { "APS-C Canon",      22.3f,  14.9f  },
    { "APS-C Nikon/Sony", 23.5f,  15.6f  },
    { "Micro Four Thirds",17.3f,  13.0f  },
    { nullptr, 0, 0 }
};
static const char* const kPresetNames[] = {
    "Custom", "Full Frame", "Super 35",
    "APS-C Canon", "APS-C Nikon/Sony", "Micro Four Thirds", nullptr
};

// ---------------------------------------------------------------------------
// Separable box blur (prefix-sum, O(w*h) per pass).
// Blurs buf in-place; tmp is a scratch buffer (resized as needed).
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
        // ---- Horizontal pass: buf -> tmp ----
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

        // ---- Vertical pass: buf -> tmp ----
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
// FlareSim
// ---------------------------------------------------------------------------

class FlareSim : public Iop
{
public:
    // ---- Knob storage ----
    const char* lens_file_;
    float       source_intensity_; // HDR boost for sampled source (default 8, internal ×1000)
    float       flare_gain_;       // artistic ghost multiplier (default 10, internal ×1000)
    int         ray_grid_;
    float       threshold_;

    bool        fov_use_sensor_;
    int         sensor_preset_;
    float       fov_h_deg_;
    float       fov_v_deg_;
    bool        fov_auto_v_;
    float       sensor_w_mm_;
    float       sensor_h_mm_;
    float       focal_length_mm_;
    float       ghost_blur_;
    int         ghost_blur_passes_;

    // Aperture
    int         aperture_blades_;
    float       aperture_rotation_;

    // Pupil sampling
    int         pupil_jitter_;
    int         jitter_seed_;
    bool        jitter_auto_seed_; // derive seed from frame number automatically

    // Spectral quality (index into {3,5,7,9,11})
    int         spectral_idx_;
    bool        spectral_jitter_;
    int         spectral_jitter_seed_;
    bool        spectral_jitter_auto_seed_;
    float       spectral_jitter_scale_;

    // Highlight compression
    bool        highlight_compress_;
    float       highlight_clip_;
    float       highlight_knee_;
    int         highlight_metric_;

    // Source position (art-directable, e.g. from tracker)
    double      manual_xy_[2];         // pixel position (animatable XY_knob)
    int         manual_sample_radius_; // averaging radius around xy (pixels)

    // Off-screen source: when Source XY is outside the frame, use these
    // instead of sampling the image.  The flare keeps rendering seamlessly.
    bool        outside_source_enable_;
    float       outside_source_color_[3];
    float       outside_source_intensity_;
    float       outside_source_falloff_;   // blend zone in pixels at frame edge

    // Per-surface toggles, gain, color, and offset (MAX_SURFS_UI pre-allocated)
    bool        surf_enabled_[MAX_SURFS_UI];
    float       surf_gain_[MAX_SURFS_UI];
    float       surf_color_[MAX_SURFS_UI][3];  // RGB tint per surface
    float       surf_offx_[MAX_SURFS_UI];      // pixel offset X per surface
    float       surf_offy_[MAX_SURFS_UI];      // pixel offset Y per surface
    float       surf_scale_[MAX_SURFS_UI];     // scale per surface (1.0 = default)
    // Per-instance labels for surface knobs (stable pointers for Knob API)
    char        surf_labels_[MAX_SURFS_UI][64];

    // ---- Runtime state ----
    LensSystem  lens_;
    std::string last_lens_file_;

    // Full enumerated pair list — updated when lens changes.
    std::vector<GhostPair> all_lens_pairs_;

    // Diagnostic counters — written atomically by do_compute().
    std::atomic<int> last_src_count_  {0};
    std::atomic<int> last_pair_count_ {0};

    // Per-pair art direction (rebuilt each frame from per-surface knobs).
    std::vector<float> pair_colors_;   // 3 per pair (RGB)
    std::vector<float> pair_offsets_;  // 2 per pair (XY px)
    std::vector<float> pair_scales_;   // 1 per pair

    // Ghost output buffers — pointers into gpu_cache_.h_pinned (FP16 pinned memory).
    // DMA writes FP16 directly here; engine() converts to FP32 per-scanline.
    uint16_t* ghost_r_  = nullptr;
    uint16_t* ghost_g_  = nullptr;
    uint16_t* ghost_b_  = nullptr;
    uint16_t* alpha_    = nullptr;
    size_t cache_npx_ = 0;   // current capacity in pixels

    int cache_width_  = 0;
    int cache_height_ = 0;

    // Persistent GPU buffer cache — avoids per-frame cudaMalloc/cudaFree.
    // Only accessed inside do_compute() which runs under compute_mutex_.
    MetalBufferCache gpu_cache_;

    // Cached ghost_filter results — reused when lens hasn't changed.
    std::vector<GhostPair> cached_filter_pairs_;
    std::vector<float>     cached_filter_boosts_;
    float                  cached_filter_shw_ = -1.0f;
    float                  cached_filter_shh_ = -1.0f;
    std::string            cached_filter_lens_;  // lens name at time of cache

    // CUDA error from last do_compute() — written under compute_mutex_.
    std::string pending_gpu_error_;

    // Lazy-compute state (all written under compute_mutex_).
    bool        needs_compute_ = false;
    int         pending_frame_ = -1;     // frame number this pending state belongs to
    int         cache_frame_   = -1;     // frame number the current cache holds
    int         pending_x0_    = 0;
    int         pending_y0_    = 0;
    int         pending_x1_    = 0;
    int         pending_y1_    = 0;
    int         pending_w_     = 0;
    int         pending_h_     = 0;
    // Format dimensions — anchor optical axis to format centre, not bbox.
    int         pending_fmt_x0_ = 0;
    int         pending_fmt_y0_ = 0;
    int         pending_fmt_w_  = 0;
    int         pending_fmt_h_  = 0;
    // Snapshot of surface count for do_compute() (safe from _validate() race).
    int pending_num_surfs_ = 0;
    std::mutex  compute_mutex_;

    // ---- Constructor ----
    explicit FlareSim(Node* node)
        : Iop(node)
        , lens_file_("")
        , source_intensity_(8.0f)
        , flare_gain_(10.0f)
        , ray_grid_(64)
        , threshold_(0.0f)
        , fov_use_sensor_(false)
        , sensor_preset_(0)
        , fov_h_deg_(40.0f)
        , fov_v_deg_(24.0f)
        , fov_auto_v_(true)
        , sensor_w_mm_(36.0f)
        , sensor_h_mm_(24.0f)
        , focal_length_mm_(50.0f)
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
        , manual_sample_radius_(4)
        , outside_source_enable_(true)
        , outside_source_intensity_(8.0f)
        , outside_source_falloff_(0.0f)
    {
        manual_xy_[0] = 960.0;  // sensible default (centre of 1920 frame)
        manual_xy_[1] = 540.0;
        outside_source_color_[0] = 1.0f;
        outside_source_color_[1] = 1.0f;
        outside_source_color_[2] = 1.0f;
        for (int k = 0; k < MAX_SURFS_UI; ++k)
            surf_enabled_[k] = true;
        for (int k = 0; k < MAX_SURFS_UI; ++k)
            surf_gain_[k] = 1.0f;
        for (int k = 0; k < MAX_SURFS_UI; ++k) {
            surf_color_[k][0] = 1.0f;
            surf_color_[k][1] = 1.0f;
            surf_color_[k][2] = 1.0f;
        }
        for (int k = 0; k < MAX_SURFS_UI; ++k) {
            surf_offx_[k] = 0.0f;
            surf_offy_[k] = 0.0f;
        }
        for (int k = 0; k < MAX_SURFS_UI; ++k)
            surf_scale_[k] = 1.0f;
        for (int k = 0; k < MAX_SURFS_UI; ++k)
            snprintf(surf_labels_[k], sizeof(surf_labels_[k]), "Surf %d", k);
    }

    // ---- Nuke boilerplate ----
    const char* Class()     const override { return "FlareSim"; }
    const char* node_help() const override
    {
        return "Physically-based lens flare simulation (ghost ray tracing).\n\nmacOS / Apple Silicon Metal compute backend.\n\n"
               "Loads a .lens prescription file and traces ghost reflections "
               "for every bright pixel in the input image.\n\n"
               "Output: ghost reflections in RGBA.\n"
               "Alpha is derived from flare luminance for compositing.\n\n"
               "Connect a second input to use it as a mask (alpha channel) "
               "that gates which bright regions drive the flare.\n\n"
               "Merge the flare over the beauty with a Merge (plus) node.";
    }

    static const Iop::Description d;

    int  maximum_inputs() const override { return 2; }
    int  minimum_inputs() const override { return 1; }
    const char* input_label(int idx, char*) const override
    {
        if (idx == 1) return "mask";
        return "";
    }

    // ---- rebuild_surf_ui ----
    // Updates the Surf tab to show one toggle + gain per surface in the
    // loaded lens system.  Also populates all_lens_pairs_ so do_compute()
    // can match surface toggles to filtered pairs.
    void rebuild_surf_ui()
    {
        // Invalidate filter cache when lens changes.
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

        // Enumerate all pairs for the physics filter (still needed internally).
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
            if (kt) {
                kt->label(surf_labels_[k]);
                kt->clear_flag(Knob::HIDDEN);
            }
            Knob* kg = knob(s_surf_knob_names.gain_names[k]);
            if (kg) {
                kg->clear_flag(Knob::HIDDEN);
            }
            Knob* kc = knob(s_surf_knob_names.color_names[k]);
            if (kc) kc->clear_flag(Knob::HIDDEN);
            Knob* kox = knob(s_surf_knob_names.offx_names[k]);
            if (kox) kox->clear_flag(Knob::HIDDEN);
            Knob* koy = knob(s_surf_knob_names.offy_names[k]);
            if (koy) koy->clear_flag(Knob::HIDDEN);
            Knob* ks = knob(s_surf_knob_names.scale_names[k]);
            if (ks) ks->clear_flag(Knob::HIDDEN);
        }
        // Hide slots beyond the surface count
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

        // Force the properties panel to re-layout so newly-visible knobs appear.
        updateUI(outputContext());
    }

    // ---- Knobs ----
    void knobs(Knob_Callback f) override
    {
        File_knob(f, &lens_file_,  "lens_file",      "Lens File");
        Tooltip(f, "Path to a .lens prescription file.");

        Divider(f, "Ghost");
        Int_knob(f,   &ray_grid_,   "ray_grid",      "Ray Grid (NxN)");
        Tooltip(f, "NxN entrance-pupil samples per source. "
                   "Higher = smoother ghosts, longer render time.");

        Float_knob(f, &flare_gain_, "flare_gain",    "Flare Gain");
        SetRange(f, 0.0, 50.0);
        Tooltip(f, "Ghost intensity multiplier.\n\n"
                   "Controls overall ghost brightness.  Typical range 1–20.  Default 10.");

        static const char* const kJitterModes[] = {
            "Off", "Stratified", "Halton", nullptr
        };
        Enumeration_knob(f, &pupil_jitter_, kJitterModes, "pupil_jitter", "Pupil Jitter");
        Tooltip(f, "Entrance-pupil sampling pattern.\n"
                   "Off: regular NxN grid — fastest, but visible dot pattern at low ray counts.\n"
                   "Stratified: one randomly-placed sample per grid cell — breaks up the dot pattern.\n"
                   "Halton: quasi-random low-discrepancy sequence — deterministic, very even coverage.");

        Int_knob(f, &jitter_seed_, "jitter_seed", "Jitter Seed");
        Tooltip(f, "Integer seed for the stratified noise pattern. "
                   "Only used when Auto Seed is off and Pupil Jitter is Stratified.");
        Bool_knob(f, &jitter_auto_seed_, "jitter_auto_seed", "Auto Seed");
        Tooltip(f, "When on (the default), the jitter seed is automatically set to the current "
                   "frame number, so sample positions change on every frame without any manual "
                   "keyframing — equivalent to animating Jitter Seed = frame.\n\n"
                   "Turn off to use the fixed Jitter Seed value, which gives a repeatable "
                   "pattern regardless of frame number (useful for still renders or when you "
                   "need the exact same noise across multiple passes).\n\n"
                   "Only affects Stratified mode; ignored for Off and Halton.");

        Divider(f, "Source");
        XY_knob(f, manual_xy_, "manual_xy", "Source XY");
        Tooltip(f, "Pixel position of the flare source.  Animatable — link to a "
                   "Tracker node's tracking output for stabilised flares.\n\n"
                   "The input image is sampled at this position: if the brightness "
                   "exceeds Threshold a flare is produced; otherwise the output is black.");
        Int_knob(f, &manual_sample_radius_, "manual_sample_radius", "Sample Radius");
        Tooltip(f, "Radius (in pixels) of the area averaged around Source XY to "
                   "determine source colour and brightness.  Larger values are more "
                   "robust to sub-pixel tracking jitter.  Default 4.");
        Float_knob(f, &source_intensity_,  "source_intensity",  "Source Intensity");
        SetRange(f, 1.0, 50.0);
        Tooltip(f, "How bright the light source is relative to the plate.\n\n"
                   "Restores the HDR brightness the camera clipped.  "
                   "Practical lamp = 2, headlight = 8, sun = 30.\n"
                   "Default 8.");
        Float_knob(f, &threshold_,  "threshold",     "Threshold");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Minimum raw plate luminance at Source XY for a flare to appear.\n\n"
                   "0 = always produce a flare (any non-black pixel).  "
                   "0.5 = only above mid-grey.  1 = only pure white.\n"
                   "Default 0.");

        Divider(f, "Outside Source");
        Bool_knob(f, &outside_source_enable_, "outside_source_enable", "Enable Outside Source");
        Tooltip(f, "When the Source XY is outside the frame, use the colour "
                   "and intensity below instead of sampling the image.  "
                   "The flare keeps rendering seamlessly as the source "
                   "leaves or re-enters the plate.");
        Color_knob(f, outside_source_color_, "outside_source_color", "Outside Color");
        SetFlags(f, Knob::STARTLINE);
        Tooltip(f, "RGB colour of the off-screen light source.  "
                   "White (1, 1, 1) gives a neutral flare.");
        Float_knob(f, &outside_source_intensity_, "outside_source_intensity", "Outside Intensity");
        SetRange(f, 0.0, 50.0);
        Tooltip(f, "Intensity of the off-screen source (same scale as Source Intensity).");
        Float_knob(f, &outside_source_falloff_, "outside_source_falloff", "Edge Falloff (px)");
        SetRange(f, 0.0, 200.0);
        Tooltip(f, "Blend zone in pixels at the frame edge.  0 = hard switch "
                   "(the image-sampled colour pops to outside colour instantly).  "
                   "Higher values smoothly interpolate between the two near the edge.");

        Divider(f, "Camera");
        Bool_knob(f, &fov_use_sensor_, "fov_use_sensor", "Use Sensor Size");
        Tooltip(f, "When enabled, FOV is computed from sensor dimensions and focal length.");
        Enumeration_knob(f, &sensor_preset_, kPresetNames, "sensor_preset", "Sensor Preset");
        if (!fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Tooltip(f, "Common sensor size presets. Selecting a preset fills Sensor Width and Height.");

        Float_knob(f, &fov_h_deg_,  "fov_h",         "FOV H (deg)");
        SetRange(f, 1.0, 180.0);
        if (fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Tooltip(f, "Horizontal field of view of the input image in degrees.");
        Bool_knob(f, &fov_auto_v_,  "fov_auto_v",    "Auto FOV V");
        if (fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Tooltip(f, "Derive vertical FOV from horizontal FOV and image aspect ratio.");
        Float_knob(f, &fov_v_deg_,  "fov_v",         "FOV V (deg)");
        if (fov_auto_v_ || fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Tooltip(f, "Vertical field of view in degrees. Ignored when Auto FOV V is enabled.");

        Float_knob(f, &sensor_w_mm_,      "sensor_w",      "Sensor Width (mm)");
        SetRange(f, 1.0, 100.0);
        if (!fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Float_knob(f, &sensor_h_mm_,      "sensor_h",      "Sensor Height (mm)");
        SetRange(f, 1.0, 100.0);
        if (!fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Float_knob(f, &focal_length_mm_,  "focal_length",  "Focal Length (mm)");
        SetRange(f, 1.0, 2000.0);
        if (!fov_use_sensor_) SetFlags(f, Knob::DISABLED);

        Divider(f, "Aperture");
        Int_knob(f, &aperture_blades_, "aperture_blades", "Aperture Blades");
        SetRange(f, 0.0, 16.0);
        Tooltip(f, "0 = circular aperture. 3–16 = regular polygon — affects ghost shape.");
        Float_knob(f, &aperture_rotation_, "aperture_rotation", "Aperture Rotation");
        SetRange(f, -180.0, 180.0);
        Tooltip(f, "Rotation of the aperture polygon in degrees.");

        Divider(f, "Spectral");
        static const char* const kSpecNames[] = { "3 (R/G/B)", "5", "7", "9", "11", "15", "21", "31", nullptr };
        Enumeration_knob(f, &spectral_idx_, kSpecNames, "spectral_samples", "Spectral Samples");
        Tooltip(f, "Number of wavelength samples for chromatic aberration. "
                   "3 = classic R/G/B (fastest). Higher = smoother colour fringing.");
        Bool_knob(f, &spectral_jitter_, "spectral_jitter", "Spectral Jitter");
        Tooltip(f, "Randomise each ray's wavelength within its spectral bin.\n"
                   "Smooths the hard colour boundaries between discrete wavelength\n"
                   "samples at no extra ray-trace cost.  The noise pattern changes\n"
                   "per frame when Auto Seed is on, enabling temporal accumulation.");
        Int_knob(f, &spectral_jitter_seed_, "spectral_jitter_seed", "Spectral Jitter Seed");
        Tooltip(f, "Fixed seed for the spectral jitter noise pattern.\n"
                   "Only used when Spectral Jitter Auto Seed is off.");
        Bool_knob(f, &spectral_jitter_auto_seed_, "spectral_jitter_auto_seed", "Auto Seed");
        Tooltip(f, "Derive spectral jitter seed from the current frame number.\n"
                   "Gives a different noise pattern each frame for temporal accumulation.");
        Float_knob(f, &spectral_jitter_scale_, "spectral_jitter_scale", "Jitter Scale");
        SetRange(f, 0.0, 3.0);
        Tooltip(f, "Multiplier on spectral jitter range.\n"
                   "1.0 = jitter covers one spectral bin width (default).\n"
                   "0.5 = subtle, half-bin jitter.  2.0 = aggressive cross-bin blending.");

        Divider(f, "Highlight");
        Bool_knob(f, &highlight_compress_, "highlight_compress", "Highlight Compression");
        Tooltip(f, "Apply soft-clip to ghost highlights.\n"
                   "Prevents hard clipped edges and super-white values.");
        static const char* const kMetricNames[] = { "Value", "Luminance", "Lightness", nullptr };
        Enumeration_knob(f, &highlight_metric_, kMetricNames, "highlight_metric", "Metric");
        Tooltip(f, "Which measure to compress:\n"
                   "Value = max(R,G,B) — clips the brightest channel.\n"
                   "Luminance = Rec.709 weighted sum — perceptually balanced.\n"
                   "Lightness = cube root of luminance — more aggressive on bright areas.");
        Float_knob(f, &highlight_clip_, "highlight_clip", "Clip");
        SetRange(f, 0.1, 10.0);
        Tooltip(f, "Maximum output value (asymptotic ceiling).  Default 2.0.");
        Float_knob(f, &highlight_knee_, "highlight_knee", "Knee");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Transition sharpness.\n"
                   "0 = very soft, gradual rolloff.\n"
                   "1 = hard clip at the Clip value.\n"
                   "0.5 = balanced (default).\n"
                   "Matches AFXToneMap convention.");

        Divider(f, "Post-process");
        Float_knob(f, &ghost_blur_,        "ghost_blur",        "Ghost Blur");
        Tooltip(f, "Post-splat blur radius as a fraction of the image diagonal. 0 = off.");
        Int_knob(f,   &ghost_blur_passes_, "ghost_blur_passes", "Ghost Blur Passes");
        Tooltip(f, "Number of box-blur passes. 3 approximates a Gaussian.");

        // ---- Per-surface toggle + gain tab ----
        Tab_knob(f, "Surfaces");
        Button(f, "surf_refresh",      "Refresh Surfaces");
        Button(f, "surf_select_all",   "Select All");
        Button(f, "surf_deselect_all", "Deselect All");
        Button(f, "surf_reset_gains",  "Reset Gains");
        for (int k = 0; k < MAX_SURFS_UI; ++k)
        {
            Bool_knob(f, &surf_enabled_[k], s_surf_knob_names.toggle_names[k], surf_labels_[k]);
            SetFlags(f, Knob::STARTLINE | Knob::HIDDEN);
            Tooltip(f, "Enable or disable ghost reflections involving this surface.");

            Float_knob(f, &surf_gain_[k], s_surf_knob_names.gain_names[k], "gain");
            SetRange(f, 0.0, 10.0);
            SetFlags(f, Knob::HIDDEN | Knob::LOG_SLIDER | Knob::STARTLINE);
            Tooltip(f, "Gain multiplier for ghost pairs involving this surface.\n"
                       "1.0 = default. 0 = effectively mutes this surface.\n"
                       "The gains of both surfaces in a pair are multiplied together.");

            Color_knob(f, surf_color_[k], s_surf_knob_names.color_names[k], "color");
            SetFlags(f, Knob::HIDDEN | Knob::STARTLINE);
            Tooltip(f, "Color tint for ghosts involving this surface.\n"
                       "White = no tint. Both surfaces' colors are multiplied together.");

            Float_knob(f, &surf_offx_[k], s_surf_knob_names.offx_names[k], "offset x");
            SetRange(f, -500.0, 500.0);
            SetFlags(f, Knob::HIDDEN | Knob::STARTLINE);
            Tooltip(f, "Horizontal pixel offset for ghosts involving this surface.\n"
                       "Both surfaces' offsets are summed together.");

            Float_knob(f, &surf_offy_[k], s_surf_knob_names.offy_names[k], "offset y");
            SetRange(f, -500.0, 500.0);
            SetFlags(f, Knob::HIDDEN);
            ClearFlags(f, Knob::STARTLINE);
            Tooltip(f, "Vertical pixel offset for ghosts involving this surface.\n"
                       "Both surfaces' offsets are summed together.");

            Float_knob(f, &surf_scale_[k], s_surf_knob_names.scale_names[k], "scale");
            SetRange(f, 0.0, 5.0);
            SetFlags(f, Knob::HIDDEN | Knob::STARTLINE);
            Tooltip(f, "Scale factor for ghost positions involving this surface.\n"
                       "1.0 = default. <1 = pull toward center, >1 = push away.\n"
                       "Both surfaces' scales are multiplied together.");
        }

        // ---- About tab ----
        Tab_knob(f, "About");
        Text_knob(f, "FlareSim\n"
                     "\n"
                     "Physically-based lens flare simulation for Nuke.\n"
                     "\n"
                     "Based on the original work by Eamonn Nugent (space55/blackhole-rt)\n"
                     "\n"
                     "Copyright \xC2\xA9 2026 Steve Watts Kennedy (LocalStarlight/flaresim_nuke)\n"
                     "Copyright \xC2\xA9 2026 Peter Mercell — GPU optimisation\n");
    }

    // ---- knob_changed ----
    int knob_changed(Knob* k) override
    {
        if (k->is("surf_refresh")) {
            rebuild_surf_ui();
            return 1;
        }
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
                Knob* kn = knob(s_surf_knob_names.gain_names[i]);
                if (kn) kn->set_value(1.0);
                surf_color_[i][0] = surf_color_[i][1] = surf_color_[i][2] = 1.0f;
                Knob* kc = knob(s_surf_knob_names.color_names[i]);
                if (kc) kc->set_value(1.0);
                surf_offx_[i] = 0.0f;
                surf_offy_[i] = 0.0f;
                Knob* kox = knob(s_surf_knob_names.offx_names[i]);
                if (kox) kox->set_value(0.0);
                Knob* koy = knob(s_surf_knob_names.offy_names[i]);
                if (koy) koy->set_value(0.0);
                surf_scale_[i] = 1.0f;
                Knob* ks = knob(s_surf_knob_names.scale_names[i]);
                if (ks) ks->set_value(1.0);
            }
            return 1;
        }
        if (k->is("fov_use_sensor") || k->is("fov_auto_v")) {
            const bool direct = !fov_use_sensor_;
            knob("fov_h")->enable(direct);
            knob("fov_auto_v")->enable(direct);
            knob("fov_v")->enable(direct && !fov_auto_v_);
            knob("sensor_preset")->enable(fov_use_sensor_);
            const bool custom = (sensor_preset_ == 0);
            knob("sensor_w")->enable(fov_use_sensor_ && custom);
            knob("sensor_h")->enable(fov_use_sensor_ && custom);
            knob("focal_length")->enable(fov_use_sensor_);
            return 1;
        }
        if (k->is("sensor_preset")) {
            if (sensor_preset_ > 0) {
                const SensorPreset& p = kSensorPresets[sensor_preset_];
                sensor_w_mm_ = p.w;
                sensor_h_mm_ = p.h;
                knob("sensor_w")->set_value(sensor_w_mm_);
                knob("sensor_h")->set_value(sensor_h_mm_);
            }
            const bool custom = (sensor_preset_ == 0);
            knob("sensor_w")->enable(custom);
            knob("sensor_h")->enable(custom);
            return 1;
        }
        if (k->is("lens_file")) {
            // Reload lens interactively and rebuild pair UI immediately.
            const std::string path(lens_file_ ? lens_file_ : "");
            if (!path.empty() && path != last_lens_file_) {
                if (lens_.load(path.c_str())) {
                    last_lens_file_ = path;
                    rebuild_surf_ui();  // also populates all_lens_pairs_
                } else {
                    fprintf(stderr, "FlareSim: failed to load lens: %s\n", path.c_str());
                }
            }
            return 1;
        }
        return Iop::knob_changed(k);
    }

    // ---- append (hash) ----
    // Always include the frame number so Nuke's tile cache never serves
    // stale results from a different frame.  The flare depends on the
    // input image content which changes every frame.
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
        if (input(1)) input(1)->validate(for_real);

        const int x0 = info_.x();
        const int y0 = info_.y();
        const int x1 = info_.r();
        const int y1 = info_.t();
        const int w  = x1 - x0;
        const int h  = y1 - y0;
        if (w <= 0 || h <= 0) return;

        // Load lens file (only when the path changes)
        const std::string cur_lens(lens_file_ ? lens_file_ : "");
        if (cur_lens != last_lens_file_)
        {
            if (!cur_lens.empty())
            {
                if (!lens_.load(cur_lens.c_str()))
                    fprintf(stderr, "FlareSim: failed to load lens file: %s\n",
                            cur_lens.c_str());
            }
            last_lens_file_ = cur_lens;
            rebuild_surf_ui();  // also populates all_lens_pairs_
        }

        {
            std::lock_guard<std::mutex> lock(compute_mutex_);
            if (!pending_gpu_error_.empty()) {
                std::string msg;
                msg.swap(pending_gpu_error_);
                error("%s", msg.c_str());
                return;
            }
            pending_x0_       = x0;
            pending_y0_       = y0;
            pending_x1_       = x1;
            pending_y1_       = y1;
            pending_w_        = w;
            pending_h_        = h;
            const Format& fmt = format();
            pending_fmt_x0_   = fmt.x();
            pending_fmt_y0_   = fmt.y();
            pending_fmt_w_    = fmt.width();
            pending_fmt_h_    = fmt.height();
            pending_num_surfs_  = lens_.num_surfaces();  // thread-safe snapshot
            pending_frame_     = (int)outputContext().frame();
            needs_compute_ = true;
        }
    }

    // ---- do_compute ----
    void do_compute()
    {
        PROF_DECL;

        const int w = pending_w_;
        const int h = pending_h_;
        const size_t npx = (size_t)w * h;

        // Ensure Metal is initialised
        if (!gpu_cache_.mtl_device) {
            std::string init_err;
            if (!metal_init(gpu_cache_, &init_err)) {
                pending_gpu_error_ = init_err;
                return;
            }
        }

        PROF_BEGIN("alloc_pinned");
        void* pinned = ensure_fp16_output(gpu_cache_, npx);
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

        const int x0 = pending_x0_;
        const int y0 = pending_y0_;
        const int x1 = pending_x1_;
        const int y1 = pending_y1_;

        float fov_h, fov_v;
        if (fov_use_sensor_) {
            const float fl = std::max(focal_length_mm_, 0.1f);
            fov_h = 2.0f * std::atan(sensor_w_mm_ / (2.0f * fl));
            fov_v = 2.0f * std::atan(sensor_h_mm_ / (2.0f * fl));
        } else {
            fov_h = (float)(fov_h_deg_ * M_PI / 180.0);
            const float aspect = (pending_fmt_h_ > 0)
                ? ((float)pending_fmt_w_ / (float)pending_fmt_h_)
                : (16.0f / 9.0f);
            fov_v = fov_auto_v_
                ? 2.0f * std::atan(std::tan(fov_h * 0.5f) / aspect)
                : (float)(fov_v_deg_ * M_PI / 180.0);
        }

        const float tan_half_h = std::tan(fov_h * 0.5f);
        const float tan_half_v = std::tan(fov_v * 0.5f);

        const float fmt_cx = pending_fmt_x0_ + pending_fmt_w_ * 0.5f;
        const float fmt_cy = pending_fmt_y0_ + pending_fmt_h_ * 0.5f;

        std::vector<BrightPixel> sources;

        // ---- Sample source at user-defined XY ----
        const int sr   = std::max(manual_sample_radius_, 1);
        const int mx   = (int)std::round(manual_xy_[0]);
        const int my   = (int)std::round(manual_xy_[1]);
        const int sy0  = std::max(y0, my - sr);
        const int sy1  = std::min(y1, my + sr + 1);
        const int sx0  = std::max(x0, mx - sr);
        const int sx1  = std::min(x1, mx + sr + 1);

        // ---- Determine how far the source is inside the format rect ----
        // Positive = inside, negative = outside.
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

        // ---- Sample image colour (when source is on screen) ----
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
                    sum_r += rp[ix];
                    sum_g += gp[ix];
                    sum_b += bp[ix];
                    ++cnt;
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

        // ---- Build the source, blending with outside colour if needed ----
        {
            // Outside source values (always computed so we can blend)
            const float out_si = outside_source_intensity_ * 1000.0f;
            const float out_r = outside_source_color_[0] * out_si;
            const float out_g = outside_source_color_[1] * out_si;
            const float out_b = outside_source_color_[2] * out_si;

            float final_r, final_g, final_b;
            bool  emit_source = false;

            if (have_image_sample && !outside_source_enable_) {
                // Normal behaviour, no outside source feature
                const float si = source_intensity_ * 1000.0f;
                final_r = img_r * si;
                final_g = img_g * si;
                final_b = img_b * si;
                emit_source = true;
            }
            else if (have_image_sample && outside_source_enable_) {
                // Source is on-screen. Check if we're in the falloff zone.
                const float falloff = std::max(outside_source_falloff_, 0.0f);
                if (falloff > 0.0f && dist_inside < falloff) {
                    // Inside the blend zone: lerp between image and outside
                    const float t = std::clamp(dist_inside / falloff, 0.0f, 1.0f);
                    const float si = source_intensity_ * 1000.0f;
                    final_r = t * (img_r * si) + (1.0f - t) * out_r;
                    final_g = t * (img_g * si) + (1.0f - t) * out_g;
                    final_b = t * (img_b * si) + (1.0f - t) * out_b;
                } else {
                    // Fully on-screen, outside falloff zone — normal image sample
                    const float si = source_intensity_ * 1000.0f;
                    final_r = img_r * si;
                    final_g = img_g * si;
                    final_b = img_b * si;
                }
                emit_source = true;
            }
            else if (!source_on_screen && outside_source_enable_) {
                // Source is off-screen — use outside colour/intensity
                final_r = out_r;
                final_g = out_g;
                final_b = out_b;
                emit_source = true;
            }
            // else: source off-screen and outside_source disabled → no flare

            if (emit_source) {
                const float ndc_x = ((float)mx - fmt_cx) / pending_fmt_w_;
                const float ndc_y = ((float)my - fmt_cy) / pending_fmt_h_;
                BrightPixel bp_out;
                bp_out.angle_x = std::atan(ndc_x * 2.0f * tan_half_h);
                bp_out.angle_y = std::atan(ndc_y * 2.0f * tan_half_v);
                bp_out.r = final_r;
                bp_out.g = final_g;
                bp_out.b = final_b;
                sources.push_back(bp_out);
            }
        }
        last_src_count_.store((int)sources.size());

        if (Op::aborted()) { zero_buffers(); return; }

        if (sources.empty()) { zero_buffers(); return; }

        const float sensor_half_w = lens_.focal_length * std::tan(fov_h * 0.5f);
        const float sensor_half_h = lens_.focal_length * std::tan(fov_v * 0.5f);

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

        // Reuse cached filter results when lens and FOV haven't changed.
        std::vector<GhostPair> active_pairs;
        std::vector<float>     area_boosts;

        PROF_BEGIN("filter_pairs");
        const bool filter_cached =
            !cached_filter_pairs_.empty() &&
            cached_filter_lens_ == lens_.name &&
            cached_filter_shw_  == sensor_half_w &&
            cached_filter_shh_  == sensor_half_h;

        if (filter_cached)
        {
            active_pairs = cached_filter_pairs_;
            area_boosts  = cached_filter_boosts_;
        }
        else
        {
            filter_ghost_pairs(lens_, sensor_half_w, sensor_half_h, cfg,
                               active_pairs, area_boosts);
            cached_filter_pairs_  = active_pairs;
            cached_filter_boosts_ = area_boosts;
            cached_filter_shw_    = sensor_half_w;
            cached_filter_shh_    = sensor_half_h;
            cached_filter_lens_   = lens_.name;
        }
        PROF_END();

        // Apply per-surface user toggles and gains: keep only pairs whose
        // both surfaces are enabled, and multiply pair boost by surface gains.
        {
            std::vector<GhostPair> enabled_pairs;
            std::vector<float>     enabled_boosts;
            std::vector<float>     enabled_colors;  // 3 per pair (RGB)
            std::vector<float>     enabled_offsets;  // 2 per pair (XY px)
            std::vector<float>     enabled_scales;   // 1 per pair
            enabled_pairs.reserve(active_pairs.size());
            enabled_boosts.reserve(active_pairs.size());
            enabled_colors.reserve(active_pairs.size() * 3);
            enabled_offsets.reserve(active_pairs.size() * 2);
            enabled_scales.reserve(active_pairs.size());

            for (int i = 0; i < (int)active_pairs.size(); ++i)
            {
                const int sa = active_pairs[i].surf_a;
                const int sb = active_pairs[i].surf_b;

                // Skip if either surface is disabled
                const bool a_on = (sa < 0 || sa >= MAX_SURFS_UI) || surf_enabled_[sa];
                const bool b_on = (sb < 0 || sb >= MAX_SURFS_UI) || surf_enabled_[sb];
                if (!a_on || !b_on)
                    continue;

                // Multiply area boost by the gain of both surfaces
                float gain_a = (sa >= 0 && sa < MAX_SURFS_UI) ? surf_gain_[sa] : 1.0f;
                float gain_b = (sb >= 0 && sb < MAX_SURFS_UI) ? surf_gain_[sb] : 1.0f;

                enabled_pairs.push_back(active_pairs[i]);
                enabled_boosts.push_back(area_boosts[i] * gain_a * gain_b);

                // Per-pair color: multiply both surfaces' tints
                float cr = 1.0f, cg = 1.0f, cb = 1.0f;
                if (sa >= 0 && sa < MAX_SURFS_UI) {
                    cr *= surf_color_[sa][0]; cg *= surf_color_[sa][1]; cb *= surf_color_[sa][2];
                }
                if (sb >= 0 && sb < MAX_SURFS_UI) {
                    cr *= surf_color_[sb][0]; cg *= surf_color_[sb][1]; cb *= surf_color_[sb][2];
                }
                enabled_colors.push_back(cr);
                enabled_colors.push_back(cg);
                enabled_colors.push_back(cb);

                // Per-pair offset: sum both surfaces' offsets
                float ox = 0.0f, oy = 0.0f;
                if (sa >= 0 && sa < MAX_SURFS_UI) { ox += surf_offx_[sa]; oy += surf_offy_[sa]; }
                if (sb >= 0 && sb < MAX_SURFS_UI) { ox += surf_offx_[sb]; oy += surf_offy_[sb]; }
                enabled_offsets.push_back(ox);
                enabled_offsets.push_back(oy);

                // Per-pair scale: multiply both surfaces' scales
                float sc = 1.0f;
                if (sa >= 0 && sa < MAX_SURFS_UI) sc *= surf_scale_[sa];
                if (sb >= 0 && sb < MAX_SURFS_UI) sc *= surf_scale_[sb];
                enabled_scales.push_back(sc);
            }
            active_pairs = std::move(enabled_pairs);
            area_boosts  = std::move(enabled_boosts);
            pair_colors_  = std::move(enabled_colors);
            pair_offsets_ = std::move(enabled_offsets);
            pair_scales_  = std::move(enabled_scales);
        }

        last_pair_count_.store((int)active_pairs.size());

        if (active_pairs.empty()) { zero_buffers(); return; }

        if (Op::aborted()) { zero_buffers(); return; }

        pending_gpu_error_.clear();
        const int fmt_x0_in_buf = pending_fmt_x0_ - x0;
        const int fmt_y0_in_buf = pending_fmt_y0_ - y0;

        PROF_BEGIN("gpu_ghost_render");
        launch_ghost_metal(lens_, active_pairs, area_boosts, sources,
                          sensor_half_w, sensor_half_h,
                          nullptr, nullptr, nullptr,
                          w, h,
                          pending_fmt_w_, pending_fmt_h_,
                          fmt_x0_in_buf, fmt_y0_in_buf,
                          cfg, gpu_cache_,
                          /*skip_readback=*/true,
                          &pending_gpu_error_,
                          &pair_colors_, &pair_offsets_, &pair_scales_);
        PROF_END();

        PROF_BEGIN("gpu_blur_alpha_readback");
        {
            int radius = 0;
            if (ghost_blur_ > 0.0f && ghost_blur_passes_ > 0)
            {
                float diag = std::sqrt((float)w * w + (float)h * h);
                radius = std::max(1, (int)std::round(ghost_blur_ * diag));
            }
            launch_blur_alpha_readback_metal(
                reinterpret_cast<float*>(ghost_r_),
                reinterpret_cast<float*>(ghost_g_),
                reinterpret_cast<float*>(ghost_b_),
                reinterpret_cast<float*>(alpha_),
                w, h, radius, ghost_blur_passes_,
                gpu_cache_, &pending_gpu_error_,
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
            std::lock_guard<std::mutex> lock(compute_mutex_);
            // Only run do_compute if the pending state matches this frame.
            // If _validate() for a later frame has already overwritten the
            // pending state, skip — Nuke will re-request this frame later
            // with a correct _validate → engine sequence.
            if (needs_compute_ && pending_frame_ == cur_frame) {
                needs_compute_ = false;
                do_compute();
                cache_frame_ = cur_frame;
            }
        }

        // If the cache doesn't belong to this frame, output black.
        const bool frame_ok = (cache_frame_ == cur_frame);

        // Pass non-managed channels through from input.
        {
            ChannelSet pass_through(channels);
            pass_through -= Chan_Red;
            pass_through -= Chan_Green;
            pass_through -= Chan_Blue;
            pass_through -= Chan_Alpha;
            if (pass_through)
                input0().get(y, x, r, pass_through, row);
        }

        const int  cache_y     = y - info_.y();
        const bool valid_cache = frame_ok && ghost_r_ != nullptr && cache_y >= 0 && cache_y < cache_height_;
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
        // Always request the full frame from input 0 (RGB) for source detection.
        input0().request(info_.x(), info_.y(), info_.r(), info_.t(),
                         Mask_RGB, count);

        // If a mask is connected, request its alpha over the full frame too.
        if (input(1))
            input(1)->request(info_.x(), info_.y(), info_.r(), info_.t(),
                              Mask_Alpha, count);

        // Pass non-RGBA channels through for the requested region only.
        ChannelSet passthru(channels);
        passthru -= Mask_RGBA;
        if (passthru)
            input0().request(x, y, r, t, passthru, count);
    }
};

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

static Iop* build(Node* node) { return new FlareSim(node); }

const Iop::Description FlareSim::d(
    "FlareSim",
    "Filter/FlareSim",
    build);
