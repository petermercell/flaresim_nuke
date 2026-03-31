# FlareSim for Nuke

A physically-based lens flare simulator for Foundry Nuke — GPU-optimised fork of [LocalStarlight/flaresim_nuke](https://github.com/LocalStarlight/flaresim_nuke).

![FlareSim](FlareSim.png)

I really appreciate Steven (LocalStarlight) for open-sourcing FlareSim. This is my version from the eyes of a compositor — built to be fast and flexible enough to handle everything a client might ask for.

The original FlareSim is a Windows/Nuke 16 plugin built on CUDA 13. This fork adds **Linux** and **macOS** support, works with **Nuke 14–17**, and brings major GPU performance improvements — async CUDA streams, FP16 output, prefix-sum blur kernels, and a per-surface art-direction UI. The core optics (ray tracing, Fresnel, lens files) are unchanged; all the work is in the GPU pipeline, the Nuke integration, and the build system.

---

## What's New

### FlareSim3D — Camera + Axis driven flares

A new node that takes a **Camera** and an **Axis** (light position) as inputs instead of a manual Source XY. The Axis world position is projected through the Camera to derive screen position and source distance automatically — no manual XY tracking needed. Connect a Camera and an Axis, and the flare tracks the 3D source through the shot.

- **Intensity Falloff** — inverse-square-law scaling based on source distance
- **Reference Distance** — distance (scene units) at which the flare has its nominal intensity
- Source behind camera → no flare (physically correct)
- Source off-screen → transitions seamlessly to Outside Source

Registered as `Filter/FlareSim3D`. Builds as a separate `.so` / `.dylib`.

### Off-Screen Source

When the light source moves outside the frame, the flare no longer vanishes. A user-defined colour and intensity takes over, so the flare renders seamlessly as the source enters or leaves the plate. Works in both FlareSim (manual XY) and FlareSim3D (Camera + Axis).

- **Enable Outside Source** — on/off toggle
- **Outside Color** — RGB colour of the off-screen light
- **Outside Intensity** — brightness of the off-screen source
- **Edge Falloff (px)** — blend zone at the frame edge for smooth transitions

### Spectral Jitter

Randomises each ray's wavelength within its spectral bin, smoothing the hard colour boundaries between discrete samples at zero extra ray-trace cost — same number of traces, each one uses a slightly different wavelength.

- **Spectral Jitter** — on/off (on by default)
- **Spectral Jitter Seed** — fixed seed for VFX reproducibility
- **Auto Seed** — derives seed from frame number for temporal variation
- **Jitter Scale** — multiplier on the randomisation range (0.5 = subtle, 1.0 = default, 2.0 = aggressive)

### Extended Spectral Samples

The spectral samples dropdown now goes up to 31: **3 (R/G/B)** · 5 · 7 · 9 · 11 · 15 · 21 · 31. With spectral jitter on, 11 samples already looks excellent. 31 gives ~10 nm bin spacing — essentially continuous spectral coverage.

### Highlight Compression

Luminance-preserving soft-clip applied on the GPU after blur. Prevents hard-clipped highlights and super-white values while maintaining colour hue through the rolloff.

- **Highlight Compression** — on/off toggle
- **Metric** — Value (max RGB), Luminance (Rec.709), or Lightness (cube root)
- **Clip** — maximum output value (asymptotic ceiling, default 2.0)
- **Knee** — transition sharpness (0 = very soft, 1 = hard clip, default 0.5)

Matches AFXToneMap convention — same Clip and Knee values produce almost the same rolloff behaviour.

### Per-Surface Art Direction

Each lens surface now has individual controls in the Surfaces tab: **gain**, **color** (RGB tint), **offset** (pixel shift x/y), and **scale** (pull toward / push away from centre). Both surfaces in a ghost pair combine together.

### Profiler

Built-in per-frame timing table. Disabled by default — flip `#define FLARESIM_PROFILE 1` in the source to enable.

---

## Performance: 205× faster

Same lens (Angenieux 180mm, 15 surfaces, 66 active pairs), same frame, same machine (RTX A5000). Buffer: 10172×5370, ghost blur radius 35 px, 2 passes.

| Stage | Original | Fork | Speedup |
|-------|----------|------|---------|
| Source detection | 3,238 ms | 8 ms | **425×** |
| Ghost filter | 8 ms | 8 ms | — |
| CUDA ghost kernel | 6,398 ms | 31 ms | **206×** |
| Ghost blur + readback | 9,359 ms | 56 ms | **167×** |
| **TOTAL** | **21,091 ms** | **103 ms** | **205×** |

21 seconds → 0.1 seconds. Interactive.

---

## Platforms

| Platform | GPU Backend | Status |
|---|---|---|
| **Linux** | CUDA | Fully supported — pre-built binaries and build-from-source |
| **macOS** | Metal | Supported — build-from-source |
| **Windows** | CUDA | Supported — build-from-source (Ninja + MSVC 2022) |

---

## Building from Source

### Linux

```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

rm -rf build && mkdir build && cd build

cmake .. -DNUKE_VERSION=14.1v8 \
         -DCMAKE_CUDA_ARCHITECTURES="86;89;90"

make -j$(nproc)
```

Output: `build/FlareSim.so` + `build/FlareSim3D.so`

Adjust `NUKE_VERSION` and `CMAKE_CUDA_ARCHITECTURES` to match your environment. `NDK_ROOT` and `NUKE_LIB_DIR` default to `/usr/local/Nuke<VERSION>`. CUDA runtime is linked statically — no runtime dependency.

### macOS (Metal)

```bash
rm -rf build && mkdir build && cd build

cmake .. -DNUKE_VERSION=14.1v8

make -j$(sysctl -n hw.ncpu)
```

Output: `build/FlareSim.dylib` + `build/FlareSim3D.dylib`

Metal shaders are compiled at plugin load time. No CUDA required.

### Windows

Requires Visual Studio 2022 Developer Command Prompt and CUDA 12.4 (or any 12.x). Uses Ninja generator — the VS generator ignores `CMAKE_CUDA_COMPILER`.

```cmd
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%

cd C:\path\to\flaresim_nuke

rmdir /s /q build 2>nul & mkdir build && cd build && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" -DNUKE_VERSION=14.1v8 -DCMAKE_CUDA_ARCHITECTURES="86;89;90" && cmake --build .
```

Output: `build\FlareSim.dll` + `build\FlareSim3D.dll`

CUDA runtime is linked statically — no `cudart64_*.dll` needed at runtime. The plugin only requires an NVIDIA driver ≥ 525 (CUDA 12.4 minimum). Any newer driver (12.8, 12.9, 13.x) works — drivers are forward-compatible.

### CUDA Architecture Reference

| Architecture | GPUs | Min CUDA |
|---|---|---|
| sm_70 | V100, Titan V | 9.0 |
| sm_75 | RTX 2000, T4 | 10.0 |
| sm_86 | RTX 3000, A5000/A6000 | 11.1 |
| sm_89 | RTX 4000 | 11.8 |
| sm_90 | H100 | 12.0 |
| sm_100 | RTX 5000, B200 | 12.8 |

Default: `86;89;90` (covers Ampere through Hopper). Add `100` for Blackwell if your CUDA toolkit supports it.

---

## Installation

1. Copy `FlareSim.so` (or `.dylib` / `.dll`) and `FlareSim3D.so` to a directory on your `NUKE_PATH`, for example `~/.nuke/plugins/`.
2. Add to your `menu.py`:
   ```python
   nuke.menu('Nodes').addCommand('Filter/FlareSim', 'nuke.createNode("FlareSim")')
   nuke.menu('Nodes').addCommand('Filter/FlareSim3D', 'nuke.createNode("FlareSim3D")')
   ```
3. Copy the `lenses/` folder somewhere accessible and point the **Lens File** knob at a `.lens` file.
4. Restart Nuke.

---

## Quick Start

**FlareSim** (manual source):
1. Connect your plate to the input.
2. Point **Lens File** at a `.lens` prescription.
3. Set **FOV H** to match your camera.
4. Set **Source XY** to the light source position, or raise **Threshold** to auto-detect bright highlights.
5. Adjust **Flare Gain** to taste.

**FlareSim3D** (3D source):
1. Connect your plate to input 0, Camera to input 1, Axis (at the light position) to input 2.
2. Point **Lens File** at a `.lens` prescription.
3. The flare tracks the Axis through the Camera automatically.
4. Enable **Intensity Falloff** and set **Reference Distance** for distance-based dimming.
5. Enable **Outside Source** so the flare persists when the source leaves the frame.

---

## Future Ideas

1. **Source reflection in the lens** — feed the detected source back into the lens system as a secondary light source, producing the bright reflection spot visible on real lens elements when shooting toward a light. This would close the loop between the ghost renderer and the physical lens geometry.
2. **Starburst as a separate node** — a standalone diffraction spike generator with more controls, decoupled from the ghost renderer.

---

## Credits

FlareSim is built on the foundational work of **Steve Watts Kennedy** ([LocalStarlight](https://github.com/LocalStarlight/flaresim_nuke)), whose original CUDA-based lens flare renderer established the core ray-tracing approach, optical physics, and lens file format.

The original physics engine is based on the work of **Eamonn Nugent** ([@space55](https://github.com/space55) · [55.dev](https://55.dev/)), whose CPU-based renderer ([blackhole-rt](https://github.com/space55/blackhole-rt/)) provided the ray-tracing foundation.

Tutorial by Steve: [https://youtu.be/yEsBOQNG16Y](https://youtu.be/yEsBOQNG16Y)

---

## License

MIT — see [LICENSE](LICENSE).

Peter Mercell — [petermercell.com](https://petermercell.com)
