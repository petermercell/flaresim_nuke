# FlareSim for Nuke

A physically-based lens flare simulator for Foundry Nuke — GPU-optimised fork of [LocalStarlight/flaresim_nuke](https://github.com/LocalStarlight/flaresim_nuke).

I really appreciate Steven (LocalStarlight) for open-sourcing FlareSim. This is my version from the eyes of a compositor — built to be fast and flexible enough to move everything a client might ask for.

The original FlareSim is a Windows/Nuke 16 plugin built on CUDA 13. This fork adds **Linux** and **macOS** support, works with **Nuke 14–17**, and brings major GPU performance improvements — async CUDA streams, FP16 output, prefix-sum blur kernels, and a per-surface art-direction UI. The core optics (ray tracing, Fresnel, lens files) are unchanged; all the work is in the GPU pipeline, the Nuke integration, and the build system.

---

## Performance: 205× faster

Same lens (Angenieux 180mm, 15 surfaces, 66 active pairs), same frame, same machine (RTX A5000). Buffer: 10172×5370, ghost blur radius 35 px, 2 passes.

| Stage | Original | Fork | Speedup |
|---|---:|---:|---|
| Source detection | 3,238 ms | 8 ms | **425×** |
| Ghost filter | 8 ms | 8 ms | — |
| CUDA ghost kernel | 6,398 ms | 31 ms | **206×** |
| Ghost blur + readback | 9,359 ms | 56 ms | **167×** |
| **TOTAL** | **21,091 ms** | **103 ms** | **205×** |

21 seconds → 0.1 seconds. Interactive.

---

## Platforms

| Platform | Status |
|---|---|
| **Linux** | Fully supported — pre-built binaries and build-from-source |
| **macOS** | Supported (Metal GPU) |
| **Windows** | Supported |

---

## Linux — Pre-built Binaries

Pre-built binaries are available. If a binary does not load or crashes, **double-check your CUDA version** — the plugin must match the CUDA runtime on your system.

The binaries were compiled and tested with **CUDA 12.1** and confirmed working on **CUDA 12.8** as well.

### Building from Source (Linux)

```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

rm -rf build && mkdir build && cd build

cmake .. -DNUKE_VERSION=17.0v1 \
         -DNDK_ROOT=/opt/Nuke17.0v1/include \
         -DNUKE_LIB_DIR=/opt/Nuke17.0v1 \
         -DCMAKE_CUDA_ARCHITECTURES="86;89;90"

make -j$(nproc)
```

Adjust `NUKE_VERSION`, `NDK_ROOT`, `NUKE_LIB_DIR`, and `CMAKE_CUDA_ARCHITECTURES` to match your environment. The build system supports Nuke 14–17, with automatic old-ABI detection for Nuke versions below 15.

---

## What Changed

### GPU (new & rewritten)

- **Fast math** — all device-side `sqrtf`/`cosf`/divisions → CUDA intrinsics, constants made `constexpr`
- **FP16 pipeline** — on-device FP32→FP16 + pinned host staging → ~50% less DMA bandwidth
- **Async streams** — dual CUDA streams, pipelined blur→alpha→readback
- **GPU blur** — new `blur_cuda.cu`: prefix-sum horizontal + sliding-window vertical, O(1)/pixel
- **GPU alpha** — luminance→alpha entirely on device
- **Cached pupil grid** — rebuilt only when config changes
- **Per-pair colours/offsets/scales** — new kernel params for art direction

### Nuke UI

- **Per-surface controls** replace per-pair toggles — gain, colour, offset, scale per lens element
- **Removed** — haze, AOV channels, source clustering, preview mode
- **Starburst dropped** — the built-in FFT starburst didn't look convincing enough and wasn't worth the overhead. A dedicated starburst node with proper controls would be the better approach (see Future Ideas).
- **Profiler** — per-stage timing to console

### Build system

- **Multi-Nuke** — Nuke 14–17, auto old-ABI for <15
- **Min sm_70** — up from sm_50, build-time validation
- **Cross-platform** — Linux static cudart, Windows MSVC matching, conditional libNdk

### Untouched (byte-identical)

`fresnel.h` · `ghost.cpp` · `ghost.h` · `lens.cpp` · `lens.h` · `trace.cpp` · `trace.h` · `vec3.h`

---

## Future Ideas

1. **Off-screen sources** — allow light sources outside the frame to still generate ghosts (e.g. sun just out of shot).
2. **Manual source placement (2–3 sources)** — skip the full-image scan entirely, let the artist place sources by hand for fast, art-directable results.
3. **Starburst as a separate node** — a standalone diffraction spike generator with more controls, decoupled from the ghost renderer.

---

*FlareSim © LocalStarlight. GPU optimisations © Peter Mercell.*
