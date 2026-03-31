// ============================================================================
// blur_cuda.h — CUDA separable box blur
//
// v3: prefix-sum horizontal, sliding-window vertical, stream-aware wrappers.
// ============================================================================
#pragma once

#include "ghost_cuda.h"   // GpuBufferCache

#include <string>

// Legacy CPU-buffer blur (kept for backward compat — not used on hot path).
void launch_box_blur_cuda(
    float* cpu_r, float* cpu_g, float* cpu_b,
    int w, int h, int radius, int passes,
    GpuBufferCache& cache, std::string* out_error = nullptr
);

// GPU-resident blur (old naive kernels — kept as fallback).
void launch_box_blur_gpu_resident(
    int w, int h, int radius, int passes,
    GpuBufferCache& cache, std::string* out_error = nullptr
);

// ---------------------------------------------------------------------------
// Stream-aware blur kernel wrappers (callable from other .cu files).
// Needed because CUDA_SEPARABLE_COMPILATION is off.
// `stream` is a cudaStream_t passed as void* to keep cuda_runtime.h
// out of the header.
// ---------------------------------------------------------------------------

// Horizontal prefix-sum box blur: O(1) per pixel.
// Falls back to naive kernel for rows wider than shared memory limit.
void launch_blur_h_on_stream(const float* in, float* out,
                             int w, int h, int radius, void* stream);

// Vertical sliding-window box blur: O(1) per pixel.
void launch_blur_v_on_stream(const float* in, float* out,
                             int w, int h, int radius, void* stream);
