#pragma once
#include <cstdint>
#include <vector>

#include "frame_types.hpp"

// CPU reference implementations. They use EXACTLY the same math, operation
// order and rounding as the OpenCL kernels so GPU output can be validated
// pixel by pixel. Outer loops are OpenMP-parallel (thread count set with
// cpu_set_threads) so the CPU baseline can be measured single- and multi-core.

void cpu_set_threads(int n);  // 0 = all hardware threads
int cpu_get_threads();

void cpu_yuv420p_to_rgb(const PlanarYUV420& in, RGBImage& out);
void cpu_rgb_to_yuv420p(const RGBImage& in, PlanarYUV420& out);
void cpu_blur_naive(const RGBImage& in, RGBImage& out);
void cpu_blur_separable(const RGBImage& in, RGBImage& out, std::vector<uint16_t>& tmp);
void cpu_sobel_luma(const PlanarYUV420& in, PlanarYUV420& out);

// Scratch buffers reused across frames (no per-frame allocation).
struct CpuScratch {
    RGBImage a, b;
    std::vector<uint16_t> tmp;
};

// Full per-frame processing chain for a given filter (same as the GPU path).
void cpu_process(const PlanarYUV420& in, PlanarYUV420& out, Filter f, CpuScratch& s);
