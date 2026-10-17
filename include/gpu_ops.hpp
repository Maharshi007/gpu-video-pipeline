#pragma once
#include <string>

#include "cl_context.hpp"
#include "frame_types.hpp"

// Per-frame GPU timings taken from OpenCL profiling events (device time).
struct GpuTimes {
    double h2d_ms = 0;     // host -> device copy of the YUV frame
    double kernel_ms = 0;  // sum of all kernels for the frame
    double d2h_ms = 0;     // device -> host copy of the result
};

// Runs the per-frame processing chain on the GPU.
// Device buffers are allocated ONCE for the frame size and reused for every
// frame: clCreateBuffer is expensive and would otherwise dominate.
class GpuPipeline {
public:
    // pinned = use page-locked staging memory (CL_MEM_ALLOC_HOST_PTR) for
    // transfers, which lets the driver DMA directly instead of first copying
    // into its own internal pinned buffer.
    GpuPipeline(const ClContext& cl, const std::string& kernel_dir, int w, int h, bool pinned);
    ~GpuPipeline();
    GpuPipeline(const GpuPipeline&) = delete;
    GpuPipeline& operator=(const GpuPipeline&) = delete;

    GpuTimes process(const PlanarYUV420& in, PlanarYUV420& out, Filter f);

private:
    cl_event launch_2d(cl_kernel k, size_t gx, size_t gy);

    const ClContext& cl_;
    int w_, h_;
    bool pinned_;
    size_t yuv_bytes_, rgb_bytes_;

    ClProgramPtr prog_yuv2rgb_, prog_rgb2yuv_, prog_blur_, prog_sobel_;
    ClKernelPtr k_yuv2rgb_, k_rgb2yuv_, k_naive_, k_sep_h_, k_sep_v_, k_tiled_, k_sobel_;
    ClMemPtr d_yuv_in_, d_yuv_out_, d_rgb_a_, d_rgb_b_, d_tmp16_;

    // Pinned host staging buffers (only when pinned_ == true).
    ClMemPtr h_in_, h_out_;
    void* h_in_ptr_ = nullptr;
    void* h_out_ptr_ = nullptr;
};
