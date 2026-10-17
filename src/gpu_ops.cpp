#include "gpu_ops.hpp"

#include <cstring>
#include <vector>

#include "timer.hpp"

static constexpr size_t WG = 16;  // 16x16 = 256 work-items per group

static size_t round_up(size_t v, size_t m) { return (v + m - 1) / m * m; }

static ClKernelPtr make_kernel(cl_program p, const char* name) {
    cl_int err = CL_SUCCESS;
    ClKernelPtr k(clCreateKernel(p, name, &err));
    CL_CHECK(err);
    return k;
}

GpuPipeline::GpuPipeline(const ClContext& cl, const std::string& dir, int w, int h, bool pinned)
    : cl_(cl), w_(w), h_(h), pinned_(pinned) {
    yuv_bytes_ = size_t(w) * h * 3 / 2;
    rgb_bytes_ = size_t(w) * h * 3;

    prog_yuv2rgb_ = cl.build_program(dir + "/yuv420p_to_rgb.cl");
    prog_rgb2yuv_ = cl.build_program(dir + "/rgb_to_yuv420p.cl");
    prog_blur_ = cl.build_program(dir + "/gaussian_blur.cl");
    prog_sobel_ = cl.build_program(dir + "/sobel.cl", "-cl-fp32-correctly-rounded-divide-sqrt");
    k_yuv2rgb_ = make_kernel(prog_yuv2rgb_.get(), "yuv420p_to_rgb");
    k_rgb2yuv_ = make_kernel(prog_rgb2yuv_.get(), "rgb_to_yuv420p");
    k_naive_ = make_kernel(prog_blur_.get(), "gaussian_blur_naive");
    k_sep_h_ = make_kernel(prog_blur_.get(), "gaussian_blur_h");
    k_sep_v_ = make_kernel(prog_blur_.get(), "gaussian_blur_v");
    k_tiled_ = make_kernel(prog_blur_.get(), "gaussian_blur_tiled");
    k_sobel_ = make_kernel(prog_sobel_.get(), "sobel");

    // Allocate every device buffer once, sized for this resolution.
    cl_int err = CL_SUCCESS;
    auto dev_buf = [&](cl_mem_flags fl, size_t n) {
        ClMemPtr m(clCreateBuffer(cl.context(), fl, n, nullptr, &err));
        CL_CHECK(err);
        return m;
    };
    d_yuv_in_ = dev_buf(CL_MEM_READ_ONLY, yuv_bytes_);
    d_yuv_out_ = dev_buf(CL_MEM_WRITE_ONLY, yuv_bytes_);
    d_rgb_a_ = dev_buf(CL_MEM_READ_WRITE, rgb_bytes_);
    d_rgb_b_ = dev_buf(CL_MEM_READ_WRITE, rgb_bytes_);
    d_tmp16_ = dev_buf(CL_MEM_READ_WRITE, rgb_bytes_ * sizeof(cl_ushort));

    if (pinned_) {
        // CL_MEM_ALLOC_HOST_PTR asks the driver for page-locked host memory.
        // Mapping it once gives a CPU pointer we keep for the whole run.
        h_in_ = dev_buf(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, yuv_bytes_);
        h_out_ = dev_buf(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, yuv_bytes_);
        h_in_ptr_ = clEnqueueMapBuffer(cl.queue(), h_in_.get(), CL_TRUE, CL_MAP_WRITE, 0,
                                       yuv_bytes_, 0, nullptr, nullptr, &err);
        CL_CHECK(err);
        h_out_ptr_ = clEnqueueMapBuffer(cl.queue(), h_out_.get(), CL_TRUE, CL_MAP_READ, 0,
                                        yuv_bytes_, 0, nullptr, nullptr, &err);
        CL_CHECK(err);
    }
}

GpuPipeline::~GpuPipeline() {
    if (h_in_ptr_) clEnqueueUnmapMemObject(cl_.queue(), h_in_.get(), h_in_ptr_, 0, nullptr, nullptr);
    if (h_out_ptr_) clEnqueueUnmapMemObject(cl_.queue(), h_out_.get(), h_out_ptr_, 0, nullptr, nullptr);
    clFinish(cl_.queue());
}

// 2-D launch with 16x16 work-groups. OpenCL 1.2 requires the global size to be
// a multiple of the local size, so it is rounded up (1080 -> 1088) and the
// kernels discard the extra work-items with a bounds check.
cl_event GpuPipeline::launch_2d(cl_kernel k, size_t gx, size_t gy) {
    size_t global[2] = {round_up(gx, WG), round_up(gy, WG)};
    size_t local[2] = {WG, WG};
    cl_event e = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(cl_.queue(), k, 2, nullptr, global, local, 0, nullptr, &e));
    return e;
}

static void set_args(cl_kernel k, cl_mem a, cl_mem b, int w, int h) {
    CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &a));
    CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &b));
    CL_CHECK(clSetKernelArg(k, 2, sizeof(int), &w));
    CL_CHECK(clSetKernelArg(k, 3, sizeof(int), &h));
}

GpuTimes GpuPipeline::process(const PlanarYUV420& in, PlanarYUV420& out, Filter f) {
    out.resize(w_, h_);
    cl_command_queue q = cl_.queue();
    cl_mem yin = d_yuv_in_.get(), yout = d_yuv_out_.get();
    cl_mem a = d_rgb_a_.get(), b = d_rgb_b_.get(), t = d_tmp16_.get();
    std::vector<ClEventPtr> kernels;

    // ---- H2D (non-blocking; the in-order queue guarantees it completes
    //      before the first kernel starts) ----
    const void* src = in.data.data();
    if (pinned_) {
        std::memcpy(h_in_ptr_, in.data.data(), yuv_bytes_);
        src = h_in_ptr_;
    }
    cl_event e_h2d = nullptr;
    CL_CHECK(clEnqueueWriteBuffer(q, yin, CL_FALSE, 0, yuv_bytes_, src, 0, nullptr, &e_h2d));
    ClEventPtr h2d(e_h2d);

    // ---- Kernels ----
    auto run = [&](cl_kernel k, cl_mem x, cl_mem y, size_t gx, size_t gy) {
        set_args(k, x, y, w_, h_);
        kernels.emplace_back(launch_2d(k, gx, gy));
    };
    const size_t W = w_, H = h_;
    switch (f) {
        case Filter::Sobel:
            run(k_sobel_.get(), yin, yout, W, H);
            break;
        case Filter::None:
            run(k_yuv2rgb_.get(), yin, a, W, H);
            run(k_rgb2yuv_.get(), a, yout, W / 2, H / 2);
            break;
        case Filter::BlurNaive:
            run(k_yuv2rgb_.get(), yin, a, W, H);
            run(k_naive_.get(), a, b, W, H);
            run(k_rgb2yuv_.get(), b, yout, W / 2, H / 2);
            break;
        case Filter::BlurSeparable:
            run(k_yuv2rgb_.get(), yin, a, W, H);
            run(k_sep_h_.get(), a, t, W, H);
            run(k_sep_v_.get(), t, b, W, H);
            run(k_rgb2yuv_.get(), b, yout, W / 2, H / 2);
            break;
        case Filter::BlurTiled:
            run(k_yuv2rgb_.get(), yin, a, W, H);
            run(k_tiled_.get(), a, b, W, H);
            run(k_rgb2yuv_.get(), b, yout, W / 2, H / 2);
            break;
    }

    // ---- D2H (blocking: when it returns, everything above has finished) ----
    void* dst = pinned_ ? h_out_ptr_ : out.data.data();
    cl_event e_d2h = nullptr;
    CL_CHECK(clEnqueueReadBuffer(q, yout, CL_TRUE, 0, yuv_bytes_, dst, 0, nullptr, &e_d2h));
    ClEventPtr d2h(e_d2h);
    if (pinned_) std::memcpy(out.data.data(), h_out_ptr_, yuv_bytes_);

    GpuTimes t_out;
    t_out.h2d_ms = event_ms(h2d.get());
    t_out.d2h_ms = event_ms(d2h.get());
    for (auto& e : kernels) t_out.kernel_ms += event_ms(e.get());
    return t_out;
}
