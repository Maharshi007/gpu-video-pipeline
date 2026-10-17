// Every GPU filter chain vs its CPU reference on a random 1080p frame.
// Usage: test_cpu_vs_gpu <kernel dir>
#include <cstdio>
#include <random>

#include "cpu_ops.hpp"
#include "gpu_ops.hpp"
#include "metrics.hpp"

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "kernels";
    try {
        const int w = 1920, h = 1080;
        PlanarYUV420 in, cpu_out, gpu_out;
        in.resize(w, h);
        std::mt19937 rng(42);
        for (auto& v : in.data) v = uint8_t(rng() & 0xFF);  // full 0..255 range: stresses clamping

        ClContext cl(false);
        GpuPipeline gpu(cl, dir, w, h, false);
        GpuPipeline gpu_pinned(cl, dir, w, h, true);
        CpuScratch s;

        int failures = 0;
        for (Filter f : {Filter::None, Filter::BlurNaive, Filter::BlurSeparable, Filter::BlurTiled,
                         Filter::Sobel}) {
            cpu_process(in, cpu_out, f, s);
            for (GpuPipeline* g : {&gpu, &gpu_pinned}) {
                g->process(in, gpu_out, f);
                int d = max_abs_diff(cpu_out.data.data(), gpu_out.data.data(), in.size_bytes());
                double p = psnr(cpu_out.data.data(), gpu_out.data.data(), in.size_bytes());
                bool ok = d <= 2 && p >= 50.0;
                failures += !ok;
                std::printf("%s  %-15s %-7s max_abs_diff=%d psnr=%.2f dB\n", ok ? "PASS" : "FAIL",
                            filter_name(f), g == &gpu ? "pageable" : "pinned", d, p);
            }
        }
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
