// kernel_bench --input <clip> [--runs 100] [--warmup 20] [--csv file]
// Isolated GPU micro-benchmark: decodes ONE frame, then runs each filter
// chain back-to-back many times and reports the median OpenCL event times.
// Back-to-back launches keep the GPU at a steady clock; in the full pipeline
// the GPU idles between frames (decode/encode) and laptop GPUs down-clock,
// which makes per-frame kernel times noisy.
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "gpu_ops.hpp"
#include "timer.hpp"
#include "video_decoder.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::string input, csv;
    int runs = 100, warmup = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--input") input = argv[i + 1];
        else if (k == "--runs") runs = std::stoi(argv[i + 1]);
        else if (k == "--warmup") warmup = std::stoi(argv[i + 1]);
        else if (k == "--csv") csv = argv[i + 1];
    }
    if (input.empty()) {
        std::cerr << "usage: kernel_bench --input <clip> [--runs N] [--warmup N] [--csv file]\n";
        return 1;
    }
    try {
        av_log_set_level(AV_LOG_ERROR);
        VideoDecoder dec(input);
        PlanarYUV420 in, out;
        if (!dec.read(in)) throw std::runtime_error("no frame");

        ClContext cl;
        fs::path exe_dir = fs::absolute(argv[0]).parent_path();
        std::string kdir = fs::exists(exe_dir / "kernels") ? (exe_dir / "kernels").string() : "kernels";
        std::string clip = fs::path(input).stem().string();

        std::ofstream f;
        if (!csv.empty()) {
            bool header = !fs::exists(csv);
            f.open(csv, std::ios::app);
            if (header) f << "clip,width,height,filter,pinned,runs,h2d_ms,kernel_ms,d2h_ms,gpu_total_ms,device\n";
        }
        for (bool pinned : {false, true}) {
            GpuPipeline gpu(cl, kdir, in.w, in.h, pinned);
            for (Filter flt : {Filter::None, Filter::BlurNaive, Filter::BlurSeparable,
                               Filter::BlurTiled, Filter::Sobel}) {
                std::vector<double> h2d, k, d2h, tot;
                for (int r = 0; r < warmup + runs; ++r) {
                    GpuTimes t = gpu.process(in, out, flt);
                    if (r < warmup) continue;
                    h2d.push_back(t.h2d_ms);
                    k.push_back(t.kernel_ms);
                    d2h.push_back(t.d2h_ms);
                    tot.push_back(t.h2d_ms + t.kernel_ms + t.d2h_ms);
                }
                std::cout << std::fixed << std::setprecision(4) << std::left << std::setw(15)
                          << filter_name(flt) << (pinned ? " pinned  " : " pageable")
                          << "  h2d " << median(h2d) << "  kernel " << median(k) << "  d2h "
                          << median(d2h) << "  total " << median(tot) << " ms\n";
                if (f)
                    f << std::fixed << std::setprecision(4) << clip << "," << in.w << "," << in.h << ","
                      << filter_name(flt) << "," << (pinned ? 1 : 0) << "," << runs << ","
                      << median(h2d) << "," << median(k) << "," << median(d2h) << "," << median(tot)
                      << "," << cl.device_name() << "\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
