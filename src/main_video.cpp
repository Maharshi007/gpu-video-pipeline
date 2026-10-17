// video_pipeline: decode -> (CPU | GPU) frame processing -> encode, with
// per-stage timing, optional CPU-vs-GPU validation and CSV output.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpu_ops.hpp"
#include "gpu_ops.hpp"
#include "metrics.hpp"
#include "timer.hpp"
#include "video_decoder.hpp"
#include "video_encoder.hpp"

namespace fs = std::filesystem;

struct Args {
    std::string input, output = "out.mp4", mode = "gpu", filter = "none", csv, clip, kernels;
    int frames = 0, warmup = 5, cpu_threads = 1;
    bool validate = false, pinned = false;
    EncoderConfig enc;
};

static void usage() {
    std::cout <<
        "video_pipeline --input <file> [--output <file.mp4>|none]\n"
        "  --mode cpu|gpu|passthrough   --filter none|blur_naive|blur_separable|blur_tiled|sobel\n"
        "  --frames N (0=all)  --warmup N  --codec libx264|libx265  --crf N  --preset P\n"
        "  --cpu-threads N (0=all cores)  --pinned  --validate  --csv <file>  --clip <name>\n"
        "  --kernels <dir>\n";
}

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--input") a.input = val();
        else if (k == "--output") a.output = val();
        else if (k == "--mode") a.mode = val();
        else if (k == "--filter") a.filter = val();
        else if (k == "--frames") a.frames = std::stoi(val());
        else if (k == "--warmup") a.warmup = std::stoi(val());
        else if (k == "--codec") a.enc.codec = val();
        else if (k == "--crf") a.enc.crf = std::stoi(val());
        else if (k == "--preset") a.enc.preset = val();
        else if (k == "--cpu-threads") a.cpu_threads = std::stoi(val());
        else if (k == "--csv") a.csv = val();
        else if (k == "--clip") a.clip = val();
        else if (k == "--kernels") a.kernels = val();
        else if (k == "--validate") a.validate = true;
        else if (k == "--pinned") a.pinned = true;
        else if (k == "--help" || k == "-h") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown option " + k);
    }
    if (a.input.empty()) { usage(); throw std::runtime_error("--input is required"); }
    if (a.clip.empty()) a.clip = fs::path(a.input).stem().string();
    return a;
}

static Filter parse_filter(const std::string& s) {
    static const std::map<std::string, Filter> m = {
        {"none", Filter::None}, {"blur_naive", Filter::BlurNaive},
        {"blur_separable", Filter::BlurSeparable}, {"blur_tiled", Filter::BlurTiled},
        {"sobel", Filter::Sobel}};
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("unknown filter " + s);
    return it->second;
}

// Kernels are copied next to the executable at build time; fall back to ./kernels.
static std::string find_kernel_dir(const std::string& user, const char* argv0) {
    if (!user.empty()) return user;
    fs::path exe_dir = fs::absolute(argv0).parent_path();
    if (fs::exists(exe_dir / "kernels" / "sobel.cl")) return (exe_dir / "kernels").string();
    return "kernels";
}

struct FrameStats { double decode, h2d, kernel, d2h, process, encode, total; };

int main(int argc, char** argv) {
    try {
        Args a = parse(argc, argv);
        av_log_set_level(AV_LOG_ERROR);  // hide x264's per-run statistics dump
        Filter filter = parse_filter(a.filter);
        if (a.mode == "passthrough") filter = Filter::None;
        const bool use_gpu = a.mode == "gpu" || (a.validate && a.mode == "cpu");
        const bool do_cpu = a.mode == "cpu" || (a.validate && a.mode == "gpu");
        cpu_set_threads(a.cpu_threads);

        VideoDecoder dec(a.input);
        const int w = dec.width(), h = dec.height();
        if (w % 2 || h % 2) throw std::runtime_error("width/height must be even for 4:2:0");
        std::cout << "[input] " << a.input << "  " << w << "x" << h << " @ "
                  << av_q2d(dec.frame_rate()) << " fps\n";

        std::unique_ptr<ClContext> cl;
        std::unique_ptr<GpuPipeline> gpu;
        if (use_gpu && a.mode != "passthrough") {
            cl = std::make_unique<ClContext>();
            gpu = std::make_unique<GpuPipeline>(*cl, find_kernel_dir(a.kernels, argv[0]), w, h, a.pinned);
        }
        std::unique_ptr<VideoEncoder> enc;
        if (a.output != "none") enc = std::make_unique<VideoEncoder>(a.output, w, h, dec.frame_rate(), a.enc);

        PlanarYUV420 in, out, ref;
        CpuScratch scratch;
        std::vector<FrameStats> stats;
        int worst_diff = 0;
        double worst_psnr = INFINITY;
        double validate_ms = 0;  // excluded from the end-to-end wall time
        auto run_start = Clock::now();

        for (;;) {
            if (a.frames > 0 && (int)stats.size() >= a.frames) break;
            FrameStats s{};
            auto t0 = Clock::now();
            if (!dec.read(in)) break;
            s.decode = ms_since(t0);

            auto tp = Clock::now();
            const PlanarYUV420* result = &in;
            if (a.mode == "gpu") {
                GpuTimes g = gpu->process(in, out, filter);
                s.h2d = g.h2d_ms; s.kernel = g.kernel_ms; s.d2h = g.d2h_ms;
                result = &out;
            } else if (a.mode == "cpu") {
                cpu_process(in, out, filter, scratch);
                result = &out;
            }
            s.process = ms_since(tp);

            auto te = Clock::now();
            if (enc) enc->write(*result);
            s.encode = ms_since(te);
            s.total = ms_since(t0);

            // Validation runs OUTSIDE the timed region so it never skews numbers.
            if (a.validate && a.mode != "passthrough") {
                auto tv = Clock::now();
                if (a.mode == "gpu") cpu_process(in, ref, filter, scratch);
                else gpu->process(in, ref, filter);
                int d = max_abs_diff(out.data.data(), ref.data.data(), out.size_bytes());
                double p = psnr(out.data.data(), ref.data.data(), out.size_bytes());
                worst_diff = std::max(worst_diff, d);
                worst_psnr = std::min(worst_psnr, p);
                validate_ms += ms_since(tv);
            }
            stats.push_back(s);
        }
        // x264/x265 buffer frames (lookahead, B-frames, frame threads), so the
        // per-frame encode() call time understates encoder cost. The final
        // flush is included here, making end_to_end_fps the honest number.
        if (enc) enc->finish();
        double run_ms = ms_since(run_start) - validate_ms;
        if (stats.empty()) throw std::runtime_error("no frames decoded");

        // Median over frames after the warm-up (first frames include JIT /
        // driver / cache warm-up and are not representative).
        size_t skip = std::min<size_t>(a.warmup, stats.size() > 1 ? stats.size() - 1 : 0);
        auto med = [&](double FrameStats::*f) {
            std::vector<double> v;
            for (size_t i = skip; i < stats.size(); ++i) v.push_back(stats[i].*f);
            return median(v);
        };
        std::vector<double> gpu_proc;
        for (size_t i = skip; i < stats.size(); ++i)
            gpu_proc.push_back(stats[i].h2d + stats[i].kernel + stats[i].d2h);

        double decode = med(&FrameStats::decode), h2d = med(&FrameStats::h2d),
               kernel = med(&FrameStats::kernel), d2h = med(&FrameStats::d2h),
               process = med(&FrameStats::process), encode = med(&FrameStats::encode),
               total = med(&FrameStats::total);
        double proc_ms = a.mode == "gpu" ? median(gpu_proc) : process;
        double processing_fps = (a.mode == "passthrough" || proc_ms <= 0) ? 0 : 1000.0 / proc_ms;
        double e2e_fps = 1000.0 * stats.size() / run_ms;  // whole run incl. encoder flush

        std::string validation = "skipped";
        if (a.validate && a.mode != "passthrough")
            validation = (worst_diff <= 2 && worst_psnr >= 50.0) ? "pass" : "FAIL";

        std::string mode_label = a.mode;
        if (a.mode == "cpu") mode_label += "_" + std::to_string(cpu_get_threads()) + "t";
        if (a.mode == "gpu" && a.pinned) mode_label += "_pinned";

        std::cout << std::fixed << std::setprecision(3)
                  << "[result] " << stats.size() << " frames, mode=" << mode_label
                  << " filter=" << a.filter << " (median of " << stats.size() - skip << ")\n"
                  << "  decode " << decode << " ms | h2d " << h2d << " | kernel " << kernel
                  << " | d2h " << d2h << " | process(wall) " << process << " | encode " << encode
                  << " | total " << total << " ms\n"
                  << "  processing_fps " << processing_fps << " | end_to_end_fps " << e2e_fps << "\n";
        if (a.validate && a.mode != "passthrough")
            std::cout << "  validation: max_abs_diff=" << worst_diff << " min_psnr=" << worst_psnr
                      << " dB -> " << validation << "\n";

        if (!a.csv.empty()) {
            bool header = !fs::exists(a.csv);
            if (fs::path(a.csv).has_parent_path()) fs::create_directories(fs::path(a.csv).parent_path());
            std::ofstream csv(a.csv, std::ios::app);
            if (header)
                csv << "clip,width,height,frames,mode,filter,decode_ms,h2d_ms,kernel_ms,d2h_ms,"
                       "process_ms,encode_ms,total_ms,processing_fps,end_to_end_fps,max_abs_diff,"
                       "psnr_cpu_vs_gpu_db,validation,cpu_threads,pinned,encoder,preset,device\n";
            std::string psnr_s = !a.validate ? "" : std::isinf(worst_psnr) ? "inf" : std::to_string(worst_psnr);
            csv << std::fixed << std::setprecision(4) << a.clip << "," << w << "," << h << ","
                << stats.size() << "," << mode_label << "," << a.filter << "," << decode << "," << h2d
                << "," << kernel << "," << d2h << "," << process << "," << encode << "," << total << ","
                << processing_fps << "," << e2e_fps << ","
                << (a.validate ? std::to_string(worst_diff) : "") << "," << psnr_s << "," << validation
                << "," << cpu_get_threads() << "," << (a.pinned ? 1 : 0) << ","
                << (enc ? a.enc.codec : "none") << "," << a.enc.preset << ","
                << (cl ? cl->device_name() : "") << "\n";
        }
        return validation == "FAIL" ? 2 : 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
