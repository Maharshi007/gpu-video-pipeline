// yuv_psnr <distorted> <reference>
// Own Y-plane PSNR, used to cross-check FFmpeg's psnr filter.
// FFmpeg's reported "average" Y-PSNR is computed from the MEAN MSE over all
// frames (not the mean of per-frame PSNRs), so we print both.
#include <cmath>
#include <iomanip>
#include <iostream>

#include "metrics.hpp"
#include "video_decoder.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: yuv_psnr <distorted> <reference>\n";
        return 1;
    }
    try {
        av_log_set_level(AV_LOG_ERROR);
        VideoDecoder dist(argv[1]), ref(argv[2]);
        PlanarYUV420 a, b;
        double mse_sum = 0, psnr_sum = 0;
        int n = 0;
        while (dist.read(a) && ref.read(b)) {
            if (a.w != b.w || a.h != b.h) throw std::runtime_error("resolution mismatch");
            double m = mse(a.Y(), b.Y(), a.luma_size());
            mse_sum += m;
            psnr_sum += std::min(psnr_from_mse(m), 100.0);
            ++n;
        }
        if (n == 0) throw std::runtime_error("no frames");
        std::cout << std::fixed << std::setprecision(4) << "frames=" << n
                  << " psnr_y_avg_mse=" << psnr_from_mse(mse_sum / n)
                  << " psnr_y_mean_of_frames=" << psnr_sum / n << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
