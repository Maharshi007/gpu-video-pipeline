// YUV -> RGB -> YUV round trip must be (almost) lossless on the Y plane.
// Usage: test_color_roundtrip [video file]  (synthetic frame if omitted)
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cpu_ops.hpp"
#include "video_decoder.hpp"

int main(int argc, char** argv) {
    PlanarYUV420 in;
    if (argc > 1) {
        VideoDecoder dec(argv[1]);
        if (!dec.read(in)) { std::printf("FAIL: no frame\n"); return 1; }
    } else {
        // Smooth gradients + noise, staying inside the legal limited range.
        in.resize(640, 360);
        std::srand(1);
        for (int y = 0; y < in.h; ++y)
            for (int x = 0; x < in.w; ++x)
                in.Y()[y * in.w + x] = uint8_t(16 + (x * 219 / in.w + std::rand() % 8) % 220);
        for (size_t i = 0; i < in.chroma_size(); ++i) {
            in.U()[i] = uint8_t(100 + (i % 56));
            in.V()[i] = uint8_t(110 + (i / in.w) % 40);
        }
    }
    RGBImage rgb;
    PlanarYUV420 back;
    cpu_yuv420p_to_rgb(in, rgb);
    cpu_rgb_to_yuv420p(rgb, back);

    double sum = 0;
    for (size_t i = 0; i < in.luma_size(); ++i) sum += std::abs(int(in.Y()[i]) - int(back.Y()[i]));
    double mae = sum / in.luma_size();
    bool ok = mae < 2.0;
    std::printf("%s: %dx%d  Y mean-abs-error after round trip = %.4f (limit 2.0)\n",
                ok ? "PASS" : "FAIL", in.w, in.h, mae);
    return ok ? 0 : 1;
}
