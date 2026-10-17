#pragma once
#include <cstdint>
#include <vector>

// One 8-bit YUV 4:2:0 frame stored as a single tightly packed buffer:
//   [ Y : w*h ][ U : (w/2)*(h/2) ][ V : (w/2)*(h/2) ]
// Keeping all three planes contiguous means a frame is uploaded to the GPU
// with ONE transfer instead of three.
struct PlanarYUV420 {
    int w = 0, h = 0;
    std::vector<uint8_t> data;

    void resize(int width, int height) {
        w = width;
        h = height;
        data.resize(size_bytes());
    }
    size_t luma_size() const { return size_t(w) * h; }
    size_t chroma_size() const { return size_t(w / 2) * (h / 2); }
    size_t size_bytes() const { return luma_size() + 2 * chroma_size(); }

    uint8_t* Y() { return data.data(); }
    uint8_t* U() { return data.data() + luma_size(); }
    uint8_t* V() { return data.data() + luma_size() + chroma_size(); }
    const uint8_t* Y() const { return data.data(); }
    const uint8_t* U() const { return data.data() + luma_size(); }
    const uint8_t* V() const { return data.data() + luma_size() + chroma_size(); }
};

// Interleaved 8-bit RGB: R,G,B,R,G,B,... (3 bytes per pixel).
struct RGBImage {
    int w = 0, h = 0;
    std::vector<uint8_t> data;
    void resize(int width, int height) {
        w = width;
        h = height;
        data.resize(size_t(w) * h * 3);
    }
};

enum class Filter { None, BlurNaive, BlurSeparable, BlurTiled, Sobel };

inline const char* filter_name(Filter f) {
    switch (f) {
        case Filter::None: return "none";
        case Filter::BlurNaive: return "blur_naive";
        case Filter::BlurSeparable: return "blur_separable";
        case Filter::BlurTiled: return "blur_tiled";
        case Filter::Sobel: return "sobel";
    }
    return "?";
}
