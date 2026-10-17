#include "cpu_ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_set_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n > 0 ? n : omp_get_num_procs());
#else
    (void)n;
#endif
}

int cpu_get_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Same clamp + round-to-nearest as to_u8() in the kernels.
static inline uint8_t to_u8(float v) {
    v = std::min(std::max(v, 0.0f), 255.0f);
    return (uint8_t)(v + 0.5f);
}

static const int WT[5] = {1, 4, 6, 4, 1};

void cpu_yuv420p_to_rgb(const PlanarYUV420& in, RGBImage& out) {
    const int w = in.w, h = in.h;
    out.resize(w, h);
    const uint8_t *Y = in.Y(), *U = in.U(), *V = in.V();
    uint8_t* rgb = out.data.data();
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int ci = (y >> 1) * (w >> 1) + (x >> 1);
            float C = (float)Y[y * w + x] - 16.0f;
            float D = (float)U[ci] - 128.0f;
            float E = (float)V[ci] - 128.0f;
            float r = 1.164f * C + 1.793f * E;
            float g = 1.164f * C - 0.213f * D - 0.533f * E;
            float b = 1.164f * C + 2.112f * D;
            int o = (y * w + x) * 3;
            rgb[o] = to_u8(r);
            rgb[o + 1] = to_u8(g);
            rgb[o + 2] = to_u8(b);
        }
    }
}

void cpu_rgb_to_yuv420p(const RGBImage& in, PlanarYUV420& out) {
    const int w = in.w, h = in.h, cw = w / 2, ch = h / 2;
    out.resize(w, h);
    const uint8_t* rgb = in.data.data();
    uint8_t *Y = out.Y(), *U = out.U(), *V = out.V();
#pragma omp parallel for
    for (int by = 0; by < ch; ++by) {
        for (int bx = 0; bx < cw; ++bx) {
            float sr = 0.0f, sg = 0.0f, sb = 0.0f;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    int x = 2 * bx + dx, y = 2 * by + dy;
                    int p = (y * w + x) * 3;
                    float r = rgb[p], g = rgb[p + 1], b = rgb[p + 2];
                    Y[y * w + x] = to_u8(16.0f + 0.183f * r + 0.614f * g + 0.062f * b);
                    sr += r; sg += g; sb += b;
                }
            }
            sr *= 0.25f; sg *= 0.25f; sb *= 0.25f;
            U[by * cw + bx] = to_u8(128.0f - 0.101f * sr - 0.339f * sg + 0.439f * sb);
            V[by * cw + bx] = to_u8(128.0f + 0.439f * sr - 0.399f * sg - 0.040f * sb);
        }
    }
}

void cpu_blur_naive(const RGBImage& in, RGBImage& out) {
    const int w = in.w, h = in.h;
    out.resize(w, h);
    const uint8_t* src = in.data.data();
    uint8_t* dst = out.data.data();
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                int sum = 0;
                for (int j = -2; j <= 2; ++j) {
                    int yy = std::clamp(y + j, 0, h - 1);
                    for (int i = -2; i <= 2; ++i) {
                        int xx = std::clamp(x + i, 0, w - 1);
                        sum += WT[j + 2] * WT[i + 2] * src[(yy * w + xx) * 3 + c];
                    }
                }
                dst[(y * w + x) * 3 + c] = (uint8_t)((sum + 128) >> 8);
            }
        }
    }
}

void cpu_blur_separable(const RGBImage& in, RGBImage& out, std::vector<uint16_t>& tmp) {
    const int w = in.w, h = in.h;
    out.resize(w, h);
    tmp.resize(size_t(w) * h * 3);
    const uint8_t* src = in.data.data();
    uint8_t* dst = out.data.data();
    uint16_t* t = tmp.data();
#pragma omp parallel for
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                int sum = 0;
                for (int i = -2; i <= 2; ++i)
                    sum += WT[i + 2] * src[(y * w + std::clamp(x + i, 0, w - 1)) * 3 + c];
                t[(y * w + x) * 3 + c] = (uint16_t)sum;
            }
#pragma omp parallel for
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                int sum = 0;
                for (int j = -2; j <= 2; ++j)
                    sum += WT[j + 2] * t[(std::clamp(y + j, 0, h - 1) * w + x) * 3 + c];
                dst[(y * w + x) * 3 + c] = (uint8_t)((sum + 128) >> 8);
            }
}

void cpu_sobel_luma(const PlanarYUV420& in, PlanarYUV420& out) {
    const int w = in.w, h = in.h;
    out.resize(w, h);
    const uint8_t* s = in.Y();
    uint8_t* d = out.Y();
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        int ym = std::max(y - 1, 0), yp = std::min(y + 1, h - 1);
        for (int x = 0; x < w; ++x) {
            int xm = std::max(x - 1, 0), xp = std::min(x + 1, w - 1);
            int p00 = s[ym * w + xm], p01 = s[ym * w + x], p02 = s[ym * w + xp];
            int p10 = s[y * w + xm], p12 = s[y * w + xp];
            int p20 = s[yp * w + xm], p21 = s[yp * w + x], p22 = s[yp * w + xp];
            int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
            int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
            float mag = std::sqrt((float)(gx * gx + gy * gy));
            d[y * w + x] = (uint8_t)std::min((int)(mag + 0.5f), 255);
        }
    }
    std::fill(out.U(), out.U() + 2 * out.chroma_size(), (uint8_t)128);
}

void cpu_process(const PlanarYUV420& in, PlanarYUV420& out, Filter f, CpuScratch& s) {
    switch (f) {
        case Filter::Sobel:
            cpu_sobel_luma(in, out);
            return;
        case Filter::None:
            cpu_yuv420p_to_rgb(in, s.a);
            cpu_rgb_to_yuv420p(s.a, out);
            return;
        case Filter::BlurNaive:
        case Filter::BlurTiled:  // tiling is a GPU memory optimisation; same result
            cpu_yuv420p_to_rgb(in, s.a);
            cpu_blur_naive(s.a, s.b);
            cpu_rgb_to_yuv420p(s.b, out);
            return;
        case Filter::BlurSeparable:
            cpu_yuv420p_to_rgb(in, s.a);
            cpu_blur_separable(s.a, s.b, s.tmp);
            cpu_rgb_to_yuv420p(s.b, out);
            return;
    }
    throw std::runtime_error("unknown filter");
}
