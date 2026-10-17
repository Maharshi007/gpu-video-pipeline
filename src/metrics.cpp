#include "metrics.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>

int max_abs_diff(const uint8_t* a, const uint8_t* b, size_t n) {
    int m = 0;
    for (size_t i = 0; i < n; ++i) {
        int d = std::abs(int(a[i]) - int(b[i]));
        if (d > m) m = d;
    }
    return m;
}

double mse(const uint8_t* a, const uint8_t* b, size_t n) {
    if (n == 0) return 0.0;
    // 64-bit integer accumulation: exact, no floating-point drift on 2M pixels.
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) {
        int d = int(a[i]) - int(b[i]);
        acc += uint64_t(d * d);
    }
    return double(acc) / double(n);
}

double psnr_from_mse(double m) {
    if (m <= 0.0) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(255.0 * 255.0 / m);
}

double psnr(const uint8_t* a, const uint8_t* b, size_t n) { return psnr_from_mse(mse(a, b, n)); }

uint64_t checksum(const uint8_t* data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}
