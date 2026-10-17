// PSNR implementation vs hand-computed values.
#include <cmath>
#include <cstdio>
#include <vector>

#include "metrics.hpp"

static int failures = 0;
static void expect_near(const char* name, double got, double want, double tol) {
    bool ok = std::fabs(got - want) <= tol;
    failures += !ok;
    std::printf("%s  %-28s got=%.6f want=%.6f\n", ok ? "PASS" : "FAIL", name, got, want);
}

int main() {
    std::vector<uint8_t> a(1000, 100), b(1000, 100);
    bool inf_ok = std::isinf(psnr(a.data(), b.data(), a.size()));
    failures += !inf_ok;
    std::printf("%s  identical -> inf\n", inf_ok ? "PASS" : "FAIL");

    // Every sample off by 1 -> MSE = 1 -> PSNR = 10*log10(65025) = 48.1308 dB
    for (auto& v : b) v = 101;
    expect_near("all diff 1 (MSE=1)", psnr(a.data(), b.data(), a.size()), 48.130804, 1e-5);

    // Every sample off by 10 -> MSE = 100 -> 28.1308 dB
    for (auto& v : b) v = 110;
    expect_near("all diff 10 (MSE=100)", psnr(a.data(), b.data(), a.size()), 28.130804, 1e-5);

    // Half the samples off by 2 -> MSE = 2 -> 45.1205 dB
    for (size_t i = 0; i < b.size(); ++i) b[i] = i % 2 ? 102 : 100;
    expect_near("half diff 2 (MSE=2)", psnr(a.data(), b.data(), a.size()), 45.120504, 1e-5);

    expect_near("max_abs_diff", max_abs_diff(a.data(), b.data(), a.size()), 2, 0);
    return failures ? 1 : 0;
}
