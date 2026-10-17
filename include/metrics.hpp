#pragma once
#include <cstddef>
#include <cstdint>

// Largest per-byte difference between two buffers.
int max_abs_diff(const uint8_t* a, const uint8_t* b, size_t n);

// Mean squared error between two 8-bit buffers.
double mse(const uint8_t* a, const uint8_t* b, size_t n);

// PSNR = 10*log10(255^2 / MSE). Returns +infinity for identical inputs.
double psnr_from_mse(double m);
double psnr(const uint8_t* a, const uint8_t* b, size_t n);

// FNV-1a hash, handy for quickly checking two outputs are bit-identical.
uint64_t checksum(const uint8_t* data, size_t n);
