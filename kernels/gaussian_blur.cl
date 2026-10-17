// 5x5 Gaussian blur on interleaved RGB, three implementations.
// Weights are the binomial [1 4 6 4 1]; the 2-D kernel is the outer product,
// which sums to 16*16 = 256. Integer math with "(sum + 128) >> 8" (round to
// nearest) makes all three variants produce IDENTICAL output, so any
// difference between them is purely performance.

__constant int WT[5] = {1, 4, 6, 4, 1};

// ---------------------------------------------------------------------------
// 1) Naive: direct 2-D convolution, 25 taps per channel per pixel, every tap
//    read straight from global memory. Borders: clamp-to-edge.
__kernel void gaussian_blur_naive(__global const uchar* src, __global uchar* dst, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int j = -2; j <= 2; ++j) {
            int yy = clamp(y + j, 0, h - 1);
            for (int i = -2; i <= 2; ++i) {
                int xx = clamp(x + i, 0, w - 1);
                sum += WT[j + 2] * WT[i + 2] * src[(yy * w + xx) * 3 + c];
            }
        }
        dst[(y * w + x) * 3 + c] = (uchar)((sum + 128) >> 8);
    }
}

// ---------------------------------------------------------------------------
// 2) Separable: a 2-D Gaussian = horizontal 1-D pass then vertical 1-D pass,
//    so 5 + 5 = 10 taps instead of 25. The intermediate is kept UN-normalised
//    in 16-bit (max 16*255 = 4080) so no precision is lost between passes.
__kernel void gaussian_blur_h(__global const uchar* src, __global ushort* tmp, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int i = -2; i <= 2; ++i) {
            int xx = clamp(x + i, 0, w - 1);
            sum += WT[i + 2] * src[(y * w + xx) * 3 + c];
        }
        tmp[(y * w + x) * 3 + c] = (ushort)sum;
    }
}

__kernel void gaussian_blur_v(__global const ushort* tmp, __global uchar* dst, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int j = -2; j <= 2; ++j) {
            int yy = clamp(y + j, 0, h - 1);
            sum += WT[j + 2] * tmp[(yy * w + x) * 3 + c];
        }
        dst[(y * w + x) * 3 + c] = (uchar)((sum + 128) >> 8);
    }
}

// ---------------------------------------------------------------------------
// 3) Tiled: each 16x16 work-group first copies its tile PLUS a 2-pixel halo
//    (20x20 pixels) into fast on-chip __local memory, then every work-item
//    convolves from local memory. Each input pixel is fetched from global
//    memory ~once per group instead of up to 25 times.
#define TILE 16
#define R 2
#define LW (TILE + 2 * R)

__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void gaussian_blur_tiled(__global const uchar* src, __global uchar* dst, int w, int h) {
    __local uchar tile[LW * LW * 3];

    int lx = get_local_id(0), ly = get_local_id(1);
    int gx0 = get_group_id(0) * TILE - R;  // top-left of the tile incl. halo
    int gy0 = get_group_id(1) * TILE - R;

    // Cooperative load: 256 work-items fill 400 tile pixels (some load two).
    for (int ty = ly; ty < LW; ty += TILE) {
        for (int tx = lx; tx < LW; tx += TILE) {
            int sx = clamp(gx0 + tx, 0, w - 1);  // clamp-to-edge, same as naive
            int sy = clamp(gy0 + ty, 0, h - 1);
            int s = (sy * w + sx) * 3, d = (ty * LW + tx) * 3;
            tile[d] = src[s];
            tile[d + 1] = src[s + 1];
            tile[d + 2] = src[s + 2];
        }
    }
    // Every work-item must finish loading before anyone reads the tile.
    // (The early return below is AFTER the barrier, so all items reach it.)
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int j = -2; j <= 2; ++j)
            for (int i = -2; i <= 2; ++i)
                sum += WT[j + 2] * WT[i + 2] * tile[((ly + R + j) * LW + (lx + R + i)) * 3 + c];
        dst[(y * w + x) * 3 + c] = (uchar)((sum + 128) >> 8);
    }
}
