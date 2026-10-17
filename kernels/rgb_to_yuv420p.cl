// Interleaved RGB -> BT.709 limited-range YUV 4:2:0.
#pragma OPENCL FP_CONTRACT OFF

inline uchar to_u8(float v) {
    v = clamp(v, 0.0f, 255.0f);
    return (uchar)(v + 0.5f);
}

// Each work-item owns one 2x2 pixel block: it writes 4 Y samples and the
// single U and V sample for that block. This avoids a second pass (or
// atomics) to compute the subsampled chroma. Global size = (w/2, h/2).
__kernel void rgb_to_yuv420p(__global const uchar* rgb, __global uchar* yuv, int w, int h) {
    int bx = get_global_id(0);
    int by = get_global_id(1);
    int cw = w / 2, ch = h / 2;
    if (bx >= cw || by >= ch) return;

    __global uchar* Y = yuv;
    __global uchar* U = yuv + w * h;
    __global uchar* V = U + cw * ch;

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
    // Chroma from the average colour of the 2x2 block.
    sr *= 0.25f; sg *= 0.25f; sb *= 0.25f;
    U[by * cw + bx] = to_u8(128.0f - 0.101f * sr - 0.339f * sg + 0.439f * sb);
    V[by * cw + bx] = to_u8(128.0f + 0.439f * sr - 0.399f * sg - 0.040f * sb);
}
