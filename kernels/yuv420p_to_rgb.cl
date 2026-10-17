// BT.709 limited-range YUV 4:2:0 -> interleaved RGB.
// Disable fused multiply-add so results match the CPU reference bit-for-bit.
#pragma OPENCL FP_CONTRACT OFF

inline uchar to_u8(float v) {
    v = clamp(v, 0.0f, 255.0f);
    return (uchar)(v + 0.5f);  // round to nearest (v >= 0, so truncation = floor)
}

// One work-item per pixel. get_global_id(0) is the COLUMN (x), so adjacent
// work-items in a warp read adjacent bytes of a row -> coalesced memory access.
__kernel void yuv420p_to_rgb(__global const uchar* yuv, __global uchar* rgb, int w, int h) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;  // global size is rounded up to the work-group size

    __global const uchar* Y = yuv;
    __global const uchar* U = yuv + w * h;
    __global const uchar* V = U + (w / 2) * (h / 2);

    // 4:2:0 -> one chroma sample is shared by a 2x2 block of luma pixels.
    int ci = (y >> 1) * (w >> 1) + (x >> 1);
    float C = (float)Y[y * w + x] - 16.0f;
    float D = (float)U[ci] - 128.0f;
    float E = (float)V[ci] - 128.0f;

    float r = 1.164f * C + 1.793f * E;
    float g = 1.164f * C - 0.213f * D - 0.533f * E;
    float b = 1.164f * C + 2.112f * D;

    int o = (y * w + x) * 3;
    rgb[o + 0] = to_u8(r);
    rgb[o + 1] = to_u8(g);
    rgb[o + 2] = to_u8(b);
}
