// Sobel edge detection on the luma (Y) plane.
// Output frame: Y = gradient magnitude, U = V = 128 (neutral chroma -> grey).
// Built with -cl-fp32-correctly-rounded-divide-sqrt so sqrt() matches the CPU.
#pragma OPENCL FP_CONTRACT OFF

__kernel void sobel(__global const uchar* yuv_in, __global uchar* yuv_out, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;

    // Clamp-to-edge neighbour fetch.
    int xm = max(x - 1, 0), xp = min(x + 1, w - 1);
    int ym = max(y - 1, 0), yp = min(y + 1, h - 1);
    int p00 = yuv_in[ym * w + xm], p01 = yuv_in[ym * w + x], p02 = yuv_in[ym * w + xp];
    int p10 = yuv_in[y * w + xm],                             p12 = yuv_in[y * w + xp];
    int p20 = yuv_in[yp * w + xm], p21 = yuv_in[yp * w + x], p22 = yuv_in[yp * w + xp];

    int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
    int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
    float mag = sqrt((float)(gx * gx + gy * gy));
    yuv_out[y * w + x] = (uchar)min((int)(mag + 0.5f), 255);

    // The top-left quarter of work-items also writes the chroma planes.
    int cw = w / 2, ch = h / 2;
    if (x < cw && y < ch) {
        yuv_out[w * h + y * cw + x] = 128;
        yuv_out[w * h + cw * ch + y * cw + x] = 128;
    }
}
