#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void DepthwiseConv2d_IMG_NHWC4(__write_only image2d_t dst_data, __read_only image2d_t src_data,
                                        __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,
                                        int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,
                                        float relu_clip_min, float relu_clip_max) {
  int X = get_global_id(1);
  int Y = get_global_id(2);
  int Z = get_global_id(0);
  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;
  FLT4 r = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  int x_offset = X * stride.x + padding.x;
  int y_offset = Y * stride.y + padding.y;
  int fx_c = Z * kernel_size.x * kernel_size.y;
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = y_offset + ky * dilation.y;
    bool outside_y = y_c < 0 || y_c >= src_size.y;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = x_offset + kx * dilation.x;
      bool outside_x = x_c < 0 || x_c >= src_size.x;
      if (!outside_x && !outside_y) {
        FLT4 flt_p = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z));
        FLT4 src_p = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c));
        r += TO_FLT4(src_p * flt_p);
      }
      fx_c++;
    }
  }
  FLT4 bias_p = bias[Z];
  FLT4 res = TO_FLT4(r) + bias_p;
  res = clamp(res, (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), res);
}

__kernel void DepthwiseConv2d_IMG_NHWC4_1x1(__write_only image2d_t dst_data, __read_only image2d_t src_data,
                                            __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,
                                            int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,
                                            float relu_clip_min, float relu_clip_max) {
  int X = get_global_id(1);
  int Y = get_global_id(2);
  int Z = get_global_id(0);
  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;
  FLT4 r = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  int x_offset = X * stride.x + padding.x;
  int y_offset = Y * stride.y + padding.y;
  int fx_c = Z;
  {
    int y_c = y_offset;
    bool outside_y = y_c < 0 || y_c >= src_size.y;
    {
      int x_c = x_offset;
      bool outside_x = x_c < 0 || x_c >= src_size.x;
      if (!outside_x && !outside_y) {
        FLT4 flt_p = READ_IMAGE(filter, smp_zero, (int2)(0, Z));
        FLT4 src_p = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c));
        r += TO_FLT4(src_p * flt_p);
      }
    }
  }
  FLT4 bias_p = bias[Z];
  FLT4 res = TO_FLT4(r) + bias_p;
  res = clamp(res, (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), res);
}
__kernel void DepthwiseConv2d_IMG_NHWC4_b222(__write_only image2d_t dst_data, __read_only image2d_t src_data,
                                             __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size, int2 stride,
                                             int2 padding, int2 dilation, int4 src_size, int4 dst_size,
                                             float relu_clip_min, float relu_clip_max) {
  int X = get_global_id(1) * 2;
  int Y = get_global_id(2) * 2;
  int Z = get_global_id(0) * 2;
  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;
  FLT4 r[8] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),
               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),
               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};
  int x_offset = X * stride.x + padding.x;
  int y_offset = Y * stride.y + padding.y;
  int f_len = kernel_size.x * kernel_size.y;
  int fx_c = Z * f_len;
  bool last_x = (get_global_id(1) == (dst_size.x + 1) / 2) && ((dst_size.x & 0x1) == 1);
  bool last_y = (get_global_id(2) == (dst_size.y + 1) / 2) && ((dst_size.y & 0x1) == 1);
  bool last_c = (get_global_id(0) == (dst_size.z + 1) / 2) && ((dst_size.z & 0x1) == 1);
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = y_offset + ky * dilation.y;
    int y_c_a1 = y_c + stride.y;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = x_offset + kx * dilation.x;
      int x_c_a1 = x_c + stride.x;
      int x_sign = x_c < 0 ? -1 : 1;
      int x_a1_sign = x_c_a1 < 0 ? -1 : 1;
      FLT4 flt_p0 = filter[fx_c];
      FLT4 flt_p1 = filter[fx_c + f_len];
      {
        FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c));
        FLT4 src_p00_c1 = READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_sign + x_c * src_size.z, y_c));
        r[0] += TO_FLT4(src_p00_c0 * flt_p0);
        r[1] += TO_FLT4(src_p00_c1 * flt_p1);
      }
      {
        FLT4 src_p01_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c));
        FLT4 src_p01_c1 = READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_a1_sign + x_c_a1 * src_size.z, y_c));
        r[2] += TO_FLT4(src_p01_c0 * flt_p0);
        r[3] += TO_FLT4(src_p01_c1 * flt_p1);
      }
      {
        FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1));
        FLT4 src_p10_c1 = READ_IMAGE(src_data, smp_zero, (int2)(Z + 1 + x_c * src_size.z, y_c_a1));
        r[4] += TO_FLT4(src_p10_c0 * flt_p0);
        r[5] += TO_FLT4(src_p10_c1 * flt_p1);
      }
      {
        FLT4 src_p11_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c_a1));
        FLT4 src_p11_c1 = READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_a1_sign + x_c_a1 * src_size.z, y_c_a1));
        r[6] += TO_FLT4(src_p11_c0 * flt_p0);
        r[7] += TO_FLT4(src_p11_c1 * flt_p1);
      }
      fx_c++;
    }
  }
  r[0] += bias[Z];
  r[1] += bias[Z + 1];
  r[2] += bias[Z];
  r[3] += bias[Z + 1];
  r[4] += bias[Z];
  r[5] += bias[Z + 1];
  r[6] += bias[Z];
  r[7] += bias[Z + 1];
  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[4] = clamp(r[4], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[5] = clamp(r[5], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[6] = clamp(r[6], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[7] = clamp(r[7], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);
  if (!last_c) {
    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y), r[1]);
  }
  if (!last_x) {
    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[2]);
    if (!last_c) {
      WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z + 1, Y), r[3]);
    }
  }
  if (!last_y) {
    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[4]);
    if (!last_c) {
      WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y + 1), r[5]);
    }
  }
  if (!last_y && !last_x) {
    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y + 1), r[6]);
    if (!last_c) {
      WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z + 1, Y + 1), r[7]);
    }
  }
}
__kernel void DepthwiseConv2d_IMG_NHWC4_b221(__write_only image2d_t dst_data, __read_only image2d_t src_data,
                                             __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size, int2 stride,
                                             int2 padding, int2 dilation, int4 src_size, int4 dst_size,
                                             float relu_clip_min, float relu_clip_max) {
  int X = get_global_id(1) * 2;
  int Y = get_global_id(2) * 2;
  int Z = get_global_id(0);
  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;
  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),
               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};
  int x_offset = X * stride.x + padding.x;
  int y_offset = Y * stride.y + padding.y;
  int f_len = kernel_size.x * kernel_size.y;
  int fx_c = Z * f_len;
  bool last_x = (get_global_id(1) == (dst_size.x + 1) / 2) && ((dst_size.x & 0x1) == 1);
  bool last_y = (get_global_id(2) == (dst_size.y + 1) / 2) && ((dst_size.y & 0x1) == 1);
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = y_offset + ky * dilation.y;
    int y_c_a1 = y_c + stride.y;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = x_offset + kx * dilation.x;
      int x_c_a1 = x_c + stride.x;
      int x_sign = x_c < 0 ? -1 : 1;
      int x_a1_sign = x_c_a1 < 0 ? -1 : 1;
      FLT4 flt_p0 = filter[fx_c];
      FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c));
      r[0] += TO_FLT4(src_p00_c0 * flt_p0);
      FLT4 src_p01_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c));
      r[1] += TO_FLT4(src_p01_c0 * flt_p0);
      FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1));
      r[2] += TO_FLT4(src_p10_c0 * flt_p0);
      FLT4 src_p11_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c_a1));
      r[3] += TO_FLT4(src_p11_c0 * flt_p0);

      fx_c++;
    }
  }
  r[0] += bias[Z];
  r[1] += bias[Z];
  r[2] += bias[Z];
  r[3] += bias[Z];
  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);
  if (!last_x) {
    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[1]);
  }
  if (!last_y) {
    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[2]);
  }
  if (!last_y && !last_x) {
    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y + 1), r[3]);
  }
}
__kernel void DepthwiseConv2d_IMG_NHWC4_1x1_b221(__write_only image2d_t dst_data, __read_only image2d_t src_data,
                                                 __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size,
                                                 int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,
                                                 float relu_clip_min, float relu_clip_max) {
  int X = get_global_id(1) * 2;
  int Y = get_global_id(2) * 2;
  int Z = get_global_id(0);
  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;
  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),
               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};
  int x_offset = X * stride.x + padding.x;
  int y_offset = Y * stride.y + padding.y;
  int f_len = kernel_size.x * kernel_size.y;
  int fx_c = Z * f_len;
  bool last_x = (get_global_id(1) == (dst_size.x + 1) / 2) && ((dst_size.x & 0x1) == 1);
  bool last_y = (get_global_id(2) == (dst_size.y + 1) / 2) && ((dst_size.y & 0x1) == 1);
  int y_c = y_offset;
  int y_c_a1 = y_c + stride.y;
  int x_c = x_offset;
  int x_c_a1 = x_c + stride.x;
  int x_sign = x_c < 0 ? -1 : 1;
  int x_a1_sign = x_c_a1 < 0 ? -1 : 1;
  FLT4 flt_p0 = filter[fx_c];
  FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c));
  r[0] += TO_FLT4(src_p00_c0 * flt_p0);
  FLT4 src_p01_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c));
  r[1] += TO_FLT4(src_p01_c0 * flt_p0);
  FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1));
  r[2] += TO_FLT4(src_p10_c0 * flt_p0);
  FLT4 src_p11_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c_a1));
  r[3] += TO_FLT4(src_p11_c0 * flt_p0);

  r[0] += bias[Z];
  r[1] += bias[Z];
  r[2] += bias[Z];
  r[3] += bias[Z];
  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);
  if (!last_x) {
    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[1]);
  }
  if (!last_y) {
    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[2]);
  }
  if (!last_y && !last_x) {
    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y + 1), r[3]);
  }
}
__kernel void DepthwiseConv2d_BUF_NC4HW4(__global FLT4 *dst_data, __global FLT4 *src_data, __global FLT4 *filter,
                                         __global FLT4 *bias, int2 kernel_size, int2 stride, int2 padding,
                                         int2 dilation, int4 src_size, int4 dst_size, float relu_clip_min,
                                         float relu_clip_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;
  FLT4 r = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  int x_offset = X * stride.x + padding.x;
  int y_offset = Y * stride.y + padding.y;
  int fx_c = Z * kernel_size.x * kernel_size.y;
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = y_offset + ky * dilation.y;
    bool outside_y = y_c < 0 || y_c >= src_size.y;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = x_offset + kx * dilation.x;
      bool outside_x = x_c < 0 || x_c >= src_size.x;
      if (!outside_x && !outside_y) {
        FLT4 flt_p = filter[fx_c];
        FLT4 src_p = src_data[(((Z)*src_size.y + (y_c)) * src_size.x + (x_c))];
        r += TO_FLT4(src_p * flt_p);
      }
      fx_c++;
    }
  }
  FLT4 bias_p = bias[Z];
  FLT4 res = TO_FLT4(r) + bias_p;
  res = clamp(res, (FLT)(relu_clip_min), (FLT)(relu_clip_max));
  dst_data[(((Z)*dst_size.y + (Y)) * dst_size.x + (X))] = res;
}
