#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void resize_nearest_neighbor_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data,
                                            int4 in_size, int4 out_size, float2 scale_factor) {
  int X = get_global_id(2);  // H * N
  int Y = get_global_id(1);  // W
  int Z = get_global_id(0);  // C4
  if (X >= out_size.x * out_size.y || Y >= out_size.z || Z >= out_size.w) {
    return;
  }
  int N = X / out_size.y;
  X = X % out_size.y;
  int src_x = (int)(X * scale_factor.x);
  int src_y = (int)(Y * scale_factor.y);
  WRITE_IMAGE(dst_data, (int2)(Y * out_size.w + Z, N * out_size.y + X),
              READ_IMAGE(src_data, smp_zero, (int2)(src_y * in_size.w + Z, N * in_size.y + src_x)));
}

__kernel void resize_nearest_neighbor_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data,
                                             int4 in_size, int4 out_size, float2 scale_factor) {
  int X = get_global_id(2);  // H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(0);  // C4
  if (X >= out_size.y || Y >= out_size.z || Z >= out_size.w) {
    return;
  }
  int src_x = (int)(X * scale_factor.x);
  int src_y = (int)(Y * scale_factor.y);
  WRITE_IMAGE(dst_data, (int2)(Y, Z * out_size.y + X),
              READ_IMAGE(src_data, smp_zero, (int2)(src_y, Z * in_size.y + src_x)));
}

__kernel void resize_bilinear_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_size,
                                    int4 out_size, float2 scale_factor) {
  int X = get_global_id(2);  // H * N
  int Y = get_global_id(1);  // W
  int Z = get_global_id(0);  // C4
  if (X >= out_size.x * out_size.y || Y >= out_size.z || Z >= out_size.w) {
    return;
  }
  int N = X / out_size.y;
  X = X % out_size.y;
  float scale_x = X * scale_factor.x;
  float scale_y = Y * scale_factor.y;
  int src_x = (int)(scale_x);
  int src_y = (int)(scale_y);
  int src_x_1 = min(src_x + 1, in_size.y - 1);
  int src_y_1 = min(src_y + 1, in_size.z - 1);
  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(src_y * in_size.w + Z, N * in_size.y + src_x));
  FLT4 src1 = READ_IMAGE(src_data, smp_zero, (int2)(src_y_1 * in_size.w + Z, N * in_size.y + src_x));
  FLT4 src2 = READ_IMAGE(src_data, smp_zero, (int2)(src_y * in_size.w + Z, N * in_size.y + src_x_1));
  FLT4 src3 = READ_IMAGE(src_data, smp_zero, (int2)(src_y_1 * in_size.w + Z, N * in_size.y + src_x_1));
  FLT4 result =
    mix(mix(src0, src1, TO_FLT(scale_y - src_y)), mix(src2, src3, TO_FLT(scale_y - src_y)), TO_FLT(scale_x - src_x));
  WRITE_IMAGE(dst_data, (int2)(Y * out_size.w + Z, N * out_size.y + X), result);
}

__kernel void resize_bilinear_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_size,
                                     int4 out_size, float2 scale_factor) {
  int X = get_global_id(2);  // H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(0);  // C4
  if (X >= out_size.y || Y >= out_size.z || Z >= out_size.w) {
    return;
  }
  float scale_x = X * scale_factor.x;
  float scale_y = Y * scale_factor.y;
  int src_x = (int)(scale_x);
  int src_y = (int)(scale_y);
  int src_x_1 = min(src_x + 1, in_size.y - 1);
  int src_y_1 = min(src_y + 1, in_size.z - 1);
  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(src_y, in_size.y * Z + src_x));
  FLT4 src1 = READ_IMAGE(src_data, smp_zero, (int2)(src_y_1, in_size.y * Z + src_x));
  FLT4 src2 = READ_IMAGE(src_data, smp_zero, (int2)(src_y, in_size.y * Z + src_x_1));
  FLT4 src3 = READ_IMAGE(src_data, smp_zero, (int2)(src_y_1, in_size.y * Z + src_x_1));
  FLT4 result =
    mix(mix(src0, src1, TO_FLT(scale_y - src_y)), mix(src2, src3, TO_FLT(scale_y - src_y)), TO_FLT(scale_x - src_x));
  WRITE_IMAGE(dst_data, (int2)(Y, out_size.w * Z + X), result);
}
