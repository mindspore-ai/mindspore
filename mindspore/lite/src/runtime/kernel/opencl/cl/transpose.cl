#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void transpose_IMG(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 HW, int2 C) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= HW.y || Y >= C.y) {
    return;
  }
  FLT4 result[4];
  result[0] = (FLT4)(0.0f);
  result[1] = (FLT4)(0.0f);
  result[2] = (FLT4)(0.0f);
  result[3] = (FLT4)(0.0f);
  FLT4 x0 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * X));
  FLT4 x1 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * X + 1));
  FLT4 x2 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * X + 2));
  FLT4 x3 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * X + 3));
  result[0].x = x0.x;
  result[0].y = x1.x;
  result[0].z = x2.x;
  result[0].w = x3.x;

  result[1].x = x0.y;
  result[1].y = x1.y;
  result[1].z = x2.y;
  result[1].w = x3.y;

  result[2].x = x0.z;
  result[2].y = x1.z;
  result[2].z = x2.z;
  result[2].w = x3.z;

  result[3].x = x0.w;
  result[3].y = x1.w;
  result[3].z = x2.w;
  result[3].w = x3.w;

  WRITE_IMAGE(dst_data, (int2)(X, 4 * Y), result[0]);
  WRITE_IMAGE(dst_data, (int2)(X, 4 * Y + 1), result[1]);
  WRITE_IMAGE(dst_data, (int2)(X, 4 * Y + 2), result[2]);
  WRITE_IMAGE(dst_data, (int2)(X, 4 * Y + 3), result[3]);
}

__kernel void transpose_BUF(__read_only image2d_t src_data, global FLT4 *dst_data, int2 HW, int2 C) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= HW.y || Y >= C.y) {
    return;
  }
  FLT4 result[4];
  result[0] = (FLT4)(0.0f);
  result[1] = (FLT4)(0.0f);
  result[2] = (FLT4)(0.0f);
  result[3] = (FLT4)(0.0f);
  FLT4 x0 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * X));
  FLT4 x1 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * X + 1));
  FLT4 x2 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * X + 2));
  FLT4 x3 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * X + 3));
  result[0].x = x0.x;
  result[0].y = x1.x;
  result[0].z = x2.x;
  result[0].w = x3.x;

  result[1].x = x0.y;
  result[1].y = x1.y;
  result[1].z = x2.y;
  result[1].w = x3.y;

  result[2].x = x0.z;
  result[2].y = x1.z;
  result[2].z = x2.z;
  result[2].w = x3.z;

  result[3].x = x0.w;
  result[3].y = x1.w;
  result[3].z = x2.w;
  result[3].w = x3.w;

  if (4 * Y < C.x) dst_data[4 * Y * HW.y + X] = result[0];
  if (4 * Y + 1 < C.x) dst_data[(4 * Y + 1) * HW.y + X] = result[1];
  if (4 * Y + 2 < C.x) dst_data[(4 * Y + 2) * HW.y + X] = result[2];
  if (4 * Y + 3 < C.x) dst_data[(4 * Y + 3) * HW.y + X] = result[3];
}
