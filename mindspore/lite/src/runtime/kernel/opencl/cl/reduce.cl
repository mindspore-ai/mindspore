#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void mean_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);  // C4
  if (X >= size.z) {
    return;
  }
  float4 result = (float4)0.f;
  for (int h = 0; h < size.x; h++) {
    for (int w = 0; w < size.y; w++) {
      result += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h)));
    }
  }
  result /= size.x * size.y;
  WRITE_IMAGE(dst_data, (int2)(X, 0), TO_FLT4(result));
}

__kernel void mean_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);  // C4
  if (X >= size.z) {
    return;
  }
  FLT4 result = (FLT4)0.f;
  for (int h = 0; h < size.x; h++) {
    for (int w = 0; w < size.y; w++) {
      result += READ_IMAGE(src_data, smp_zero, (int2)(w, X * size.x + h));
    }
  }
  result /= size.x * size.y;
  WRITE_IMAGE(dst_data, (int2)(0, X), result);
}

__kernel void sum_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);  // C4
  if (X >= size.z) {
    return;
  }
  FLT4 result = (FLT4)0.f;
  for (int h = 0; h < size.x; h++) {
    for (int w = 0; w < size.y; w++) {
      result += READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h));
    }
  }
  WRITE_IMAGE(dst_data, (int2)(X, 0), result);
}

__kernel void sum_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);  // C4
  if (X >= size.z) {
    return;
  }
  FLT4 result = (FLT4)0.f;
  for (int h = 0; h < size.x; h++) {
    for (int w = 0; w < size.y; w++) {
      result += READ_IMAGE(src_data, smp_zero, (int2)(w, X * size.x + h));
    }
  }
  WRITE_IMAGE(dst_data, (int2)(0, X), result);
}
