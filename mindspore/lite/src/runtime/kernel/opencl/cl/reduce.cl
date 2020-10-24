#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define LOCAL_CACHE_THREAD 16
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

__kernel void mean_local_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);  // C4
  int localy = get_local_id(1);
  int localz = get_local_id(2);
  if (X >= size.z) return;
  __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];
  temp[localy][localz] = (float4)0.f;
  for (int h = localy; h < size.x; h += LOCAL_CACHE_THREAD) {
    for (int w = localz; w < size.y; w += LOCAL_CACHE_THREAD) {
      temp[localy][localz] += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h)));
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (localz == 0) {
    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
      temp[localy][0] += temp[localy][i];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float4 result = temp[0][0];
  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
    result += temp[i][0];
  }
  result /= size.x * size.y;
  WRITE_IMAGE(dst_data, (int2)(X, 0), TO_FLT4(result));
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

__kernel void sum_local_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);  // C4
  int localy = get_local_id(1);
  int localz = get_local_id(2);
  if (X >= size.z) return;
  __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];
  temp[localy][localz] = (float4)0.f;
  for (int h = localy; h < size.x; h += LOCAL_CACHE_THREAD) {
    for (int w = localz; w < size.y; w += LOCAL_CACHE_THREAD) {
      temp[localy][localz] += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h)));
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (localz == 0) {
    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
      temp[localy][0] += temp[localy][i];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float4 result = temp[0][0];
  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
    result += temp[i][0];
  }
  WRITE_IMAGE(dst_data, (int2)(X, 0), TO_FLT4(result));
}
