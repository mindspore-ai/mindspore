#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C4NUM 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void FullConnection_NHWC4(__read_only image2d_t input, __global FLT16 *weight, __read_only image2d_t bias,
                                   __write_only image2d_t output, int4 in_shape, int2 out_shape) {
  int gidx = get_global_id(0);  // CO4
  int gidz = get_global_id(2);  // N
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int hwci4 = ci4 * in_shape.y * in_shape.z;
  int co4 = UP_DIV(out_shape.y, C4NUM);
  int n = out_shape.x;
  bool inside = gidx < co4 && gidz < n;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < hwci4 && inside; i += 4) {
    int index_h = i / (ci4 * in_shape.z);
    int index_wci4 = i % (ci4 * in_shape.z);
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(index_wci4, gidz * in_shape.y + index_h));
    FLT16 w = weight[i * co4 + gidx];
    result.x += dot(v, w.s0123);
    result.y += dot(v, w.s4567);
    result.z += dot(v, w.s89ab);
    result.w += dot(v, w.scdef);
  }
  __local FLT4 temp[32][4];
  temp[lidx][lidy] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidy == 0 && inside) {
    result += temp[lidx][1];
    result += temp[lidx][2];
    result += temp[lidx][3];
    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));
    WRITE_IMAGE(output, (int2)(gidx, gidz), result);
  }
}

__kernel void FullConnection_NHWC4_ReLU(__read_only image2d_t input, __global FLT16 *weight, __read_only image2d_t bias,
                                        __write_only image2d_t output, int4 in_shape, int2 out_shape) {
  int gidx = get_global_id(0);  // CO4
  int gidz = get_global_id(2);  // N
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int hwci4 = ci4 * in_shape.y * in_shape.z;
  int co4 = UP_DIV(out_shape.y, C4NUM);
  int n = out_shape.x;
  bool inside = gidx < co4 && gidz < n;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < hwci4 && inside; i += 4) {
    int index_h = i / (ci4 * in_shape.z);
    int index_wci4 = i % (ci4 * in_shape.z);
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(index_wci4, gidz * in_shape.y + index_h));
    FLT16 w = weight[i * co4 + gidx];
    result.x += dot(v, w.s0123);
    result.y += dot(v, w.s4567);
    result.z += dot(v, w.s89ab);
    result.w += dot(v, w.scdef);
  }
  __local FLT4 temp[32][4];
  temp[lidx][lidy] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidy == 0 && inside) {
    result += temp[lidx][1];
    result += temp[lidx][2];
    result += temp[lidx][3];
    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));
    result = max(result, (FLT4)(0.f));
    WRITE_IMAGE(output, (int2)(gidx, gidz), result);
  }
}

__kernel void FullConnection_NC4HW4(__read_only image2d_t input, __global FLT16 *weight, __read_only image2d_t bias,
                                    __write_only image2d_t output, int4 in_shape, int2 out_shape) {
  int gidx = get_global_id(0);  // CO4
  int gidz = get_global_id(2);  // N
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int hwci4 = ci4 * in_shape.y * in_shape.z;
  int co4 = UP_DIV(out_shape.y, C4NUM);
  int n = out_shape.x;
  bool inside = gidx < co4 && gidz < n;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < hwci4 && inside; i += 4) {
    int index_ci4h = i / in_shape.z;
    int index_w = i % in_shape.z;
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(index_w, gidz * in_shape.y * ci4 + index_ci4h));
    FLT16 w = weight[i * co4 + gidx];
    result.x += dot(v, w.s0123);
    result.y += dot(v, w.s4567);
    result.z += dot(v, w.s89ab);
    result.w += dot(v, w.scdef);
  }
  __local FLT4 temp[32][4];
  temp[lidx][lidy] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidy == 0 && inside) {
    result += temp[lidx][1];
    result += temp[lidx][2];
    result += temp[lidx][3];
    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));
    WRITE_IMAGE(output, (int2)(0, gidz * co4 + gidx), result);
  }
}
