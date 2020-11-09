#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C4NUM 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void FullConnection_NHWC4(__read_only image2d_t input, __write_only image2d_t output, __global FLT16 *weight,
                                   __read_only image2d_t bias, int4 in_shape, int2 out_shape, float act_min,
                                   float act_max) {
  int gidx = get_global_id(0);  // CO4
  int gidz = get_global_id(2);  // N
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int hwci4 = ci4 * in_shape.y * in_shape.z;
  int wci4 = ci4 * in_shape.z;
  int co4 = UP_DIV(out_shape.y, C4NUM);
  int n = out_shape.x;
  bool inside = gidx < co4 && gidz < n;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < hwci4 && inside; i += 4) {
    int index_h = i / wci4;
    int index_wci4 = i % wci4;
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
    result = clamp(result, (FLT)(act_min), (FLT)(act_max));
    WRITE_IMAGE(output, (int2)(gidx, gidz), result);
  }
}
