#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C4NUM 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void MatMul_2d(__read_only image2d_t input, __write_only image2d_t output, __global FLT16 *weight,
                        __read_only image2d_t bias, int4 in_shape, int4 out_shape) {
  int gidx = get_global_id(0);  // CO4
  int gidz = get_global_id(2);  // N
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int co4 = UP_DIV(out_shape.w, C4NUM);
  int n = out_shape.z;
  bool inside = gidx < co4 && gidz < n;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < ci4 && inside; i += 4) {
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(i, gidz));
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

__kernel void MatMul_4d(__read_only image2d_t input, __write_only image2d_t output, __global FLT16 *weight,
                        __read_only image2d_t bias, int4 in_shape, int4 out_shape) {
  int gidx = get_global_id(0);  // CO4
  int gidy = get_global_id(1);  // N * H * 4
  int gidz = get_global_id(2);  // W
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int co4 = UP_DIV(out_shape.w, C4NUM);
  int n = out_shape.x;
  int h = out_shape.y;
  int w = out_shape.z;
  int nh_index = gidy / 4;
  bool inside = gidx < co4 && gidz < w && nh_index < n * h;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < ci4 && inside; i += 4) {
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(gidz * ci4 + i, nh_index));
    FLT16 weight_value = weight[nh_index * ci4 * co4 + i * co4 + gidx];
    result.x += dot(v, weight_value.s0123);
    result.y += dot(v, weight_value.s4567);
    result.z += dot(v, weight_value.s89ab);
    result.w += dot(v, weight_value.scdef);
  }
  __local FLT4 temp[32][4];
  temp[lidx][lidy] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidy == 0 && inside) {
    result += temp[lidx][1];
    result += temp[lidx][2];
    result += temp[lidx][3];
    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));
    WRITE_IMAGE(output, (int2)(gidz * co4 + gidx, nh_index), result);
  }
}

__kernel void MatMulActWeightTransposeB_4d(__read_only image2d_t input, __write_only image2d_t output,
                                           __read_only image2d_t weight, __read_only image2d_t bias, int4 in_shape,
                                           int4 out_shape) {
  int gidx = get_global_id(0);  // CO4
  int gidy = get_global_id(1);  // N * H * 4
  int gidz = get_global_id(2);  // W
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int co4 = UP_DIV(out_shape.w, C4NUM);
  int n = out_shape.x;
  int h = out_shape.y;
  int w = out_shape.z;
  int nh_index = gidy / 4;
  bool inside = gidx < co4 && gidz < w && nh_index < n * h;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < ci4 && inside; i += 4) {
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(gidz * ci4 + i, nh_index));
    FLT4 weight_value0 = READ_IMAGE(weight, smp_zero, (int2)(gidx * 4 * ci4 + i, nh_index));
    result.x += dot(v, weight_value0);
    FLT4 weight_value1 = READ_IMAGE(weight, smp_zero, (int2)((gidx * 4 + 1) * ci4 + i, nh_index));
    result.y += dot(v, weight_value1);
    FLT4 weight_value2 = READ_IMAGE(weight, smp_zero, (int2)((gidx * 4 + 2) * ci4 + i, nh_index));
    result.z += dot(v, weight_value2);
    FLT4 weight_value3 = READ_IMAGE(weight, smp_zero, (int2)((gidx * 4 + 3) * ci4 + i, nh_index));
    result.w += dot(v, weight_value3);
  }
  __local FLT4 temp[32][4];
  temp[lidx][lidy] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidy == 0 && inside) {
    result += temp[lidx][1];
    result += temp[lidx][2];
    result += temp[lidx][3];
    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));
    WRITE_IMAGE(output, (int2)(gidz * co4 + gidx, nh_index), result);
  }
}

__kernel void MatMulActWeight_4d(__read_only image2d_t input, __write_only image2d_t output,
                                 __read_only image2d_t weight, __read_only image2d_t bias, int4 in_shape,
                                 int4 out_shape) {
  int gidx = get_global_id(0);  // CO4
  int gidy = get_global_id(1);  // N * H * 4
  int gidz = get_global_id(2);  // W
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int co4 = UP_DIV(out_shape.w, C4NUM);
  int n = out_shape.x;
  int h = out_shape.y;
  int w = out_shape.z;
  int nh_index = gidy / 4;
  bool inside = gidx < co4 && gidz < w && nh_index < n * h;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < ci4 && inside; i += 4) {
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(gidz * ci4 + i, nh_index));
    FLT4 weight_value0 = READ_IMAGE(weight, smp_zero, (int2)(i * 4 * co4 + gidx, nh_index));
    result += v.x * weight_value0;
    FLT4 weight_value1 = READ_IMAGE(weight, smp_zero, (int2)((i * 4 + 1) * co4 + gidx, nh_index));
    result += v.y * weight_value1;
    FLT4 weight_value2 = READ_IMAGE(weight, smp_zero, (int2)((i * 4 + 2) * co4 + gidx, nh_index));
    result += v.z * weight_value2;
    FLT4 weight_value3 = READ_IMAGE(weight, smp_zero, (int2)((i * 4 + 3) * co4 + gidx, nh_index));
    result += v.w * weight_value3;
  }
  __local FLT4 temp[32][4];
  temp[lidx][lidy] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidy == 0 && inside) {
    result += temp[lidx][1];
    result += temp[lidx][2];
    result += temp[lidx][3];
    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));
    WRITE_IMAGE(output, (int2)(gidz * co4 + gidx, nh_index), result);
  }
}
