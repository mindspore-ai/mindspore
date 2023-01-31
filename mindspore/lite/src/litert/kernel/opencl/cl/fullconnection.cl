#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C4NUM 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void FullConnection(__read_only image2d_t input, __write_only image2d_t output, __global FLT16 *weight,
                             __read_only image2d_t bias, int N, int CI4, int CO4, int2 in_img_shape, int act_type) {
  int gidx = get_global_id(0);  // CO4
  int gidz = get_global_id(2);  // N
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  bool inside = gidx < CO4 && gidz < N;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < CI4 && inside; i += 4) {
    int index = gidz * CI4 + i;
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(index % in_img_shape.y, index / in_img_shape.y));
    FLT16 w = weight[i * CO4 + gidx];
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
    if (act_type == ActivationType_RELU) {
      result = max(result, (FLT4)(0.0f));
    } else if (act_type == ActivationType_RELU6) {
      result = clamp(result, (FLT4)(0.0f), (FLT4)(6.0f));
    } else if (act_type == ActivationType_TANH) {
      result = tanh(clamp(result, (FLT)(-10.0f), (FLT)(10.0f)));
    } else if (act_type == ActivationType_SIGMOID) {
      result = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-result));
    }
    WRITE_IMAGE(output, (int2)(gidx, gidz), result);
  }
}

__kernel void FullConnectionWeightVar(__read_only image2d_t input, __write_only image2d_t output,
                                      __read_only image2d_t weight, __read_only image2d_t bias, int N, int CI4, int CO4,
                                      int2 in_img_shape, int act_type) {
  int gidx = get_global_id(0);  // CO4
  int gidz = get_global_id(2);  // N
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  bool inside = gidx < CO4 && gidz < N;
  FLT4 result = (FLT4)(0.0f);
  for (uint i = lidy; i < CI4 && inside; i += 4) {
    int index = gidz * CI4 + i;
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(index % in_img_shape.y, index / in_img_shape.y));
    FLT4 weight0 = READ_IMAGE(weight, smp_zero, (int2)(i, gidx * 4));
    result.x += dot(v, weight0);
    FLT4 weight1 = READ_IMAGE(weight, smp_zero, (int2)(i, gidx * 4 + 1));
    result.y += dot(v, weight1);
    FLT4 weight2 = READ_IMAGE(weight, smp_zero, (int2)(i, gidx * 4 + 2));
    result.z += dot(v, weight2);
    FLT4 weight3 = READ_IMAGE(weight, smp_zero, (int2)(i, gidx * 4 + 3));
    result.w += dot(v, weight3);
  }
  __local FLT4 temp[32][4];
  temp[lidx][lidy] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidy == 0 && inside) {
    result += temp[lidx][1];
    result += temp[lidx][2];
    result += temp[lidx][3];
    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));
    if (act_type == ActivationType_RELU) {
      result = max(result, (FLT4)(0.0f));
    } else if (act_type == ActivationType_RELU6) {
      result = clamp(result, (FLT4)(0.0f), (FLT4)(6.0f));
    } else if (act_type == ActivationType_TANH) {
      result = tanh(clamp(result, (FLT)(-10.0f), (FLT)(10.0f)));
    } else if (act_type == ActivationType_SIGMOID) {
      result = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-result));
    }
    WRITE_IMAGE(output, (int2)(gidx, gidz), result);
  }
}
