#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C4NUM 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void MatMul_Strassen_NHWC4_2d(__read_only image2d_t input, __write_only image2d_t output, __global FLT *weight,
                                       int4 in_shape, int4 out_shape) {
  int gidx = get_global_id(0);  // CO4
  int gidz = get_global_id(2);  // N
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int co4 = UP_DIV(out_shape.w, C4NUM);
  int weight_stride = in_shape.w;
  FLT sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  bool inside = gidx < co4 && gidz < weight_stride;
  for (uint i = lidy; i < ci4 && inside; i += 4) {
    FLT4 result_in = READ_IMAGE(input, smp_zero, (int2)(i, gidz));
    int index_x = (i * C4NUM) * weight_stride + gidx * C4NUM;
    int index_y = index_x + weight_stride;
    int index_z = index_y + weight_stride;
    int index_w = index_z + weight_stride;
    for (int j = 0; j < C4NUM; ++j) {
      FLT4 result_weight = {weight[index_x + j], weight[index_y + j], weight[index_z + j], weight[index_w + j]};
      sum[j] += dot(result_in, result_weight);
    }
  }
  FLT4 result = {sum[0], sum[1], sum[2], sum[3]};
  __local FLT4 temp[32][4];
  temp[lidx][lidy] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidy == 0 && inside) {
    result += temp[lidx][1];
    result += temp[lidx][2];
    result += temp[lidx][3];
    WRITE_IMAGE(output, (int2)(gidx, gidz), result);
  }
}

// flag = 0 : represent add, otherwise, sub
__kernel void MatMul_BUF_Add_Sub_2(__global FLT4 *input, __global FLT4 *output, int4 shape, int4 offset, int flag) {
  int gidy = get_global_id(0);  // W*C4
  int gidx = get_global_id(2);  // N*H
  if (gidx >= shape.x * shape.y || gidy >= shape.z * shape.w) {
    return;
  }
  int ci_co_4 = shape.w;
  const int origin_shape = 2 * ci_co_4;
  int index_1 = (gidx + offset.x) * origin_shape + gidy + offset.y;
  int index_2 = (gidx + offset.z) * origin_shape + gidy + offset.w;
  FLT4 result1 = input[index_1];
  FLT4 result2 = input[index_2];
  FLT4 result;
  if (flag == 0) {
    result = result1 + result2;
  } else {
    result = result1 - result2;
  }
  output[gidx * ci_co_4 + gidy] = result;
}

__kernel void MatMul_IMG_Add_Sub_2(__read_only image2d_t input, __write_only image2d_t output, int4 shape, int4 offset,
                                   int flag) {
  int gidy = get_global_id(0);  // W*C4
  int gidx = get_global_id(2);  // N*H
  if (gidx >= shape.x * shape.y || gidy >= shape.z * shape.w) {
    return;
  }
  FLT4 result1 = READ_IMAGE(input, smp_zero, (int2)(gidy + offset.y, gidx + offset.x));
  FLT4 result2 = READ_IMAGE(input, smp_zero, (int2)(gidy + offset.w, gidx + offset.z));
  FLT4 result;
  if (flag == 0) {
    result = result1 + result2;
  } else {
    result = result1 - result2;
  }
  WRITE_IMAGE(output, (int2)(gidy, gidx), result);
}

__kernel void Strassen_Back_Result(__read_only image2d_t input1, __read_only image2d_t input2,
                                   __read_only image2d_t input3, __read_only image2d_t input4,
                                   __read_only image2d_t input5, __read_only image2d_t input6,
                                   __read_only image2d_t input7, __write_only image2d_t output, int4 shape) {
  int gidy = get_global_id(0);  // W*C4
  int gidx = get_global_id(2);  // N*H
  int offset_x = shape.x * shape.y, offset_y = shape.z * shape.w;
  if (gidx >= offset_x || gidy >= offset_y) {
    return;
  }
  FLT4 result_M1 = READ_IMAGE(input1, smp_zero, (int2)(gidy, gidx));
  FLT4 result_M2 = READ_IMAGE(input2, smp_zero, (int2)(gidy, gidx));
  FLT4 result_M3 = READ_IMAGE(input3, smp_zero, (int2)(gidy, gidx));
  FLT4 result_M4 = READ_IMAGE(input4, smp_zero, (int2)(gidy, gidx));
  FLT4 result_M5 = READ_IMAGE(input5, smp_zero, (int2)(gidy, gidx));
  FLT4 result_M6 = READ_IMAGE(input6, smp_zero, (int2)(gidy, gidx));
  FLT4 result_M7 = READ_IMAGE(input7, smp_zero, (int2)(gidy, gidx));
  FLT4 result1 = result_M4 + result_M5 + result_M6 - result_M2;  // C11
  FLT4 result2 = result_M1 + result_M2;                          // C12
  FLT4 result3 = result_M3 + result_M4;                          // C21
  FLT4 result4 = result_M1 + result_M5 - result_M3 - result_M7;  // C22
  WRITE_IMAGE(output, (int2)(gidy, gidx), result1);
  WRITE_IMAGE(output, (int2)(gidy + offset_y, gidx), result2);
  WRITE_IMAGE(output, (int2)(gidy, gidx + offset_x), result3);
  WRITE_IMAGE(output, (int2)(gidy + offset_y, gidx + offset_x), result4);
}

__kernel void MatMul_IMG_Filled(__read_only image2d_t input, __write_only image2d_t output, int4 shape, int2 offset) {
  int gidy = get_global_id(0);  // W*C4
  int gidx = get_global_id(2);  // N*H
  if (gidx >= shape.x * shape.y || gidy >= shape.z * shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input, smp_zero, (int2)(gidy + offset.y, gidx + offset.x));
  WRITE_IMAGE(output, (int2)(gidy, gidx), result);
}

__kernel void MatMul_BUF_Filled(__global FLT4 *input, __global FLT4 *output, int4 shape, int2 offset) {
  int gidy = get_global_id(0);  // W*C4
  int gidx = get_global_id(2);  // N*H
  if (gidx >= shape.x * shape.y || gidy >= shape.z * shape.w) {
    return;
  }
  int stride_out = shape.z * shape.w;
  int index_out = gidx * stride_out + gidy;
  const int stride_origin = 2 * stride_out;
  int index_in = (gidx + offset.x) * stride_origin + gidy + offset.y;
  FLT4 result = input[index_in];
  output[index_out] = result;
}
