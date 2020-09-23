#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define divide_no_check(a, b) (a / b)
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void SoftMax_NHWC4_BUF(__read_only image2d_t input, __global FLT4 *output, const int4 input_shape) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int H = input_shape.x;
  int W = input_shape.y;
  int C = input_shape.z;
  int S = input_shape.w;

  if (X >= H || Y >= W) return;

  float sum = 0.0f;
  for (int d = 0; d < S; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * S + d, X)));
    sum += exp(t.x);
    if (d * 4 + 1 < C) sum += exp(t.y);
    if (d * 4 + 2 < C) sum += exp(t.z);
    if (d * 4 + 3 < C) sum += exp(t.w);
  }

  for (int d = 0; d < S; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * S + d, X)));
    t = divide_no_check(exp(t), sum);
    __global FLT *output_flt = (__global FLT *)output;
    output_flt += (X * W + Y) * C + 4 * d;
    FLT4 result = TO_FLT4(t);
    output_flt[0] = result.x;
    if (d * 4 + 1 < C) output_flt[1] += result.y;
    if (d * 4 + 2 < C) output_flt[2] += result.z;
    if (d * 4 + 3 < C) output_flt[3] += result.w;
  }
}

__kernel void SoftMax_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int H = input_shape.x;
  int W = input_shape.y;
  int C = input_shape.z;
  int S = input_shape.w;

  if (X >= H || Y >= W) return;

  float sum = 0.0f;
  for (int d = 0; d < S; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * S + d, X)));
    sum += exp(t.x);
    if (d * 4 + 1 < C) sum += exp(t.y);
    if (d * 4 + 2 < C) sum += exp(t.z);
    if (d * 4 + 3 < C) sum += exp(t.w);
  }

  for (int d = 0; d < S; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * S + d, X)));
    t = exp(t) / sum;
    WRITE_IMAGE(output, (int2)(Y * S + d, X), TO_FLT4(t));
  }
}

__kernel void SoftMax_NC4HW4_BUF(__read_only image2d_t input, __global FLT4 *output, const int4 input_shape) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int H = input_shape.x;
  int W = input_shape.y;
  int C = input_shape.z;
  int S = input_shape.w;

  if (X >= H || Y >= W) return;

  float sum = 0.0f;
  for (int d = 0; d < S; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y, d * H + X)));
    sum += exp(t.x);
    if (d * 4 + 1 < C) sum += exp(t.y);
    if (d * 4 + 2 < C) sum += exp(t.z);
    if (d * 4 + 3 < C) sum += exp(t.w);
  }

  for (int d = 0; d < S; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y, d * H + X)));
    t = divide_no_check(exp(t), sum);
    __global FLT *output_flt = (__global FLT *)output;
    output_flt += (X * W + Y) * C + 4 * d;
    FLT4 result = TO_FLT4(t);
    output_flt[0] = result.x;
    if (d * 4 + 1 < C) output_flt[1] += result.y;
    if (d * 4 + 2 < C) output_flt[2] += result.z;
    if (d * 4 + 3 < C) output_flt[3] += result.w;
  }
}

__kernel void SoftMax_NC4HW4_IMG(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int H = input_shape.x;
  int W = input_shape.y;
  int C = input_shape.z;
  int S = input_shape.w;

  if (X >= H || Y >= W) return;

  float sum = 0.0f;
  for (int d = 0; d < S; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y, d * H + X)));
    sum += exp(t.x);
    if (d * 4 + 1 < C) sum += exp(t.y);
    if (d * 4 + 2 < C) sum += exp(t.z);
    if (d * 4 + 3 < C) sum += exp(t.w);
  }

  for (int d = 0; d < S; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y, d * H + X)));
    t = exp(t) / sum;
    WRITE_IMAGE(output, (int2)(Y, d * H + X), TO_FLT4(t));
  }
}

__kernel void SoftMax1x1_NHWC4_BUF(__read_only image2d_t input, __global FLT4 *output, const float4 mask,
                                   const int slices, const int slices_x32) {
  int tid = get_local_id(0);
  float sum = 0.0f;
  for (size_t i = tid; i < slices - 1; i += 32) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, 0)));
    sum += dot((float4)(1.0f), exp(src));
  }
  if ((slices - 1) % 32 == tid) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(slices - 1, 0)));
    sum += dot(convert_float4(mask), exp(src));
  }

  __local float4 tmp[8];
  __local float *tmpx1 = (__local float *)tmp;
  tmpx1[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    sum = dot((float4)(1.0f), tmp[0]);
    sum += dot((float4)(1.0f), tmp[1]);
    sum += dot((float4)(1.0f), tmp[2]);
    sum += dot((float4)(1.0f), tmp[3]);
    sum += dot((float4)(1.0f), tmp[4]);
    sum += dot((float4)(1.0f), tmp[5]);
    sum += dot((float4)(1.0f), tmp[6]);
    sum += dot((float4)(1.0f), tmp[7]);
    tmpx1[0] = divide_no_check(1.0f, sum);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  sum = tmpx1[0];
  for (size_t i = tid; i < slices - 1; i += 32) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, 0)));
    result = exp(result) * sum;
    output[i] = TO_FLT4(result);
  }
  if ((slices - 1) % 32 == tid) {
    float4 result_float = convert_float4(READ_IMAGE(input, smp_zero, (int2)(slices - 1, 0)));
    result_float = exp(result_float) * sum;
    FLT4 result = TO_FLT4(result_float);
    __global FLT4 *remain_ptr4 = output;
    remain_ptr4 += slices - 1;
    __global FLT *remain_ptr = (__global FLT *)remain_ptr4;
    remain_ptr[0] = result.x;
    if (mask.y > 0.f) {
      remain_ptr[1] = result.y;
    }
    if (mask.z > 0.f) {
      remain_ptr[2] = result.z;
    }
    if (mask.w > 0.f) {
      remain_ptr[3] = result.w;
    }
  }
}

__kernel void SoftMax1x1_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                                   const int slices, const int slices_x32) {
  int tid = get_local_id(0);
  float sum = 0.0f;
  for (size_t i = tid; i < slices - 1; i += 32) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, 0)));
    sum += dot((float4)(1.0f), exp(src));
  }
  if ((slices - 1) % 32 == tid) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(slices - 1, 0)));

    sum += dot(convert_float4(mask), exp(src));
  }

  __local float4 tmp[8];
  __local float *tmpx1 = (__local float *)tmp;
  tmpx1[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    sum = dot((float4)(1.0f), tmp[0]);
    sum += dot((float4)(1.0f), tmp[1]);
    sum += dot((float4)(1.0f), tmp[2]);
    sum += dot((float4)(1.0f), tmp[3]);
    sum += dot((float4)(1.0f), tmp[4]);
    sum += dot((float4)(1.0f), tmp[5]);
    sum += dot((float4)(1.0f), tmp[6]);
    sum += dot((float4)(1.0f), tmp[7]);
    tmpx1[0] = divide_no_check(1.0f, sum);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  sum = tmpx1[0];
  for (size_t i = tid; i < slices; i += 32) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, 0)));
    result = exp(result) * sum;
    WRITE_IMAGE(output, (int2)(i, 0), TO_FLT4(result));
  }
}

__kernel void SoftMax1x1_NC4HW4_BUF(__read_only image2d_t input, __global FLT4 *output, const float4 mask,
                                    const int slices, const int slices_x32) {
  int tid = get_local_id(0);
  float sum = 0.0f;
  for (size_t i = tid; i < slices - 1; i += 32) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(0, i)));
    sum += dot((float4)(1.0f), exp(src));
  }
  if ((slices - 1) % 32 == tid) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(0, slices - 1)));

    sum += dot(convert_float4(mask), exp(src));
  }

  __local float4 tmp[8];
  __local float *tmpx1 = (__local float *)tmp;
  tmpx1[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    sum = dot((float4)(1.0f), tmp[0]);
    sum += dot((float4)(1.0f), tmp[1]);
    sum += dot((float4)(1.0f), tmp[2]);
    sum += dot((float4)(1.0f), tmp[3]);
    sum += dot((float4)(1.0f), tmp[4]);
    sum += dot((float4)(1.0f), tmp[5]);
    sum += dot((float4)(1.0f), tmp[6]);
    sum += dot((float4)(1.0f), tmp[7]);
    tmpx1[0] = divide_no_check(1.0f, sum);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  sum = tmpx1[0];
  for (size_t i = tid; i < slices - 1; i += 32) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(0, i)));
    result = exp(result) * sum;
    output[i] = TO_FLT4(result);
  }
  if ((slices - 1) % 32 == tid) {
    float4 result_float = convert_float4(READ_IMAGE(input, smp_zero, (int2)(0, slices - 1)));
    result_float = exp(result_float) * sum;
    FLT4 result = TO_FLT4(result_float);
    __global FLT4 *remain_ptr4 = output;
    remain_ptr4 += slices - 1;
    __global FLT *remain_ptr = (__global FLT *)remain_ptr4;
    remain_ptr[0] = result.x;
    if (mask.y > 0.f) {
      remain_ptr[1] = result.y;
    }
    if (mask.z > 0.f) {
      remain_ptr[2] = result.z;
    }
    if (mask.w > 0.f) {
      remain_ptr[3] = result.w;
    }
  }
}

__kernel void SoftMax1x1_NC4HW4_IMG(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                                    const int slices, const int slices_x32) {
  int tid = get_local_id(0);
  float sum = 0.0f;
  for (size_t i = tid; i < slices - 1; i += 32) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(0, i)));
    sum += dot((float4)(1.0f), exp(src));
  }
  if ((slices - 1) % 32 == tid) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(0, slices - 1)));
    sum += dot(convert_float4(mask), exp(src));
  }

  __local float4 tmp[8];
  __local float *tmpx1 = (__local float *)tmp;
  tmpx1[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    sum = dot((float4)(1.0f), tmp[0]);
    sum += dot((float4)(1.0f), tmp[1]);
    sum += dot((float4)(1.0f), tmp[2]);
    sum += dot((float4)(1.0f), tmp[3]);
    sum += dot((float4)(1.0f), tmp[4]);
    sum += dot((float4)(1.0f), tmp[5]);
    sum += dot((float4)(1.0f), tmp[6]);
    sum += dot((float4)(1.0f), tmp[7]);
    tmpx1[0] = divide_no_check(1.0f, sum);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  sum = tmpx1[0];
  for (size_t i = tid; i < slices; i += 32) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(0, i)));
    result = exp(result) * sum;
    WRITE_IMAGE(output, (int2)(0, i), TO_FLT4(result));
  }
}
