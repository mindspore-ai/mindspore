#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define divide_no_check(a, b) (a / b)
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void SoftMaxAxis3_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                                 const int4 input_shape) {
  int X = get_global_id(1);  // H
  int Y = get_global_id(0);  // W
  int H = input_shape.y;
  int W = input_shape.z;
  int C4 = input_shape.w;

  if (X >= H || Y >= W) return;

  // get max
  float4 last = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + C4 - 1, X)));
  float input_max = last.x;
  if (mask.y > 0.5f) input_max = max(input_max, last.y);
  if (mask.z > 0.5f) input_max = max(input_max, last.z);
  if (mask.w > 0.5f) input_max = max(input_max, last.w);
  for (int d = 0; d < C4 - 1; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + d, X)));
    input_max = max(input_max, t.x);
    input_max = max(input_max, t.y);
    input_max = max(input_max, t.z);
    input_max = max(input_max, t.w);
  }
  float4 input_max_f4 = (float4)(input_max, input_max, input_max, input_max);

  float sum = 0.0f;
  for (int d = 0; d < C4 - 1; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + d, X)));
    sum += dot(exp(t - input_max_f4), (float4)(1.f));
  }
  float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + C4 - 1, X)));
  sum += dot(exp(min(t - input_max_f4, 0)), mask);
  for (int d = 0; d < C4 - 1; ++d) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + d, X)));
    result = exp(result - input_max_f4) / sum;
    WRITE_IMAGE(output, (int2)(Y * C4 + d, X), TO_FLT4(result));
  }
  float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + C4 - 1, X)));
  result = exp(min(result - input_max_f4, 0)) / sum;
  result = result * mask;
  WRITE_IMAGE(output, (int2)(Y * C4 + C4 - 1, X), TO_FLT4(result));
}

__kernel void SoftMaxAxis1_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                                 const int4 input_shape) {
  int X = get_global_id(1);  // W
  int Y = get_global_id(0);  // C4
  int H = input_shape.y;
  int W = input_shape.z;
  int C4 = input_shape.w;

  if (X >= W || Y >= C4) return;

  float4 sum = 0.0f;
  for (int d = 0; d < H; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(X * C4 + Y, d)));
    sum += exp(t);
  }
  for (int d = 0; d < H; ++d) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(X * C4 + Y, d)));
    result = exp(result) / sum;
    WRITE_IMAGE(output, (int2)(X * C4 + Y, d), TO_FLT4(result));
  }
}

__kernel void SoftMaxAxis2_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                                 const int4 input_shape) {
  int X = get_global_id(1);  // H
  int Y = get_global_id(0);  // C4
  int H = input_shape.y;
  int W = input_shape.z;
  int C4 = input_shape.w;

  if (X >= H || Y >= C4) return;

  float4 sum = 0.0f;
  for (int d = 0; d < W; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(d * C4 + Y, X)));
    sum += exp(t);
  }
  for (int d = 0; d < W; ++d) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(d * C4 + Y, X)));
    result = exp(result) / sum;
    WRITE_IMAGE(output, (int2)(d * C4 + Y, X), TO_FLT4(result));
  }
}

__kernel void SoftMax1x1_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                               const int4 input_shape) {
  int tid = get_local_id(0);
  int C4 = input_shape.w;
  float sum = 0.0f;
  for (size_t i = tid; i < C4 - 1; i += 32) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, 0)));
    sum += dot((float4)(1.0f), exp(src));
  }
  if ((C4 - 1) % 32 == tid) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(C4 - 1, 0)));
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
  for (size_t i = tid; i < C4; i += 32) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, 0)));
    result = exp(result) * sum;
    WRITE_IMAGE(output, (int2)(i, 0), TO_FLT4(result));
  }
}
