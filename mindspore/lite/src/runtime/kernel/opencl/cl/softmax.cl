#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define divide_no_check(a, b) (a / b)
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void SoftmaxAxis3_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                                 const int4 input_shape) {
  int X = get_global_id(1);  // H
  int Y = get_global_id(0);  // W
  int n = get_global_id(2);  // N
  int H = input_shape.y;
  int W = input_shape.z;
  int C4 = input_shape.w;

  if (n >= input_shape.x || X >= H || Y >= W) return;

  // get max
  float4 last = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + C4 - 1, n * H + X)));
  float input_max = last.x;
  if (mask.y > 0.5f) input_max = max(input_max, last.y);
  if (mask.z > 0.5f) input_max = max(input_max, last.z);
  if (mask.w > 0.5f) input_max = max(input_max, last.w);
  for (int d = 0; d < C4 - 1; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + d, n * H + X)));
    input_max = max(input_max, t.x);
    input_max = max(input_max, t.y);
    input_max = max(input_max, t.z);
    input_max = max(input_max, t.w);
  }
  float4 input_max_f4 = (float4)(input_max, input_max, input_max, input_max);

  float sum = 0.0f;
  for (int d = 0; d < C4 - 1; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + d, n * H + X)));
    sum += dot(exp(t - input_max_f4), (float4)(1.f));
  }
  float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + C4 - 1, n * H + X)));
  sum += dot(exp(min(t - input_max_f4, (float4)(0.f))), mask);
  for (int d = 0; d < C4 - 1; ++d) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + d, n * H + X)));
    result = exp(result - input_max_f4) / sum;
    WRITE_IMAGEOUT(output, (int2)(Y * C4 + d, n * H + X), OUT_FLT4(result));
  }
  float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + C4 - 1, n * H + X)));
  result = exp(min(result - input_max_f4, (float4)(0.f))) / sum;
  result = result * mask;
  WRITE_IMAGEOUT(output, (int2)(Y * C4 + C4 - 1, n * H + X), OUT_FLT4(result));
}

__kernel void SoftmaxAxis1_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                                 const int4 input_shape) {
  int X = get_global_id(1);  // W
  int Y = get_global_id(0);  // C4
  int n = get_global_id(2);  // N
  int H = input_shape.y;
  int W = input_shape.z;
  int C4 = input_shape.w;

  if (n >= input_shape.x || X >= W || Y >= C4) return;

  // get max
  float input_max = 0.0f;
  for (int d = 0; d < H; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(X * C4 + Y, n * H + d)));
    input_max = max(input_max, t.x);
    input_max = max(input_max, t.y);
    input_max = max(input_max, t.z);
    input_max = max(input_max, t.w);
  }
  float4 input_max_f4 = (float4)(input_max, input_max, input_max, input_max);

  // get sum
  float4 sum = 0.0f;
  for (int d = 0; d < H; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(X * C4 + Y, n * H + d)));
    sum += exp(t - input_max_f4);
  }
  for (int d = 0; d < H; ++d) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(X * C4 + Y, n * H + d)));
    result = exp(result - input_max_f4) / sum;
    WRITE_IMAGEOUT(output, (int2)(X * C4 + Y, n * H + d), OUT_FLT4(result));
  }
}

__kernel void SoftmaxAxis2_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                                 const int4 input_shape) {
  int X = get_global_id(1);  // H
  int Y = get_global_id(0);  // C4
  int n = get_global_id(2);  // n
  int H = input_shape.y;
  int W = input_shape.z;
  int C4 = input_shape.w;

  if (n >= input_shape.x || X >= H || Y >= C4) return;

  // get max
  float input_max = 0.0f;
  for (int d = 0; d < W; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(d * C4 + Y, n * H + X)));
    input_max = max(input_max, t.x);
    input_max = max(input_max, t.y);
    input_max = max(input_max, t.z);
    input_max = max(input_max, t.w);
  }
  float4 input_max_f4 = (float4)(input_max, input_max, input_max, input_max);

  // get sum
  float4 sum = 0.0f;
  for (int d = 0; d < W; ++d) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(d * C4 + Y, n * H + X)));
    sum += exp(t - input_max);
  }
  for (int d = 0; d < W; ++d) {
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(d * C4 + Y, n * H + X)));
    result = exp(result - input_max) / sum;
    WRITE_IMAGEOUT(output, (int2)(d * C4 + Y, n * H + X), OUT_FLT4(result));
  }
}

__kernel void Softmax1x1_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                               const int4 input_shape) {
  int tid = get_local_id(0);
  int n = get_global_id(1);
  if (n >= input_shape.x) return;
  int C4 = input_shape.w;

  // get max
  float4 last = convert_float4(READ_IMAGE(input, smp_zero, (int2)(C4 - 1, n)));
  float input_max = last.x;
  if (mask.y > 0.5f) input_max = max(input_max, last.y);
  if (mask.z > 0.5f) input_max = max(input_max, last.z);
  if (mask.w > 0.5f) input_max = max(input_max, last.w);
  for (size_t i = tid; i < C4 - 1; i += 32) {
    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, n)));
    input_max = max(input_max, t.x);
    input_max = max(input_max, t.y);
    input_max = max(input_max, t.z);
    input_max = max(input_max, t.w);
  }
  __local float4 tmp[8];
  __local float *tmpx1 = (__local float *)tmp;
  tmpx1[tid] = input_max;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    input_max = max(input_max, tmpx1[0]);
    input_max = max(input_max, tmpx1[1]);
    input_max = max(input_max, tmpx1[2]);
    input_max = max(input_max, tmpx1[3]);
    input_max = max(input_max, tmpx1[4]);
    input_max = max(input_max, tmpx1[5]);
    input_max = max(input_max, tmpx1[6]);
    input_max = max(input_max, tmpx1[7]);
    tmpx1[0] = input_max;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  float4 input_max_f4 = (float4)(tmpx1[0], tmpx1[0], tmpx1[0], tmpx1[0]);

  // get sum
  float sum = 0.0f;
  for (size_t i = tid; i < C4 - 1; i += 32) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, n)));
    sum += dot((float4)(1.0f), exp(src - input_max_f4));
  }
  if ((C4 - 1) % 32 == tid) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(C4 - 1, n)));
    sum += dot(convert_float4(mask), exp(src - input_max_f4));
  }

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
    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, n)));
    result = exp(result - input_max_f4) * sum;
    WRITE_IMAGEOUT(output, (int2)(i, n), OUT_FLT4(result));
  }
}

__kernel void Softmax1x1_32_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                                  const int4 input_shape) {
  int n = get_global_id(1);
  if (n >= input_shape.x) return;

  int C4 = input_shape.w;

  // Calc input Max Value
  float4 input_max_vec4 = convert_float4(READ_IMAGE(input, smp_zero, (int2)(C4 - 1, n))) * mask;
  for (size_t index = 0; index < C4 - 1; index++) {
    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(index, n)));
    input_max_vec4 = max(input_max_vec4, src);
  }

  float input_max = max(input_max_vec4.x, input_max_vec4.y);
  input_max = max(input_max, input_max_vec4.z);
  input_max = max(input_max, input_max_vec4.w);
  float4 input_max_f4 = (float4)(input_max, input_max, input_max, input_max);

  // Calc input sum value
  float4 element_vec4[8];  // 8 : MAX_C4_NUM
  float4 sum_vec4 = convert_float4(mask);
  sum_vec4 *= exp(convert_float4(READ_IMAGE(input, smp_zero, (int2)(C4 - 1, n))) - input_max_f4);
  element_vec4[C4 - 1] = sum_vec4;

  for (size_t index = 0; index < C4 - 1; index++) {
    float4 src = exp(convert_float4(READ_IMAGE(input, smp_zero, (int2)(index, n))) - input_max_f4);
    sum_vec4 += src;
    element_vec4[index] = src;
  }
  float input_sum = divide_no_check(1.0f, (dot((float4)(1.0f), sum_vec4)));

  // calc softmax1x1 value
  for (size_t index = 0; index < C4; index++) {
    WRITE_IMAGEOUT(output, (int2)(index, n), OUT_FLT4(element_vec4[index] * input_sum));
  }
}
