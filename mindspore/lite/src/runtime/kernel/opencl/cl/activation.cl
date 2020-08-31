#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SLICES 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define MIN(X, Y) (X < Y ? X : Y)
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void LeakyRelu(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                        __global FLT *alpha) {
  int C = input_shape.w;     // channel size
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    tmp.x = in_c4.x > 0.0f ? in_c4.x : in_c4.x * alpha[0];
    tmp.y = in_c4.y > 0.0f ? in_c4.y : in_c4.y * alpha[0];
    tmp.z = in_c4.z > 0.0f ? in_c4.z : in_c4.z * alpha[0];
    tmp.w = in_c4.w > 0.0f ? in_c4.w : in_c4.w * alpha[0];
    WRITE_IMAGE(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}

__kernel void Relu(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int C = input_shape.w;     // channel size
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    tmp.x = in_c4.x > 0.0f ? in_c4.x : 0.0f;
    tmp.y = in_c4.y > 0.0f ? in_c4.y : 0.0f;
    tmp.z = in_c4.z > 0.0f ? in_c4.z : 0.0f;
    tmp.w = in_c4.w > 0.0f ? in_c4.w : 0.0f;
    WRITE_IMAGE(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}

__kernel void Relu6(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int C = input_shape.w;     // channel size
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    tmp.x = in_c4.x > 0.0f ? MIN(in_c4.x, 6.0f) : 0.0f;
    tmp.y = in_c4.y > 0.0f ? MIN(in_c4.y, 6.0f) : 0.0f;
    tmp.z = in_c4.z > 0.0f ? MIN(in_c4.z, 6.0f) : 0.0f;
    tmp.w = in_c4.w > 0.0f ? MIN(in_c4.w, 6.0f) : 0.0f;
    WRITE_IMAGE(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}

__kernel void Sigmoid(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int C = input_shape.w;     // channel size
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    tmp.x = 1.0f / (1.0f + exp(-in_c4.x));
    tmp.y = 1.0f / (1.0f + exp(-in_c4.y));
    tmp.z = 1.0f / (1.0f + exp(-in_c4.z));
    tmp.w = 1.0f / (1.0f + exp(-in_c4.w));
    WRITE_IMAGE(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}
