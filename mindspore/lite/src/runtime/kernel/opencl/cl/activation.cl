#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SLICES 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define MIN(X, Y) (X < Y ? X : Y)
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void LeakyRelu_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const int4 img_shape,
                              __global FLT4 *alpha, const int4 input_shape) {
  int Y = get_global_id(0);  // H
  int X = get_global_id(1);  // W C4
  if (X >= img_shape.z || Y >= img_shape.y) return;
  int C = X % UP_DIV(input_shape.w, SLICES);
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 tmp;
  tmp.x = in_c4.x > 0.0f ? in_c4.x : in_c4.x * alpha[C].x;
  tmp.y = in_c4.y > 0.0f ? in_c4.y : in_c4.y * alpha[C].y;
  tmp.z = in_c4.z > 0.0f ? in_c4.z : in_c4.z * alpha[C].z;
  tmp.w = in_c4.w > 0.0f ? in_c4.w : in_c4.w * alpha[C].w;
  WRITE_IMAGE(output, (int2)(X, Y), tmp);
}

__kernel void LeakyRelu_NC4HW4(__read_only image2d_t input, __write_only image2d_t output, const int4 img_shape,
                               __global FLT4 *alpha, const int4 input_shape) {
  int Y = get_global_id(0);  // C4 H
  int X = get_global_id(1);  // W
  if (X >= img_shape.z || Y >= img_shape.y) return;
  int C = Y / input_shape.y;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 tmp;
  tmp.x = in_c4.x > 0.0f ? in_c4.x : in_c4.x * alpha[C].x;
  tmp.y = in_c4.y > 0.0f ? in_c4.y : in_c4.y * alpha[C].y;
  tmp.z = in_c4.z > 0.0f ? in_c4.z : in_c4.z * alpha[C].z;
  tmp.w = in_c4.w > 0.0f ? in_c4.w : in_c4.w * alpha[C].w;
  WRITE_IMAGE(output, (int2)(X, Y), tmp);
}

__kernel void Relu(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int Y = get_global_id(0);
  int X = get_global_id(1);
  if (X >= input_shape.z || Y >= input_shape.y) return;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 tmp;
  tmp.x = in_c4.x > 0.0f ? in_c4.x : 0.0f;
  tmp.y = in_c4.y > 0.0f ? in_c4.y : 0.0f;
  tmp.z = in_c4.z > 0.0f ? in_c4.z : 0.0f;
  tmp.w = in_c4.w > 0.0f ? in_c4.w : 0.0f;
  WRITE_IMAGE(output, (int2)(X, Y), tmp);
}

__kernel void Relu6(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int Y = get_global_id(0);
  int X = get_global_id(1);
  if (X >= input_shape.z || Y >= input_shape.y) return;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 tmp;
  tmp.x = in_c4.x > 0.0f ? MIN(in_c4.x, 6.0f) : 0.0f;
  tmp.y = in_c4.y > 0.0f ? MIN(in_c4.y, 6.0f) : 0.0f;
  tmp.z = in_c4.z > 0.0f ? MIN(in_c4.z, 6.0f) : 0.0f;
  tmp.w = in_c4.w > 0.0f ? MIN(in_c4.w, 6.0f) : 0.0f;
  WRITE_IMAGE(output, (int2)(X, Y), tmp);
}

__kernel void Sigmoid(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int Y = get_global_id(0);
  int X = get_global_id(1);
  if (X >= input_shape.z || Y >= input_shape.y) return;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 tmp;
  tmp.x = 1.0f / (1.0f + exp(-in_c4.x));
  tmp.y = 1.0f / (1.0f + exp(-in_c4.y));
  tmp.z = 1.0f / (1.0f + exp(-in_c4.z));
  tmp.w = 1.0f / (1.0f + exp(-in_c4.w));
  WRITE_IMAGE(output, (int2)(X, Y), tmp);
}
