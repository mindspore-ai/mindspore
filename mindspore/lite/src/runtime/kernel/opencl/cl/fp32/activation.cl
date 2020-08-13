#pragma OPENCL EXTENSION cl_arm_printf : enable

#define SLICES 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define FLT4 float4
#define MIN(X, Y) (X < Y ? X : Y)
#define READ_FLT4 read_imagef
#define WRITE_FLT4 write_imagef
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void ReluScalar(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                         const float alpha) {
  int C = input_shape.w;     // channel size
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_FLT4(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    tmp.x = in_c4.x >= 0 ? in_c4.x : in_c4.x * alpha;
    tmp.y = in_c4.y >= 0 ? in_c4.y : in_c4.y * alpha;
    tmp.z = in_c4.z >= 0 ? in_c4.z : in_c4.z * alpha;
    tmp.w = in_c4.w >= 0 ? in_c4.w : in_c4.w * alpha;
    WRITE_FLT4(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}

__kernel void Relu(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int C = input_shape.w;     // channel size
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_FLT4(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    tmp.x = in_c4.x >= 0 ? in_c4.x : 0;
    tmp.y = in_c4.y >= 0 ? in_c4.y : 0;
    tmp.z = in_c4.z >= 0 ? in_c4.z : 0;
    tmp.w = in_c4.w >= 0 ? in_c4.w : 0;
    WRITE_FLT4(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}

__kernel void Relu6(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int C = input_shape.w;     // channel size
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_FLT4(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    tmp.x = in_c4.x >= 0 ? MIN(in_c4.x, 6) : 0;
    tmp.y = in_c4.y >= 0 ? MIN(in_c4.y, 6) : 0;
    tmp.z = in_c4.z >= 0 ? MIN(in_c4.z, 6) : 0;
    tmp.w = in_c4.w >= 0 ? MIN(in_c4.w, 6) : 0;
    WRITE_FLT4(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}

__kernel void Sigmoid(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int C = input_shape.w;     // channel size
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_FLT4(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    tmp.x = 1 / (1 + exp(-in_c4.x));
    tmp.y = 1 / (1 + exp(-in_c4.y));
    tmp.z = 1 / (1 + exp(-in_c4.z));
    tmp.w = 1 / (1 + exp(-in_c4.w));
    WRITE_FLT4(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}
