#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define SLICES 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void PRelu(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                    __read_only image2d_t alpha, const int dim) {
  int C = input_shape.w;  // channel size

  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    if (dim == 1) {
      FLT4 weight = READ_IMAGE(alpha, smp_zero, (int2)(0, 0));
      tmp.x = in_c4.x >= 0 ? in_c4.x : in_c4.x * weight.x;
      tmp.y = in_c4.y >= 0 ? in_c4.y : in_c4.y * weight.x;
      tmp.z = in_c4.z >= 0 ? in_c4.z : in_c4.z * weight.x;
      tmp.w = in_c4.w >= 0 ? in_c4.w : in_c4.w * weight.x;
    } else {
      FLT4 weight = READ_IMAGE(alpha, smp_zero, (int2)(num, 0));
      tmp.x = in_c4.x >= 0 ? in_c4.x : in_c4.x * weight.x;
      tmp.y = in_c4.y >= 0 ? in_c4.y : in_c4.y * weight.y;
      tmp.z = in_c4.z >= 0 ? in_c4.z : in_c4.z * weight.z;
      tmp.w = in_c4.w >= 0 ? in_c4.w : in_c4.w * weight.w;
    }
    WRITE_IMAGE(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}
