#pragma OPENCL EXTENSION cl_arm_printf : enable

#define SLICES 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define FLT4 float4
#define READ_FLT4 read_imagef
#define WRITE_FLT4 write_imagef
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void BiasAdd(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                      __global float *alpha, const int dim) {
  int C = input_shape.w;  // channel size

  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  for (int num = 0; num < UP_DIV(C, SLICES); ++num) {
    FLT4 in_c4 = READ_FLT4(input, smp_zero, (int2)(X * UP_DIV(C, SLICES) + num, Y));  // NHWC4: H WC
    FLT4 tmp;
    int index = 0;
    if (dim == 2) {
      index = X * 4;
    } else {
      index = num * 4;
    }
    tmp.x = in_c4.x + alpha[index];
    tmp.y = in_c4.y + alpha[index + 1];
    tmp.z = in_c4.z + alpha[index + 2];
    tmp.w = in_c4.w + alpha[index + 3];
    WRITE_FLT4(output, (int2)(X * UP_DIV(C, SLICES) + num, Y), tmp);  // NHWC4: H WC
  }
}
