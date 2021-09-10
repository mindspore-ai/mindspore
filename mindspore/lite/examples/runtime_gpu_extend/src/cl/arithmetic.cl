#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void ElementAdd(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                         const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a + b;

  WRITE_IMAGE(output, (int2)(X, Y), result);
}
