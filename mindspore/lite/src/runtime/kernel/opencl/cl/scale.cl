#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void Scale_IMG(__read_only image2d_t input, __read_only image2d_t scale, __read_only image2d_t offset,
                        __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 in = read_imagef(input, smp_none, (int2)(X, Y));
  FLT4 s = read_imagef(scale, smp_none, (int2)(X, Y));
  FLT4 o = read_imagef(offset, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), in * s + o);
}

__kernel void BoardcastScale_IMG(__read_only image2d_t input, float scale, float offset, __write_only image2d_t output,
                                 const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 in = read_imagef(input, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), in * (FLT)scale + (FLT)offset);
}

__kernel void Scale_C_IMG(__read_only image2d_t input, __read_only image2d_t scale, __read_only image2d_t offset,
                          __write_only image2d_t output, const int2 output_shape, const int C) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 in = read_imagef(input, smp_none, (int2)(X, Y));
  FLT4 s = read_imagef(scale, smp_none, (int2)(X % C, 0));
  FLT4 o = read_imagef(offset, smp_none, (int2)(X % C, 0));
  WRITE_IMAGE(output, (int2)(X, Y), in * s + o);
}
