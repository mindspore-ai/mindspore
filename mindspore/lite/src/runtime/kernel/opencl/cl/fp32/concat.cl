// #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define FLT4 float4
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void Concat(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,
                     int2 input_channels, int4 output_shape) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  if (Z < input_channels.x) {
    FLT4 result = read_imagef(input0, smp_none, (int2)((Y)*input_channels.x + Z, (X)));
    write_imagef(output, (int2)((Y)*output_shape.w + Z, (X)), result);
  } else {
    FLT4 result = read_imagef(input1, smp_none, (int2)((Y)*input_channels.y + Z - input_channels.x, (X)));
    write_imagef(output, (int2)((Y)*output_shape.w + Z, (X)), result);
  }
}

__kernel void Concat3input(__read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,
                           __write_only image2d_t output, int3 input_channels, int4 output_shape) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  if (Z < input_channels.x) {
    FLT4 result0 = read_imagef(input0, smp_none, (int2)((Y)*input_channels.x + Z, (X)));
    write_imagef(output, (int2)((Y)*output_shape.w + Z, (X)), result0);
  } else if (Z < (input_channels.x + input_channels.y)) {
    FLT4 result1 = read_imagef(input1, smp_none, (int2)((Y)*input_channels.y + Z - input_channels.x, (X)));
    write_imagef(output, (int2)((Y)*output_shape.w + Z, (X)), result1);
  } else {
    FLT4 result2 =
      read_imagef(input2, smp_none, (int2)((Y)*input_channels.z + Z - input_channels.x - input_channels.y, (X)));
    write_imagef(output, (int2)((Y)*output_shape.w + Z, (X)), result2);
  }
}
