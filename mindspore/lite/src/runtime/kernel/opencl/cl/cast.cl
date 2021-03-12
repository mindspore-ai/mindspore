#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void Cast_fp32_to_fp16(__read_only image2d_t input, __write_only image2d_t output, int2 XY) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= XY.x || y >= XY.y) {
    return;
  }
  half4 result = convert_half4(read_imagef(input, smp_none, (int2)(x, y)));
  write_imageh(output, (int2)(x, y), result);
}

__kernel void Cast_fp32_to_fp32(__read_only image2d_t input, __write_only image2d_t output, int2 XY) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= XY.x || y >= XY.y) {
    return;
  }
  float4 result = read_imagef(input, smp_none, (int2)(x, y));
  write_imagef(output, (int2)(x, y), result);
}

__kernel void Cast_fp16_to_fp16(__read_only image2d_t input, __write_only image2d_t output, int2 XY) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= XY.x || y >= XY.y) {
    return;
  }
  half4 result = read_imageh(input, smp_none, (int2)(x, y));
  write_imageh(output, (int2)(x, y), result);
}

__kernel void Cast_fp16_to_fp32(__read_only image2d_t input, __write_only image2d_t output, int2 XY) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= XY.x || y >= XY.y) {
    return;
  }
  float4 result = convert_float4(read_imageh(input, smp_none, (int2)(x, y)));
  write_imagef(output, (int2)(x, y), result);
}
