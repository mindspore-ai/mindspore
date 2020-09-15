#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void Cast_Fp32ToFp16_NHWC4(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  half4 result = convert_half4(READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X))));
  write_imageh(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Cast_Fp32ToFp16_NC4HW4(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  half4 result = convert_half4(READ_IMAGE(input0, smp_none, (int2)((Y), (Z * output_shape.y + X))));
  write_imageh(output, (int2)((Y), (Z * output_shape.y + X)), result);
}

__kernel void Cast_Fp16ToFp32_NHWC4(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  float4 result = convert_float4(READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X))));
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Cast_Fp16ToFp32_NC4HW4(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  float4 result = convert_float4(READ_IMAGE(input0, smp_none, (int2)((Y), (Z * output_shape.y + X))));
  WRITE_IMAGE(output, (int2)((Y), (Z * output_shape.y + X)), result);
}
