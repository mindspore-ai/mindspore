#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define INT4 int4
#define INT2 int2
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__kernel void batch_normalization(__read_only image2d_t input, __read_only image2d_t scale,
                                  __read_only image2d_t offset, __read_only image2d_t mean,
                                  __read_only image2d_t variance, __write_only image2d_t output, const INT4 input_shape,
                                  float epsilon) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // C/4
  if (X >= input_shape.y || Y >= input_shape.z || Z >= input_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input, smp_none, (int2)((Y)*input_shape.w + Z, (X)));

  FLT4 result_mean = READ_IMAGE(mean, smp_none, (int2)((Z), (0)));
  FLT4 result_var = READ_IMAGE(variance, smp_none, (int2)((Z), (0)));
  FLT4 result_scale = READ_IMAGE(scale, smp_none, (int2)((Z), (0)));
  FLT4 result_offset = READ_IMAGE(offset, smp_none, (int2)((Z), (0)));

  result.x = result_scale.x * ((result.x - result_mean.x) / sqrt(result_var.x + epsilon)) + result_offset.x;
  result.y = result_scale.y * ((result.y - result_mean.y) / sqrt(result_var.y + epsilon)) + result_offset.y;
  result.z = result_scale.z * ((result.z - result_mean.z) / sqrt(result_var.z + epsilon)) + result_offset.z;
  result.w = result_scale.w * ((result.w - result_mean.w) / sqrt(result_var.w + epsilon)) + result_offset.w;
  WRITE_IMAGE(output, (int2)((Y)*input_shape.w + Z, (X)), result);
}
