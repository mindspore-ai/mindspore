#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define INT4 int4
#define INT2 int2
#define C4NUM 4
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__kernel void Batch_normalization_NHWC4(__read_only image2d_t input, __global FLT *scale, __global FLT *offset,
                                        __global FLT *mean, __global FLT *variance, __write_only image2d_t output,
                                        const INT4 input_shape, float epsilon, int unalign_input_w) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // C/4
  if (X >= input_shape.y || Y >= input_shape.z || Z >= input_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input, smp_none, (int2)((Y)*input_shape.w + Z, (X)));

  FLT result_mean[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  FLT result_var[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  FLT result_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  FLT result_offset[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if ((Z + 1) * C4NUM <= unalign_input_w) {
    for (int i = 0; i < C4NUM; ++i) {
      result_mean[i] = mean[Z * C4NUM + i];
      result_var[i] = variance[Z * C4NUM + i];
      result_scale[i] = scale[Z * C4NUM + i];
      result_offset[i] = offset[Z * C4NUM + i];
    }
  } else {
    for (int i = 0; i < unalign_input_w % C4NUM; ++i) {
      result_mean[i] = mean[Z * C4NUM + i];
      result_var[i] = variance[Z * C4NUM + i];
      result_scale[i] = scale[Z * C4NUM + i];
      result_offset[i] = offset[Z * C4NUM + i];
    }
  }
  result.x = result_scale[0] * ((result.x - result_mean[0]) / sqrt(result_var[0] + epsilon)) + result_offset[0];
  result.y = result_scale[1] * ((result.y - result_mean[1]) / sqrt(result_var[1] + epsilon)) + result_offset[1];
  result.z = result_scale[2] * ((result.z - result_mean[2]) / sqrt(result_var[2] + epsilon)) + result_offset[2];
  result.w = result_scale[3] * ((result.w - result_mean[3]) / sqrt(result_var[3] + epsilon)) + result_offset[3];
  WRITE_IMAGE(output, (int2)((Y)*input_shape.w + Z, (X)), result);
}
