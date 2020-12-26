#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define C4NUM 4
#define CHECK_IDX                                                                                                  \
  int X = get_global_id(0);                                                                                        \
  int Y = get_global_id(1);                                                                                        \
  int Z = get_global_id(2);                                                                                        \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w || output_shape.y == 0) { \
    return;                                                                                                        \
  }

FLT OptimizedPowerImpl(FLT x, int exponent) {
  int exp = abs(exponent);
  FLT result = 1.0f;
  FLT iterator = x;
  while (exp) {
    if (exp % 2) {
      result *= iterator;
    }
    iterator *= iterator;
    exp = exp / 2;
  }
  return exponent >= 0 ? result : 1 / result;
}

__kernel void power(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,
                    int4 output_shape, FLT4 parameter) {
  CHECK_IDX;
  int n = X / output_shape.y;
  int h = X % output_shape.y;
  FLT4 result;
  FLT4 result0 = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)));
  FLT4 result1 = READ_IMAGE(input1, smp_none, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)));

  FLT tmp_result[4];
  FLT tmp_result0[4] = {result0.x, result0.y, result0.z, result0.w};
  FLT tmp_result1[4] = {result1.x, result1.y, result1.z, result1.w};

  for (int i = 0; i < 4; ++i) {
    tmp_result0[i] = tmp_result0[i] * parameter.z + parameter.y;
    if (floor(tmp_result1[i]) == tmp_result1[i]) {
      int exponent = tmp_result1[i];
      tmp_result[i] = OptimizedPowerImpl(tmp_result0[i], exponent);
    } else {
      tmp_result[i] = pow(tmp_result0[i], tmp_result1[i]);
    }
  }
  result.x = tmp_result[0];
  result.y = tmp_result[1];
  result.z = tmp_result[2];
  result.w = tmp_result[3];
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)), result);
}

__kernel void power_broadcast(__read_only image2d_t input, __write_only image2d_t output, int4 output_shape,
                              FLT4 parameter) {
  CHECK_IDX;
  int n = X / output_shape.y;
  int h = X % output_shape.y;
  FLT4 result;
  FLT4 result0 = READ_IMAGE(input, smp_none, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)));
  FLT tmp_result0[4] = {result0.x, result0.y, result0.z, result0.w};
  FLT tmp_result[4];

  bool flag = floor(parameter.x) == parameter.x ? false : true;
  for (int i = 0; i < 4; ++i) {
    tmp_result0[i] = tmp_result0[i] * parameter.z + parameter.y;
    if (flag) {
      int exponent = parameter.x;
      tmp_result[i] = OptimizedPowerImpl(tmp_result0[i], exponent);
    } else {
      tmp_result[i] = pow(tmp_result0[i], parameter.x);
    }
  }
  result.x = tmp_result[0];
  result.y = tmp_result[1];
  result.z = tmp_result[2];
  result.w = tmp_result[3];
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)), result);
}
