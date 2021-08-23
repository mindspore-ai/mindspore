__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void ElementAddInt8(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max,
                             const float4 scale, const char4 zero_point) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }
  char4 a = convert_char4(read_imagei(input_a, smp_none, (int2)(X, Y)));
  char4 b = convert_char4(read_imagei(input_b, smp_none, (int2)(X, Y)));

  float4 real_a = convert_float4(a - zero_point.x) * scale.x;
  float4 real_b = convert_float4(b - zero_point.y) * scale.y;
  int4 result = convert_int4(round((real_a + real_b) / scale.z)) + zero_point.z;
  result = clamp(result, (int)(act_min), (int)(act_max));
  write_imagei(output, (int2)(X, Y), result);
}
