__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void ElementAdd(__read_only image2d_t *input_a, __read_only image2d_t *input_b, __write_only image2d_t *output,
                         const int4 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= output_shape.x || Y >= output_shape.y || Z >= output_shape.w) return;

  if (idx >= n) return;
  float4 a = read_imagef(input_a, smp_none, (int2)(X, Y * output_shape.w + Z));
  float4 b = read_imagef(input_b, smp_none, (int2)(X, Y * output_shape.w + Z));
  src = a + b;
  write_imagef(output, (int2)(0, 0), src);
}
