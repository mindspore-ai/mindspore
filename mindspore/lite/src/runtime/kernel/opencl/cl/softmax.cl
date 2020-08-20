__kernel void SoftMax_BUF(__global float4 *input, __global float4 *output, const int4 input_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int H = input_shape.x;
  int W = input_shape.y;
  int C = input_shape.z;
  int S = input_shape.w;

  if (X >= W || Y >= H) return;

  float sum = 0.0f;
  for (int d = 0; d < S; ++d) {
    float4 t = input[(Y * W + X * H) * C + d];
    sum += exp(t.x);
    if (d * 4 + 1 < C) sum += exp(t.y);
    if (d * 4 + 2 < C) sum += exp(t.z);
    if (d * 4 + 3 < C) sum += exp(t.w);
  }

  for (int d = 0; d < S; ++d) {
    float4 t = input[(Y * W + X * H) * C + d];
    t = exp(t) / sum;
    float4 result = convert_float4(t);
    output[(Y * W + X * H) * C + d] = result;
  }
}

__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void SoftMax_IMG(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= input_shape.x || Y >= input_shape.y) return;

  float sum = 0.0f;
  for (int d = 0; d < input_shape.w; ++d) {
    float4 t = read_imagef(input, smp_none, (int2)(Y * input_shape.w + d, X));
    sum += exp(t.x);
    if (d * 4 + 1 < input_shape.z) sum += exp(t.y);
    if (d * 4 + 2 < input_shape.z) sum += exp(t.z);
    if (d * 4 + 3 < input_shape.z) sum += exp(t.w);
  }

  for (int d = 0; d < input_shape.w; ++d) {
    float4 t = read_imagef(input, smp_none, (int2)(Y * input_shape.w + d, X));
    t = exp(t) / sum;
    float4 result = convert_float4(t);
    write_imagef(output, (int2)(Y * input_shape.w + d, X), result);
  }
}
