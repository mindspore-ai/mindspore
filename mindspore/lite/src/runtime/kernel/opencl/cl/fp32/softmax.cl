#define SLICES 4

int DivideRoundUp(int n, int div)
{
    int q = n / div;
    return n % div == 0 ? q : q + 1;
}

__kernel void SoftMax(__global float4 *input,
                      __global float4 *output,
                      const int4 input_shape) {
  int X = get_global_id(0); // width
  int Y = get_global_id(1); // height
  int H = input_shape.y;
  int W = input_shape.z;
  int C = input_shape.w;

  if (X >= W || Y >= H) return;

  float sum = 0.0f;
  for (int d = 0; d < DivideRoundUp(C, SLICES); ++d) {
    float4 t = input[(Y * W + X * H) * C + d];
    sum += exp(t.x);
    if (d * 4 + 1 < C) sum += exp(t.y);
    if (d * 4 + 2 < C) sum += exp(t.z);
    if (d * 4 + 3 < C) sum += exp(t.w);
  }

  for (int d = 0; d < DivideRoundUp(C, SLICES); ++d) {
    float4 t = input[(Y * W + X * H) * C + d];
    t = exp(t) / sum;
    float4 result = convert_float4(t);
    output[(Y * W + X * H) * C + d] = result;
  }
}