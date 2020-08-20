__kernel void MaxPooling2d_BUF(__global float4 *input, __global float4 *output, const int4 input_shape,
                               const int4 output_shape, const int2 stride, const int2 kernel_size, const int2 padding) {
  // axis to dst tensor coordinate
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);

  // boundary check
  if (X >= output_shape.x || Y >= output_shape.y || Z >= output_shape.w) {
    return;
  }

  float4 maximum = (float4)(-10000.0f);
  int xs = X * stride.x + padding.x;
  int ys = Y * stride.y + padding.y;

  for (int kx = 0; kx < kernel_size.x; ++kx) {
    int x_c = xs + kx;
    if (x_c < 0 || x_c >= input_shape.x) {
      continue;
    }
    for (int ky = 0; ky < kernel_size.y; ++ky) {
      int y_c = ys + ky;
      if (y_c < 0 || y_c >= input_shape.y) {
        continue;
      }
      float4 src = input[(input_shape.y * x_c + y_c) * input_shape.w + Z];
      maximum = max(src, maximum);
    }
  }
  output[(output_shape.y * X + Y) * output_shape.w + Z] = maximum;
}

__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void MaxPooling2d_IMG(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                               const int4 output_shape, const int2 stride, const int2 kernel_size, const int2 padding) {
  // axis to dst tensor coordinate
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);

  // boundary check
  if (X >= output_shape.x || Y >= output_shape.y || Z >= output_shape.w) {
    return;
  }

  float4 maximum = (float4)(-10000.0f);
  int xs = X * stride.x + padding.x;
  int ys = Y * stride.y + padding.y;
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = ys + ky;
    if (y_c < 0 || y_c >= input_shape.y) continue;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = xs + kx;
      if (x_c < 0 || x_c >= input_shape.x) continue;
      float4 src = read_imagef(input, smp_none, (int2)(y_c * input_shape.w + Z, x_c));
      maximum = max(src, maximum);
    }
  }
  write_imagef(output, (int2)(Y * output_shape.w + Z, X), maximum);
}
