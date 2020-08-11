__kernel void AvgPooling2d_BUF(__global float4 *input, __global float4 *output, const int4 input_shape,
                               const int4 output_shape, const int2 stride, const int2 kernel_size, const int2 padding) {
  // axis to dst tensor coordinate
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);

  // boundary check
  if (X >= output_shape.x || Y >= output_shape.y || Z >= output_shape.w) {
    return;
  }

  float4 r = (float4)(0.0f);
  float window_size = 0.0f;
  int xs = X * stride.x + padding.x;
  int ys = Y * stride.y + padding.y;

  for (int kx = 0; kx < kernel_size.x; ++kx) {
    int x_c = xs + kx;
    bool outside_x = x_c < 0 || x_c >= input_shape.x;
    for (int ky = 0; ky < kernel_size.y; ++ky) {
      int y_c = ys + ky;
      bool outside = outside_x || y_c < 0 || y_c >= input_shape.y;
      r += !outside ? input[(input_shape.y * x_c + y_c) * output_shape.w + Z] : (float4)(0.0f);
      window_size += !outside ? 1.0f : 0.0f;
    }
  }
  float4 result = convert_float4(r / window_size);
  output[(output_shape.y * X + Y) * output_shape.w + Z] = result;
}

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void AvgPooling2d_IMG(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                               const int4 output_shape, const int2 stride, const int2 kernel_size, const int2 padding) {
  // axis to dst tensor coordinate
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);

  // boundary check
  if (X >= output_shape.x || Y >= output_shape.y || Z >= output_shape.w) {
    return;
  }

  float4 r = (float4)(0.0f);
  float window_size = 0.0f;
  int xs = X * stride.x + padding.x;
  int ys = Y * stride.y + padding.y;

  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = ys + ky;
    bool outside_y = y_c < 0 || y_c >= input_shape.y;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = xs + kx;
      bool outside = outside_y || x_c < 0 || x_c >= input_shape.x;
      r += read_imagef(input, smp_zero, (int2)(y_c * input_shape.w + Z, x_c));
      window_size += !outside ? 1.0f : 0.0f;
    }
  }
  float4 result = convert_float4(r / window_size);
  write_imagef(output, (int2)(Y * output_shape.w + Z, X), result);
}
