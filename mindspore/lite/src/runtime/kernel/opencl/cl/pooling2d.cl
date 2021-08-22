#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define divide_no_check(a, b) (a / b)
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void AvgPooling2d_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                                     const int4 output_shape, const int2 stride, const int2 kernel_size,
                                     const int2 padding) {
  // axis to dst tensor coordinate
  int X = get_global_id(2);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(0);  // C4
  int N = X / output_shape.y;
  X = X % output_shape.y;
  // boundary check
  if (N >= output_shape.x || X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }

  FLT4 r = (FLT4)(0.0f);
  FLT window_size = 0.0f;
  int xs = X * stride.x - padding.x;
  int ys = Y * stride.y - padding.y;

  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = ys + ky;
    bool outside_y = y_c < 0 || y_c >= input_shape.z;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = xs + kx;
      bool outside = outside_y || x_c < 0 || x_c >= input_shape.y;
      r +=
        !outside ? READ_IMAGE(input, smp_zero, (int2)(y_c * input_shape.w + Z, N * input_shape.y + x_c)) : (FLT4)(0.0f);
      window_size += !outside ? 1.0f : 0.0f;
    }
  }
  FLT4 result = TO_FLT4(divide_no_check(r, window_size));
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + Z, N * output_shape.y + X), result);
}

__kernel void AvgPooling2d_ReLU_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output,
                                          const int4 input_shape, const int4 output_shape, const int2 stride,
                                          const int2 kernel_size, const int2 padding) {
  // axis to dst tensor coordinate
  int X = get_global_id(2);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(0);  // C4
  int N = X / output_shape.y;
  X = X % output_shape.y;
  // boundary check
  if (N >= output_shape.x || X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }

  FLT4 r = (FLT4)(0.0f);
  FLT window_size = 0.0f;
  int xs = X * stride.x - padding.x;
  int ys = Y * stride.y - padding.y;

  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = ys + ky;
    bool outside_y = y_c < 0 || y_c >= input_shape.z;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = xs + kx;
      bool outside = outside_y || x_c < 0 || x_c >= input_shape.y;
      r +=
        !outside ? READ_IMAGE(input, smp_zero, (int2)(y_c * input_shape.w + Z, N * input_shape.y + x_c)) : (FLT4)(0.0f);
      window_size += !outside ? 1.0f : 0.0f;
    }
  }
  FLT4 result = TO_FLT4(divide_no_check(r, window_size));
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + Z, N * output_shape.y + X), max(result, (FLT4)(0.f)));
}

__kernel void MaxPooling2d_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                                     const int4 output_shape, const int2 stride, const int2 kernel_size,
                                     const int2 padding) {
  // axis to dst tensor coordinate
  int X = get_global_id(2);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(0);  // C4
  int N = X / output_shape.y;
  X = X % output_shape.y;
  // boundary check
  if (N >= output_shape.x || X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }

  FLT4 maximum = (FLT4)(-10000.0f);
  int xs = X * stride.x - padding.x;
  int ys = Y * stride.y - padding.y;
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = ys + ky;
    if (y_c < 0 || y_c >= input_shape.z) continue;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = xs + kx;
      if (x_c < 0 || x_c >= input_shape.y) continue;
      FLT4 src = READ_IMAGE(input, smp_zero, (int2)(y_c * input_shape.w + Z, N * input_shape.y + x_c));
      maximum = max(src, maximum);
    }
  }
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + Z, N * output_shape.y + X), maximum);
}

__kernel void MaxPooling2d_ReLU_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output,
                                          const int4 input_shape, const int4 output_shape, const int2 stride,
                                          const int2 kernel_size, const int2 padding) {
  // axis to dst tensor coordinate
  int X = get_global_id(2);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(0);  // C4
  int N = X / output_shape.y;
  X = X % output_shape.y;
  // boundary check
  if (N >= output_shape.x || X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }

  FLT4 maximum = (FLT4)(-10000.0f);
  int xs = X * stride.x - padding.x;
  int ys = Y * stride.y - padding.y;
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = ys + ky;
    if (y_c < 0 || y_c >= input_shape.z) continue;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = xs + kx;
      if (x_c < 0 || x_c >= input_shape.y) continue;
      FLT4 src = READ_IMAGE(input, smp_zero, (int2)(y_c * input_shape.w + Z, N * input_shape.y + x_c));
      maximum = max(src, maximum);
    }
  }
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + Z, N * output_shape.y + X), max(maximum, (FLT4)(0.f)));
}
