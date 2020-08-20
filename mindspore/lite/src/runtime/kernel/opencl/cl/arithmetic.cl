#define divide_no_check(a, b) (a / b)
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void ElementAdd_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), a + b);
}

__kernel void ElementSub_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), a - b);
}

__kernel void ElementMul_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), a * b);
}

__kernel void ElementDiv_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), divide_no_check(a, b));
}

__kernel void BoardcastArith_IMG(__read_only image2d_t input_a, float weight, float bias, __write_only image2d_t output,
                                 const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), weight * a + bias);
}

__kernel void ElementAdd_BUF(__global float *input_a, __global float *input_b, __global float *output,
                             const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] + input_b[idx];
}

__kernel void ElementSub_BUF(__global float *input_a, __global float *input_b, __global float *output,
                             const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] - input_b[idx];
}

__kernel void ElementMul_BUF(__global float *input_a, __global float *input_b, __global float *output,
                             const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] * input_b[idx];
}

__kernel void ElementDiv_BUF(__global float *input_a, __global float *input_b, __global float *output,
                             const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] * input_b[idx];
}

__kernel void BoardcastArith_BUF(__global float *input_a, float weight, float bias, __global float *output,
                                 const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = weight * input_a[idx] + bias;
}
