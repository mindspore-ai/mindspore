__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void ElementAdd(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                         const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  float4 a = read_imagef(input_a, smp_none, (int2)(X, Y));
  float4 b = read_imagef(input_b, smp_none, (int2)(X, Y));
  write_imagef(output, (int2)(X, Y), a + b);
}

__kernel void ElementSub(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                         const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  float4 a = read_imagef(input_a, smp_none, (int2)(X, Y));
  float4 b = read_imagef(input_b, smp_none, (int2)(X, Y));
  write_imagef(output, (int2)(X, Y), a - b);
}

__kernel void ElementMul(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                         const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  float4 a = read_imagef(input_a, smp_none, (int2)(X, Y));
  float4 b = read_imagef(input_b, smp_none, (int2)(X, Y));
  write_imagef(output, (int2)(X, Y), a * b);
}

__kernel void ElementDiv(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                         const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  float4 a = read_imagef(input_a, smp_none, (int2)(X, Y));
  float4 b = read_imagef(input_b, smp_none, (int2)(X, Y));
  if (b == 0) {
    return;
  }
  write_imagef(output, (int2)(X, Y), a / b);
}

__kernel void BoardcastArith(__read_only image2d_t input_a, float weight, float bias, __write_only image2d_t output,
                             const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  float4 a = read_imagef(input_a, smp_none, (int2)(X, Y));
  write_imagef(output, (int2)(X, Y), weight * a + bias);
}
