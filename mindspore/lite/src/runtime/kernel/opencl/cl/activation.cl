#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void LeakyRelu(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape,
                        const float alpha) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= img_shape.x || Y >= img_shape.y) return;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 tmp;
  FLT alpha_f = TO_FLT(alpha);
  tmp.x = in_c4.x > 0.0f ? in_c4.x : in_c4.x * alpha_f;
  tmp.y = in_c4.y > 0.0f ? in_c4.y : in_c4.y * alpha_f;
  tmp.z = in_c4.z > 0.0f ? in_c4.z : in_c4.z * alpha_f;
  tmp.w = in_c4.w > 0.0f ? in_c4.w : in_c4.w * alpha_f;
  WRITE_IMAGE(output, (int2)(X, Y), tmp);
}

__kernel void Relu(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= img_shape.x || Y >= img_shape.y) return;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  in_c4 = max(in_c4, (FLT)(0.f));
  WRITE_IMAGE(output, (int2)(X, Y), in_c4);
}

__kernel void Relu6(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= img_shape.x || Y >= img_shape.y) return;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  in_c4 = clamp(in_c4, (FLT)(0.f), (FLT)(6.f));
  WRITE_IMAGE(output, (int2)(X, Y), in_c4);
}

__kernel void Sigmoid(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= img_shape.x || Y >= img_shape.y) return;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  in_c4 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-in_c4));
  WRITE_IMAGE(output, (int2)(X, Y), in_c4);
}

__kernel void Tanh(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= img_shape.x || Y >= img_shape.y) return;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 exp0 = exp(in_c4);
  FLT4 exp1 = exp(-in_c4);
  in_c4 = (exp0 - exp1) / (exp0 + exp1);
  WRITE_IMAGE(output, (int2)(X, Y), in_c4);
}

__kernel void Swish(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= img_shape.x || Y >= img_shape.y) return;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  in_c4 = in_c4 * ((FLT4)(1.f) / ((FLT4)(1.f) + exp(-in_c4)));
  WRITE_IMAGE(output, (int2)(X, Y), in_c4);
}
