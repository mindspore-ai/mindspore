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

__kernel void Sigmoid(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape, const int c4,
                      const int last_c4) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= img_shape.x || Y >= img_shape.y || c4 == 0) return;
  int C4 = X % c4;
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  if (C4 < c4 - 1) {
    in_c4 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-in_c4));
  } else {
    in_c4.x = (FLT)(1.f) / ((FLT)(1.f) + exp(-in_c4.x));
    if (last_c4 > 1) {
      in_c4.y = (FLT)(1.f) / ((FLT)(1.f) + exp(-in_c4.y));
    }
    if (last_c4 > 2) {
      in_c4.z = (FLT)(1.f) / ((FLT)(1.f) + exp(-in_c4.z));
    }
    if (last_c4 > 3) {
      in_c4.w = (FLT)(1.f) / ((FLT)(1.f) + exp(-in_c4.w));
    }
  }
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

__kernel void HSwish(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {
  int X = get_global_id(0);  // w*c
  int Y = get_global_id(1);  // n*h
  if (X >= img_shape.x || Y >= img_shape.y) return;
  FLT4 temp = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 result = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  result.x = temp.x * (temp.x <= -3 ? 0 : (temp.x >= 3 ? 1 : temp.x / 6 + 0.5f));
  result.y = temp.y * (temp.y <= -3 ? 0 : (temp.y >= 3 ? 1 : temp.y / 6 + 0.5f));
  result.z = temp.z * (temp.z <= -3 ? 0 : (temp.z >= 3 ? 1 : temp.z / 6 + 0.5f));
  result.w = temp.w * (temp.w <= -3 ? 0 : (temp.w >= 3 ? 1 : temp.w / 6 + 0.5f));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void HSigmoid(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {
  int X = get_global_id(0);  // w*c
  int Y = get_global_id(1);  // n*h
  if (X >= img_shape.x || Y >= img_shape.y) return;
  FLT4 temp = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 result = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  result.x = temp.x <= -3 ? 0 : (temp.x >= 3 ? 1 : temp.x / 6 + 0.5f);
  result.y = temp.y <= -3 ? 0 : (temp.y >= 3 ? 1 : temp.y / 6 + 0.5f);
  result.z = temp.z <= -3 ? 0 : (temp.z >= 3 ? 1 : temp.z / 6 + 0.5f);
  result.w = temp.w <= -3 ? 0 : (temp.w >= 3 ? 1 : temp.w / 6 + 0.5f);
  WRITE_IMAGE(output, (int2)(X, Y), result);
}
