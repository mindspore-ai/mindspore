#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define SLICES 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void PRelu(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                    __read_only image2d_t alpha, const int data_type, const int bias_dim) {
  int H = input_shape.y;
  int C = input_shape.w;  // channel size
  C = UP_DIV(C, SLICES);
  if (C == 0 || H == 0) {
    return;
  }
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id
  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 tmp;
  int index = 0;
  if (data_type == 1) {  // NHWC4
    index = X % C;
  } else if (data_type == 2) {  // NC4HW4
    index = Y / H;
  } else {
    return;
  }
  if (bias_dim == 1) {
    index = 0;
  }
  FLT4 weight = READ_IMAGE(alpha, smp_zero, (int2)(index, 0));
  FLT4 bias = weight;
  if (bias_dim == 1) {
    bias.y = weight.x;
    bias.z = weight.x;
    bias.w = weight.x;
  }
  tmp.x = in_c4.x > 0.0f ? in_c4.x : in_c4.x * bias.x;
  tmp.y = in_c4.y > 0.0f ? in_c4.y : in_c4.y * bias.y;
  tmp.z = in_c4.z > 0.0f ? in_c4.z : in_c4.z * bias.z;
  tmp.w = in_c4.w > 0.0f ? in_c4.w : in_c4.w * bias.w;
  WRITE_IMAGE(output, (int2)(X, Y), tmp);
}
