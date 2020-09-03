#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C4NUM 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void BiasAdd(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,
                      __read_only image2d_t alpha, const int data_type) {
  int H = input_shape.y;
  int C = input_shape.w;  // channel size
  C = UP_DIV(C, C4NUM);
  if ((C == 0 || H == 0) && data_type != 1) {
    return;
  }
  int Y = get_global_id(0);  // height id
  int X = get_global_id(1);  // weight id

  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));
  FLT4 tmp = in_c4;
  int index = 0;
  if (data_type == 1) {  // NC
    index = X;
  } else if (data_type == 2) {  // NHWC4
    index = X % C;
  } else {  // NC4HW4
    index = Y / H;
  }
  tmp += READ_IMAGE(alpha, smp_zero, (int2)(index, 0));
  WRITE_IMAGE(output, (int2)(X, Y), tmp);
}
