#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define INT4 int4
#define C4NUM 4
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
#define CHECK_IDX_FOR_STACK                                                                 \
  int X = get_global_id(0);                                                                 \
  int Y = get_global_id(1);                                                                 \
  int Z = get_global_id(2);                                                                 \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) { \
    return;                                                                                 \
  }                                                                                         \
  FLT4 result;

// input -1D
__kernel void stack_2input_3axis_1inshape(__read_only image2d_t input0, __read_only image2d_t input1,
                                          __write_only image2d_t output, int4 input_shape, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W*C
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z * output_shape.w) {
    return;
  }
  FLT4 result1 = READ_IMAGE(input0, smp_none, (int2)(X, 0));
  FLT result1_temp[4] = {result1.x, result1.y, result1.z, result1.w};
  FLT4 result2 = READ_IMAGE(input1, smp_none, (int2)(X, 0));
  FLT result2_temp[4] = {result2.x, result2.y, result2.z, result2.w};
  for (int i = 0; i < C4NUM; ++i) {
    FLT4 result = {result1_temp[i], result2_temp[i], 0, 0};
    WRITE_IMAGE(output, (int2)(Y, (X * C4NUM + i)), result);
  }
}

// input -2D -axis = 1
__kernel void stack_2input_1axis_2inshape(__read_only image2d_t input0, __read_only image2d_t input1,
                                          __write_only image2d_t output, int4 input_shape, int4 output_shape) {
  CHECK_IDX_FOR_STACK;
  int IN = X / output_shape.y;
  int IH = X % output_shape.y;
  int boundary0 = input_shape.z;
  if (Y < boundary0) {
    int coordinate_x = Y * input_shape.w + Z;
    int coordinate_y = IN * input_shape.y + IH;
    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y));
  } else {
    int coordinate_x = (Y - boundary0) * input_shape.w + Z;
    int coordinate_y = IN * input_shape.y + IH;
    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y));
  }
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (IN * output_shape.y + IH)), result);
}

// input -3D -axis = 1
__kernel void stack_2input_1axis_3inshape(__read_only image2d_t input0, __read_only image2d_t input1,
                                          __write_only image2d_t output, int4 input_shape, int4 output_shape) {
  CHECK_IDX_FOR_STACK;
  int IN = X / output_shape.y;
  int IH = X % output_shape.y;
  int boundary0 = input_shape.y;
  if (IH < boundary0) {
    int coordinate_x = Y * input_shape.w + Z;
    int coordinate_y = IN * input_shape.y + IH;
    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y));
  } else {
    int coordinate_x = Y * input_shape.w + Z;
    int coordinate_y = IN * input_shape.y + IH - boundary0;
    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y));
  }
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (IN * output_shape.y + IH)), result);
}

// input -3D -axis = 2
__kernel void stack_2input_2axis_3inshape(__read_only image2d_t input0, __read_only image2d_t input1,
                                          __write_only image2d_t output, int4 input_shape, int4 output_shape) {
  CHECK_IDX_FOR_STACK;
  int boundary0 = input_shape.y;
  int IN = X / output_shape.y;
  int IW = X % output_shape.y;
  int IC = Z;
  int coordinate_x = IW * input_shape.w + IC;
  int coordinate_y = IN * input_shape.y;
  if (Y < boundary0) {
    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y));
  } else {
    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y));
  }
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, IN * output_shape.y + IW), result);
}

// input -3D -axis = 3  and input -2D -axis = 2  boundary stack
__kernel void stack_2input_boundary(__global float *input0, __global float *input1, __global float *output,
                                    int4 input_shape, int4 output_shape, int2 stride_w) {
  int X = get_global_id(0);  // N
  int Y = get_global_id(1);  // H
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }
  int IW = output_shape.z;
  int Align_out = output_shape.w * C4NUM;
  int Align_in = input_shape.w * C4NUM;
  int index_out = X * output_shape.y * stride_w.x + Y * stride_w.x;
  int index_in = X * input_shape.y * stride_w.y + Y * Align_in;
  for (int iw = 0; iw < IW; iw++) {
    int index_out_tmp = index_out + iw * Align_out;
    int index_in_tmp = index_in + iw;
    output[index_out_tmp] = input0[index_in_tmp];
    index_out_tmp++;
    output[index_out_tmp] = input1[index_in_tmp];
  }
}
