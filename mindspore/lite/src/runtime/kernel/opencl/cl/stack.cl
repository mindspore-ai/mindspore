#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define INT4 int4
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
#define CHECK_IDX_FOR_STACK                                                                 \
  int X = get_global_id(0);                                                                 \
  int Y = get_global_id(1);                                                                 \
  int Z = get_global_id(2);                                                                 \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) { \
    return;                                                                                 \
  }                                                                                         \
  FLT4 result;

__kernel void stack8inputaxis1(__read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,
                               __read_only image2d_t input3, __read_only image2d_t input4, __read_only image2d_t input5,
                               __read_only image2d_t input6, __read_only image2d_t input7,
                               __write_only image2d_t output, int4 input_shape0, int4 input_shape1, int4 input_shape2,
                               int4 input_shape3, int4 input_shape4, int4 input_shape5, int4 input_shape6,
                               int4 input_shape7, int4 output_shape) {
  CHECK_IDX_FOR_STACK;
  if (X < input_shape0.y) {
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
  } else if (X < (input_shape0.y + input_shape1.y)) {
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z, (X - input_shape0.y)));
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y)) {
    result = READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z, (X - input_shape0.y - input_shape1.y)));
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y + input_shape3.y)) {
    result = READ_IMAGE(input3, smp_none,
                        (int2)((Y)*input_shape3.w + Z, (X - input_shape0.y - input_shape1.y - input_shape2.y)));
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y + input_shape3.y + input_shape4.y)) {
    result = READ_IMAGE(
      input4, smp_none,
      (int2)((Y)*input_shape4.w + Z, (X - input_shape0.y - input_shape1.y - input_shape2.y - input_shape3.y)));
  } else if (X <
             (input_shape0.y + input_shape1.y + input_shape2.y + input_shape3.y + input_shape4.y + input_shape5.y)) {
    result = READ_IMAGE(input5, smp_none,
                        (int2)((Y)*input_shape5.w + Z, (X - input_shape0.y - input_shape1.y - input_shape2.y -
                                                        input_shape3.y - input_shape4.y)));
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y + input_shape3.y + input_shape4.y + input_shape5.y +
                  input_shape6.y)) {
    result = READ_IMAGE(input6, smp_none,
                        (int2)((Y)*input_shape6.w + Z, (X - input_shape0.y - input_shape1.y - input_shape2.y -
                                                        input_shape3.y - input_shape4.y - input_shape5.y)));
  } else {
    result =
      READ_IMAGE(input7, smp_none,
                 (int2)((Y)*input_shape7.w + Z, (X - input_shape0.y - input_shape1.y - input_shape2.y - input_shape3.y -
                                                 input_shape4.y - input_shape5.y - input_shape6.y)));
  }
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void stack8inputaxis2(__read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,
                               __read_only image2d_t input3, __read_only image2d_t input4, __read_only image2d_t input5,
                               __read_only image2d_t input6, __read_only image2d_t input7,
                               __write_only image2d_t output, int4 input_shape0, int4 input_shape1, int4 input_shape2,
                               int4 input_shape3, int4 input_shape4, int4 input_shape5, int4 input_shape6,
                               int4 input_shape7, int4 output_shape) {
  CHECK_IDX_FOR_STACK;
  if (Y < input_shape0.z) {
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
  } else if (Y < (input_shape0.z + input_shape1.z)) {
    result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z) * input_shape1.w + Z, (X)));
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z)) {
    result = READ_IMAGE(input2, smp_none, (int2)((Y - input_shape0.z - input_shape1.z) * input_shape2.w + Z, (X)));
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z + input_shape3.z)) {
    result = READ_IMAGE(input3, smp_none,
                        (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z) * input_shape3.w + Z, (X)));
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z + input_shape3.z + input_shape4.z)) {
    result = READ_IMAGE(
      input4, smp_none,
      (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z - input_shape3.z) * input_shape4.w + Z, (X)));
  } else if (Y <
             (input_shape0.z + input_shape1.z + input_shape2.z + input_shape3.z + input_shape4.z + input_shape5.z)) {
    result = READ_IMAGE(
      input5, smp_none,
      (int2)(
        (Y - input_shape0.z - input_shape1.z - input_shape2.z - input_shape3.z - input_shape4.z) * input_shape5.w + Z,
        (X)));
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z + input_shape3.z + input_shape4.z + input_shape5.z +
                  input_shape6.z)) {
    result = READ_IMAGE(
      input6, smp_none,
      (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z - input_shape3.z - input_shape4.z - input_shape5.z) *
                 input_shape6.w +
               Z,
             (X)));
  } else {
    result = READ_IMAGE(input7, smp_none,
                        (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z - input_shape3.z - input_shape4.z -
                                input_shape5.z - input_shape6.z) *
                                   input_shape7.w +
                                 Z,
                               (X)));
  }
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void stack8inputaxis3(__read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,
                               __read_only image2d_t input3, __read_only image2d_t input4, __read_only image2d_t input5,
                               __read_only image2d_t input6, __read_only image2d_t input7,
                               __write_only image2d_t output, int4 input_shape0, int4 input_shape1, int4 input_shape2,
                               int4 input_shape3, int4 input_shape4, int4 input_shape5, int4 input_shape6,
                               int4 input_shape7, int4 output_shape) {
  CHECK_IDX_FOR_STACK;
  if (Z < input_shape0.w) {
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
  } else if (Z < (input_shape0.w + input_shape1.w)) {
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z - input_shape0.w, (X)));
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w)) {
    result = READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z - input_shape0.w - input_shape1.w, (X)));
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w + input_shape3.w)) {
    result = READ_IMAGE(input3, smp_none,
                        (int2)((Y)*input_shape3.w + Z - input_shape0.w - input_shape1.w - input_shape2.w, (X)));
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w + input_shape3.w + input_shape4.w)) {
    result = READ_IMAGE(
      input4, smp_none,
      (int2)((Y)*input_shape4.w + Z - input_shape0.w - input_shape1.w - input_shape2.w - input_shape3.w, (X)));
  } else if (Z <
             (input_shape0.w + input_shape1.w + input_shape2.w + input_shape3.w + input_shape4.w + input_shape5.w)) {
    result = READ_IMAGE(input5, smp_none,
                        (int2)((Y)*input_shape5.w + Z - input_shape0.w - input_shape1.w - input_shape2.w -
                                 input_shape3.w - input_shape4.w,
                               (X)));
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w + input_shape3.w + input_shape4.w + input_shape5.w +
                  input_shape6.w)) {
    result = READ_IMAGE(input6, smp_none,
                        (int2)((Y)*input_shape6.w + Z - input_shape0.w - input_shape1.w - input_shape2.w -
                                 input_shape3.w - input_shape4.w - input_shape5.w,
                               (X)));
  } else {
    result = READ_IMAGE(input7, smp_none,
                        (int2)((Y)*input_shape7.w + Z - input_shape0.w - input_shape1.w - input_shape2.w -
                                 input_shape3.w - input_shape4.w - input_shape5.w - input_shape6.w,
                               (X)));
  }
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}
