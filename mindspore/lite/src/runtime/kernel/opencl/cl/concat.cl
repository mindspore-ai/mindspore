#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#define CHECK_IDXConcat2input_NHWC4                                                         \
  int X = get_global_id(0);                                                                 \
  int Y = get_global_id(1);                                                                 \
  int Z = get_global_id(2);                                                                 \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) { \
    return;                                                                                 \
  }                                                                                         \
  FLT4 result;

#define DOConcat2inputaxis1_NHWC4                                                                \
  if (X < input_shape0.y) {                                                                      \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                  \
  } else {                                                                                       \
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z, (X - input_shape0.y))); \
  }                                                                                              \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define DOConcat2inputaxis2_NHWC4                                                                  \
  if (Y < input_shape0.z) {                                                                        \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                    \
  } else {                                                                                         \
    result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z) * input_shape1.w + Z, (X))); \
  }                                                                                                \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define DOConcat2inputaxis3_NHWC4                                                                \
  if (Z < input_shape0.w) {                                                                      \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                  \
  } else {                                                                                       \
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z - input_shape0.w, (X))); \
  }                                                                                              \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define CHECK_IDXConcat2input_NC4HW4                                                                                \
  int X = get_global_id(0);                                                                                         \
  int Y = get_global_id(1);                                                                                         \
  int Z = get_global_id(2);                                                                                         \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {                         \
    return;                                                                                                         \
  }                                                                                                                 \
  if (input_shape0.y == 0 || input_shape1.y == 0 || output_shape.y == 0) {                                          \
    return;                                                                                                         \
  }                                                                                                                 \
  int in_postion_x;                                                                                                 \
  int out_pos_x = (X / output_shape.y) * output_shape.w * output_shape.y + Z * output_shape.y + X % output_shape.y; \
  FLT4 result;

#define DOConcat2inputaxis1_NC4HW4                                                                                   \
  if (X < input_shape0.y) {                                                                                          \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y; \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                \
  } else {                                                                                                           \
    in_postion_x = ((X - input_shape0.y) / input_shape1.y) * input_shape1.w * input_shape1.y + Z * input_shape1.y +  \
                   ((X - input_shape0.y) % input_shape1.y);                                                          \
    result = READ_IMAGE(input1, smp_none, (int2)((Y), in_postion_x));                                                \
  }                                                                                                                  \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define DOConcat2inputaxis2_NC4HW4                                                                                     \
  if (Y < input_shape0.z) {                                                                                            \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y;   \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                  \
  } else {                                                                                                             \
    in_postion_x = (X / input_shape1.y) * input_shape1.w * input_shape1.y + Z * input_shape1.y + (X % input_shape1.y); \
    result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z), in_postion_x));                                 \
  }                                                                                                                    \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define DOConcat2inputaxis3_NC4HW4                                                                                   \
  if (Z < input_shape0.w) {                                                                                          \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y; \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                \
  } else {                                                                                                           \
    in_postion_x = (X / input_shape1.y) * input_shape1.w * input_shape1.y + (Z - input_shape0.w) * input_shape1.y +  \
                   (X % input_shape1.y);                                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y), in_postion_x));                                                \
  }                                                                                                                  \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define CHECK_IDXConcat3input_NC4HW4                                                                                \
  int X = get_global_id(0);                                                                                         \
  int Y = get_global_id(1);                                                                                         \
  int Z = get_global_id(2);                                                                                         \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {                         \
    return;                                                                                                         \
  }                                                                                                                 \
  if (input_shape0.y == 0 || input_shape1.y == 0 || input_shape2.y == 0 || output_shape.y == 0) {                   \
    return;                                                                                                         \
  }                                                                                                                 \
  int in_postion_x;                                                                                                 \
  int out_pos_x = (X / output_shape.y) * output_shape.w * output_shape.y + Z * output_shape.y + X % output_shape.y; \
  FLT4 result;

#define DOConcat3inputaxis1_NC4HW4                                                                                   \
  if (X < input_shape0.y) {                                                                                          \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y; \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (X < input_shape0.y + input_shape1.y) {                                                                  \
    in_postion_x = ((X - input_shape0.y) / input_shape1.y) * input_shape1.w * input_shape1.y + Z * input_shape1.y +  \
                   ((X - input_shape0.y) % input_shape1.y);                                                          \
    result = READ_IMAGE(input1, smp_none, (int2)((Y), in_postion_x));                                                \
  } else {                                                                                                           \
    in_postion_x = ((X - input_shape0.y - input_shape1.y) / input_shape2.y) * input_shape2.w * input_shape2.y +      \
                   Z * input_shape2.y + ((X - input_shape0.y - input_shape1.y) % input_shape2.y);                    \
    result = READ_IMAGE(input2, smp_none, (int2)((Y), in_postion_x));                                                \
  }                                                                                                                  \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define DOConcat3inputaxis2_NC4HW4                                                                                     \
  if (Y < input_shape0.z) {                                                                                            \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y;   \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                  \
  } else if (Y < input_shape0.z + input_shape1.z) {                                                                    \
    in_postion_x = (X / input_shape1.y) * input_shape1.w * input_shape1.y + Z * input_shape1.y + (X % input_shape1.y); \
    result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z), in_postion_x));                                 \
  } else {                                                                                                             \
    in_postion_x = (X / input_shape2.y) * input_shape2.w * input_shape2.y + Z * input_shape2.y + (X % input_shape2.y); \
    result = READ_IMAGE(input2, smp_none, (int2)((Y - input_shape0.z - input_shape1.z), in_postion_x));                \
  }                                                                                                                    \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define DOConcat3inputaxis3_NC4HW4                                                                                   \
  if (Z < input_shape0.w) {                                                                                          \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y; \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (Z < input_shape0.w + input_shape1.w) {                                                                  \
    in_postion_x = (X / input_shape1.y) * input_shape1.w * input_shape1.y + (Z - input_shape0.w) * input_shape1.y +  \
                   (X % input_shape1.y);                                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y), in_postion_x));                                                \
  } else {                                                                                                           \
    in_postion_x = (X / input_shape2.y) * input_shape2.w * input_shape2.y +                                          \
                   (Z - input_shape0.w - input_shape1.w) * input_shape2.y + (X % input_shape2.y);                    \
    result = READ_IMAGE(input2, smp_none, (int2)((Y), in_postion_x));                                                \
  }                                                                                                                  \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define CHECK_IDXConcat3input_NHWC4                                                         \
  int X = get_global_id(0);                                                                 \
  int Y = get_global_id(1);                                                                 \
  int Z = get_global_id(2);                                                                 \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) { \
    return;                                                                                 \
  }                                                                                         \
  FLT4 result;

#define DOConcat3inputaxis1_NHWC4                                                                                 \
  if (X < input_shape0.y) {                                                                                       \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                                   \
  } else if (X < (input_shape0.y + input_shape1.y)) {                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z, (X - input_shape0.y)));                  \
  } else {                                                                                                        \
    result = READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z, (X - input_shape0.y - input_shape1.y))); \
  }                                                                                                               \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define DOConcat3inputaxis2_NHWC4                                                                                   \
  if (Y < input_shape0.z) {                                                                                         \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                                     \
  } else if (Y < (input_shape0.z + input_shape1.z)) {                                                               \
    result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z) * input_shape1.w + Z, (X)));                  \
  } else {                                                                                                          \
    result = READ_IMAGE(input2, smp_none, (int2)((Y - input_shape0.z - input_shape1.z) * input_shape2.w + Z, (X))); \
  }                                                                                                                 \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define DOConcat3inputaxis3_NHWC4                                                                                 \
  if (Z < input_shape0.w) {                                                                                       \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                                   \
  } else if (Z < (input_shape0.w + input_shape1.w)) {                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z - input_shape0.w, (X)));                  \
  } else {                                                                                                        \
    result = READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z - input_shape0.w - input_shape1.w, (X))); \
  }                                                                                                               \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define CHECK_IDXConcat4input_NHWC4                                                         \
  int X = get_global_id(0);                                                                 \
  int Y = get_global_id(1);                                                                 \
  int Z = get_global_id(2);                                                                 \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) { \
    return;                                                                                 \
  }                                                                                         \
  FLT4 result;

#define DOConcat4inputaxis1_NHWC4                                                                                 \
  if (X < input_shape0.y) {                                                                                       \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                                   \
  } else if (X < (input_shape0.y + input_shape1.y)) {                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z, (X - input_shape0.y)));                  \
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y)) {                                            \
    result = READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z, (X - input_shape0.y - input_shape1.y))); \
  } else {                                                                                                        \
    result = READ_IMAGE(input3, smp_none,                                                                         \
                        (int2)((Y)*input_shape3.w + Z, (X - input_shape0.y - input_shape1.y - input_shape2.y)));  \
  }                                                                                                               \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define DOConcat4inputaxis2_NHWC4                                                                                   \
  if (Y < input_shape0.z) {                                                                                         \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                                     \
  } else if (Y < (input_shape0.z + input_shape1.z)) {                                                               \
    result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z) * input_shape1.w + Z, (X)));                  \
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z)) {                                              \
    result = READ_IMAGE(input2, smp_none, (int2)((Y - input_shape0.z - input_shape1.z) * input_shape2.w + Z, (X))); \
  } else {                                                                                                          \
    result = READ_IMAGE(input3, smp_none,                                                                           \
                        (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z) * input_shape3.w + Z, (X)));  \
  }                                                                                                                 \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define DOConcat4inputaxis3_NHWC4                                                                                 \
  if (Z < input_shape0.w) {                                                                                       \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                                   \
  } else if (Z < (input_shape0.w + input_shape1.w)) {                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z - input_shape0.w, (X)));                  \
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w)) {                                            \
    result = READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z - input_shape0.w - input_shape1.w, (X))); \
  } else {                                                                                                        \
    result = READ_IMAGE(input3, smp_none,                                                                         \
                        (int2)((Y)*input_shape3.w + Z - input_shape0.w - input_shape1.w - input_shape2.w, (X)));  \
  }                                                                                                               \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define CHECK_IDXConcat4input_NC4HW4                                                                                \
  int X = get_global_id(0);                                                                                         \
  int Y = get_global_id(1);                                                                                         \
  int Z = get_global_id(2);                                                                                         \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {                         \
    return;                                                                                                         \
  }                                                                                                                 \
  if (input_shape0.y == 0 || input_shape1.y == 0 || input_shape2.y == 0 || input_shape3.y == 0 ||                   \
      output_shape.y == 0) {                                                                                        \
    return;                                                                                                         \
  }                                                                                                                 \
  int in_postion_x;                                                                                                 \
  int out_pos_x = (X / output_shape.y) * output_shape.w * output_shape.y + Z * output_shape.y + X % output_shape.y; \
  FLT4 result;

#define DOConcat4inputaxis1_NC4HW4                                                                                   \
  if (X < input_shape0.y) {                                                                                          \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y; \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (X < input_shape0.y + input_shape1.y) {                                                                  \
    in_postion_x = ((X - input_shape0.y) / input_shape1.y) * input_shape1.w * input_shape1.y + Z * input_shape1.y +  \
                   ((X - input_shape0.y) % input_shape1.y);                                                          \
    result = READ_IMAGE(input1, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (X < input_shape0.y + input_shape1.y + input_shape2.y) {                                                 \
    in_postion_x = ((X - input_shape0.y - input_shape1.y) / input_shape2.y) * input_shape2.w * input_shape2.y +      \
                   Z * input_shape2.y + ((X - input_shape0.y - input_shape1.y) % input_shape2.y);                    \
    result = READ_IMAGE(input2, smp_none, (int2)((Y), in_postion_x));                                                \
  } else {                                                                                                           \
    in_postion_x =                                                                                                   \
      ((X - input_shape0.y - input_shape1.y - input_shape2.y) / input_shape3.y) * input_shape3.w * input_shape3.y +  \
      Z * input_shape3.y + ((X - input_shape0.y - input_shape1.y - input_shape2.y) % input_shape3.y);                \
    result = READ_IMAGE(input3, smp_none, (int2)((Y), in_postion_x));                                                \
  }                                                                                                                  \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define DOConcat4inputaxis2_NC4HW4                                                                                     \
  if (Y < input_shape0.z) {                                                                                            \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y;   \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                  \
  } else if (Y < input_shape0.z + input_shape1.z) {                                                                    \
    in_postion_x = (X / input_shape1.y) * input_shape1.w * input_shape1.y + Z * input_shape1.y + (X % input_shape1.y); \
    result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z), in_postion_x));                                 \
  } else if (Y < input_shape0.z + input_shape1.z + input_shape2.z) {                                                   \
    in_postion_x = (X / input_shape2.y) * input_shape2.w * input_shape2.y + Z * input_shape2.y + (X % input_shape2.y); \
    result = READ_IMAGE(input2, smp_none, (int2)((Y - input_shape0.z - input_shape1.z), in_postion_x));                \
  } else {                                                                                                             \
    in_postion_x = (X / input_shape3.y) * input_shape3.w * input_shape3.y + Z * input_shape3.y + (X % input_shape3.y); \
    result =                                                                                                           \
      READ_IMAGE(input3, smp_none, (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z), in_postion_x));      \
  }                                                                                                                    \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define DOConcat4inputaxis3_NC4HW4                                                                                   \
  if (Z < input_shape0.w) {                                                                                          \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y; \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (Z < input_shape0.w + input_shape1.w) {                                                                  \
    in_postion_x = (X / input_shape1.y) * input_shape1.w * input_shape1.y + (Z - input_shape0.w) * input_shape1.y +  \
                   (X % input_shape1.y);                                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (Z < input_shape0.w + input_shape1.w + input_shape2.w) {                                                 \
    in_postion_x = (X / input_shape2.y) * input_shape2.w * input_shape2.y +                                          \
                   (Z - input_shape0.w - input_shape1.w) * input_shape2.y + (X % input_shape2.y);                    \
    result = READ_IMAGE(input2, smp_none, (int2)((Y), in_postion_x));                                                \
  } else {                                                                                                           \
    in_postion_x = (X / input_shape3.y) * input_shape3.w * input_shape3.y +                                          \
                   (Z - input_shape0.w - input_shape1.w - input_shape2.w) * input_shape3.y + (X % input_shape3.y);   \
    result = READ_IMAGE(input3, smp_none, (int2)((Y), in_postion_x));                                                \
  }                                                                                                                  \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

__kernel void Concat4input_NC4HW4(__read_only image2d_t input0, __read_only image2d_t input1,
                                  __read_only image2d_t input2, __read_only image2d_t input3,
                                  __write_only image2d_t output, int4 input_shape0, int4 input_shape1,
                                  int4 input_shape2, int4 input_shape3, int4 output_shape, const int axis) {}

#define CHECK_IDXConcat6input_NHWC4                                                         \
  int X = get_global_id(0);                                                                 \
  int Y = get_global_id(1);                                                                 \
  int Z = get_global_id(2);                                                                 \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) { \
    return;                                                                                 \
  }                                                                                         \
  FLT4 result;

#define DOConcat6inputaxis1_NHWC4                                                                                 \
  if (X < input_shape0.y) {                                                                                       \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                                   \
  } else if (X < (input_shape0.y + input_shape1.y)) {                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z, (X - input_shape0.y)));                  \
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y)) {                                            \
    result = READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z, (X - input_shape0.y - input_shape1.y))); \
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y + input_shape3.y)) {                           \
    result = READ_IMAGE(input3, smp_none,                                                                         \
                        (int2)((Y)*input_shape3.w + Z, (X - input_shape0.y - input_shape1.y - input_shape2.y)));  \
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y + input_shape3.y + input_shape4.y)) {          \
    result = READ_IMAGE(                                                                                          \
      input4, smp_none,                                                                                           \
      (int2)((Y)*input_shape4.w + Z, (X - input_shape0.y - input_shape1.y - input_shape2.y - input_shape3.y)));   \
  } else {                                                                                                        \
    result = READ_IMAGE(input5, smp_none,                                                                         \
                        (int2)((Y)*input_shape5.w + Z, (X - input_shape0.y - input_shape1.y - input_shape2.y -    \
                                                        input_shape3.y - input_shape4.y)));                       \
  }                                                                                                               \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define DOConcat6inputaxis2_NHWC4                                                                                      \
  if (Y < input_shape0.z) {                                                                                            \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                                        \
  } else if (Y < (input_shape0.z + input_shape1.z)) {                                                                  \
    result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z) * input_shape1.w + Z, (X)));                     \
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z)) {                                                 \
    result = READ_IMAGE(input2, smp_none, (int2)((Y - input_shape0.z - input_shape1.z) * input_shape2.w + Z, (X)));    \
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z + input_shape3.z)) {                                \
    result = READ_IMAGE(input3, smp_none,                                                                              \
                        (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z) * input_shape3.w + Z, (X)));     \
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z + input_shape3.z + input_shape4.z)) {               \
    result = READ_IMAGE(                                                                                               \
      input4, smp_none,                                                                                                \
      (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z - input_shape3.z) * input_shape4.w + Z, (X)));      \
  } else {                                                                                                             \
    result = READ_IMAGE(                                                                                               \
      input5, smp_none,                                                                                                \
      (int2)(                                                                                                          \
        (Y - input_shape0.z - input_shape1.z - input_shape2.z - input_shape3.z - input_shape4.z) * input_shape5.w + Z, \
        (X)));                                                                                                         \
  }                                                                                                                    \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define DOConcat6inputaxis3_NHWC4                                                                                 \
  if (Z < input_shape0.w) {                                                                                       \
    result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));                                   \
  } else if (Z < (input_shape0.w + input_shape1.w)) {                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z - input_shape0.w, (X)));                  \
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w)) {                                            \
    result = READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z - input_shape0.w - input_shape1.w, (X))); \
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w + input_shape3.w)) {                           \
    result = READ_IMAGE(input3, smp_none,                                                                         \
                        (int2)((Y)*input_shape3.w + Z - input_shape0.w - input_shape1.w - input_shape2.w, (X)));  \
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w + input_shape3.w + input_shape4.w)) {          \
    result = READ_IMAGE(                                                                                          \
      input4, smp_none,                                                                                           \
      (int2)((Y)*input_shape4.w + Z - input_shape0.w - input_shape1.w - input_shape2.w - input_shape3.w, (X)));   \
  } else {                                                                                                        \
    result = READ_IMAGE(input5, smp_none,                                                                         \
                        (int2)((Y)*input_shape5.w + Z - input_shape0.w - input_shape1.w - input_shape2.w -        \
                                 input_shape3.w - input_shape4.w,                                                 \
                               (X)));                                                                             \
  }                                                                                                               \
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);

#define CHECK_IDXConcat6input_NC4HW4                                                              \
  int X = get_global_id(0);                                                                       \
  int Y = get_global_id(1);                                                                       \
  int Z = get_global_id(2);                                                                       \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {       \
    return;                                                                                       \
  }                                                                                               \
  if (input_shape0.y == 0 || input_shape1.y == 0 || input_shape2.y == 0 || input_shape3.y == 0 || \
      input_shape4.y == 0 || input_shape5.y == 0 || output_shape.y == 0) {                        \
    return;                                                                                       \
  }                                                                                               \
  int in_postion_x;                                                                               \
  FLT4 result;                                                                                    \
  int out_pos_x = (X / output_shape.y) * output_shape.w * output_shape.y + Z * output_shape.y + X % output_shape.y;

#define DOConcat6inputaxis1_NC4HW4                                                                                   \
  if (X < input_shape0.y) {                                                                                          \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y; \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (X < (input_shape0.y + input_shape1.y)) {                                                                \
    in_postion_x = ((X - input_shape0.y) / input_shape1.y) * input_shape1.w * input_shape1.y + Z * input_shape1.y +  \
                   ((X - input_shape0.y) % input_shape1.y);                                                          \
    result = READ_IMAGE(input1, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y)) {                                               \
    in_postion_x = ((X - input_shape0.y - input_shape1.y) / input_shape2.y) * input_shape2.w * input_shape2.y +      \
                   Z * input_shape2.y + ((X - input_shape0.y - input_shape1.y) % input_shape2.y);                    \
    result = READ_IMAGE(input2, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y + input_shape3.y)) {                              \
    in_postion_x =                                                                                                   \
      ((X - input_shape0.y - input_shape1.y - input_shape2.y) / input_shape3.y) * input_shape3.w * input_shape3.y +  \
      Z * input_shape3.y + ((X - input_shape0.y - input_shape1.y - input_shape2.y) % input_shape3.y);                \
    result = READ_IMAGE(input3, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (X < (input_shape0.y + input_shape1.y + input_shape2.y + input_shape3.y + input_shape4.y)) {             \
    in_postion_x = ((X - input_shape0.y - input_shape1.y - input_shape2.y - input_shape3.y) / input_shape4.y) *      \
                     input_shape4.w * input_shape4.y +                                                               \
                   Z * input_shape4.y +                                                                              \
                   ((X - input_shape0.y - input_shape1.y - input_shape2.y - input_shape3.y) % input_shape4.y);       \
    result = READ_IMAGE(input4, smp_none, (int2)((Y), in_postion_x));                                                \
  } else {                                                                                                           \
    in_postion_x =                                                                                                   \
      ((X - input_shape0.y - input_shape1.y - input_shape2.y - input_shape3.y - input_shape4.y) / input_shape5.y) *  \
        input_shape5.w * input_shape5.y +                                                                            \
      Z * input_shape5.y +                                                                                           \
      ((X - input_shape0.y - input_shape1.y - input_shape2.y - input_shape3.y - input_shape4.y) % input_shape5.y);   \
    result = READ_IMAGE(input5, smp_none, (int2)((Y), in_postion_x));                                                \
  }                                                                                                                  \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define DOConcat6inputaxis2_NC4HW4                                                                                     \
  if (Y < input_shape0.z) {                                                                                            \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y;   \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                  \
  } else if (Y < (input_shape0.z + input_shape1.z)) {                                                                  \
    in_postion_x = (X / input_shape1.y) * input_shape1.w * input_shape1.y + Z * input_shape1.y + (X % input_shape1.y); \
    result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z), in_postion_x));                                 \
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z)) {                                                 \
    in_postion_x = (X / input_shape2.y) * input_shape2.w * input_shape2.y + Z * input_shape2.y + (X % input_shape2.y); \
    result = READ_IMAGE(input2, smp_none, (int2)((Y - input_shape0.z - input_shape1.z), in_postion_x));                \
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z + input_shape3.z)) {                                \
    in_postion_x = (X / input_shape3.y) * input_shape3.w * input_shape3.y + Z * input_shape3.y + (X % input_shape3.y); \
    result =                                                                                                           \
      READ_IMAGE(input3, smp_none, (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z), in_postion_x));      \
  } else if (Y < (input_shape0.z + input_shape1.z + input_shape2.z + input_shape3.z + input_shape4.z)) {               \
    in_postion_x = (X / input_shape4.y) * input_shape4.w * input_shape4.y + Z * input_shape4.y + (X % input_shape4.y); \
    result =                                                                                                           \
      READ_IMAGE(input4, smp_none,                                                                                     \
                 (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z - input_shape3.z), in_postion_x));       \
  } else {                                                                                                             \
    in_postion_x = (X / input_shape5.y) * input_shape5.w * input_shape5.y + Z * input_shape5.y + (X % input_shape5.y); \
    result = READ_IMAGE(                                                                                               \
      input5, smp_none,                                                                                                \
      (int2)((Y - input_shape0.z - input_shape1.z - input_shape2.z - input_shape3.z - input_shape4.z), in_postion_x)); \
  }                                                                                                                    \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define DOConcat6inputaxis3_NC4HW4                                                                                   \
  if (Z < input_shape0.w) {                                                                                          \
    in_postion_x = (X / input_shape0.y) * input_shape0.w * input_shape0.y + Z * input_shape0.y + X % input_shape0.y; \
    result = READ_IMAGE(input0, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (Z < (input_shape0.w + input_shape1.w)) {                                                                \
    in_postion_x = (X / input_shape1.y) * input_shape1.w * input_shape1.y + (Z - input_shape0.w) * input_shape1.y +  \
                   (X % input_shape1.y);                                                                             \
    result = READ_IMAGE(input1, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w)) {                                               \
    in_postion_x = (X / input_shape2.y) * input_shape2.w * input_shape2.y +                                          \
                   (Z - input_shape0.w - input_shape1.w) * input_shape2.y + (X % input_shape2.y);                    \
    result = READ_IMAGE(input2, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w + input_shape3.w)) {                              \
    in_postion_x = (X / input_shape3.y) * input_shape3.w * input_shape3.y +                                          \
                   (Z - input_shape0.w - input_shape1.w - input_shape2.w) * input_shape3.y + (X % input_shape3.y);   \
    result = READ_IMAGE(input3, smp_none, (int2)((Y), in_postion_x));                                                \
  } else if (Z < (input_shape0.w + input_shape1.w + input_shape2.w + input_shape3.w + input_shape4.w)) {             \
    in_postion_x = (X / input_shape4.y) * input_shape4.w * input_shape4.y +                                          \
                   (Z - input_shape0.w - input_shape1.w - input_shape2.w - input_shape3.w) * input_shape4.y +        \
                   (X % input_shape4.y);                                                                             \
    result = READ_IMAGE(input4, smp_none, (int2)((Y), in_postion_x));                                                \
  } else {                                                                                                           \
    in_postion_x =                                                                                                   \
      (X / input_shape5.y) * input_shape5.w * input_shape5.y +                                                       \
      (Z - input_shape0.w - input_shape1.w - input_shape2.w - input_shape3.w - input_shape4.w) * input_shape5.y +    \
      (X % input_shape5.y);                                                                                          \
    result = READ_IMAGE(input5, smp_none, (int2)((Y), in_postion_x));                                                \
  }                                                                                                                  \
  WRITE_IMAGE(output, (int2)((Y), out_pos_x), result);

#define CONCAT6(Inputnum, Axis, ToFormat)                                                                      \
  __kernel void Concat##Inputnum##Axis##ToFormat(                                                              \
    __read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,                  \
    __read_only image2d_t input3, __read_only image2d_t input4, __read_only image2d_t input5,                  \
    __write_only image2d_t output, int4 input_shape0, int4 input_shape1, int4 input_shape2, int4 input_shape3, \
    int4 input_shape4, int4 input_shape5, int4 output_shape, const int axis) {                                 \
    CHECK_IDXConcat6input##ToFormat;                                                                           \
    DOConcat##Inputnum##Axis##ToFormat;                                                                        \
  }

#define CONCAT4(Inputnum, Axis, ToFormat)                                                              \
  __kernel void Concat##Inputnum##Axis##ToFormat(                                                      \
    __read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,          \
    __read_only image2d_t input3, __write_only image2d_t output, int4 input_shape0, int4 input_shape1, \
    int4 input_shape2, int4 input_shape3, int4 output_shape, const int axis) {                         \
    CHECK_IDXConcat4input##ToFormat;                                                                   \
    DOConcat##Inputnum##Axis##ToFormat;                                                                \
  }

#define CONCAT3(Inputnum, Axis, ToFormat)                                                                     \
  __kernel void Concat##Inputnum##Axis##ToFormat(__read_only image2d_t input0, __read_only image2d_t input1,  \
                                                 __read_only image2d_t input2, __write_only image2d_t output, \
                                                 int4 input_shape0, int4 input_shape1, int4 input_shape2,     \
                                                 int4 output_shape, const int axis) {                         \
    CHECK_IDXConcat3input##ToFormat;                                                                          \
    DOConcat##Inputnum##Axis##ToFormat;                                                                       \
  }

#define CONCAT2(Inputnum, Axis, ToFormat)                                                                             \
  __kernel void Concat##Inputnum##Axis##ToFormat(__read_only image2d_t input0, __read_only image2d_t input1,          \
                                                 __write_only image2d_t output, int4 input_shape0, int4 input_shape1, \
                                                 int4 output_shape, const int axis) {                                 \
    CHECK_IDXConcat2input##ToFormat;                                                                                  \
    DOConcat##Inputnum##Axis##ToFormat;                                                                               \
  }

// nc4hw4
CONCAT6(6input, axis1, _NC4HW4)
CONCAT6(6input, axis2, _NC4HW4)
CONCAT6(6input, axis3, _NC4HW4)
CONCAT4(4input, axis1, _NC4HW4)
CONCAT4(4input, axis2, _NC4HW4)
CONCAT4(4input, axis3, _NC4HW4)
CONCAT3(3input, axis1, _NC4HW4)
CONCAT3(3input, axis2, _NC4HW4)
CONCAT3(3input, axis3, _NC4HW4)
CONCAT2(2input, axis1, _NC4HW4)
CONCAT2(2input, axis2, _NC4HW4)
CONCAT2(2input, axis3, _NC4HW4)

// nhwc4
CONCAT6(6input, axis1, _NHWC4)
CONCAT6(6input, axis2, _NHWC4)
CONCAT6(6input, axis3, _NHWC4)
CONCAT4(4input, axis1, _NHWC4)
CONCAT4(4input, axis2, _NHWC4)
CONCAT4(4input, axis3, _NHWC4)
CONCAT3(3input, axis1, _NHWC4)
CONCAT3(3input, axis2, _NHWC4)
CONCAT3(3input, axis3, _NHWC4)
CONCAT2(2input, axis1, _NHWC4)
CONCAT2(2input, axis2, _NHWC4)
CONCAT2(2input, axis3, _NHWC4)
