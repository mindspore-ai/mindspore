#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define C4NUM 4

// Align in Axis C for concat
#define CHECK_IDX                                                                           \
  int X = get_global_id(0);                                                                 \
  int Y = get_global_id(1);                                                                 \
  int Z = get_global_id(2);                                                                 \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) { \
    return;                                                                                 \
  }                                                                                         \
  FLT4 result;

// axis = 1
#define DOConcat2inputaxis1_NHWC4                                              \
  int IN = X / output_shape.y;                                                 \
  int IH = X % output_shape.y;                                                 \
  int boundary0 = input_shape0.y;                                              \
  int boundary1 = boundary0 + input_shape1.y;                                  \
  if (IH < boundary0) {                                                        \
    int coordinate_x = Y * input_shape0.w + Z;                                 \
    int coordinate_y = IN * input_shape0.y + IH;                               \
    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y)); \
  } else if (IH < boundary1) {                                                 \
    int coordinate_x = Y * input_shape1.w + Z;                                 \
    int coordinate_y = IN * input_shape1.y + IH - boundary0;                   \
    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat3inputaxis1_NHWC4                                              \
  DOConcat2inputaxis1_NHWC4;                                                   \
  int boundary2 = boundary1 + input_shape2.y;                                  \
  if (IH >= boundary1 && IH < boundary2) {                                     \
    int coordinate_x = Y * input_shape2.w + Z;                                 \
    int coordinate_y = IN * input_shape2.y + IH - boundary1;                   \
    result = READ_IMAGE(input2, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat4inputaxis1_NHWC4                                              \
  DOConcat3inputaxis1_NHWC4;                                                   \
  int boundary3 = boundary2 + input_shape3.y;                                  \
  if (IH >= boundary2 && IH < boundary3) {                                     \
    int coordinate_x = Y * input_shape3.w + Z;                                 \
    int coordinate_y = IN * input_shape3.y + IH - boundary2;                   \
    result = READ_IMAGE(input3, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat5inputaxis1_NHWC4                                              \
  DOConcat4inputaxis1_NHWC4;                                                   \
  int boundary4 = boundary3 + input_shape4.y;                                  \
  if (IH >= boundary3 && IH < boundary4) {                                     \
    int coordinate_x = Y * input_shape4.w + Z;                                 \
    int coordinate_y = IN * input_shape4.y + IH - boundary3;                   \
    result = READ_IMAGE(input4, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat6inputaxis1_NHWC4                                              \
  DOConcat5inputaxis1_NHWC4;                                                   \
  int boundary5 = boundary4 + input_shape5.y;                                  \
  if (IH >= boundary4 && IH < boundary5) {                                     \
    int coordinate_x = Y * input_shape5.w + Z;                                 \
    int coordinate_y = IN * input_shape5.y + IH - boundary4;                   \
    result = READ_IMAGE(input5, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

// axis = 2
#define DOConcat2inputaxis2_NHWC4                                              \
  int boundary0 = input_shape0.z;                                              \
  int boundary1 = boundary0 + input_shape1.z;                                  \
  if (Y < boundary0) {                                                         \
    int coordinate_x = Y * input_shape0.w + Z;                                 \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y)); \
  } else if (Y < boundary1) {                                                  \
    int coordinate_x = (Y - boundary0) * input_shape1.w + Z;                   \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat3inputaxis2_NHWC4                                              \
  DOConcat2inputaxis2_NHWC4;                                                   \
  int boundary2 = boundary1 + input_shape2.z;                                  \
  if (Y >= boundary1 && Y < boundary2) {                                       \
    int coordinate_x = (Y - boundary1) * input_shape2.w + Z;                   \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input2, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat4inputaxis2_NHWC4                                              \
  DOConcat3inputaxis2_NHWC4;                                                   \
  int boundary3 = boundary2 + input_shape3.z;                                  \
  if (Y >= boundary2 && Y < boundary3) {                                       \
    int coordinate_x = (Y - boundary2) * input_shape3.w + Z;                   \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input3, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat5inputaxis2_NHWC4                                              \
  DOConcat4inputaxis2_NHWC4;                                                   \
  int boundary4 = boundary3 + input_shape4.z;                                  \
  if (Y >= boundary3 && Y < boundary4) {                                       \
    int coordinate_x = (Y - boundary3) * input_shape4.w + Z;                   \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input4, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat6inputaxis2_NHWC4                                              \
  DOConcat5inputaxis2_NHWC4;                                                   \
  int boundary5 = boundary4 + input_shape5.z;                                  \
  if (Y >= boundary4 && Y < boundary5) {                                       \
    int coordinate_x = (Y - boundary4) * input_shape5.w + Z;                   \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input5, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

// axis = 3
#define DOConcat2inputaxis3_NHWC4                                              \
  int boundary0 = input_shape0.w;                                              \
  int boundary1 = boundary0 + input_shape1.w;                                  \
  if (Z < boundary0) {                                                         \
    int coordinate_x = Y * input_shape0.w + Z;                                 \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y)); \
  } else if (Z < boundary1) {                                                  \
    int coordinate_x = Y * input_shape1.w + Z - boundary0;                     \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat3inputaxis3_NHWC4                                              \
  DOConcat2inputaxis3_NHWC4;                                                   \
  int boundary2 = boundary1 + input_shape2.w;                                  \
  if (Z >= boundary1 && Z < boundary2) {                                       \
    int coordinate_x = Y * input_shape2.w + Z - boundary1;                     \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input2, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat4inputaxis3_NHWC4                                              \
  DOConcat3inputaxis3_NHWC4;                                                   \
  int boundary3 = boundary2 + input_shape3.w;                                  \
  if (Z >= boundary2 && Z < boundary3) {                                       \
    int coordinate_x = Y * input_shape3.w + Z - boundary2;                     \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input3, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat5inputaxis3_NHWC4                                              \
  DOConcat4inputaxis3_NHWC4;                                                   \
  int boundary4 = boundary3 + input_shape4.w;                                  \
  if (Z >= boundary3 && Z < boundary4) {                                       \
    int coordinate_x = Y * input_shape4.w + Z - boundary3;                     \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input4, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define DOConcat6inputaxis3_NHWC4                                              \
  DOConcat5inputaxis3_NHWC4;                                                   \
  int boundary5 = boundary4 + input_shape5.w;                                  \
  if (Z >= boundary4 && Z < boundary5) {                                       \
    int coordinate_x = Y * input_shape5.w + Z - boundary4;                     \
    int coordinate_y = X;                                                      \
    result = READ_IMAGE(input5, smp_none, (int2)(coordinate_x, coordinate_y)); \
  }

#define CONCAT6(Inputnum, Axis, ToFormat)                                                                      \
  __kernel void Concat##Inputnum##Axis##ToFormat(                                                              \
    __read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,                  \
    __read_only image2d_t input3, __read_only image2d_t input4, __read_only image2d_t input5,                  \
    __write_only image2d_t output, int4 input_shape0, int4 input_shape1, int4 input_shape2, int4 input_shape3, \
    int4 input_shape4, int4 input_shape5, int4 output_shape) {                                                 \
    CHECK_IDX;                                                                                                 \
    DOConcat##Inputnum##Axis##ToFormat;                                                                        \
    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                          \
  }

#define CONCAT5(Inputnum, Axis, ToFormat)                                                                         \
  __kernel void Concat##Inputnum##Axis##ToFormat(                                                                 \
    __read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,                     \
    __read_only image2d_t input3, __read_only image2d_t input4, __write_only image2d_t output, int4 input_shape0, \
    int4 input_shape1, int4 input_shape2, int4 input_shape3, int4 input_shape4, int4 output_shape) {              \
    CHECK_IDX;                                                                                                    \
    DOConcat##Inputnum##Axis##ToFormat;                                                                           \
    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                             \
  }

#define CONCAT4(Inputnum, Axis, ToFormat)                                                                             \
  __kernel void Concat##Inputnum##Axis##ToFormat(__read_only image2d_t input0, __read_only image2d_t input1,          \
                                                 __read_only image2d_t input2, __read_only image2d_t input3,          \
                                                 __write_only image2d_t output, int4 input_shape0, int4 input_shape1, \
                                                 int4 input_shape2, int4 input_shape3, int4 output_shape) {           \
    CHECK_IDX                                                                                                         \
    DOConcat##Inputnum##Axis##ToFormat;                                                                               \
    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                                 \
  }

#define CONCAT3(Inputnum, Axis, ToFormat)                                                                        \
  __kernel void Concat##Inputnum##Axis##ToFormat(                                                                \
    __read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,                    \
    __write_only image2d_t output, int4 input_shape0, int4 input_shape1, int4 input_shape2, int4 output_shape) { \
    CHECK_IDX                                                                                                    \
    DOConcat##Inputnum##Axis##ToFormat;                                                                          \
    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                            \
  }

#define CONCAT2(Inputnum, Axis, ToFormat)                                                                             \
  __kernel void Concat##Inputnum##Axis##ToFormat(__read_only image2d_t input0, __read_only image2d_t input1,          \
                                                 __write_only image2d_t output, int4 input_shape0, int4 input_shape1, \
                                                 int4 output_shape) {                                                 \
    CHECK_IDX                                                                                                         \
    DOConcat##Inputnum##Axis##ToFormat;                                                                               \
    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                                 \
  }

// axis = 1
CONCAT6(6input, axis1, _NHWC4)
CONCAT5(5input, axis1, _NHWC4)
CONCAT4(4input, axis1, _NHWC4)
CONCAT3(3input, axis1, _NHWC4)
CONCAT2(2input, axis1, _NHWC4)

// axis = 2
CONCAT6(6input, axis2, _NHWC4)
CONCAT5(5input, axis2, _NHWC4)
CONCAT4(4input, axis2, _NHWC4)
CONCAT3(3input, axis2, _NHWC4)
CONCAT2(2input, axis2, _NHWC4)

// axis = 3
CONCAT6(6input, axis3, _NHWC4)
CONCAT5(5input, axis3, _NHWC4)
CONCAT4(4input, axis3, _NHWC4)
CONCAT3(3input, axis3, _NHWC4)
CONCAT2(2input, axis3, _NHWC4)

// UnAlign in Axis C for concat
#define CHECK_IDX_UNALIGN                                                                         \
  int X = get_global_id(0);                                                                       \
  int Y = get_global_id(1);                                                                       \
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z) {                              \
    return;                                                                                       \
  }                                                                                               \
  int IN = X / output_shape.y, IH = X % output_shape.y;                                           \
  int IW = Y;                                                                                     \
  int Align_Shape0 = UP_DIV(input_shape0.w, C4NUM), Align_Shape1 = UP_DIV(input_shape1.w, C4NUM); \
  int Align_OutShape = output_shape.w;                                                            \
  int index_output = (IN * output_shape.y + IH) * stride_w + IW * Align_OutShape * C4NUM;

int doconcat(__read_only image2d_t input, __global FLT *output, int Align_Shape, int4 input_shape, int IN, int IH,
             int Y, int index_output) {
  int Remainder = input_shape.w % C4NUM;
  for (int i = 0; i < Align_Shape; ++i) {
    FLT4 result = READ_IMAGE(input, smp_none, (int2)((Y * Align_Shape + i), (IN * input_shape.y + IH)));
    FLT result_temp[4] = {result.x, result.y, result.z, result.w};
    if ((i + 1) * C4NUM <= input_shape.w) {
      for (int j = 0; j < C4NUM; ++j) {
        output[index_output++] = result_temp[j];
      }
    } else {
      for (int j = 0; j < Remainder; ++j) {
        output[index_output++] = result_temp[j];
      }
    }
  }
  return index_output;
}

__kernel void ConcatInput2UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,
                                        __global FLT *output, int4 input_shape0, int4 input_shape1, int stride_w,
                                        int4 output_shape) {
  CHECK_IDX_UNALIGN;
  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);
  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);
}

__kernel void ConcatInput3UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,
                                        __read_only image2d_t input2, __global FLT *output, int4 input_shape0,
                                        int4 input_shape1, int4 input_shape2, int stride_w, int4 output_shape) {
  CHECK_IDX_UNALIGN;
  int Align_Shape2 = UP_DIV(input_shape2.w, C4NUM);
  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);
  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);
  index_output = doconcat(input2, output, Align_Shape2, input_shape2, IN, IH, Y, index_output);
}

__kernel void ConcatInput4UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,
                                        __read_only image2d_t input2, __read_only image2d_t input3,
                                        __global FLT *output, int4 input_shape0, int4 input_shape1, int4 input_shape2,
                                        int4 input_shape3, int stride_w, int4 output_shape) {
  CHECK_IDX_UNALIGN;
  int Align_Shape2 = UP_DIV(input_shape2.w, C4NUM), Align_Shape3 = UP_DIV(input_shape3.w, C4NUM);
  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);
  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);
  index_output = doconcat(input2, output, Align_Shape2, input_shape2, IN, IH, Y, index_output);
  index_output = doconcat(input3, output, Align_Shape3, input_shape3, IN, IH, Y, index_output);
}

__kernel void ConcatInput5UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,
                                        __read_only image2d_t input2, __read_only image2d_t input3,
                                        __read_only image2d_t input4, __global FLT *output, int4 input_shape0,
                                        int4 input_shape1, int4 input_shape2, int4 input_shape3, int4 input_shape4,
                                        int stride_w, int4 output_shape) {
  CHECK_IDX_UNALIGN;
  int Align_Shape2 = UP_DIV(input_shape2.w, C4NUM), Align_Shape3 = UP_DIV(input_shape3.w, C4NUM);
  int Align_Shape4 = UP_DIV(input_shape4.w, C4NUM);
  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);
  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);
  index_output = doconcat(input2, output, Align_Shape2, input_shape2, IN, IH, Y, index_output);
  index_output = doconcat(input3, output, Align_Shape3, input_shape3, IN, IH, Y, index_output);
  index_output = doconcat(input4, output, Align_Shape4, input_shape4, IN, IH, Y, index_output);
}

__kernel void ConcatInput6UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,
                                        __read_only image2d_t input2, __read_only image2d_t input3,
                                        __read_only image2d_t input4, __read_only image2d_t input5,
                                        __global FLT *output, int4 input_shape0, int4 input_shape1, int4 input_shape2,
                                        int4 input_shape3, int4 input_shape4, int4 input_shape5, int stride_w,
                                        int4 output_shape) {
  CHECK_IDX_UNALIGN;
  int Align_Shape2 = UP_DIV(input_shape2.w, C4NUM), Align_Shape3 = UP_DIV(input_shape3.w, C4NUM);
  int Align_Shape4 = UP_DIV(input_shape4.w, C4NUM), Align_Shape5 = UP_DIV(input_shape5.w, C4NUM);
  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);
  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);
  index_output = doconcat(input2, output, Align_Shape2, input_shape2, IN, IH, Y, index_output);
  index_output = doconcat(input3, output, Align_Shape3, input_shape3, IN, IH, Y, index_output);
  index_output = doconcat(input4, output, Align_Shape4, input_shape4, IN, IH, Y, index_output);
  index_output = doconcat(input5, output, Align_Shape5, input_shape5, IN, IH, Y, index_output);
}
