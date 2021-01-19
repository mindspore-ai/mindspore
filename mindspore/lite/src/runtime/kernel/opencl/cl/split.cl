#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define C4NUM 4

#define CHECK_IDX_ALIGN                                                                     \
  const int X = get_global_id(0);                                                           \
  const int Y = get_global_id(1);                                                           \
  const int Z = get_global_id(2);                                                           \
  if (X > in_shape.x * in_shape.y || Y > in_shape.z || Z > in_shape.w || in_shape.y == 0) { \
    return;                                                                                 \
  }

#define ARGS_ALIGN                         \
  const int IN = X / in_shape.y;           \
  const int IH = X % in_shape.y;           \
  int coordinate_x = IN * in_shape.y + IH; \
  int coordinate_y = Y * in_shape.w + Z;   \
  FLT4 result = READ_IMAGE(input, smp_none, (int2)(coordinate_y, coordinate_x));

__kernel void split_out2_axis3(__read_only image2d_t input, __write_only image2d_t output1,
                               __write_only image2d_t output2, __global int *split_sizes_, int4 in_shape,
                               int4 out_shape1, int4 out_shape2) {
  CHECK_IDX_ALIGN;
  ARGS_ALIGN;
  int boundary = UP_DIV(split_sizes_[0], C4NUM);
  if (Z < boundary) {
    coordinate_x = IN * out_shape1.y + IH;
    coordinate_y = Y * out_shape1.w + Z;
    WRITE_IMAGE(output1, (int2)(coordinate_y, coordinate_x), result);
  } else {
    coordinate_x = IN * out_shape2.y + IH;
    coordinate_y = Y * out_shape2.w + Z - boundary;
    WRITE_IMAGE(output2, (int2)(coordinate_y, coordinate_x), result);
  }
}

__kernel void split_out2_axis2(__read_only image2d_t input, __write_only image2d_t output1,
                               __write_only image2d_t output2, __global int *split_sizes_, int4 in_shape,
                               int4 out_shape1, int4 out_shape2) {
  CHECK_IDX_ALIGN;
  ARGS_ALIGN;
  if (Y < split_sizes_[0]) {
    coordinate_x = IN * out_shape1.y + IH;
    coordinate_y = Y * out_shape1.w + Z;
    WRITE_IMAGE(output1, (int2)(coordinate_y, coordinate_x), result);
  } else {
    coordinate_x = IN * out_shape2.y + IH;
    coordinate_y = (Y - split_sizes_[0]) * out_shape2.w + Z;
    WRITE_IMAGE(output2, (int2)(coordinate_y, coordinate_x), result);
  }
}

__kernel void split_out2_axis1(__read_only image2d_t input, __write_only image2d_t output1,
                               __write_only image2d_t output2, __global int *split_sizes_, int4 in_shape,
                               int4 out_shape1, int4 out_shape2) {
  CHECK_IDX_ALIGN;
  ARGS_ALIGN;
  if (IH < split_sizes_[0]) {
    coordinate_x = IN * out_shape1.y + IH;
    coordinate_y = Y * out_shape1.w + Z;
    WRITE_IMAGE(output1, (int2)(coordinate_y, coordinate_x), result);
  } else {
    coordinate_x = IN * out_shape2.y + IH - split_sizes_[0];
    coordinate_y = Y * out_shape2.w + Z;
    WRITE_IMAGE(output2, (int2)(coordinate_y, coordinate_x), result);
  }
}

// UnAlign in Axis C for concat
#define CHECK_IDX_UNALIGN                                                   \
  const int X = get_global_id(0);                                           \
  const int Y = get_global_id(1);                                           \
  if (X >= in_shape.x * in_shape.y || Y >= in_shape.z || in_shape.y == 0) { \
    return;                                                                 \
  }

#define ARGS_UNALIGN                                   \
  const int IN = X / in_shape.y, IH = X % in_shape.y;  \
  const int IW = Y;                                    \
  const int Align_inShape = UP_DIV(in_shape.w, C4NUM); \
  int index_input = (IN * in_shape.y + IH) * stride_w + IW * Align_inShape * C4NUM;

int dosplit(__global FLT *input, __write_only image2d_t output, int4 out_shape, int IN, int IH, int IW,
            int index_input) {
  int Remainder = out_shape.w % C4NUM;
  int coordinate_x = IN * out_shape.y + IH;
  int align_w = UP_DIV(out_shape.w, C4NUM);
  for (int i = 0; i < align_w; ++i) {
    int coordinate_y = IW * align_w + i;
    if ((i + 1) * C4NUM <= out_shape.w) {
      FLT4 result = {input[index_input], input[index_input + 1], input[index_input + 2], input[index_input + 3]};
      WRITE_IMAGE(output, (int2)(coordinate_y, coordinate_x), result);
      index_input += 4;
    } else {
      FLT result_temp[4] = {};
      for (int j = 0; j < Remainder; ++j) {
        result_temp[j] = input[index_input++];
      }
      FLT4 result = {result_temp[0], result_temp[1], result_temp[2], result_temp[3]};
      WRITE_IMAGE(output, (int2)(coordinate_y, coordinate_x), result);
    }
  }
  return index_input;
}

__kernel void split_out2_axis3_unalign(__global FLT *input, __write_only image2d_t output1,
                                       __write_only image2d_t output2, __global int *split_sizes_, int4 in_shape,
                                       int4 out_shape1, int4 out_shape2, int stride_w) {
  CHECK_IDX_UNALIGN;
  ARGS_UNALIGN;
  index_input = dosplit(input, output1, out_shape1, IN, IH, IW, index_input);
  index_input = dosplit(input, output2, out_shape2, IN, IH, IW, index_input);
}
