#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define Pad(dataformat, in_x, in_y, out_x, out_y)                                                              \
  __kernel void Pad_##dataformat(__read_only image2d_t input, __write_only image2d_t output, int4 input_shape, \
                                 int4 output_shape, int2 pad, float constant_value) {                          \
    int oh = get_global_id(0);                                                                                 \
    int ow = get_global_id(1);                                                                                 \
    int co_slice = get_global_id(2);                                                                           \
    int OH = output_shape.y;                                                                                   \
    int OW = output_shape.z;                                                                                   \
    int CO_SLICES = output_shape.w;                                                                            \
                                                                                                               \
    if (oh >= OH || ow >= OW || co_slice >= CO_SLICES) {                                                       \
      return;                                                                                                  \
    }                                                                                                          \
                                                                                                               \
    int IH = input_shape.y;                                                                                    \
    int IW = input_shape.z;                                                                                    \
    int CI_SLICES = input_shape.w;                                                                             \
                                                                                                               \
    int pad_top = pad.x;                                                                                       \
    int pad_left = pad.y;                                                                                      \
    int ih = oh - pad_top;                                                                                     \
    int iw = ow - pad_left;                                                                                    \
                                                                                                               \
    FLT4 result = (FLT4)(constant_value);                                                                      \
    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {                                                            \
      result = READ_IMAGE(input, smp_zero, (int2)(in_x, in_y));                                                \
    }                                                                                                          \
    WRITE_IMAGE(output, (int2)(out_x, out_y), result);                                                         \
  }

Pad(NHWC4, iw *CI_SLICES + co_slice, ih, ow *CO_SLICES + co_slice, oh);
Pad(NC4HW4, iw, co_slice *IH + ih, ow, co_slice *OH + oh);
