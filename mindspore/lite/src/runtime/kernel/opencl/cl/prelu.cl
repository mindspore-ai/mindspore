#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define NHWC4 2
#define NC4HW4 100

__kernel void PRelu_scalar(__read_only image2d_t input, __write_only image2d_t output, float weight, int4 shape,
                           int data_format) {
  int h = get_global_id(0);
  int w = get_global_id(1);
  int slice = get_global_id(2);
  int H = shape.y;
  int W = shape.z;
  int SLICES = shape.w;
  if (h >= H || w >= W || slice >= SLICES) {
    return;
  }

  int x, y;
  if (data_format == 2) {
    x = w * SLICES + slice;
    y = h;
  } else {
    x = w;
    y = slice * H + h;
  }

  FLT4 out = READ_IMAGE(input, smp_zero, (int2)(x, y));
  if (out.x < 0) {
    out.x *= weight;
  }
  if (out.y < 0) {
    out.y *= weight;
  }
  if (out.z < 0) {
    out.z *= weight;
  }
  if (out.w < 0) {
    out.w *= weight;
  }
  WRITE_IMAGE(output, (int2)(x, y), out);
}

__kernel void PRelu_vector(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight_vector,
                           int4 shape, int data_format) {
  int h = get_global_id(0);
  int w = get_global_id(1);
  int slice = get_global_id(2);
  int H = shape.y;
  int W = shape.z;
  int SLICES = shape.w;
  if (h >= H || w >= W || slice >= SLICES) {
    return;
  }
  FLT4 weight = weight_vector[slice];

  int x, y;
  if (data_format == 2) {
    x = w * SLICES + slice;
    y = h;
  } else {
    x = w;
    y = slice * H + h;
  }

  FLT4 out = READ_IMAGE(input, smp_zero, (int2)(x, y));
  if (out.x < 0) {
    out.x *= weight.x;
  }
  if (out.y < 0) {
    out.y *= weight.y;
  }
  if (out.z < 0) {
    out.z *= weight.z;
  }
  if (out.w < 0) {
    out.w *= weight.w;
  }
  WRITE_IMAGE(output, (int2)(x, y), out);
}
