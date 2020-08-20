#define CI_TILE 4
#define CO_TILE 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
// #define __global
// #pragma OPENCL EXTENSION cl_arm_printf : enable
__kernel void convolution_NHWC_OHWI(__global float *input, __global float *weight, __global float *bias,
                                    __global float *output,
                                    const int4 input_shape,    // NHWC
                                    const int4 output_shape,   // NHWC
                                    const int4 kernel_stride,  // kernelHW_strideHW
                                    const int4 pad) {
  int ow = get_global_id(0);
  int oh = get_global_id(1);
  int co_slice = get_global_id(2);

  int CI = input_shape.w, IH = input_shape.y, IW = input_shape.z;
  int CO = output_shape.w, OH = output_shape.y, OW = output_shape.z;
  int KH = kernel_stride.x, KW = kernel_stride.y;
  int strideH = kernel_stride.z, strideW = kernel_stride.w;
  int padTop = pad.x, padLeft = pad.z;
  int CI_SLICES = UP_DIV(CI, CI_TILE);
  int CO_SLICES = UP_DIV(CO, CO_TILE);

  if (oh >= OH || ow >= OW || co_slice >= CO_SLICES) return;

  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  for (int kh = 0; kh < KH; ++kh) {
    int ih = kh + oh * strideH - padTop;
    for (int kw = 0; kw < KW; ++kw) {
      int iw = kw + ow * strideW - padLeft;
      for (int ci_slice = 0; ci_slice < CI_SLICES; ++ci_slice) {
        for (int ci_inner = 0; ci_inner < CI_TILE; ++ci_inner) {
          int ci = ci_slice * CI_TILE + ci_inner;
          if (ci >= CI) break;

          int input_idx = ih * IW * CI + iw * CI + ci;
          float value = 0;
          if (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
            value = 0;
          else
            value = input[input_idx];

          int CO_OFFSET = KH * KW * CI;
          int weight_idx = (co_slice * CO_TILE) * CO_OFFSET + kh * KW * CI + kw * CI + ci;
          acc.x += weight[weight_idx + 0 * CO_OFFSET] * value;
          acc.y += weight[weight_idx + 1 * CO_OFFSET] * value;
          acc.z += weight[weight_idx + 2 * CO_OFFSET] * value;
          acc.w += weight[weight_idx + 3 * CO_OFFSET] * value;
        }
      }
    }
  }
  int output_idx = oh * OW * CO + ow * CO + (co_slice * CO_TILE);
  if (co_slice < CO_SLICES - 1 || CO % CO_TILE == 0) {
    output[output_idx + 0] = acc.x + bias[co_slice * CO_TILE + 0];
    output[output_idx + 1] = acc.y + bias[co_slice * CO_TILE + 1];
    output[output_idx + 2] = acc.z + bias[co_slice * CO_TILE + 2];
    output[output_idx + 3] = acc.w + bias[co_slice * CO_TILE + 3];
  } else if (CO % CO_TILE == 1) {
    output[output_idx + 0] = acc.x + bias[co_slice * CO_TILE + 0];
  } else if (CO % CO_TILE == 2) {
    output[output_idx + 0] = acc.x + bias[co_slice * CO_TILE + 0];
    output[output_idx + 1] = acc.y + bias[co_slice * CO_TILE + 1];
  } else if (CO % CO_TILE == 3) {
    output[output_idx + 0] = acc.x + bias[co_slice * CO_TILE + 0];
    output[output_idx + 1] = acc.y + bias[co_slice * CO_TILE + 1];
    output[output_idx + 2] = acc.z + bias[co_slice * CO_TILE + 2];
  }
}

// #pragma OPENCL EXTENSION cl_khr_fp16 : enable
// #define FLT4 half4
#define FLT4 float4
__kernel void convolution_NHWC4_OHWIIO_float8(__global FLT4 *input, __global FLT4 *weight, __global FLT4 *bias,
                                              __global FLT4 *output,
                                              const int4 input_shape,    // NHWC
                                              const int4 output_shape,   // NHWC
                                              const int4 kernel_stride,  // kernelHW_strideHW
                                              const int4 pad) {
  int oh = get_global_id(0);        // [0, OH)
  int ow = get_global_id(1);        // [0, OW)
  int co_slice = get_global_id(2);  // [0, UP_DIV(CO, CO_TILE) )

  int CI = input_shape.w, IH = input_shape.y, IW = input_shape.z;
  int CO = output_shape.w, OH = output_shape.y, OW = output_shape.z;
  int CI_SLICES = UP_DIV(CI, CI_TILE);
  int CO_SLICES = UP_DIV(CO, CO_TILE);
  int KH = kernel_stride.x, KW = kernel_stride.y;
  int strideH = kernel_stride.z, strideW = kernel_stride.w;
  int padTop = pad.x, padLeft = pad.z;

  if (oh >= OH || ow >= OW || 2 * co_slice >= CO_SLICES) return;
  if (2 * co_slice + 1 >= CO_SLICES) {
    FLT4 out0_c4 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
    __global FLT4 *w0_ic1_oc4 = weight + (2 * co_slice + 0) * KH * KW * CI_SLICES * CI_TILE;
    for (int kh = 0; kh < KH; ++kh) {
      int ih = kh + oh * strideH - padTop;
      for (int kw = 0; kw < KW; ++kw) {
        int iw = kw + ow * strideW - padLeft;
        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
          for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {
            FLT4 in_c4 = input[ih * IW * CI_SLICES + iw * CI_SLICES + ci_slice];
            out0_c4 += w0_ic1_oc4[0] * in_c4.x;
            out0_c4 += w0_ic1_oc4[1] * in_c4.y;
            out0_c4 += w0_ic1_oc4[2] * in_c4.z;
            out0_c4 += w0_ic1_oc4[3] * in_c4.w;
            w0_ic1_oc4 += 4;
          }
        } else {
          w0_ic1_oc4 += 4 * CI_SLICES;
        }
      }
    }
    output[oh * OW * CO_SLICES + ow * CO_SLICES + 2 * co_slice + 0] = out0_c4 + bias[2 * co_slice + 0];
  } else {
    FLT4 out0_c4 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
    FLT4 out1_c4 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
    __global FLT4 *w0_ic1_oc4 = weight + (2 * co_slice + 0) * KH * KW * CI_SLICES * CI_TILE;
    __global FLT4 *w1_ic1_oc4 = weight + (2 * co_slice + 1) * KH * KW * CI_SLICES * CI_TILE;
    for (int kh = 0; kh < KH; ++kh) {
      int ih = kh + oh * strideH - padTop;
      for (int kw = 0; kw < KW; ++kw) {
        int iw = kw + ow * strideW - padLeft;
        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
          int idx = ih * IW * CI_SLICES + iw * CI_SLICES;
          for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {
            FLT4 in_c4 = input[idx + ci_slice];

            out0_c4 += w0_ic1_oc4[0] * in_c4.x;
            out0_c4 += w0_ic1_oc4[1] * in_c4.y;
            out0_c4 += w0_ic1_oc4[2] * in_c4.z;
            out0_c4 += w0_ic1_oc4[3] * in_c4.w;
            w0_ic1_oc4 += 4;

            out1_c4 += w1_ic1_oc4[0] * in_c4.x;
            out1_c4 += w1_ic1_oc4[1] * in_c4.y;
            out1_c4 += w1_ic1_oc4[2] * in_c4.z;
            out1_c4 += w1_ic1_oc4[3] * in_c4.w;
            w1_ic1_oc4 += 4;
          }
        } else {
          w0_ic1_oc4 += 4 * CI_SLICES;
          w1_ic1_oc4 += 4 * CI_SLICES;
        }
      }
    }
    output[oh * OW * CO_SLICES + ow * CO_SLICES + 2 * co_slice + 0] = out0_c4 + bias[2 * co_slice + 0];
    output[oh * OW * CO_SLICES + ow * CO_SLICES + 2 * co_slice + 1] = out1_c4 + bias[2 * co_slice + 1];
  }
}
