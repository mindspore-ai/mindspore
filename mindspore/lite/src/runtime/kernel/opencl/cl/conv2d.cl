#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define CI_TILE 4
#define MAX_IMAGE2D_SIZE 65535
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

#define ActType_Relu 1
#define ActType_Relu6 3

#define DEFINE_ARGS                                                               \
  const int N = input_shape.x;                                                    \
  const int IH = input_shape.y, IW = input_shape.z, CI_SLICES = input_shape.w;    \
  const int OH = output_shape.y, OW = output_shape.z, CO_SLICES = output_shape.w; \
  const int KH = kernel_stride.x, KW = kernel_stride.y;                           \
  const int strideH = kernel_stride.z, strideW = kernel_stride.w;                 \
  const int padTop = pad.x, padBottom = pad.y, padLeft = pad.z, padRight = pad.w; \
  const int dilationH = dilation.x, dilationW = dilation.y;                       \
                                                                                  \
  const int n_oh = get_global_id(0);                                              \
  const int ow = get_global_id(1) * BlockW;                                       \
  const int co_slice = get_global_id(2) * BlockC;                                 \
  const int OH_SLICES = UP_DIV(OH, BlockH);                                       \
  const int n = n_oh / OH_SLICES;                                                 \
  const int oh = (n_oh % OH_SLICES) * BlockH;                                     \
  if (n >= N || oh >= OH || ow >= OW || co_slice >= CO_SLICES) {                  \
    return;                                                                       \
  }

__kernel void Conv2D_H1W1C1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,
                            __global FLT4 *bias, const int4 input_shape, const int4 output_shape,
                            const int4 kernel_stride, const int4 pad, const int2 dilation, const int act_type) {
  const int BlockH = 1;
  const int BlockW = 1;
  const int BlockC = 1;
  DEFINE_ARGS;

  const int oh0 = oh + 0;
  const int n_oh0 = n * OH + oh0;
  const int ow0 = ow + 0;
  const int co_slice0 = co_slice + 0;

  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;

  for (int kh = 0; kh < KH; ++kh) {
    const int ih0 = kh * dilationH + oh0 * strideH - padTop;
    const int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;

    for (int kw = 0; kw < KW; ++kw) {
      const int iw0 = kw * dilationW + ow0 * strideW - padLeft;
      int x_idx0 = iw0 * CI_SLICES;

      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {
        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));
        x_idx0++;

        out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;
        out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;
        out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;
        out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;

        weight_ptr += 4;
      }
    }
  }

  if (bias) {
    out_h0_w0_c0 += bias[co_slice0];
  }

  if (act_type == ActType_Relu) {
    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));
  } else if (act_type == ActType_Relu6) {
    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
  }

  if (OW * CO_SLICES <= MAX_IMAGE2D_SIZE) {
    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);
  } else {
    WRITE_IMAGE(output, (int2)(n_oh0 * CO_SLICES + co_slice0, ow0), out_h0_w0_c0);
  }
}

__kernel void Conv2D_H2W1C1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,
                            __global FLT4 *bias, const int4 input_shape, const int4 output_shape,
                            const int4 kernel_stride, const int4 pad, const int2 dilation, const int act_type) {
  const int BlockH = 2;
  const int BlockW = 1;
  const int BlockC = 1;
  DEFINE_ARGS;

  const int oh0 = oh + 0;
  const int oh1 = oh + 1;
  const int n_oh0 = n * OH + oh0;
  const int n_oh1 = n * OH + oh1;
  const int ow0 = ow + 0;
  const int co_slice0 = co_slice + 0;

  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;

  for (int kh = 0; kh < KH; ++kh) {
    const int ih0 = kh * dilationH + oh0 * strideH - padTop;
    // no need to check oh1, finally write out will check (oh1 < OH)
    const int ih1 = kh * dilationH + oh1 * strideH - padTop;
    // check ih0 and ih1
    const int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;
    const int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;

    for (int kw = 0; kw < KW; ++kw) {
      const int iw0 = kw * dilationW + ow0 * strideW - padLeft;
      int x_idx0 = iw0 * CI_SLICES;

      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {
        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));
        FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));
        x_idx0++;

        out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;
        out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;
        out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;
        out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;
        out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;
        out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;
        out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;
        out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;

        weight_ptr += 4;
      }
    }
  }

  if (bias) {
    out_h0_w0_c0 += bias[co_slice0];
    out_h1_w0_c0 += bias[co_slice0];
  }

  if (act_type == ActType_Relu) {
    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));
    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));
  } else if (act_type == ActType_Relu6) {
    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
  }

  if (OW * CO_SLICES <= MAX_IMAGE2D_SIZE) {
    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);
    }  // end if (oh1 < OH)
  } else {
    WRITE_IMAGE(output, (int2)(n_oh0 * CO_SLICES + co_slice0, ow0), out_h0_w0_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(n_oh1 * CO_SLICES + co_slice0, ow0), out_h1_w0_c0);
    }  // end (oh1 < OH)
  }
}

__kernel void Conv2D_H2W1C2(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,
                            __global FLT4 *bias, const int4 input_shape, const int4 output_shape,
                            const int4 kernel_stride, const int4 pad, const int2 dilation, const int act_type) {
  const int BlockH = 2;
  const int BlockW = 1;
  const int BlockC = 2;
  DEFINE_ARGS;

  const int oh0 = oh + 0;
  const int oh1 = oh + 1;
  const int n_oh0 = n * OH + oh0;
  const int n_oh1 = n * OH + oh1;
  const int ow0 = ow + 0;
  const int co_slice0 = co_slice + 0;
  const int co_slice1 = co_slice + 1;

  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;

  for (int kh = 0; kh < KH; ++kh) {
    const int ih0 = kh * dilationH + oh0 * strideH - padTop;
    // no need to check oh1, finally write out will check (oh1 < OH)
    const int ih1 = kh * dilationH + oh1 * strideH - padTop;
    // check ih0 and ih1
    const int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;
    const int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;

    for (int kw = 0; kw < KW; ++kw) {
      const int iw0 = kw * dilationW + ow0 * strideW - padLeft;
      int x_idx0 = iw0 * CI_SLICES;

      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {
        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));
        FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));
        x_idx0++;

        out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;
        out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;
        out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;
        out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;
        out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;
        out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;
        out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;
        out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;

        out_h0_w0_c1 += weight_ptr[4] * in_h0_w0.x;
        out_h1_w0_c1 += weight_ptr[4] * in_h1_w0.x;
        out_h0_w0_c1 += weight_ptr[5] * in_h0_w0.y;
        out_h1_w0_c1 += weight_ptr[5] * in_h1_w0.y;
        out_h0_w0_c1 += weight_ptr[6] * in_h0_w0.z;
        out_h1_w0_c1 += weight_ptr[6] * in_h1_w0.z;
        out_h0_w0_c1 += weight_ptr[7] * in_h0_w0.w;
        out_h1_w0_c1 += weight_ptr[7] * in_h1_w0.w;

        weight_ptr += 8;
      }
    }
  }

  if (bias) {
    out_h0_w0_c0 += bias[co_slice0];
    out_h1_w0_c0 += bias[co_slice0];
    out_h0_w0_c1 += bias[co_slice1];
    out_h1_w0_c1 += bias[co_slice1];
  }

  if (act_type == ActType_Relu) {
    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));
    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));
    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));
    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));
  } else if (act_type == ActType_Relu6) {
    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
  }

  if (OW * CO_SLICES <= MAX_IMAGE2D_SIZE) {
    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);
    }  // end if (oh1 < OH)
    if (co_slice1 < CO_SLICES) {
      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh0), out_h0_w0_c1);
      if (oh1 < OH) {
        WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh1), out_h1_w0_c1);
      }  // end if (oh1 < OH)
    }    // end if (co_slice1 < CO_SLICES)
  } else {
    WRITE_IMAGE(output, (int2)(n_oh0 * CO_SLICES + co_slice0, ow0), out_h0_w0_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(n_oh1 * CO_SLICES + co_slice0, ow0), out_h1_w0_c0);
    }  // end (oh1 < OH)
    if (co_slice1 < CO_SLICES) {
      WRITE_IMAGE(output, (int2)(n_oh0 * CO_SLICES + co_slice1, ow0), out_h0_w0_c1);
      if (oh1 < OH) {
        WRITE_IMAGE(output, (int2)(n_oh1 * CO_SLICES + co_slice1, ow0), out_h1_w0_c1);
      }  // end if (oh1 < OH)
    }    // end if (co_slice1 < CO_SLICES)
  }
}

__kernel void Conv2D_H2W2C2(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,
                            __global FLT4 *bias, const int4 input_shape, const int4 output_shape,
                            const int4 kernel_stride, const int4 pad, const int2 dilation, const int act_type) {
  const int BlockH = 2;
  const int BlockW = 2;
  const int BlockC = 2;
  DEFINE_ARGS;

  const int oh0 = oh + 0;
  const int oh1 = oh + 1;
  const int n_oh0 = n * OH + oh0;
  const int n_oh1 = n * OH + oh1;
  const int ow0 = ow + 0;
  const int ow1 = ow + 1;
  const int co_slice0 = co_slice + 0;
  const int co_slice1 = co_slice + 1;

  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h0_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h0_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;

  for (int kh = 0; kh < KH; ++kh) {
    const int ih0 = kh * dilationH + oh0 * strideH - padTop;
    // no need to check oh1, finally write out will check (oh1 < OH)
    const int ih1 = kh * dilationH + oh1 * strideH - padTop;
    // check ih0 and ih1
    const int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;
    const int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;

    for (int kw = 0; kw < KW; ++kw) {
      const int iw0 = kw * dilationW + ow0 * strideW - padLeft;
      int iw1 = (ow1 < OW) ? kw * dilationW + ow1 * strideW - padLeft : -2;
      int x_idx0 = iw0 * CI_SLICES;
      int x_idx1 = iw1 * CI_SLICES;

      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {
        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));
        FLT4 in_h0_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx0));
        FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));
        FLT4 in_h1_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx1));
        x_idx0++;
        x_idx1++;

        out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;
        out_h0_w1_c0 += weight_ptr[0] * in_h0_w1.x;
        out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;
        out_h1_w1_c0 += weight_ptr[0] * in_h1_w1.x;
        out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;
        out_h0_w1_c0 += weight_ptr[1] * in_h0_w1.y;
        out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;
        out_h1_w1_c0 += weight_ptr[1] * in_h1_w1.y;
        out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;
        out_h0_w1_c0 += weight_ptr[2] * in_h0_w1.z;
        out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;
        out_h1_w1_c0 += weight_ptr[2] * in_h1_w1.z;
        out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;
        out_h0_w1_c0 += weight_ptr[3] * in_h0_w1.w;
        out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;
        out_h1_w1_c0 += weight_ptr[3] * in_h1_w1.w;

        out_h0_w0_c1 += weight_ptr[4] * in_h0_w0.x;
        out_h0_w1_c1 += weight_ptr[4] * in_h0_w1.x;
        out_h1_w0_c1 += weight_ptr[4] * in_h1_w0.x;
        out_h1_w1_c1 += weight_ptr[4] * in_h1_w1.x;
        out_h0_w0_c1 += weight_ptr[5] * in_h0_w0.y;
        out_h0_w1_c1 += weight_ptr[5] * in_h0_w1.y;
        out_h1_w0_c1 += weight_ptr[5] * in_h1_w0.y;
        out_h1_w1_c1 += weight_ptr[5] * in_h1_w1.y;
        out_h0_w0_c1 += weight_ptr[6] * in_h0_w0.z;
        out_h0_w1_c1 += weight_ptr[6] * in_h0_w1.z;
        out_h1_w0_c1 += weight_ptr[6] * in_h1_w0.z;
        out_h1_w1_c1 += weight_ptr[6] * in_h1_w1.z;
        out_h0_w0_c1 += weight_ptr[7] * in_h0_w0.w;
        out_h0_w1_c1 += weight_ptr[7] * in_h0_w1.w;
        out_h1_w0_c1 += weight_ptr[7] * in_h1_w0.w;
        out_h1_w1_c1 += weight_ptr[7] * in_h1_w1.w;

        weight_ptr += 8;
      }
    }
  }

  if (bias) {
    out_h0_w0_c0 += bias[co_slice0];
    out_h0_w1_c0 += bias[co_slice0];
    out_h1_w0_c0 += bias[co_slice0];
    out_h1_w1_c0 += bias[co_slice0];
    out_h0_w0_c1 += bias[co_slice1];
    out_h0_w1_c1 += bias[co_slice1];
    out_h1_w0_c1 += bias[co_slice1];
    out_h1_w1_c1 += bias[co_slice1];
  }

  if (act_type == ActType_Relu) {
    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));
    out_h0_w1_c0 = max(out_h0_w1_c0, (FLT4)(0.0f));
    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));
    out_h1_w1_c0 = max(out_h1_w1_c0, (FLT4)(0.0f));
    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));
    out_h0_w1_c1 = max(out_h0_w1_c1, (FLT4)(0.0f));
    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));
    out_h1_w1_c1 = max(out_h1_w1_c1, (FLT4)(0.0f));
  } else if (act_type == ActType_Relu6) {
    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w1_c0 = clamp(out_h0_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w1_c0 = clamp(out_h1_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w1_c1 = clamp(out_h0_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w1_c1 = clamp(out_h1_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));
  }

  if (OW * CO_SLICES <= MAX_IMAGE2D_SIZE) {
    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);
    WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh0), out_h0_w1_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);
      WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh1), out_h1_w1_c0);
    }  // end if (oh1 < OH)
    if (co_slice1 < CO_SLICES) {
      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh0), out_h0_w0_c1);
      WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh0), out_h0_w1_c1);
      if (oh1 < OH) {
        WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh1), out_h1_w0_c1);
        WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh1), out_h1_w1_c1);
      }  // end if (oh1 < OH)
    }    // end if (co_slice1 < CO_SLICES)
  } else {
    WRITE_IMAGE(output, (int2)(n_oh0 * CO_SLICES + co_slice0, ow0), out_h0_w0_c0);
    WRITE_IMAGE(output, (int2)(n_oh0 * CO_SLICES + co_slice0, ow1), out_h0_w1_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(n_oh1 * CO_SLICES + co_slice0, ow0), out_h1_w0_c0);
      WRITE_IMAGE(output, (int2)(n_oh1 * CO_SLICES + co_slice0, ow1), out_h1_w1_c0);
    }  // end (oh1 < OH)
    if (co_slice1 < CO_SLICES) {
      WRITE_IMAGE(output, (int2)(n_oh0 * CO_SLICES + co_slice1, ow0), out_h0_w0_c1);
      WRITE_IMAGE(output, (int2)(n_oh0 * CO_SLICES + co_slice1, ow1), out_h0_w1_c1);
      if (oh1 < OH) {
        WRITE_IMAGE(output, (int2)(n_oh1 * CO_SLICES + co_slice1, ow0), out_h1_w0_c1);
        WRITE_IMAGE(output, (int2)(n_oh1 * CO_SLICES + co_slice1, ow1), out_h1_w1_c1);
      }  // end if (oh1 < OH)
    }    // end if (co_slice1 < CO_SLICES)
  }
}
