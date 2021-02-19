#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define CI_TILE 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

#define DEFINE_ARGS                                                         \
  int N = input_shape.x;                                                    \
  int IH = input_shape.y, IW = input_shape.z, CI_SLICES = input_shape.w;    \
  int OH = output_shape.y, OW = output_shape.z, CO_SLICES = output_shape.w; \
  int KH = kernel_stride.x, KW = kernel_stride.y;                           \
  int strideH = kernel_stride.z, strideW = kernel_stride.w;                 \
  int padTop = pad.x, padBottom = pad.y, padLeft = pad.z, padRight = pad.w; \
  int dilationH = dilation.x, dilationW = dilation.y;                       \
                                                                            \
  int n_oh = get_global_id(0);                                              \
  int ow = get_global_id(1) * BlockW;                                       \
  int co_slice = get_global_id(2) * BlockC;                                 \
  int OH_SLICES = UP_DIV(OH, BlockH);                                       \
  int n = n_oh / OH_SLICES;                                                 \
  int oh = (n_oh % OH_SLICES) * BlockH;                                     \
  if (n >= N || oh >= OH || ow >= OW || co_slice >= CO_SLICES) {            \
    return;                                                                 \
  }

#define DO_TANH(data) \
  exp0 = exp(data);   \
  exp1 = exp(-data);  \
  data = (exp0 - exp1) / (exp0 + exp1);

#define DO_LEAKY_RELU(data, alpha)               \
  data.x = data.x > 0 ? data.x : data.x * alpha; \
  data.y = data.y > 0 ? data.y : data.y * alpha; \
  data.z = data.z > 0 ? data.z : data.z * alpha; \
  data.w = data.w > 0 ? data.w : data.w * alpha;

__kernel void Conv2D_H1W1C1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,
                            __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,
                            int2 dilation, int act_type, float alpha) {
  const int BlockH = 1;
  const int BlockW = 1;
  const int BlockC = 1;
  DEFINE_ARGS;

  int oh0 = oh + 0;
  int n_oh0 = n * OH + oh0;
  int ow0 = ow + 0;
  int co_slice0 = co_slice + 0;

  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;

  for (int kh = 0; kh < KH; ++kh) {
    int ih0 = kh * dilationH + oh0 * strideH - padTop;
    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;

    for (int kw = 0; kw < KW; ++kw) {
      int iw0 = kw * dilationW + ow0 * strideW - padLeft;
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

  if (bias != 0) {
    out_h0_w0_c0 += bias[co_slice0];
  }

  if (act_type == ActivationType_RELU) {
    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));
  } else if (act_type == ActivationType_RELU6) {
    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
  } else if (act_type == ActivationType_TANH) {
    FLT4 exp0, exp1;
    DO_TANH(out_h0_w0_c0);
  } else if (act_type == ActivationType_LEAKY_RELU) {
    DO_LEAKY_RELU(out_h0_w0_c0, alpha);
  } else if (act_type == ActivationType_SIGMOID) {
    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));
  }

  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {
    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);
  } else {
    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);
  }
}

__kernel void Conv2D_H2W1C1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,
                            __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,
                            int2 dilation, int act_type, float alpha) {
  const int BlockH = 2;
  const int BlockW = 1;
  const int BlockC = 1;
  DEFINE_ARGS;

  int oh0 = oh + 0;
  int oh1 = oh + 1;
  int n_oh0 = n * OH + oh0;
  int n_oh1 = n * OH + oh1;
  int ow0 = ow + 0;
  int co_slice0 = co_slice + 0;

  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;

  for (int kh = 0; kh < KH; ++kh) {
    int ih0 = kh * dilationH + oh0 * strideH - padTop;
    // no need to check oh1, finally write out will check (oh1 < OH)
    int ih1 = kh * dilationH + oh1 * strideH - padTop;
    // check ih0 and ih1
    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;
    int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;

    for (int kw = 0; kw < KW; ++kw) {
      int iw0 = kw * dilationW + ow0 * strideW - padLeft;
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

  if (bias != 0) {
    out_h0_w0_c0 += bias[co_slice0];
    out_h1_w0_c0 += bias[co_slice0];
  }

  if (act_type == ActivationType_RELU) {
    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));
    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));
  } else if (act_type == ActivationType_RELU6) {
    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
  } else if (act_type == ActivationType_TANH) {
    FLT4 exp0, exp1;
    DO_TANH(out_h0_w0_c0);
    DO_TANH(out_h1_w0_c0);
  } else if (act_type == ActivationType_LEAKY_RELU) {
    DO_LEAKY_RELU(out_h0_w0_c0, alpha);
    DO_LEAKY_RELU(out_h1_w0_c0, alpha);
  } else if (act_type == ActivationType_SIGMOID) {
    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));
    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));
  }

  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {
    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);
    }  // end if (oh1 < OH)
  } else {
    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);
    }  // end (oh1 < OH)
  }
}

__kernel void Conv2D_H2W1C2(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,
                            __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,
                            int2 dilation, int act_type, float alpha) {
  const int BlockH = 2;
  const int BlockW = 1;
  const int BlockC = 2;
  DEFINE_ARGS;

  int oh0 = oh + 0;
  int oh1 = oh + 1;
  int n_oh0 = n * OH + oh0;
  int n_oh1 = n * OH + oh1;
  int ow0 = ow + 0;
  int co_slice0 = co_slice + 0;
  int co_slice1 = co_slice + 1;

  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;

  for (int kh = 0; kh < KH; ++kh) {
    int ih0 = kh * dilationH + oh0 * strideH - padTop;
    // no need to check oh1, finally write out will check (oh1 < OH)
    int ih1 = kh * dilationH + oh1 * strideH - padTop;
    // check ih0 and ih1
    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;
    int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;

    for (int kw = 0; kw < KW; ++kw) {
      int iw0 = kw * dilationW + ow0 * strideW - padLeft;
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

  if (bias != 0) {
    out_h0_w0_c0 += bias[co_slice0];
    out_h1_w0_c0 += bias[co_slice0];
    out_h0_w0_c1 += bias[co_slice1];
    out_h1_w0_c1 += bias[co_slice1];
  }

  if (act_type == ActivationType_RELU) {
    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));
    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));
    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));
    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));
  } else if (act_type == ActivationType_RELU6) {
    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
  } else if (act_type == ActivationType_TANH) {
    FLT4 exp0, exp1;
    DO_TANH(out_h0_w0_c0);
    DO_TANH(out_h1_w0_c0);
    DO_TANH(out_h0_w0_c1);
    DO_TANH(out_h1_w0_c1);
  } else if (act_type == ActivationType_LEAKY_RELU) {
    DO_LEAKY_RELU(out_h0_w0_c0, alpha);
    DO_LEAKY_RELU(out_h1_w0_c0, alpha);
    DO_LEAKY_RELU(out_h0_w0_c1, alpha);
    DO_LEAKY_RELU(out_h1_w0_c1, alpha);
  } else if (act_type == ActivationType_SIGMOID) {
    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));
    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));
    out_h0_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c1));
    out_h1_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c1));
  }

  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {
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
    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);
    }  // end (oh1 < OH)
    if (co_slice1 < CO_SLICES) {
      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow0), out_h0_w0_c1);
      if (oh1 < OH) {
        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow0), out_h1_w0_c1);
      }  // end if (oh1 < OH)
    }    // end if (co_slice1 < CO_SLICES)
  }
}

__kernel void Conv2D_H2W2C2(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,
                            __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,
                            int2 dilation, int act_type, float alpha) {
  const int BlockH = 2;
  const int BlockW = 2;
  const int BlockC = 2;
  DEFINE_ARGS;

  int oh0 = oh + 0;
  int oh1 = oh + 1;
  int n_oh0 = n * OH + oh0;
  int n_oh1 = n * OH + oh1;
  int ow0 = ow + 0;
  int ow1 = ow + 1;
  int co_slice0 = co_slice + 0;
  int co_slice1 = co_slice + 1;

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
    int ih0 = kh * dilationH + oh0 * strideH - padTop;
    // no need to check oh1, finally write out will check (oh1 < OH)
    int ih1 = kh * dilationH + oh1 * strideH - padTop;
    // check ih0 and ih1
    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;
    int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;

    for (int kw = 0; kw < KW; ++kw) {
      int iw0 = kw * dilationW + ow0 * strideW - padLeft;
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

  if (bias != 0) {
    out_h0_w0_c0 += bias[co_slice0];
    out_h0_w1_c0 += bias[co_slice0];
    out_h1_w0_c0 += bias[co_slice0];
    out_h1_w1_c0 += bias[co_slice0];
    out_h0_w0_c1 += bias[co_slice1];
    out_h0_w1_c1 += bias[co_slice1];
    out_h1_w0_c1 += bias[co_slice1];
    out_h1_w1_c1 += bias[co_slice1];
  }

  if (act_type == ActivationType_RELU) {
    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));
    out_h0_w1_c0 = max(out_h0_w1_c0, (FLT4)(0.0f));
    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));
    out_h1_w1_c0 = max(out_h1_w1_c0, (FLT4)(0.0f));
    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));
    out_h0_w1_c1 = max(out_h0_w1_c1, (FLT4)(0.0f));
    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));
    out_h1_w1_c1 = max(out_h1_w1_c1, (FLT4)(0.0f));
  } else if (act_type == ActivationType_RELU6) {
    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w1_c0 = clamp(out_h0_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w1_c0 = clamp(out_h1_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w1_c1 = clamp(out_h0_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w1_c1 = clamp(out_h1_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));
  } else if (act_type == ActivationType_TANH) {
    FLT4 exp0, exp1;
    DO_TANH(out_h0_w0_c0);
    DO_TANH(out_h0_w1_c0);
    DO_TANH(out_h1_w0_c0);
    DO_TANH(out_h1_w1_c0);
    DO_TANH(out_h0_w0_c1);
    DO_TANH(out_h0_w1_c1);
    DO_TANH(out_h1_w0_c1);
    DO_TANH(out_h1_w1_c1);
  } else if (act_type == ActivationType_LEAKY_RELU) {
    DO_LEAKY_RELU(out_h0_w0_c0, alpha);
    DO_LEAKY_RELU(out_h0_w1_c0, alpha);
    DO_LEAKY_RELU(out_h1_w0_c0, alpha);
    DO_LEAKY_RELU(out_h1_w1_c0, alpha);
    DO_LEAKY_RELU(out_h0_w0_c1, alpha);
    DO_LEAKY_RELU(out_h0_w1_c1, alpha);
    DO_LEAKY_RELU(out_h1_w0_c1, alpha);
    DO_LEAKY_RELU(out_h1_w1_c1, alpha);
  } else if (act_type == ActivationType_SIGMOID) {
    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));
    out_h0_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c0));
    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));
    out_h1_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c0));
    out_h0_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c1));
    out_h0_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c1));
    out_h1_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c1));
    out_h1_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c1));
  }

  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {
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
    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);
    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow1), out_h0_w1_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);
      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow1), out_h1_w1_c0);
    }  // end (oh1 < OH)
    if (co_slice1 < CO_SLICES) {
      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow0), out_h0_w0_c1);
      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow1), out_h0_w1_c1);
      if (oh1 < OH) {
        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow0), out_h1_w0_c1);
        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow1), out_h1_w1_c1);
      }  // end if (oh1 < OH)
    }    // end if (co_slice1 < CO_SLICES)
  }
}

__kernel void Conv2D_H2W2C2_Img(__read_only image2d_t input, __write_only image2d_t output,
                                __read_only image2d_t weight, __global FLT4 *bias, int4 input_shape, int4 output_shape,
                                int4 kernel_stride, int4 pad, int2 dilation, int act_type, float alpha) {
  const int BlockH = 2;
  const int BlockW = 2;
  const int BlockC = 2;
  DEFINE_ARGS;

  int oh0 = oh + 0;
  int oh1 = oh + 1;
  int n_oh0 = n * OH + oh0;
  int n_oh1 = n * OH + oh1;
  int ow0 = ow + 0;
  int ow1 = ow + 1;
  int co_slice0 = co_slice + 0;
  int co_slice1 = co_slice + 1;

  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h0_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h0_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out_h1_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  int filter_offset = 0;
  for (int kh = 0; kh < KH; ++kh) {
    int ih0 = kh * dilationH + oh0 * strideH - padTop;
    // no need to check oh1, finally write out will check (oh1 < OH)
    int ih1 = kh * dilationH + oh1 * strideH - padTop;
    // check ih0 and ih1
    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;
    int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;

    for (int kw = 0; kw < KW; ++kw) {
      int iw0 = kw * dilationW + ow0 * strideW - padLeft;
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

        FLT4 filter_ci0_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 0));
        FLT4 filter_ci1_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 1));
        FLT4 filter_ci2_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 2));
        FLT4 filter_ci3_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 3));
        FLT4 filter_ci0_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 0));
        FLT4 filter_ci1_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 1));
        FLT4 filter_ci2_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 2));
        FLT4 filter_ci3_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 3));
        filter_offset += 4;

        out_h0_w0_c0 += filter_ci0_co0 * in_h0_w0.x;
        out_h0_w1_c0 += filter_ci0_co0 * in_h0_w1.x;
        out_h1_w0_c0 += filter_ci0_co0 * in_h1_w0.x;
        out_h1_w1_c0 += filter_ci0_co0 * in_h1_w1.x;
        out_h0_w0_c0 += filter_ci1_co0 * in_h0_w0.y;
        out_h0_w1_c0 += filter_ci1_co0 * in_h0_w1.y;
        out_h1_w0_c0 += filter_ci1_co0 * in_h1_w0.y;
        out_h1_w1_c0 += filter_ci1_co0 * in_h1_w1.y;
        out_h0_w0_c0 += filter_ci2_co0 * in_h0_w0.z;
        out_h0_w1_c0 += filter_ci2_co0 * in_h0_w1.z;
        out_h1_w0_c0 += filter_ci2_co0 * in_h1_w0.z;
        out_h1_w1_c0 += filter_ci2_co0 * in_h1_w1.z;
        out_h0_w0_c0 += filter_ci3_co0 * in_h0_w0.w;
        out_h0_w1_c0 += filter_ci3_co0 * in_h0_w1.w;
        out_h1_w0_c0 += filter_ci3_co0 * in_h1_w0.w;
        out_h1_w1_c0 += filter_ci3_co0 * in_h1_w1.w;

        out_h0_w0_c1 += filter_ci0_co1 * in_h0_w0.x;
        out_h0_w1_c1 += filter_ci0_co1 * in_h0_w1.x;
        out_h1_w0_c1 += filter_ci0_co1 * in_h1_w0.x;
        out_h1_w1_c1 += filter_ci0_co1 * in_h1_w1.x;
        out_h0_w0_c1 += filter_ci1_co1 * in_h0_w0.y;
        out_h0_w1_c1 += filter_ci1_co1 * in_h0_w1.y;
        out_h1_w0_c1 += filter_ci1_co1 * in_h1_w0.y;
        out_h1_w1_c1 += filter_ci1_co1 * in_h1_w1.y;
        out_h0_w0_c1 += filter_ci2_co1 * in_h0_w0.z;
        out_h0_w1_c1 += filter_ci2_co1 * in_h0_w1.z;
        out_h1_w0_c1 += filter_ci2_co1 * in_h1_w0.z;
        out_h1_w1_c1 += filter_ci2_co1 * in_h1_w1.z;
        out_h0_w0_c1 += filter_ci3_co1 * in_h0_w0.w;
        out_h0_w1_c1 += filter_ci3_co1 * in_h0_w1.w;
        out_h1_w0_c1 += filter_ci3_co1 * in_h1_w0.w;
        out_h1_w1_c1 += filter_ci3_co1 * in_h1_w1.w;
      }
    }
  }

  if (bias != 0) {
    out_h0_w0_c0 += bias[co_slice0];
    out_h0_w1_c0 += bias[co_slice0];
    out_h1_w0_c0 += bias[co_slice0];
    out_h1_w1_c0 += bias[co_slice0];
    out_h0_w0_c1 += bias[co_slice1];
    out_h0_w1_c1 += bias[co_slice1];
    out_h1_w0_c1 += bias[co_slice1];
    out_h1_w1_c1 += bias[co_slice1];
  }

  if (act_type == ActivationType_RELU) {
    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));
    out_h0_w1_c0 = max(out_h0_w1_c0, (FLT4)(0.0f));
    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));
    out_h1_w1_c0 = max(out_h1_w1_c0, (FLT4)(0.0f));
    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));
    out_h0_w1_c1 = max(out_h0_w1_c1, (FLT4)(0.0f));
    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));
    out_h1_w1_c1 = max(out_h1_w1_c1, (FLT4)(0.0f));
  } else if (act_type == ActivationType_RELU6) {
    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w1_c0 = clamp(out_h0_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w1_c0 = clamp(out_h1_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h0_w1_c1 = clamp(out_h0_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));
    out_h1_w1_c1 = clamp(out_h1_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));
  } else if (act_type == ActivationType_TANH) {
    FLT4 exp0, exp1;
    DO_TANH(out_h0_w0_c0);
    DO_TANH(out_h0_w1_c0);
    DO_TANH(out_h1_w0_c0);
    DO_TANH(out_h1_w1_c0);
    DO_TANH(out_h0_w0_c1);
    DO_TANH(out_h0_w1_c1);
    DO_TANH(out_h1_w0_c1);
    DO_TANH(out_h1_w1_c1);
  } else if (act_type == ActivationType_LEAKY_RELU) {
    DO_LEAKY_RELU(out_h0_w0_c0, alpha);
    DO_LEAKY_RELU(out_h0_w1_c0, alpha);
    DO_LEAKY_RELU(out_h1_w0_c0, alpha);
    DO_LEAKY_RELU(out_h1_w1_c0, alpha);
    DO_LEAKY_RELU(out_h0_w0_c1, alpha);
    DO_LEAKY_RELU(out_h0_w1_c1, alpha);
    DO_LEAKY_RELU(out_h1_w0_c1, alpha);
    DO_LEAKY_RELU(out_h1_w1_c1, alpha);
  } else if (act_type == ActivationType_SIGMOID) {
    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));
    out_h0_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c0));
    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));
    out_h1_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c0));
    out_h0_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c1));
    out_h0_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c1));
    out_h1_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c1));
    out_h1_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c1));
  }

  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {
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
    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);
    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow1), out_h0_w1_c0);
    if (oh1 < OH) {
      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);
      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow1), out_h1_w1_c0);
    }  // end (oh1 < OH)
    if (co_slice1 < CO_SLICES) {
      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow0), out_h0_w0_c1);
      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow1), out_h0_w1_c1);
      if (oh1 < OH) {
        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow0), out_h1_w0_c1);
        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow1), out_h1_w1_c1);
      }  // end if (oh1 < OH)
    }    // end if (co_slice1 < CO_SLICES)
  }
}
