#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

constant FLT Bt[36] = {
  1.0000000000f, 0.0000000000f,  -2.5000004768f, -0.0000001192f, 1.0000001192f,  0.0000000000f,
  0.0000000000f, 0.9428091049f,  1.3333333731f,  -0.4714044929f, -0.6666667461f, 0.0000000000f,
  0.0000000000f, -0.9428089857f, 1.3333334923f,  0.4714045525f,  -0.6666667461f, 0.0000000000f,
  0.0000000000f, -0.1178511307f, -0.0833333358f, 0.2357022613f,  0.1666666865f,  0.0000000000f,
  0.0000000000f, 0.1178511307f,  -0.0833333507f, -0.2357022911f, 0.1666666865f,  0.0000000000f,
  0.0000000000f, 0.9999998808f,  -0.0000000596f, -2.5000000000f, 0.0000000000f,  1.0000000000f,
};

__kernel void Winograd4x4To36(__read_only image2d_t input, __write_only image2d_t output,
                              int4 input_shape,     // N H W CI_SLICES
                              int4 output_shape) {  // N 36 H/4*W/4 CI_SLICES
#define PAD 1
  int tile_xy = get_global_id(0);
  int row = get_global_id(1);
  int slice = get_global_id(2);

  int TILE_XY = output_shape.z;
  int SLICES = input_shape.w;
  if (tile_xy >= TILE_XY || row >= 6 || slice >= SLICES) {
    return;
  }

  int IH = input_shape.y, IW = input_shape.z;
  int TILE_X = UP_DIV(IW, 4);
  int tile_x = tile_xy % TILE_X;
  int tile_y = tile_xy / TILE_X;

  constant FLT *Bt_row = Bt + row * 6;
  FLT4 BtD_row[6] = {0};

  int ih = tile_y * 4 - PAD;
  int iw = tile_x * 4 - PAD;
  for (int y = 0; y < 6; y++) {
    int x_idx = iw * SLICES + slice;
    for (int x = 0; x < 6; x++) {
      // no need to check iw: because slice is in [0, SLICES). when iw<0, x_idx<0; iw>=IW, x_idx>=IW*SLICES
      // if (iw < 0 || iw >= IW) { continue; }
      BtD_row[x] += Bt_row[y] * READ_IMAGE(input, smp_zero, (int2)(x_idx, ih));
      x_idx += SLICES;
    }
    ih++;
  }

  int y_idx = slice * 36 + row * 6;
  for (int y = 0; y < 6; y++) {
    FLT4 acc = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int x = 0; x < 6; x++) {
      acc += BtD_row[x] * Bt[y * 6 + x];
    }
    WRITE_IMAGE(output, (int2)(tile_xy, y_idx + y), acc);  // CH W  H=36
  }
#undef PAD
}

__kernel void WinogradConvolution(__read_only image2d_t input, __write_only image2d_t output, __global FLT16 *weight,
                                  int4 input_shape,     // N 36 H/4*W/4 CI_SLICES
                                  int4 output_shape) {  // N 36 H/4*W/4 CO_SLICES
#define H 36
  int w = get_global_id(0) * 2;
  int h = get_global_id(1);
  int co_slice = get_global_id(2) * 2;

  int CI_SLICES = input_shape.w;
  int W = input_shape.z;
  int CO_SLICES = output_shape.w;

  if (h >= H || w >= W || co_slice >= CO_SLICES) {
    return;
  }

  FLT4 out00 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out01 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out10 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out11 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  int y_idx = h;
  __global FLT16 *weight_ptr = weight + (co_slice / 2 * 36 + h) * CI_SLICES * 2;
  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {
    FLT4 in0 = READ_IMAGE(input, smp_zero, (int2)(w + 0, y_idx));
    FLT4 in1 = READ_IMAGE(input, smp_zero, (int2)(w + 1, y_idx));
    y_idx += 36;

    FLT16 weight0 = weight_ptr[0], weight1 = weight_ptr[1];
    weight_ptr += 2;

    out00 += in0.x * weight0.s0123;
    out00 += in0.y * weight0.s4567;
    out00 += in0.z * weight0.s89ab;
    out00 += in0.w * weight0.scdef;

    out01 += in1.x * weight0.s0123;
    out01 += in1.y * weight0.s4567;
    out01 += in1.z * weight0.s89ab;
    out01 += in1.w * weight0.scdef;

    out10 += in0.x * weight1.s0123;
    out10 += in0.y * weight1.s4567;
    out10 += in0.z * weight1.s89ab;
    out10 += in0.w * weight1.scdef;

    out11 += in1.x * weight1.s0123;
    out11 += in1.y * weight1.s4567;
    out11 += in1.z * weight1.s89ab;
    out11 += in1.w * weight1.scdef;
  }

  WRITE_IMAGE(output, (int2)(w + 0, (co_slice + 0) * H + h), out00);
  if (w + 1 < W) {
    WRITE_IMAGE(output, (int2)(w + 1, (co_slice + 0) * H + h), out01);
  }

  if (co_slice + 1 < CO_SLICES) {
    WRITE_IMAGE(output, (int2)(w + 0, (co_slice + 1) * H + h), out10);
    if (w + 1 < W) {
      WRITE_IMAGE(output, (int2)(w + 1, (co_slice + 1) * H + h), out11);
    }
  }
#undef H
}

constant FLT At[24] = {1.0000000000f, 1.0000000000f, 1.0000000000f,  1.0000000000f, 1.0000000000f,  0.0000000000f,
                       0.0000000000f, 0.7071067691f, -0.7071067691f, 1.4142135382f, -1.4142135382f, 0.0000000000f,
                       0.0000000000f, 0.4999999702f, 0.4999999702f,  1.9999998808f, 1.9999998808f,  0.0000000000f,
                       0.0000000000f, 0.3535533845f, -0.3535533845f, 2.8284270763f, -2.8284270763f, 1.0000000000f};

__kernel void Winograd36To4x4(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *bias,
                              int4 input_shape,   // N 36 H/4*W/4 CO_SLICES
                              int4 output_shape,  // N H W CO_SLICES
                              int act_type, float alpha) {
  int tile_xy = get_global_id(0);
  int row = get_global_id(1);
  int slice = get_global_id(2);

  int TILE_XY = input_shape.z;
  int SLICES = input_shape.w;
  int OH = output_shape.y;
  int OW = output_shape.z;

  if (tile_xy >= TILE_XY || row >= 4 || slice >= SLICES) {
    return;
  }

  constant FLT *At_row = At + row * 6;
  FLT4 AtM_row[6] = {0};
  for (int y = 0, idx = slice * 36; y < 6; y++) {
    for (int x = 0; x < 6; x++, idx++) {
      AtM_row[x] += At_row[y] * READ_IMAGE(input, smp_zero, (int2)(tile_xy, idx));
    }
  }

  int TILE_X = UP_DIV(OW, 4);
  int tile_x = tile_xy % TILE_X;
  int tile_y = tile_xy / TILE_X;
  int oh = tile_y * 4 + row;
  int ow = tile_x * 4;
  int x_idx = ow * SLICES + slice;

  for (int x = 0, idx = 0; x < 4; x++) {
    FLT4 acc = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int y = 0; y < 6; y++, idx++) {
      acc += AtM_row[y] * At[idx];
    }

    if (bias != 0) {
      acc += bias[slice];
    }

    if (act_type == ActivationType_RELU) {
      acc = max(acc, (FLT4)(0.0f));
    } else if (act_type == ActivationType_RELU6) {
      acc = clamp(acc, (FLT4)(0.0f), (FLT4)(6.0f));
    } else if (act_type == ActivationType_TANH) {
      FLT4 exp0 = exp(acc);
      FLT4 exp1 = exp(-acc);
      acc = (exp0 - exp1) / (exp0 + exp1);
    } else if (act_type == ActivationType_LEAKY_RELU) {
      if (acc.x < 0) acc.x *= alpha;
      if (acc.y < 0) acc.y *= alpha;
      if (acc.z < 0) acc.z *= alpha;
      if (acc.w < 0) acc.w *= alpha;
    } else if (act_type == ActivationType_SIGMOID) {
      acc = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-acc));
    }

    WRITE_IMAGE(output, (int2)(x_idx, oh), acc);
    x_idx += SLICES;
  }
}
