#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define CI_TILE 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

constant FLT Bt[36] = {
  1.0000000000f, 0.0000000000f,  -2.5000004768f, -0.0000001192f, 1.0000001192f,  0.0000000000f,
  0.0000000000f, 0.9428091049f,  1.3333333731f,  -0.4714044929f, -0.6666667461f, 0.0000000000f,
  0.0000000000f, -0.9428089857f, 1.3333334923f,  0.4714045525f,  -0.6666667461f, 0.0000000000f,
  0.0000000000f, -0.1178511307f, -0.0833333358f, 0.2357022613f,  0.1666666865f,  0.0000000000f,
  0.0000000000f, 0.1178511307f,  -0.0833333507f, -0.2357022911f, 0.1666666865f,  0.0000000000f,
  0.0000000000f, 0.9999998808f,  -0.0000000596f, -2.5000000000f, 0.0000000000f,  1.0000000000f,
};

__kernel void Winograd4x4To36(__read_only image2d_t input,    // height=N*H             width=W*CI_SLICES
                              __write_only image2d_t output,  // height=CI_SLICES*36    width=H/4*W/4
                              int4 input_shape,               // N H W CI_SLICES
                              int TILE_HW, int pad) {
  int tile_hw = get_global_id(0);
  int row = get_global_id(1);
  int ci_slice = get_global_id(2);
  int H = input_shape.y;
  int W = input_shape.z;
  int CI_SLICES = input_shape.w;
  if (tile_hw >= TILE_HW || row >= 6 || ci_slice >= CI_SLICES) {
    return;
  }
  int TILE_W = UP_DIV(W, 4);
  int tile_w = tile_hw % TILE_W;
  int tile_h = tile_hw / TILE_W;

  constant FLT *Bt_row = Bt + row * 6;
  FLT4 BtD_row[6] = {0};
  int h = tile_h * 4 - pad;
  int w = tile_w * 4 - pad;

  int x_idx = w * CI_SLICES + ci_slice;
  FLT bt0 = Bt_row[0], bt1 = Bt_row[1], bt2 = Bt_row[2], bt3 = Bt_row[3], bt4 = Bt_row[4], bt5 = Bt_row[5];
  BtD_row[0] =
    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +
    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +
    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));
  x_idx += CI_SLICES;
  BtD_row[1] =
    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +
    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +
    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));
  x_idx += CI_SLICES;
  BtD_row[2] =
    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +
    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +
    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));
  x_idx += CI_SLICES;
  BtD_row[3] =
    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +
    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +
    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));
  x_idx += CI_SLICES;
  BtD_row[4] =
    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +
    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +
    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));
  x_idx += CI_SLICES;
  BtD_row[5] =
    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +
    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +
    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));

#if FP16_ENABLE
#ifndef HALF_MAX  // adreno not exist
#define HALF_MAX 0x1.ffcp15h
#endif
#define LimitAcc()          \
  acc = min(acc, HALF_MAX); \
  acc = max(acc, -HALF_MAX);
#else
#define LimitAcc() \
  {}
#endif

  int y_idx = ci_slice * 36 + row * 6;
  FLT4 acc = BtD_row[0] + (FLT)(-2.5f) * BtD_row[2] + BtD_row[4];
  LimitAcc();
  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);

  FLT4 tmp0 = (FLT)(0.9428091049f) * BtD_row[1] + (FLT)(-0.4714044929f) * BtD_row[3];
  FLT4 tmp1 = (FLT)(1.3333333731f) * BtD_row[2] + (FLT)(-0.6666667461f) * BtD_row[4];
  acc = tmp0 + tmp1;
  LimitAcc();
  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);

  acc = -tmp0 + tmp1;
  LimitAcc();
  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);

  tmp0 = (FLT)(-0.1178511307f) * BtD_row[1] + (FLT)(0.2357022613f) * BtD_row[3];
  tmp1 = (FLT)(-0.0833333358f) * BtD_row[2] + (FLT)(0.1666666865f) * BtD_row[4];
  acc = tmp0 + tmp1;
  LimitAcc();
  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);

  acc = -tmp0 + tmp1;
  LimitAcc();
  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);

  acc = BtD_row[1] + (FLT)(-2.5f) * BtD_row[3] + BtD_row[5];
  LimitAcc();
  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);
}

__kernel void WinogradConv2D(__read_only image2d_t input,    // height=CI_SLICES*36    width=TILE_HW
                             __write_only image2d_t output,  // height=CO_SLICES*36    width=TILE_HW
                             __global FLT16 *weight, int TILE_HW, int CI_SLICES, int CO_SLICES) {
  int tile_hw = get_global_id(0) * 2;
  int h = get_global_id(1);
  int co_slice = get_global_id(2) * 2;
  if (h >= 36 || tile_hw >= TILE_HW || co_slice >= CO_SLICES) {
    return;
  }

  FLT4 out00 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out01 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out10 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out11 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  int y_idx = h;
  __global FLT16 *weight_ptr = weight + (co_slice / 2 * 36 + h) * CI_SLICES * 2;
  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {
    FLT4 in0 = READ_IMAGE(input, smp_zero, (int2)(tile_hw + 0, y_idx));
    FLT4 in1 = READ_IMAGE(input, smp_zero, (int2)(tile_hw + 1, y_idx));
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

  WRITE_IMAGE(output, (int2)(tile_hw + 0, (co_slice + 0) * 36 + h), out00);
  WRITE_IMAGE(output, (int2)(tile_hw + 1, (co_slice + 0) * 36 + h), out01);
  WRITE_IMAGE(output, (int2)(tile_hw + 0, (co_slice + 1) * 36 + h), out10);
  WRITE_IMAGE(output, (int2)(tile_hw + 1, (co_slice + 1) * 36 + h), out11);
}

__kernel void WinogradConv2D_Img(__read_only image2d_t input,    // height=CI_SLICES*36    width=TILE_HW
                                 __write_only image2d_t output,  // height=CO_SLICES*36    width=TILE_HW
                                 __read_only image2d_t weight, int TILE_HW, int CI_SLICES, int CO_SLICES) {
  int tile_hw = get_global_id(0) * 2;
  int h = get_global_id(1);
  int co_slice = get_global_id(2) * 2;
  if (h >= 36 || tile_hw >= TILE_HW || co_slice >= CO_SLICES) {
    return;
  }
  int CI_ALIGN = CI_SLICES * CI_TILE;

  FLT4 out00 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out01 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out10 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  FLT4 out11 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);

  int y_idx = h;
  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {
    FLT4 in0 = READ_IMAGE(input, smp_zero, (int2)(tile_hw + 0, y_idx));
    FLT4 in1 = READ_IMAGE(input, smp_zero, (int2)(tile_hw + 1, y_idx));
    y_idx += 36;

    FLT4 filter_ci0_co0 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 0, co_slice + 0));
    FLT4 filter_ci1_co0 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 1, co_slice + 0));
    FLT4 filter_ci2_co0 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 2, co_slice + 0));
    FLT4 filter_ci3_co0 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 3, co_slice + 0));
    FLT4 filter_ci0_co1 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 0, co_slice + 1));
    FLT4 filter_ci1_co1 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 1, co_slice + 1));
    FLT4 filter_ci2_co1 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 2, co_slice + 1));
    FLT4 filter_ci3_co1 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 3, co_slice + 1));

    out00 += in0.x * filter_ci0_co0;
    out00 += in0.y * filter_ci1_co0;
    out00 += in0.z * filter_ci2_co0;
    out00 += in0.w * filter_ci3_co0;

    out01 += in1.x * filter_ci0_co0;
    out01 += in1.y * filter_ci1_co0;
    out01 += in1.z * filter_ci2_co0;
    out01 += in1.w * filter_ci3_co0;

    out10 += in0.x * filter_ci0_co1;
    out10 += in0.y * filter_ci1_co1;
    out10 += in0.z * filter_ci2_co1;
    out10 += in0.w * filter_ci3_co1;

    out11 += in1.x * filter_ci0_co1;
    out11 += in1.y * filter_ci1_co1;
    out11 += in1.z * filter_ci2_co1;
    out11 += in1.w * filter_ci3_co1;
  }

  WRITE_IMAGE(output, (int2)(tile_hw + 0, (co_slice + 0) * 36 + h), out00);
  WRITE_IMAGE(output, (int2)(tile_hw + 1, (co_slice + 0) * 36 + h), out01);
  WRITE_IMAGE(output, (int2)(tile_hw + 0, (co_slice + 1) * 36 + h), out10);
  WRITE_IMAGE(output, (int2)(tile_hw + 1, (co_slice + 1) * 36 + h), out11);
}

#define DO_LEAKY_RELU(data, alpha)               \
  data.x = data.x > 0 ? data.x : data.x * alpha; \
  data.y = data.y > 0 ? data.y : data.y * alpha; \
  data.z = data.z > 0 ? data.z : data.z * alpha; \
  data.w = data.w > 0 ? data.w : data.w * alpha;

constant FLT At[24] = {1.0000000000f, 1.0000000000f, 1.0000000000f,  1.0000000000f, 1.0000000000f,  0.0000000000f,
                       0.0000000000f, 0.7071067691f, -0.7071067691f, 1.4142135382f, -1.4142135382f, 0.0000000000f,
                       0.0000000000f, 0.4999999702f, 0.4999999702f,  1.9999998808f, 1.9999998808f,  0.0000000000f,
                       0.0000000000f, 0.3535533845f, -0.3535533845f, 2.8284270763f, -2.8284270763f, 1.0000000000f};

#define UpdateAcc()                                   \
  if (bias != 0) acc += bias[co_slice];               \
  if (act_type == ActivationType_RELU) {              \
    acc = max(acc, (FLT4)(0.0f));                     \
  } else if (act_type == ActivationType_RELU6) {      \
    acc = clamp(acc, (FLT4)(0.0f), (FLT4)(6.0f));     \
  } else if (act_type == ActivationType_TANH) {       \
    FLT4 exp0 = exp(acc);                             \
    FLT4 exp1 = exp(-acc);                            \
    acc = (exp0 - exp1) / (exp0 + exp1);              \
  } else if (act_type == ActivationType_LEAKY_RELU) { \
    DO_LEAKY_RELU(acc, alpha);                        \
  } else if (act_type == ActivationType_SIGMOID) {    \
    acc = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-acc));    \
  }

__kernel void Winograd36To4x4(__read_only image2d_t input,    // height=CO_SLICES*36    width=TILE_HW
                              __write_only image2d_t output,  // height=N*H             width=W*CO_SLICES
                              __global FLT4 *bias,
                              int4 output_shape,  // N H W CO_SLICES
                              int TILE_HW, int act_type, float alpha) {
  int tile_hw = get_global_id(0);
  int row = get_global_id(1);
  int co_slice = get_global_id(2);
  int H = output_shape.y;
  int W = output_shape.z;
  int CO_SLICES = output_shape.w;
  if (tile_hw >= TILE_HW || row >= 4 || co_slice >= CO_SLICES) {
    return;
  }

  constant FLT *At_row = At + row * 6;
  FLT4 AtM_row[6] = {0};
  int idx = co_slice * 36;
  FLT at = At_row[0];
  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  at = At_row[1];
  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  at = At_row[2];
  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  at = At_row[3];
  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  at = At_row[4];
  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  at = At_row[5];
  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));
  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));

  int TILE_W = UP_DIV(W, 4);
  int tile_w = tile_hw % TILE_W;
  int tile_h = tile_hw / TILE_W;
  int h = tile_h * 4 + row;
  int w = tile_w * 4;
  int x_idx = w * CO_SLICES + co_slice;

  FLT4 acc = AtM_row[0] + AtM_row[1] + AtM_row[2] + AtM_row[3] + AtM_row[4];
  UpdateAcc();
  WRITE_IMAGE(output, (int2)(x_idx, h), acc);
  x_idx += CO_SLICES;

  acc = (FLT)(0.7071067691f) * (AtM_row[1] - AtM_row[2]) + (FLT)(1.4142135382f) * (AtM_row[3] - AtM_row[4]);
  UpdateAcc();
  WRITE_IMAGE(output, (int2)(x_idx, h), acc);
  x_idx += CO_SLICES;

  acc = (FLT)(0.5f) * (AtM_row[1] + AtM_row[2]) + (FLT)(2.0f) * (AtM_row[3] + AtM_row[4]);
  UpdateAcc();
  WRITE_IMAGE(output, (int2)(x_idx, h), acc);
  x_idx += CO_SLICES;

  acc =
    (FLT)(0.3535533845f) * (AtM_row[1] - AtM_row[2]) + (FLT)(2.8284270763f) * (AtM_row[3] - AtM_row[4]) + AtM_row[5];
  UpdateAcc();
  WRITE_IMAGE(output, (int2)(x_idx, h), acc);
}
