/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/fp16/winograd_utils_fp16.h"
#include "nnacl/fp16/matrix_fp16.h"

#define MIN_UNIT 2
#define MAX_UNIT 8

void GeneralInputTransformUnitFp16(const float16_t *src_data, float16_t *dst_data, float16_t *matrix_b,
                                   float16_t *matrix_bt, int src_step, int dst_step, int in_unit) {
  int len = in_unit * in_unit;
  if (len > MAX_LEN) return;
  float16x8_t src[MAX_LEN];
  float16x8_t t[MAX_LEN];
  float16x8_t m[MAX_LEN];
  float16x8_t vec_b[MAX_LEN];
  float16x8_t vec_bt[MAX_LEN];
  for (int i = 0; i < len; i++) {
    src[i] = vld1q_f16(src_data + i * src_step);
    vec_b[i] = vdupq_n_f16(matrix_b[i]);
    vec_bt[i] = vdupq_n_f16(matrix_bt[i]);
  }
  MatrixMultiplyVecFp16(vec_bt, src, t, NULL, in_unit, in_unit, in_unit);
  MatrixMultiplyVecFp16(t, vec_b, m, NULL, in_unit, in_unit, in_unit);
  for (int i = 0; i < len; i++) {
    int dst_step_offset = i * dst_step;
    vst1_f16(dst_data + dst_step_offset, vget_low_f16(m[i]));
    vst1_f16(dst_data + dst_step_offset + 64, vget_high_f16(m[i]));
  }
}

void GeneralOutputTransformUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    float16_t *matrix_a, float16_t *matrix_at, int src_step, int dst_step, int in_unit,
                                    int out_unit) {
  int src_len = in_unit * in_unit;
  if (src_len > MAX_LEN) {
    return;
  }
  float16x8_t src[MAX_LEN];
  float16x8_t t[MAX_LEN];
  float16x8_t m[MAX_LEN];
  float16x8_t vec_a[MAX_LEN];
  float16x8_t vec_at[MAX_LEN];
  int tmp_len = in_unit * out_unit;
  if (tmp_len > MAX_LEN) return;

  for (int i = 0; i < tmp_len; i++) {
    vec_a[i] = vdupq_n_f16(matrix_a[i]);
    vec_at[i] = vdupq_n_f16(matrix_at[i]);
  }
  for (int i = 0; i < src_len; i++) {
    src[i] = vld1q_f16(src_data + i * src_step);
  }
  MatrixMultiplyVecFp16(vec_at, src, t, NULL, out_unit, in_unit, in_unit);
  MatrixMultiplyVecFp16(t, vec_a, m, bias_data, out_unit, in_unit, out_unit);

  for (int i = 0; i < out_unit; i++) {
    int dst_k_offset = i * dst_step * C8NUM;
    int m_k_offset = i * out_unit;
    for (int j = 0; j < out_unit; j++) {
      vst1q_f16(dst_data + dst_k_offset + j * C8NUM, m[m_k_offset + j]);
    }
  }
}

static InputTransFp16Func InputTransFp16FuncList[] = {
  NULL, NULL, NULL, NULL, InputTransform4x4UnitFp16, NULL, InputTransform6x6UnitFp16, NULL, InputTransform8x8UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncList4[] = {NULL, NULL, OutputTransform4x2UnitFp16,
                                                         OutputTransform4x3UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncReluList4[] = {NULL, NULL, OutputTransform4x2ReluUnitFp16,
                                                             OutputTransform4x3ReluUnitFp16};
static OutputTransFp16Func OutputTransFp16FuncRelu6List4[] = {NULL, NULL, OutputTransform4x2Relu6UnitFp16,
                                                              OutputTransform4x3Relu6UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncList6[] = {NULL,
                                                         NULL,
                                                         OutputTransform6x2UnitFp16,
                                                         OutputTransform6x3UnitFp16,
                                                         OutputTransform6x4UnitFp16,
                                                         OutputTransform6x5UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncReluList6[] = {NULL,
                                                             NULL,
                                                             OutputTransform6x2ReluUnitFp16,
                                                             OutputTransform6x3ReluUnitFp16,
                                                             OutputTransform6x4ReluUnitFp16,
                                                             OutputTransform6x5ReluUnitFp16};

static OutputTransFp16Func OutputTransFp16FuncRelu6List6[] = {NULL,
                                                              NULL,
                                                              OutputTransform6x2Relu6UnitFp16,
                                                              OutputTransform6x3Relu6UnitFp16,
                                                              OutputTransform6x4Relu6UnitFp16,
                                                              OutputTransform6x5Relu6UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncList8[] = {NULL,
                                                         NULL,
                                                         OutputTransform8x2UnitFp16,
                                                         OutputTransform8x3UnitFp16,
                                                         OutputTransform8x4UnitFp16,
                                                         OutputTransform8x5UnitFp16,
                                                         OutputTransform8x6UnitFp16,
                                                         OutputTransform8x7UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncReluList8[] = {NULL,
                                                             NULL,
                                                             OutputTransform8x2ReluUnitFp16,
                                                             OutputTransform8x3ReluUnitFp16,
                                                             OutputTransform8x4ReluUnitFp16,
                                                             OutputTransform8x5ReluUnitFp16,
                                                             OutputTransform8x6ReluUnitFp16,
                                                             OutputTransform8x7ReluUnitFp16};

static OutputTransFp16Func OutputTransFp16FuncRelu6List8[] = {NULL,
                                                              NULL,
                                                              OutputTransform8x2Relu6UnitFp16,
                                                              OutputTransform8x3Relu6UnitFp16,
                                                              OutputTransform8x4Relu6UnitFp16,
                                                              OutputTransform8x5Relu6UnitFp16,
                                                              OutputTransform8x6Relu6UnitFp16,
                                                              OutputTransform8x7Relu6UnitFp16};

InputTransFp16Func GetInputTransFp16Func(int input_unit) { return InputTransFp16FuncList[input_unit]; }

void InputTransform4x4UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step) {
  float16x8_t src[16];
  float16x8_t t[16];
  float16x8_t m[16];
  Load16DataFp16;
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = vsubq_f16(src[offset], src[2 + offset]);
    t[4 + l] = vaddq_f16(src[1 + offset], src[2 + offset]);
    t[8 + l] = vsubq_f16(src[2 + offset], src[1 + offset]);
    t[12 + l] = vsubq_f16(src[3 + offset], src[1 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    m[l] = vsubq_f16(t[offset], t[2 + offset]);
    m[4 + l] = vaddq_f16(t[1 + offset], t[2 + offset]);
    m[8 + l] = vsubq_f16(t[2 + offset], t[1 + offset]);
    m[12 + l] = vsubq_f16(t[3 + offset], t[1 + offset]);
  }
  for (int i = 0; i < 16; i++) {
    int dst_offset = i * dst_step;
    vst1_f16(dst_data + dst_offset, vget_low_f16(m[i]));
    vst1_f16(dst_data + dst_offset + 64, vget_high_f16(m[i]));
  }
}

void InputTransform6x6UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step) {
  float16x8_t src[36];
  float16x8_t t[36];
  float16x8_t m[36];
  Load36DataFp16;
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vsubq_f16(src[3 + offset], src[1 + offset]);
    float16x8_t tmp2 = vsubq_f16(src[4 + offset], src[2 + offset]);
    t[l] = vaddq_f16(vsubq_f16(vmulq_n_f16(src[offset], 4), vmulq_n_f16(src[2 + offset], 5)), src[4 + offset]);
    t[6 + l] = vaddq_f16(vmulq_n_f16(vaddq_f16(src[1 + offset], src[2 + offset]), -4),
                         vaddq_f16(src[3 + offset], src[4 + offset]));
    t[12 + l] = vaddq_f16(vmulq_n_f16(vsubq_f16(src[1 + offset], src[2 + offset]), 4),
                          vsubq_f16(src[4 + offset], src[3 + offset]));
    t[18 + l] = vaddq_f16(vmulq_n_f16(tmp1, 2), tmp2);
    t[24 + l] = vaddq_f16(vmulq_n_f16(tmp1, -2), tmp2);
    t[30 + l] = vaddq_f16(vsubq_f16(vmulq_n_f16(src[1 + offset], 4), vmulq_n_f16(src[3 + offset], 5)), src[5 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vsubq_f16(t[3 + offset], t[1 + offset]);
    float16x8_t tmp2 = vsubq_f16(t[4 + offset], t[2 + offset]);
    m[l] = vaddq_f16(vsubq_f16(vmulq_n_f16(t[offset], 4), vmulq_n_f16(t[2 + offset], 5)), t[4 + offset]);
    m[6 + l] =
      vaddq_f16(vmulq_n_f16(vaddq_f16(t[1 + offset], t[2 + offset]), -4), vaddq_f16(t[3 + offset], t[4 + offset]));
    m[12 + l] =
      vaddq_f16(vmulq_n_f16(vsubq_f16(t[1 + offset], t[2 + offset]), 4), vsubq_f16(t[4 + offset], t[3 + offset]));
    m[18 + l] = vaddq_f16(vmulq_n_f16(tmp1, 2), tmp2);
    m[24 + l] = vaddq_f16(vmulq_n_f16(tmp1, -2), tmp2);
    m[30 + l] = vaddq_f16(vsubq_f16(vmulq_n_f16(t[1 + offset], 4), vmulq_n_f16(t[3 + offset], 5)), t[5 + offset]);
  }
  for (int i = 0; i < 36; i++) {
    int dst_offset = i * dst_step;
    vst1_f16(dst_data + dst_offset, vget_low_f16(m[i]));
    vst1_f16(dst_data + dst_offset + 64, vget_high_f16(m[i]));
  }
}

void InputTransform8x8UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step) {
  float16x8_t src[64];
  float16x8_t t[64];
  float16x8_t m[64];
  Load64DataFp16;
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    t[l] = vsubq_f16(vaddq_f16(vsubq_f16(vmulq_n_f16(src[offset], 0.5625), vmulq_n_f16(src[2 + offset], 3.0625)),
                               vmulq_n_f16(src[4 + offset], 3.5)),
                     src[6 + offset]);
    float16x8_t tmp1 = vaddq_f16(vmulq_n_f16(src[1 + offset], 1.125), vmulq_n_f16(src[5 + offset], 0.5));
    float16x8_t tmp2 = vsubq_f16(vmulq_n_f16(src[2 + offset], 2.25), vmulq_n_f16(src[4 + offset], 3.25));
    t[8 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(src[3 + offset], 1.625)), src[6 + offset]);
    t[16 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(src[3 + offset], 1.625)), src[6 + offset]);
    tmp1 = vaddq_f16(vmulq_n_f16(src[1 + offset], 0.5625), src[5 + offset]);
    tmp2 = vsubq_f16(vmulq_n_f16(src[2 + offset], 0.5625), vmulq_n_f16(src[4 + offset], 2.5));
    t[24 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(src[3 + offset], 2.5)), src[6 + offset]);
    t[32 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(src[3 + offset], 2.5)), src[6 + offset]);
    tmp1 = vaddq_f16(vmulq_n_f16(src[1 + offset], 0.375), vmulq_n_f16(src[5 + offset], 1.5));
    tmp2 = vsubq_f16(vmulq_n_f16(src[2 + offset], 0.25), vmulq_n_f16(src[4 + offset], 1.25));
    t[40 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(src[3 + offset], 1.875)), src[6 + offset]);
    t[48 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(src[3 + offset], 1.875)), src[6 + offset]);
    t[56 + l] =
      vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src[1 + offset], -0.5625), vmulq_n_f16(src[3 + offset], 3.0625)),
                          vmulq_n_f16(src[5 + offset], 3.5)),
                src[7 + offset]);
  }
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    m[l] = vsubq_f16(vaddq_f16(vsubq_f16(vmulq_n_f16(t[offset], 0.5625), vmulq_n_f16(t[2 + offset], 3.0625)),
                               vmulq_n_f16(t[4 + offset], 3.5)),
                     t[6 + offset]);
    float16x8_t tmp1 = vaddq_f16(vmulq_n_f16(t[1 + offset], 1.125), vmulq_n_f16(t[5 + offset], 0.5));
    float16x8_t tmp2 = vsubq_f16(vmulq_n_f16(t[2 + offset], 2.25), vmulq_n_f16(t[4 + offset], 3.25));
    m[8 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(t[3 + offset], 1.625)), t[6 + offset]);
    m[16 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(t[3 + offset], 1.625)), t[6 + offset]);
    tmp1 = vaddq_f16(vmulq_n_f16(t[1 + offset], 0.5625), t[5 + offset]);
    tmp2 = vsubq_f16(vmulq_n_f16(t[2 + offset], 0.5625), vmulq_n_f16(t[4 + offset], 2.5));
    m[24 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(t[3 + offset], 2.5)), t[6 + offset]);
    m[32 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(t[3 + offset], 2.5)), t[6 + offset]);
    tmp1 = vaddq_f16(vmulq_n_f16(t[1 + offset], 0.375), vmulq_n_f16(t[5 + offset], 1.5));
    tmp2 = vsubq_f16(vmulq_n_f16(t[2 + offset], 0.25), vmulq_n_f16(t[4 + offset], 1.25));
    m[40 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(t[3 + offset], 1.875)), t[6 + offset]);
    m[48 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(t[3 + offset], 1.875)), t[6 + offset]);
    m[56 + l] = vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t[1 + offset], -0.5625), vmulq_n_f16(t[3 + offset], 3.0625)),
                                    vmulq_n_f16(t[5 + offset], 3.5)),
                          t[7 + offset]);
  }
  for (int i = 0; i < 64; i++) {
    int dst_offset = i * dst_step;
    vst1_f16(dst_data + dst_offset, vget_low_f16(m[i]));
    vst1_f16(dst_data + dst_offset + 64, vget_high_f16(m[i]));
  }
}

OutputTransFp16Func GetOutputTransFp16Func(int input_unit, int output_unit, ActType act_type) {
  if (input_unit == 4 && output_unit < 4) {
    if (act_type == ActType_Relu) {
      return OutputTransFp16FuncReluList4[output_unit];
    } else if (act_type == ActType_Relu6) {
      return OutputTransFp16FuncRelu6List4[output_unit];
    } else {
      return OutputTransFp16FuncList4[output_unit];
    }
  } else if (input_unit == 6 && output_unit < 6) {
    if (act_type == ActType_Relu) {
      return OutputTransFp16FuncReluList6[output_unit];
    } else if (act_type == ActType_Relu6) {
      return OutputTransFp16FuncRelu6List6[output_unit];
    } else {
      return OutputTransFp16FuncList6[output_unit];
    }
  } else if (input_unit == 8 && output_unit < 8) {
    if (act_type == ActType_Relu) {
      return OutputTransFp16FuncReluList8[output_unit];
    } else if (act_type == ActType_Relu6) {
      return OutputTransFp16FuncRelu6List8[output_unit];
    } else {
      return OutputTransFp16FuncList8[output_unit];
    }
  } else {
    return NULL;
  }
}

void OutputTransform4x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[16];
  float16x8_t t[8];
  float16x8_t m[4];
  Load16DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform4x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[16];
  float16x8_t t[8];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  Load16DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform4x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[16];
  float16x8_t t[8];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load16DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
    m[l + 2] = vminq_f16(six, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform4x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[16];
  float16x8_t t[12];
  float16x8_t m[9];
  Load16DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(src[1 + offset], src[2 + offset]);
    t[l] = vaddq_f16(src[offset], tmp);
    t[l + 4] = vsubq_f16(src[1 + offset], src[2 + offset]);
    t[l + 8] = vaddq_f16(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(t[1 + offset], t[2 + offset]);
    m[l] = vaddq_f16(vaddq_f16(t[offset], tmp), bias_ptr);
    m[l + 3] = vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(tmp, t[3 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform4x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[16];
  float16x8_t t[12];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  Load16DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(src[1 + offset], src[2 + offset]);
    t[l] = vaddq_f16(src[offset], tmp);
    t[l + 4] = vsubq_f16(src[1 + offset], src[2 + offset]);
    t[l + 8] = vaddq_f16(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(t[1 + offset], t[2 + offset]);
    m[l] = vaddq_f16(vaddq_f16(t[offset], tmp), bias_ptr);
    m[l + 3] = vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(tmp, t[3 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform4x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[16];
  float16x8_t t[12];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load16DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(src[1 + offset], src[2 + offset]);
    t[l] = vaddq_f16(src[offset], tmp);
    t[l + 4] = vsubq_f16(src[1 + offset], src[2 + offset]);
    t[l + 8] = vaddq_f16(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(t[1 + offset], t[2 + offset]);
    m[l] = vaddq_f16(vaddq_f16(t[offset], tmp), bias_ptr);
    m[l + 3] = vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(tmp, t[3 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 3] = vminq_f16(six, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
    m[l + 6] = vminq_f16(six, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[12];
  float16x8_t m[4];
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                     src[4 + offset]);
    t[l + 6] = vaddq_f16(vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                                   vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2)),
                         src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]), t[4 + offset]),
      bias_ptr);
    m[l + 2] = vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]),
                                             vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
                                   t[5 + offset]),
                         bias_ptr);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[12];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                     src[4 + offset]);
    t[l + 6] = vaddq_f16(vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                                   vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2)),
                         src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]), t[4 + offset]),
      bias_ptr);
    m[l + 2] = vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]),
                                             vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
                                   t[5 + offset]),
                         bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[12];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                     src[4 + offset]);
    t[l + 6] = vaddq_f16(vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                                   vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2)),
                         src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]), t[4 + offset]),
      bias_ptr);
    m[l + 2] = vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]),
                                             vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
                                   t[5 + offset]),
                         bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
    m[l + 2] = vminq_f16(six, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[18];
  float16x8_t m[9];
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                         vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = vaddq_f16(
      vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
      bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[18];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                         vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = vaddq_f16(
      vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
      bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[18];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                         vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = vaddq_f16(
      vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
      bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 3] = vminq_f16(six, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
    m[l + 6] = vminq_f16(six, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[24];
  float16x8_t m[16];
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x4ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[24];
  float16x8_t m[16];
  float16x8_t zero = vdupq_n_f16(0);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 4] = vmaxq_f16(zero, m[l + 4]);
    m[l + 8] = vmaxq_f16(zero, m[l + 8]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x4Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[24];
  float16x8_t m[16];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 4] = vmaxq_f16(zero, m[l + 4]);
    m[l + 4] = vminq_f16(six, m[l + 4]);
    m[l + 8] = vmaxq_f16(zero, m[l + 8]);
    m[l + 8] = vminq_f16(six, m[l + 8]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
    m[l + 12] = vminq_f16(six, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[30];
  float16x8_t m[25];
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8));
    t[l + 24] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), bias_ptr);
    m[l + 20] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x5ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[30];
  float16x8_t m[25];
  float16x8_t zero = vdupq_n_f16(0);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8));
    t[l + 24] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), bias_ptr);
    m[l + 20] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 5] = vmaxq_f16(zero, m[l + 5]);
    m[l + 10] = vmaxq_f16(zero, m[l + 10]);
    m[l + 15] = vmaxq_f16(zero, m[l + 15]);
    m[l + 20] = vmaxq_f16(zero, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x5Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[30];
  float16x8_t m[25];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8));
    t[l + 24] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), bias_ptr);
    m[l + 20] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 5] = vmaxq_f16(zero, m[l + 5]);
    m[l + 5] = vminq_f16(six, m[l + 5]);
    m[l + 10] = vmaxq_f16(zero, m[l + 10]);
    m[l + 10] = vminq_f16(six, m[l + 10]);
    m[l + 15] = vmaxq_f16(zero, m[l + 15]);
    m[l + 15] = vminq_f16(six, m[l + 15]);
    m[l + 20] = vmaxq_f16(zero, m[l + 20]);
    m[l + 20] = vminq_f16(six, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[16];
  float16x8_t m[4];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), t[7 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[16];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), t[7 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[16];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), t[7 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
    m[l + 2] = vminq_f16(six, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[24];
  float16x8_t m[9];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 6] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), t[7 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[24];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 6] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), t[7 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[24];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 6] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), t[7 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 3] = vminq_f16(six, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
    m[l + 6] = vminq_f16(six, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[32];
  float16x8_t m[16];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 12] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x4ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[32];
  float16x8_t m[16];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 12] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 4] = vmaxq_f16(zero, m[l + 4]);
    m[l + 8] = vmaxq_f16(zero, m[l + 8]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x4Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[32];
  float16x8_t m[16];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 12] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 4] = vmaxq_f16(zero, m[l + 4]);
    m[l + 4] = vminq_f16(six, m[l + 4]);
    m[l + 8] = vmaxq_f16(zero, m[l + 8]);
    m[l + 8] = vminq_f16(six, m[l + 8]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
    m[l + 12] = vminq_f16(six, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[40];
  float16x8_t m[25];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 20] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x5ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[40];
  float16x8_t m[25];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 20] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 5] = vmaxq_f16(zero, m[l + 5]);
    m[l + 10] = vmaxq_f16(zero, m[l + 10]);
    m[l + 15] = vmaxq_f16(zero, m[l + 15]);
    m[l + 20] = vmaxq_f16(zero, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x5Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[40];
  float16x8_t m[25];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 20] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 5] = vmaxq_f16(zero, m[l + 5]);
    m[l + 5] = vminq_f16(six, m[l + 5]);
    m[l + 10] = vmaxq_f16(zero, m[l + 10]);
    m[l + 10] = vminq_f16(six, m[l + 10]);
    m[l + 15] = vmaxq_f16(zero, m[l + 15]);
    m[l + 15] = vminq_f16(six, m[l + 15]);
    m[l + 20] = vmaxq_f16(zero, m[l + 20]);
    m[l + 20] = vminq_f16(six, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[48];
  float16x8_t m[36];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
    t[l + 40] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 18] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 24] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
    m[l + 30] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 6 && r_w == 6) {
    for (int i = 0; i < 6; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 6;
      vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x6ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[48];
  float16x8_t m[36];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
    t[l + 40] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 18] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 24] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
    m[l + 30] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
    m[l + 18] = vmaxq_f16(zero, m[l + 18]);
    m[l + 24] = vmaxq_f16(zero, m[l + 24]);
    m[l + 30] = vmaxq_f16(zero, m[l + 30]);
  }
  if (r_c == C8NUM && r_h == 6 && r_w == 6) {
    for (int i = 0; i < 6; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 6;
      vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x6Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[48];
  float16x8_t m[36];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
    t[l + 40] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 18] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 24] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
    m[l + 30] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
    m[l + 6] = vminq_f16(six, m[l + 6]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
    m[l + 12] = vminq_f16(six, m[l + 12]);
    m[l + 18] = vmaxq_f16(zero, m[l + 18]);
    m[l + 18] = vminq_f16(six, m[l + 18]);
    m[l + 24] = vmaxq_f16(zero, m[l + 24]);
    m[l + 24] = vminq_f16(six, m[l + 24]);
    m[l + 30] = vmaxq_f16(zero, m[l + 30]);
    m[l + 30] = vminq_f16(six, m[l + 30]);
  }
  if (r_c == C8NUM && r_h == 6 && r_w == 6) {
    for (int i = 0; i < 6; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 6;
      vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x7UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[56];
  float16x8_t m[49];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
    t[l + 40] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375));
    t[l + 48] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 14] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 21] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 28] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
    m[l + 35] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      vst1q_f16(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 7;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x7ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[56];
  float16x8_t m[49];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
    t[l + 40] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375));
    t[l + 48] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 14] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 21] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 28] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
    m[l + 35] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 7] = vmaxq_f16(zero, m[l + 7]);
    m[l + 14] = vmaxq_f16(zero, m[l + 14]);
    m[l + 21] = vmaxq_f16(zero, m[l + 21]);
    m[l + 28] = vmaxq_f16(zero, m[l + 28]);
    m[l + 35] = vmaxq_f16(zero, m[l + 35]);
    m[l + 42] = vmaxq_f16(zero, m[l + 42]);
  }
  if (r_c == C8NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      vst1q_f16(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 7;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x7Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[56];
  float16x8_t m[49];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
    t[l + 40] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375));
    t[l + 48] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 14] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 21] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 28] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
    m[l + 35] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 7] = vmaxq_f16(zero, m[l + 7]);
    m[l + 7] = vminq_f16(six, m[l + 7]);
    m[l + 14] = vmaxq_f16(zero, m[l + 14]);
    m[l + 14] = vminq_f16(six, m[l + 14]);
    m[l + 21] = vmaxq_f16(zero, m[l + 21]);
    m[l + 21] = vminq_f16(six, m[l + 21]);
    m[l + 28] = vmaxq_f16(zero, m[l + 28]);
    m[l + 28] = vminq_f16(six, m[l + 28]);
    m[l + 35] = vmaxq_f16(zero, m[l + 35]);
    m[l + 35] = vminq_f16(six, m[l + 35]);
    m[l + 42] = vmaxq_f16(zero, m[l + 42]);
    m[l + 42] = vminq_f16(six, m[l + 42]);
  }
  if (r_c == C8NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      vst1q_f16(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 7;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}
