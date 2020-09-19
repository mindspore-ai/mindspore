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

#include "nnacl/winograd_utils.h"
#include <stdio.h>
#include "nnacl/minimal_filtering_generator.h"

#define MIN_UNIT 2
#define MAX_UNIT 8

static InputTransFunc InputTransFuncList[] = {
  NULL, NULL, NULL, NULL, InputTransform4x4Unit, NULL, InputTransform6x6Unit, NULL, InputTransform8x8Unit};

static OutputTransFunc OutputTransFuncList4[] = {NULL, NULL, OutputTransform4x2Unit, OutputTransform4x3Unit};

static OutputTransFunc OutputTransFuncList6[] = {
  NULL, NULL, OutputTransform6x2Unit, OutputTransform6x3Unit, OutputTransform6x4Unit, OutputTransform6x5Unit};

static OutputTransFunc OutputTransFuncList8[] = {NULL,
                                                 NULL,
                                                 OutputTransform8x2Unit,
                                                 OutputTransform8x3Unit,
                                                 OutputTransform8x4Unit,
                                                 OutputTransform8x5Unit,
                                                 OutputTransform8x6Unit,
                                                 OutputTransform8x7Unit};
//
// static bool InputUnitList[] = {false, false, false, false, true, false, true, false, true};

void GeneralInputTransformUnit(const float *src_data, float *dst_data, float *matrix_b, float *matrix_bt, int src_step,
                               int dst_step, int in_unit) {
  int len = in_unit * in_unit;
  if (len > MAX_LEN) return;
#ifdef ENABLE_ARM
  float32x4_t src[MAX_LEN];
  float32x4_t t[MAX_LEN];
  float32x4_t m[MAX_LEN];
  float32x4_t vec_b[MAX_LEN];
  float32x4_t vec_bt[MAX_LEN];
  for (int i = 0; i < len; i++) {
    src[i] = vld1q_f32(src_data + i * src_step);
    vec_b[i] = vdupq_n_f32(matrix_b[i]);
    vec_bt[i] = vdupq_n_f32(matrix_bt[i]);
  }
  MatrixMultiplyVec(vec_bt, src, t, NULL, in_unit, in_unit, in_unit);
  MatrixMultiplyVec(t, vec_b, m, NULL, in_unit, in_unit, in_unit);
  for (int i = 0; i < len; i++) {
    vst1q_f32(dst_data + i * dst_step, m[i]);
  }
#else
  float src[MAX_LEN];
  float t[MAX_LEN];
  float m[MAX_LEN];
  for (int i = 0; i < C4NUM; ++i) {
    for (int j = 0; j < len; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    MatrixMultiply(matrix_bt, src, t, in_unit, in_unit, in_unit);
    MatrixMultiply(t, matrix_b, m, in_unit, in_unit, in_unit);
    for (int k = 0; k < len; ++k) {
      dst_data[i + k * dst_step] = m[k];
    }
  }
#endif
}

void GeneralOutputTransformUnit(const float *src_data, float *dst_data, const float *bias_data, float *matrix_a,
                                float *matrix_at, int src_step, int dst_step, int in_unit, int out_unit) {
  int src_len = in_unit * in_unit;
  if (src_len > MAX_LEN) {
    return;
  }
#ifdef ENABLE_ARM
  float32x4_t src[MAX_LEN];
  float32x4_t t[MAX_LEN];
  float32x4_t m[MAX_LEN];
  float32x4_t vec_a[MAX_LEN];
  float32x4_t vec_at[MAX_LEN];
  int tmp_len = in_unit * out_unit;
  if (tmp_len > MAX_LEN) return;

  for (int i = 0; i < tmp_len; i++) {
    vec_a[i] = vdupq_n_f32(matrix_a[i]);
    vec_at[i] = vdupq_n_f32(matrix_at[i]);
  }
  for (int i = 0; i < src_len; i++) {
    src[i] = vld1q_f32(src_data + i * src_step);
  }
  MatrixMultiplyVec(vec_at, src, t, NULL, out_unit, in_unit, in_unit);
  MatrixMultiplyVec(t, vec_a, m, bias_data, out_unit, in_unit, out_unit);

  for (int i = 0; i < out_unit; i++) {
    int dst_k_offset = i * dst_step * C4NUM;
    int m_k_offset = i * out_unit;
    for (int j = 0; j < out_unit; j++) {
      vst1q_f32(dst_data + dst_k_offset + j * C4NUM, m[m_k_offset + j]);
    }
  }
#else
  float src[MAX_LEN];
  float t[MAX_LEN];
  float m[MAX_LEN];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < src_len; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    // AT * x * A
    MatrixMultiply(matrix_at, src, t, out_unit, in_unit, in_unit);
    MatrixMultiply(t, matrix_a, m, out_unit, in_unit, out_unit);

    // store output
    for (int k = 0; k < out_unit; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * out_unit;
      for (int j = 0; j < out_unit; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

InputTransFunc GetInputTransFunc(int input_unit) { return InputTransFuncList[input_unit]; }

void InputTransform4x4Unit(const float *src_data, float *dst_data, int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[16];
  float32x4_t t[16];
  float32x4_t m[16];
  Load16Data;
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = vsubq_f32(src[offset], src[2 + offset]);
    t[4 + l] = vaddq_f32(src[1 + offset], src[2 + offset]);
    t[8 + l] = vsubq_f32(src[2 + offset], src[1 + offset]);
    t[12 + l] = vsubq_f32(src[3 + offset], src[1 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    m[l] = vsubq_f32(t[offset], t[2 + offset]);
    m[4 + l] = vaddq_f32(t[1 + offset], t[2 + offset]);
    m[8 + l] = vsubq_f32(t[2 + offset], t[1 + offset]);
    m[12 + l] = vsubq_f32(t[3 + offset], t[1 + offset]);
  }
  for (int i = 0; i < 16; i++) {
    vst1q_f32(dst_data + i * dst_step, m[i]);
  }
#else
  float src[16];
  float t[16];
  float m[16];
  for (int i = 0; i < C4NUM; ++i) {
    for (int j = 0; j < 16; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = src[offset] - src[2 + offset];
      t[4 + l] = src[1 + offset] + src[2 + offset];
      t[8 + l] = src[2 + offset] - src[1 + offset];
      t[12 + l] = src[3 + offset] - src[1 + offset];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      m[l] = t[offset] - t[2 + offset];
      m[4 + l] = t[1 + offset] + t[2 + offset];
      m[8 + l] = t[2 + offset] - t[1 + offset];
      m[12 + l] = t[3 + offset] - t[1 + offset];
    }
    for (int k = 0; k < 16; ++k) {
      dst_data[i + k * dst_step] = m[k];
    }
  }
#endif
}

void InputTransform6x6Unit(const float *src_data, float *dst_data, int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[36];
  float32x4_t t[36];
  float32x4_t m[36];
  Load36Data;
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float32x4_t tmp1 = vsubq_f32(src[3 + offset], src[1 + offset]);
    float32x4_t tmp2 = vsubq_f32(src[4 + offset], src[2 + offset]);
    t[l] = vaddq_f32(vsubq_f32(vmulq_n_f32(src[offset], 4), vmulq_n_f32(src[2 + offset], 5)), src[4 + offset]);
    t[6 + l] = vaddq_f32(vmulq_n_f32(vaddq_f32(src[1 + offset], src[2 + offset]), -4),
                         vaddq_f32(src[3 + offset], src[4 + offset]));
    t[12 + l] = vaddq_f32(vmulq_n_f32(vsubq_f32(src[1 + offset], src[2 + offset]), 4),
                          vsubq_f32(src[4 + offset], src[3 + offset]));
    t[18 + l] = vaddq_f32(vmulq_n_f32(tmp1, 2), tmp2);
    t[24 + l] = vaddq_f32(vmulq_n_f32(tmp1, -2), tmp2);
    t[30 + l] = vaddq_f32(vsubq_f32(vmulq_n_f32(src[1 + offset], 4), vmulq_n_f32(src[3 + offset], 5)), src[5 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float32x4_t tmp1 = vsubq_f32(t[3 + offset], t[1 + offset]);
    float32x4_t tmp2 = vsubq_f32(t[4 + offset], t[2 + offset]);
    m[l] = vaddq_f32(vsubq_f32(vmulq_n_f32(t[offset], 4), vmulq_n_f32(t[2 + offset], 5)), t[4 + offset]);
    m[6 + l] =
      vaddq_f32(vmulq_n_f32(vaddq_f32(t[1 + offset], t[2 + offset]), -4), vaddq_f32(t[3 + offset], t[4 + offset]));
    m[12 + l] =
      vaddq_f32(vmulq_n_f32(vsubq_f32(t[1 + offset], t[2 + offset]), 4), vsubq_f32(t[4 + offset], t[3 + offset]));
    m[18 + l] = vaddq_f32(vmulq_n_f32(tmp1, 2), tmp2);
    m[24 + l] = vaddq_f32(vmulq_n_f32(tmp1, -2), tmp2);
    m[30 + l] = vaddq_f32(vsubq_f32(vmulq_n_f32(t[1 + offset], 4), vmulq_n_f32(t[3 + offset], 5)), t[5 + offset]);
  }
  for (int i = 0; i < 36; i++) {
    vst1q_f32(dst_data + i * dst_step, m[i]);
  }
#else
  float src[36];
  float t[36];
  float m[36];
  for (int i = 0; i < C4NUM; ++i) {
    for (int j = 0; j < 36; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      float tmp1 = src[3 + offset] - src[1 + offset];
      float tmp2 = src[4 + offset] - src[2 + offset];
      t[l] = 4 * src[offset] - 5 * src[2 + offset] + src[4 + offset];
      t[6 + l] = -4 * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]);
      t[12 + l] = 4 * (src[1 + offset] - src[2 + offset]) + (src[4 + offset] - src[3 + offset]);
      t[18 + l] = 2 * tmp1 + tmp2;
      t[24 + l] = -2 * tmp1 + tmp2;
      t[30 + l] = 4 * src[1 + offset] - 5 * src[3 + offset] + src[5 + offset];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      float tmp1 = t[3 + offset] - t[1 + offset];
      float tmp2 = t[4 + offset] - t[2 + offset];
      m[l] = 4 * t[offset] - 5 * t[2 + offset] + t[4 + offset];
      m[6 + l] = -4 * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]);
      m[12 + l] = 4 * (t[1 + offset] - t[2 + offset]) + (t[4 + offset] - t[3 + offset]);
      m[18 + l] = 2 * tmp1 + tmp2;
      m[24 + l] = -2 * tmp1 + tmp2;
      m[30 + l] = 4 * t[1 + offset] - 5 * t[3 + offset] + t[5 + offset];
    }
    for (int k = 0; k < 36; ++k) {
      dst_data[i + k * dst_step] = m[k];
    }
  }
#endif
}

void InputTransform8x8Unit(const float *src_data, float *dst_data, int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[64];
  float32x4_t t[64];
  float32x4_t m[64];
  Load64Data;
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    t[l] = vsubq_f32(vaddq_f32(vsubq_f32(vmulq_n_f32(src[offset], 36), vmulq_n_f32(src[2 + offset], 49)),
                               vmulq_n_f32(src[4 + offset], 14)),
                     src[6 + offset]);
    float32x4_t tmp1 = vaddq_f32(vmulq_n_f32(src[1 + offset], 36), src[5 + offset]);
    float32x4_t tmp2 = vsubq_f32(vmulq_n_f32(src[2 + offset], 36), vmulq_n_f32(src[4 + offset], 13));
    t[8 + l] = vaddq_f32(vsubq_f32(vaddq_f32(tmp1, tmp2), vmulq_n_f32(src[3 + offset], 13)), src[6 + offset]);
    t[16 + l] = vaddq_f32(vaddq_f32(vsubq_f32(tmp2, tmp1), vmulq_n_f32(src[3 + offset], 13)), src[6 + offset]);
    tmp1 = vaddq_f32(vmulq_n_f32(src[1 + offset], 18), vmulq_n_f32(src[5 + offset], 2));
    tmp2 = vsubq_f32(vmulq_n_f32(src[2 + offset], 9), vmulq_n_f32(src[4 + offset], 10));
    t[24 + l] = vaddq_f32(vsubq_f32(vaddq_f32(tmp1, tmp2), vmulq_n_f32(src[3 + offset], 20)), src[6 + offset]);
    t[32 + l] = vaddq_f32(vaddq_f32(vsubq_f32(tmp2, tmp1), vmulq_n_f32(src[3 + offset], 20)), src[6 + offset]);
    tmp1 = vaddq_f32(vmulq_n_f32(src[1 + offset], 12), vmulq_n_f32(src[5 + offset], 3));
    tmp2 = vsubq_f32(vmulq_n_f32(src[2 + offset], 4), vmulq_n_f32(src[4 + offset], 5));
    t[40 + l] = vaddq_f32(vsubq_f32(vaddq_f32(tmp1, tmp2), vmulq_n_f32(src[3 + offset], 15)), src[6 + offset]);
    t[48 + l] = vaddq_f32(vaddq_f32(vsubq_f32(tmp2, tmp1), vmulq_n_f32(src[3 + offset], 15)), src[6 + offset]);
    t[56 + l] = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src[1 + offset], -36), vmulq_n_f32(src[3 + offset], 49)),
                                    vmulq_n_f32(src[5 + offset], 14)),
                          src[7 + offset]);
  }
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    m[l] = vsubq_f32(
      vaddq_f32(vsubq_f32(vmulq_n_f32(t[offset], 36), vmulq_n_f32(t[2 + offset], 49)), vmulq_n_f32(t[4 + offset], 14)),
      t[6 + offset]);
    float32x4_t tmp1 = vaddq_f32(vmulq_n_f32(t[1 + offset], 36), t[5 + offset]);
    float32x4_t tmp2 = vsubq_f32(vmulq_n_f32(t[2 + offset], 36), vmulq_n_f32(t[4 + offset], 13));
    m[8 + l] = vaddq_f32(vsubq_f32(vaddq_f32(tmp1, tmp2), vmulq_n_f32(t[3 + offset], 13)), t[6 + offset]);
    m[16 + l] = vaddq_f32(vaddq_f32(vsubq_f32(tmp2, tmp1), vmulq_n_f32(t[3 + offset], 13)), t[6 + offset]);
    tmp1 = vaddq_f32(vmulq_n_f32(t[1 + offset], 18), vmulq_n_f32(t[5 + offset], 2));
    tmp2 = vsubq_f32(vmulq_n_f32(t[2 + offset], 9), vmulq_n_f32(t[4 + offset], 10));
    m[24 + l] = vaddq_f32(vsubq_f32(vaddq_f32(tmp1, tmp2), vmulq_n_f32(t[3 + offset], 20)), t[6 + offset]);
    m[32 + l] = vaddq_f32(vaddq_f32(vsubq_f32(tmp2, tmp1), vmulq_n_f32(t[3 + offset], 20)), t[6 + offset]);
    tmp1 = vaddq_f32(vmulq_n_f32(t[1 + offset], 12), vmulq_n_f32(t[5 + offset], 3));
    tmp2 = vsubq_f32(vmulq_n_f32(t[2 + offset], 4), vmulq_n_f32(t[4 + offset], 5));
    m[40 + l] = vaddq_f32(vsubq_f32(vaddq_f32(tmp1, tmp2), vmulq_n_f32(t[3 + offset], 15)), t[6 + offset]);
    m[48 + l] = vaddq_f32(vaddq_f32(vsubq_f32(tmp2, tmp1), vmulq_n_f32(t[3 + offset], 15)), t[6 + offset]);
    m[56 + l] = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t[1 + offset], -36), vmulq_n_f32(t[3 + offset], 49)),
                                    vmulq_n_f32(t[5 + offset], 14)),
                          t[7 + offset]);
  }
  for (int i = 0; i < 64; i++) {
    vst1q_f32(dst_data + i * dst_step, m[i]);
  }
#else
  float src[64];
  float t[64];
  float m[64];
  for (int i = 0; i < C4NUM; ++i) {
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = 36 * src[offset] - 49 * src[2 + offset] + 14 * src[4 + offset] - src[6 + offset];
      float tmp1 = 36 * src[1 + offset] + src[5 + offset];
      float tmp2 = 36 * src[2 + offset] - 13 * src[4 + offset];
      t[8 + l] = tmp1 + tmp2 - 13 * src[3 + offset] + src[6 + offset];
      t[16 + l] = tmp2 - tmp1 + 13 * src[3 + offset] + src[6 + offset];
      tmp1 = 18 * src[1 + offset] + 2 * src[5 + offset];
      tmp2 = 9 * src[2 + offset] - 10 * src[4 + offset];
      t[24 + l] = tmp1 + tmp2 - 20 * src[3 + offset] + src[6 + offset];
      t[32 + l] = tmp2 - tmp1 + 20 * src[3 + offset] + src[6 + offset];
      tmp1 = 12 * src[1 + offset] + 3 * src[5 + offset];
      tmp2 = 4 * src[2 + offset] - 5 * src[4 + offset];
      t[40 + l] = tmp1 + tmp2 - 15 * src[3 + offset] + src[6 + offset];
      t[48 + l] = tmp2 - tmp1 + 15 * src[3 + offset] + src[6 + offset];
      t[56 + l] = -36 * src[1 + offset] + 49 * src[3 + offset] - 14 * src[5 + offset] + src[7 + offset];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      m[l] = 36 * t[offset] - 49 * t[2 + offset] + 14 * t[4 + offset] - t[6 + offset];
      float tmp1 = 36 * t[1 + offset] + t[5 + offset];
      float tmp2 = 36 * t[2 + offset] - 13 * t[4 + offset];
      m[8 + l] = tmp1 + tmp2 - 13 * t[3 + offset] + t[6 + offset];
      m[16 + l] = tmp2 - tmp1 + 13 * t[3 + offset] + t[6 + offset];
      tmp1 = 18 * t[1 + offset] + 2 * t[5 + offset];
      tmp2 = 9 * t[2 + offset] - 10 * t[4 + offset];
      m[24 + l] = tmp1 + tmp2 - 20 * t[3 + offset] + t[6 + offset];
      m[32 + l] = tmp2 - tmp1 + 20 * t[3 + offset] + t[6 + offset];
      tmp1 = 12 * t[1 + offset] + 3 * t[5 + offset];
      tmp2 = 4 * t[2 + offset] - 5 * t[4 + offset];
      m[40 + l] = tmp1 + tmp2 - 15 * t[3 + offset] + t[6 + offset];
      m[48 + l] = tmp2 - tmp1 + 15 * t[3 + offset] + t[6 + offset];
      m[56 + l] = -36 * t[1 + offset] + 49 * t[3 + offset] - 14 * t[5 + offset] + t[7 + offset];
    }
    for (int k = 0; k < 64; ++k) {
      dst_data[i + k * dst_step] = m[k];
    }
  }
#endif
}

OutputTransFunc GetOutputTransFunc(int input_unit, int output_unit) {
  if (input_unit == 4 && output_unit < 4) {
      return OutputTransFuncList4[output_unit];
  } else if (input_unit == 6 && output_unit < 6) {
      return OutputTransFuncList6[output_unit];
  } else if (input_unit == 8 && output_unit < 8) {
      return OutputTransFuncList8[output_unit];
  } else {
    return NULL;
  }
}

void OutputTransform4x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[16];
  float32x4_t t[8];
  float32x4_t m[4];
  Load16Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = vaddq_f32(vaddq_f32(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = vaddq_f32(vsubq_f32(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = vaddq_f32(vaddq_f32(vaddq_f32(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = vaddq_f32(vaddq_f32(vsubq_f32(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
  }
  Store4Data;
#else
  float src[16];
  float t[8];
  float m[4];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 16; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset];
      t[l + 4] = src[1 + offset] - src[2 + offset] + src[3 + offset];
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset];
      m[l + 2] = t[1 + offset] - t[2 + offset] + t[3 + offset];
    }
    // store output
    for (int k = 0; k < 2; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 2;
      for (int j = 0; j < 2; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform4x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[16];
  float32x4_t t[12];
  float32x4_t m[9];
  Load16Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    float32x4_t tmp = vaddq_f32(src[1 + offset], src[2 + offset]);
    t[l] = vaddq_f32(src[offset], tmp);
    t[l + 4] = vsubq_f32(src[1 + offset], src[2 + offset]);
    t[l + 8] = vaddq_f32(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    float32x4_t tmp = vaddq_f32(t[1 + offset], t[2 + offset]);
    m[l] = vaddq_f32(vaddq_f32(t[offset], tmp), bias_ptr);
    m[l + 3] = vaddq_f32(vsubq_f32(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = vaddq_f32(vaddq_f32(tmp, t[3 + offset]), bias_ptr);
  }
  Store9Data;
#else
  float src[16];
  float t[12];
  float m[9];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 16; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = src[0 + offset] + src[1 + offset] + src[2 + offset];
      t[l + 4] = src[1 + offset] - src[2 + offset];
      t[l + 8] = src[1 + offset] + src[2 + offset] + src[3 + offset];
    }
    for (int l = 0; l < 3; ++l) {
      int offset = l * 4;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset];
      m[l + 3] = t[1 + offset] - t[2 + offset];
      m[l + 6] = t[1 + offset] + t[2 + offset] + t[3 + offset];
    }
    // store output
    for (int k = 0; k < 3; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 3;
      for (int j = 0; j < 3; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform6x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[36];
  float32x4_t t[12];
  float32x4_t m[4];
  Load36Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                     src[4 + offset]);
    t[l + 6] = vaddq_f32(vaddq_f32(vsubq_f32(src[1 + offset], src[2 + offset]),
                                   vmulq_n_f32(vsubq_f32(src[3 + offset], src[4 + offset]), 2)),
                         src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]), t[4 + offset]),
      bias_ptr);
    m[l + 2] = vaddq_f32(vaddq_f32(vaddq_f32(vsubq_f32(t[1 + offset], t[2 + offset]),
                                             vmulq_n_f32(vsubq_f32(t[3 + offset], t[4 + offset]), 2)),
                                   t[5 + offset]),
                         bias_ptr);
  }
  Store4Data;
#else
  float src[36];
  float t[12];
  float m[4];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 36; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset];
      t[l + 6] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]) + src[5 + offset];
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 6;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset];
      m[l + 2] = t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]) + t[5 + offset];
    }
    // store output
    for (int k = 0; k < 2; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 2;
      for (int j = 0; j < 2; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}
void OutputTransform6x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[36];
  float32x4_t t[18];
  float32x4_t m[9];
  Load36Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float32x4_t tmp1 = vaddq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f32(vaddq_f32(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f32(vsubq_f32(src[1 + offset], src[2 + offset]),
                         vmulq_n_f32(vsubq_f32(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    float32x4_t tmp1 = vaddq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f32(vaddq_f32(vaddq_f32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = vaddq_f32(
      vaddq_f32(vsubq_f32(t[1 + offset], t[2 + offset]), vmulq_n_f32(vsubq_f32(t[3 + offset], t[4 + offset]), 2)),
      bias_ptr);
    m[l + 6] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), t[5 + offset]), bias_ptr);
  }
  Store9Data;
#else
  float src[36];
  float t[18];
  float m[9];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 36; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset];
      t[l + 6] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]);
      t[l + 12] = src[1 + offset] + src[2 + offset] + 4 * (src[3 + offset] + src[4 + offset]) + src[5 + offset];
    }
    for (int l = 0; l < 3; ++l) {
      int offset = l * 6;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset];
      m[l + 3] = t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]);
      m[l + 6] = t[1 + offset] + t[2 + offset] + 4 * (t[3 + offset] + t[4 + offset]) + t[5 + offset];
    }
    // store output
    for (int k = 0; k < 3; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 3;
      for (int j = 0; j < 3; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}
void OutputTransform6x4Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[36];
  float32x4_t t[24];
  float32x4_t m[16];
  Load36Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float32x4_t tmp1 = vaddq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp3 = vsubq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp4 = vsubq_f32(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f32(vaddq_f32(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f32(tmp3, vmulq_n_f32(tmp4, 2));
    t[l + 12] = vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4));
    t[l + 18] = vaddq_f32(vaddq_f32(tmp3, vmulq_n_f32(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    float32x4_t tmp1 = vaddq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp3 = vsubq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp4 = vsubq_f32(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f32(vaddq_f32(vaddq_f32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = vaddq_f32(vaddq_f32(tmp3, vmulq_n_f32(tmp4, 2)), bias_ptr);
    m[l + 8] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), bias_ptr);
    m[l + 12] = vaddq_f32(vaddq_f32(vaddq_f32(tmp3, vmulq_n_f32(tmp4, 8)), t[5 + offset]), bias_ptr);
  }
  Store16Data;
#else
  float src[36];
  float t[24];
  float m[16];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 36; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset];
      t[l + 6] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]);
      t[l + 12] = src[1 + offset] + src[2 + offset] + 4 * (src[3 + offset] + src[4 + offset]);
      t[l + 18] = src[1 + offset] - src[2 + offset] + 8 * (src[3 + offset] - src[4 + offset]) + src[5 + offset];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 6;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset];
      m[l + 4] = t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]);
      m[l + 8] = t[1 + offset] + t[2 + offset] + 4 * (t[3 + offset] + t[4 + offset]);
      m[l + 12] = t[1 + offset] - t[2 + offset] + 8 * (t[3 + offset] - t[4 + offset]) + t[5 + offset];
    }
    // store output
    for (int k = 0; k < 4; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 4;
      for (int j = 0; j < 4; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}
void OutputTransform6x5Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[36];
  float32x4_t t[30];
  float32x4_t m[25];
  Load36Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float32x4_t tmp1 = vaddq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp3 = vsubq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp4 = vsubq_f32(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f32(vaddq_f32(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f32(tmp3, vmulq_n_f32(tmp4, 2));
    t[l + 12] = vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4));
    t[l + 18] = vaddq_f32(tmp3, vmulq_n_f32(tmp4, 8));
    t[l + 24] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    float32x4_t tmp1 = vaddq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp3 = vsubq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp4 = vsubq_f32(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f32(vaddq_f32(vaddq_f32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = vaddq_f32(vaddq_f32(tmp3, vmulq_n_f32(tmp4, 2)), bias_ptr);
    m[l + 10] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), bias_ptr);
    m[l + 15] = vaddq_f32(vaddq_f32(tmp3, vmulq_n_f32(tmp4, 8)), bias_ptr);
    m[l + 20] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 16)), t[5 + offset]), bias_ptr);
  }
  Store25Data;
#else
  float src[36];
  float t[30];
  float m[25];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 36; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset];
      t[l + 6] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]);
      t[l + 12] = src[1 + offset] + src[2 + offset] + 4 * (src[3 + offset] + src[4 + offset]);
      t[l + 18] = src[1 + offset] - src[2 + offset] + 8 * (src[3 + offset] - src[4 + offset]);
      t[l + 24] = src[1 + offset] + src[2 + offset] + 16 * (src[3 + offset] + src[4 + offset]) + src[5 + offset];
    }
    for (int l = 0; l < 5; ++l) {
      int offset = l * 6;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset];
      m[l + 5] = t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]);
      m[l + 10] = t[1 + offset] + t[2 + offset] + 4 * (t[3 + offset] + t[4 + offset]);
      m[l + 15] = t[1 + offset] - t[2 + offset] + 8 * (t[3 + offset] - t[4 + offset]);
      m[l + 20] = t[1 + offset] + t[2 + offset] + 16 * (t[3 + offset] + t[4 + offset]) + t[5 + offset];
    }
    // store output
    for (int k = 0; k < 5; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 5;
      for (int j = 0; j < 5; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform8x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[64];
  float32x4_t t[16];
  float32x4_t m[4];
  Load64Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    t[l] = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src[offset], src[1 + offset]), src[2 + offset]),
                                                   src[3 + offset]),
                                         src[4 + offset]),
                               src[5 + offset]),
                     src[6 + offset]);
    t[l + 8] = vaddq_f32(vaddq_f32(vaddq_f32(vsubq_f32(src[1 + offset], src[2 + offset]),
                                             vmulq_n_f32(vsubq_f32(src[3 + offset], src[4 + offset]), 2)),
                                   vmulq_n_f32(vsubq_f32(src[5 + offset], src[6 + offset]), 3)),
                         src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    m[l] = vaddq_f32(
      vaddq_f32(
        vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]),
                            t[4 + offset]),
                  t[5 + offset]),
        t[6 + offset]),
      bias_ptr);
    m[l + 2] = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vsubq_f32(t[1 + offset], t[2 + offset]),
                                                       vmulq_n_f32(vsubq_f32(t[3 + offset], t[4 + offset]), 2)),
                                             vmulq_n_f32(vsubq_f32(t[5 + offset], t[6 + offset]), 3)),
                                   t[7 + offset]),
                         bias_ptr);
  }
  Store4Data;
#else
  float src[64];
  float t[16];
  float m[4];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]) +
                 3 * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 2] = t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]) +
                 3 * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < 2; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 2;
      for (int j = 0; j < 2; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}
void OutputTransform8x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[64];
  float32x4_t t[24];
  float32x4_t m[9];
  Load64Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(src[5 + offset], src[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f32(vaddq_f32(vaddq_f32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3));
    t[l + 16] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(t[5 + offset], t[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3)), bias_ptr);
    m[l + 6] = vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9)), t[7 + offset]), bias_ptr);
  }
  Store9Data;
#else
  float src[64];
  float t[24];
  float m[9];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]) +
                 3 * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = src[1 + offset] + src[2 + offset] + 4 * (src[3 + offset] + src[4 + offset]) +
                  9 * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 3; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 3] =
        t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]) + 3 * (t[5 + offset] - t[6 + offset]);
      m[l + 6] = t[1 + offset] + t[2 + offset] + 4 * (t[3 + offset] + t[4 + offset]) +
                 9 * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < 3; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 3;
      for (int j = 0; j < 3; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}
void OutputTransform8x4Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[64];
  float32x4_t t[32];
  float32x4_t m[16];
  Load64Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(src[5 + offset], src[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f32(vaddq_f32(vaddq_f32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3));
    t[l + 16] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9));
    t[l + 24] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 8)), vmulq_n_f32(tmp6, 27)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(t[5 + offset], t[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3)), bias_ptr);
    m[l + 8] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9)), bias_ptr);
    m[l + 12] = vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 8)), vmulq_n_f32(tmp6, 27)), t[7 + offset]), bias_ptr);
  }
  Store16Data;
#else
  float src[64];
  float t[32];
  float m[16];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]) +
                 3 * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = src[1 + offset] + src[2 + offset] + 4 * (src[3 + offset] + src[4 + offset]) +
                  9 * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = src[1 + offset] - src[2 + offset] + 8 * (src[3 + offset] - src[4 + offset]) +
                  27 * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 4] =
        t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]) + 3 * (t[5 + offset] - t[6 + offset]);
      m[l + 8] =
        t[1 + offset] + t[2 + offset] + 4 * (t[3 + offset] + t[4 + offset]) + 9 * (t[5 + offset] + t[6 + offset]);
      m[l + 12] = t[1 + offset] - t[2 + offset] + 8 * (t[3 + offset] - t[4 + offset]) +
                  27 * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < 4; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 4;
      for (int j = 0; j < 4; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}
void OutputTransform8x5Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[64];
  float32x4_t t[40];
  float32x4_t m[25];
  Load64Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(src[5 + offset], src[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f32(vaddq_f32(vaddq_f32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3));
    t[l + 16] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9));
    t[l + 24] = vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 8)), vmulq_n_f32(tmp6, 27));
    t[l + 32] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 16)), vmulq_n_f32(tmp3, 81)), src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(t[5 + offset], t[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3)), bias_ptr);
    m[l + 10] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9)), bias_ptr);
    m[l + 15] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 8)), vmulq_n_f32(tmp6, 27)), bias_ptr);
    m[l + 20] = vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 16)), vmulq_n_f32(tmp3, 81)), t[7 + offset]), bias_ptr);
  }
  Store25Data;
#else
  float src[64];
  float t[40];
  float m[25];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]) +
                 3 * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = src[1 + offset] + src[2 + offset] + 4 * (src[3 + offset] + src[4 + offset]) +
                  9 * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = src[1 + offset] - src[2 + offset] + 8 * (src[3 + offset] - src[4 + offset]) +
                  27 * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = src[1 + offset] + src[2 + offset] + 16 * (src[3 + offset] + src[4 + offset]) +
                  81 * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 5; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 5] =
        t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]) + 3 * (t[5 + offset] - t[6 + offset]);
      m[l + 10] =
        t[1 + offset] + t[2 + offset] + 4 * (t[3 + offset] + t[4 + offset]) + 9 * (t[5 + offset] + t[6 + offset]);
      m[l + 15] =
        t[1 + offset] - t[2 + offset] + 8 * (t[3 + offset] - t[4 + offset]) + 27 * (t[5 + offset] - t[6 + offset]);
      m[l + 20] = t[1 + offset] + t[2 + offset] + 16 * (t[3 + offset] + t[4 + offset]) +
                  81 * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < 5; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 5;
      for (int j = 0; j < 5; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}
void OutputTransform8x6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[64];
  float32x4_t t[48];
  float32x4_t m[36];
  Load64Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(src[5 + offset], src[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f32(vaddq_f32(vaddq_f32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3));
    t[l + 16] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9));
    t[l + 24] = vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 8)), vmulq_n_f32(tmp6, 27));
    t[l + 32] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 16)), vmulq_n_f32(tmp3, 81));
    t[l + 40] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 32)), vmulq_n_f32(tmp6, 243)), src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(t[5 + offset], t[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3)), bias_ptr);
    m[l + 12] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9)), bias_ptr);
    m[l + 18] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 8)), vmulq_n_f32(tmp6, 27)), bias_ptr);
    m[l + 24] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 16)), vmulq_n_f32(tmp3, 81)), bias_ptr);
    m[l + 30] = vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 32)), vmulq_n_f32(tmp6, 243)), t[7 + offset]), bias_ptr);
  }
  for (int i = 0; i < 6; i++) {
    int dst_k_offset = i * dst_step * C4NUM;
    int m_k_offset = i * 6;
    vst1q_f32(dst_data + dst_k_offset + 0 * C4NUM, m[m_k_offset]);
    vst1q_f32(dst_data + dst_k_offset + 1 * C4NUM, m[m_k_offset + 1]);
    vst1q_f32(dst_data + dst_k_offset + 2 * C4NUM, m[m_k_offset + 2]);
    vst1q_f32(dst_data + dst_k_offset + 3 * C4NUM, m[m_k_offset + 3]);
    vst1q_f32(dst_data + dst_k_offset + 4 * C4NUM, m[m_k_offset + 4]);
    vst1q_f32(dst_data + dst_k_offset + 5 * C4NUM, m[m_k_offset + 5]);
  }
#else
  float src[64];
  float t[48];
  float m[36];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]) +
                 3 * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = src[1 + offset] + src[2 + offset] + 4 * (src[3 + offset] + src[4 + offset]) +
                  9 * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = src[1 + offset] - src[2 + offset] + 8 * (src[3 + offset] - src[4 + offset]) +
                  27 * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = src[1 + offset] + src[2 + offset] + 16 * (src[3 + offset] + src[4 + offset]) +
                  81 * (src[5 + offset] + src[6 + offset]);
      t[l + 40] = src[1 + offset] - src[2 + offset] + 32 * (src[3 + offset] - src[4 + offset]) +
                  243 * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 6] =
        t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]) + 3 * (t[5 + offset] - t[6 + offset]);
      m[l + 12] =
        t[1 + offset] + t[2 + offset] + 4 * (t[3 + offset] + t[4 + offset]) + 9 * (t[5 + offset] + t[6 + offset]);
      m[l + 18] =
        t[1 + offset] - t[2 + offset] + 8 * (t[3 + offset] - t[4 + offset]) + 27 * (t[5 + offset] - t[6 + offset]);
      m[l + 24] =
        t[1 + offset] + t[2 + offset] + 16 * (t[3 + offset] + t[4 + offset]) + 81 * (t[5 + offset] + t[6 + offset]);
      m[l + 30] = t[1 + offset] - t[2 + offset] + 32 * (t[3 + offset] - t[4 + offset]) +
                  243 * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < 6; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 6;
      for (int j = 0; j < 6; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}
void OutputTransform8x7Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src[64];
  float32x4_t t[56];
  float32x4_t m[49];
  Load64Data;
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(src[5 + offset], src[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(src[1 + offset], src[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(src[3 + offset], src[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f32(vaddq_f32(vaddq_f32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3));
    t[l + 16] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9));
    t[l + 24] = vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 8)), vmulq_n_f32(tmp6, 27));
    t[l + 32] = vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 16)), vmulq_n_f32(tmp3, 81));
    t[l + 40] = vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 32)), vmulq_n_f32(tmp6, 243));
    t[l + 48] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 64)), vmulq_n_f32(tmp3, 729)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    float32x4_t tmp1 = vaddq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp2 = vaddq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp3 = vaddq_f32(t[5 + offset], t[6 + offset]);
    float32x4_t tmp4 = vsubq_f32(t[1 + offset], t[2 + offset]);
    float32x4_t tmp5 = vsubq_f32(t[3 + offset], t[4 + offset]);
    float32x4_t tmp6 = vsubq_f32(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 2)), vmulq_n_f32(tmp6, 3)), bias_ptr);
    m[l + 14] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 4)), vmulq_n_f32(tmp3, 9)), bias_ptr);
    m[l + 21] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 8)), vmulq_n_f32(tmp6, 27)), bias_ptr);
    m[l + 28] = vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 16)), vmulq_n_f32(tmp3, 81)), bias_ptr);
    m[l + 35] = vaddq_f32(vaddq_f32(vaddq_f32(tmp4, vmulq_n_f32(tmp5, 32)), vmulq_n_f32(tmp6, 243)), bias_ptr);
    m[l + 42] = vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(tmp1, vmulq_n_f32(tmp2, 64)), vmulq_n_f32(tmp3, 729)), t[7 + offset]), bias_ptr);
  }
  for (int i = 0; i < 7; i++) {
    int dst_k_offset = i * dst_step * C4NUM;
    int m_k_offset = i * 7;
    vst1q_f32(dst_data + dst_k_offset + 0 * C4NUM, m[m_k_offset]);
    vst1q_f32(dst_data + dst_k_offset + 1 * C4NUM, m[m_k_offset + 1]);
    vst1q_f32(dst_data + dst_k_offset + 2 * C4NUM, m[m_k_offset + 2]);
    vst1q_f32(dst_data + dst_k_offset + 3 * C4NUM, m[m_k_offset + 3]);
    vst1q_f32(dst_data + dst_k_offset + 4 * C4NUM, m[m_k_offset + 4]);
    vst1q_f32(dst_data + dst_k_offset + 5 * C4NUM, m[m_k_offset + 5]);
    vst1q_f32(dst_data + dst_k_offset + 6 * C4NUM, m[m_k_offset + 6]);
  }
#else
  float src[64];
  float t[56];
  float m[49];
  for (int i = 0; i < C4NUM; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = src[1 + offset] - src[2 + offset] + 2 * (src[3 + offset] - src[4 + offset]) +
                 3 * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = src[1 + offset] + src[2 + offset] + 4 * (src[3 + offset] + src[4 + offset]) +
                  9 * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = src[1 + offset] - src[2 + offset] + 8 * (src[3 + offset] - src[4 + offset]) +
                  27 * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = src[1 + offset] + src[2 + offset] + 16 * (src[3 + offset] + src[4 + offset]) +
                  81 * (src[5 + offset] + src[6 + offset]);
      t[l + 40] = src[1 + offset] - src[2 + offset] + 32 * (src[3 + offset] - src[4 + offset]) +
                  243 * (src[5 + offset] - src[6 + offset]);
      t[l + 48] = src[1 + offset] + src[2 + offset] + 64 * (src[3 + offset] + src[4 + offset]) +
                  729 * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 7; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 7] =
        t[1 + offset] - t[2 + offset] + 2 * (t[3 + offset] - t[4 + offset]) + 3 * (t[5 + offset] - t[6 + offset]);
      m[l + 14] =
        t[1 + offset] + t[2 + offset] + 4 * (t[3 + offset] + t[4 + offset]) + 9 * (t[5 + offset] + t[6 + offset]);
      m[l + 21] =
        t[1 + offset] - t[2 + offset] + 8 * (t[3 + offset] - t[4 + offset]) + 27 * (t[5 + offset] - t[6 + offset]);
      m[l + 28] =
        t[1 + offset] + t[2 + offset] + 16 * (t[3 + offset] + t[4 + offset]) + 81 * (t[5 + offset] + t[6 + offset]);
      m[l + 35] =
        t[1 + offset] - t[2 + offset] + 32 * (t[3 + offset] - t[4 + offset]) + 243 * (t[5 + offset] - t[6 + offset]);
      m[l + 42] = t[1 + offset] + t[2 + offset] + 64 * (t[3 + offset] + t[4 + offset]) +
                  729 * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < 7; ++k) {
      int dst_k_offset = k * dst_step * C4NUM;
      int m_k_offset = k * 7;
      for (int j = 0; j < 7; ++j) {
        dst_data[i + dst_k_offset + j * C4NUM] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

// Reference to the paper "Fast Algorithms for Convolutional Neural Networks"
// Utilize cost model to compute performance gain.
// If the gain is greater than got from Im2col, winograd algorithm will be chosen.
int SelectOutputUnit(ConvParameter *conv_param) {
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_c = conv_param->input_channel_;
  int out_w = conv_param->output_w_;
  int out_h = conv_param->output_h_;
  int out_c = conv_param->output_channel_;
  int unit2 = UP_DIV(out_w * out_h, C12NUM * conv_param->op_parameter_.thread_num_);
  int max_out_unit = (int)(sqrtf((float)unit2));
  max_out_unit = max_out_unit < MAX_UNIT ? max_out_unit : MAX_UNIT;
  max_out_unit = max_out_unit > MIN_UNIT ? max_out_unit : MIN_UNIT;

  int unit = 0;
  float max_rate = 0.0f;
  float common_cost = (float)out_h * out_w * in_c * out_c * kernel_h * kernel_w;

  for (int i = MIN_UNIT; i <= max_out_unit; ++i) {
    int input_unit = i + kernel_w - 1;
    if (!GetOutputTransFunc(input_unit, i)) {
      continue;
    }
    float penalty = ((float)input_unit * input_unit) / ((float)kernel_h * kernel_w) * 0.12f;
    float wino_cost = ((2 + out_c) * (float)input_unit * input_unit * in_c + ((float)input_unit + i) * i * out_c) *
                      UP_DIV(out_w, i) * UP_DIV(out_h, i);
    float reduce_rate = common_cost / wino_cost - penalty;
    if (reduce_rate > max_rate) {
      max_rate = reduce_rate;
      unit = i;
    }
  }
  if (max_rate < 1.0f) {
    return 1;
  }
  // If output_unit is 1, then it is conventional convolution
  return unit;
}

void CheckIfUseWinograd(bool *use_winograd, int *output_unit, ConvParameter *conv_param) {
  if (conv_param->kernel_w_ == conv_param->kernel_h_ && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1 &&
      conv_param->stride_h_ == 1 && conv_param->stride_w_ == 1) {
    *output_unit = SelectOutputUnit(conv_param);
    if (*output_unit > 1) {
      *use_winograd = true;
    }
  } else {
    *use_winograd = false;
  }
}
