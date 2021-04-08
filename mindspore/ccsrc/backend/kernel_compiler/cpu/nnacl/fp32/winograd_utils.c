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

#include "nnacl/fp32/winograd_utils.h"
#include "nnacl/base/minimal_filtering_generator.h"

#define MIN_UNIT 2
#define MAX_UNIT 8

static InputTransFunc InputTransFuncList[] = {
  NULL, NULL, NULL, NULL, InputTransform4x4Unit, NULL, InputTransform6x6Unit, NULL, InputTransform8x8Unit};

static OutputTransFunc OutputTransFuncList4[] = {NULL, NULL, OutputTransform4x2Unit, OutputTransform4x3Unit};

static OutputTransFunc OutputTransFuncReluList4[] = {NULL, NULL, OutputTransform4x2ReluUnit,
                                                     OutputTransform4x3ReluUnit};
static OutputTransFunc OutputTransFuncRelu6List4[] = {NULL, NULL, OutputTransform4x2Relu6Unit,
                                                      OutputTransform4x3Relu6Unit};

static OutputTransFunc OutputTransFuncList6[] = {
  NULL, NULL, OutputTransform6x2Unit, OutputTransform6x3Unit, OutputTransform6x4Unit, OutputTransform6x5Unit};

static OutputTransFunc OutputTransFuncReluList6[] = {NULL,
                                                     NULL,
                                                     OutputTransform6x2ReluUnit,
                                                     OutputTransform6x3ReluUnit,
                                                     OutputTransform6x4ReluUnit,
                                                     OutputTransform6x5ReluUnit};

static OutputTransFunc OutputTransFuncRelu6List6[] = {NULL,
                                                      NULL,
                                                      OutputTransform6x2Relu6Unit,
                                                      OutputTransform6x3Relu6Unit,
                                                      OutputTransform6x4Relu6Unit,
                                                      OutputTransform6x5Relu6Unit};

static OutputTransFunc OutputTransFuncList8[] = {NULL,
                                                 NULL,
                                                 OutputTransform8x2Unit,
                                                 OutputTransform8x3Unit,
                                                 OutputTransform8x4Unit,
                                                 OutputTransform8x5Unit,
                                                 OutputTransform8x6Unit,
                                                 OutputTransform8x7Unit};

static OutputTransFunc OutputTransFuncReluList8[] = {NULL,
                                                     NULL,
                                                     OutputTransform8x2ReluUnit,
                                                     OutputTransform8x3ReluUnit,
                                                     OutputTransform8x4ReluUnit,
                                                     OutputTransform8x5ReluUnit,
                                                     OutputTransform8x6ReluUnit,
                                                     OutputTransform8x7ReluUnit};

static OutputTransFunc OutputTransFuncRelu6List8[] = {NULL,
                                                      NULL,
                                                      OutputTransform8x2Relu6Unit,
                                                      OutputTransform8x3Relu6Unit,
                                                      OutputTransform8x4Relu6Unit,
                                                      OutputTransform8x5Relu6Unit,
                                                      OutputTransform8x6Relu6Unit,
                                                      OutputTransform8x7Relu6Unit};

void GeneralInputTransformUnit(const float *src_data, float *dst_data, const float *matrix_b, const float *matrix_bt,
                               int src_step, int dst_step, int in_unit) {
  int len = in_unit * in_unit;
  if (len > MAX_LEN) return;
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[MAX_LEN];
  MS_FLOAT32X4 t[MAX_LEN];
  MS_FLOAT32X4 m[MAX_LEN];
  MS_FLOAT32X4 vec_b[MAX_LEN];
  MS_FLOAT32X4 vec_bt[MAX_LEN];
  for (int i = 0; i < len; i++) {
    src[i] = MS_LDQ_F32(src_data + i * src_step);
    vec_b[i] = MS_MOVQ_F32(matrix_b[i]);
    vec_bt[i] = MS_MOVQ_F32(matrix_bt[i]);
  }
  MatrixMultiplyVec(vec_bt, src, t, NULL, in_unit, in_unit, in_unit);
  MatrixMultiplyVec(t, vec_b, m, NULL, in_unit, in_unit, in_unit);
  for (int i = 0; i < len; i++) {
    MS_STQ_F32(dst_data + i * dst_step, m[i]);
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

void GeneralOutputTransformUnit(const float *src_data, float *dst_data, const float *bias_data, const float *matrix_a,
                                const float *matrix_at, int src_step, int dst_step, int in_unit, int out_unit) {
  int src_len = in_unit * in_unit;
  if (src_len > MAX_LEN) {
    return;
  }
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[MAX_LEN];
  MS_FLOAT32X4 t[MAX_LEN];
  MS_FLOAT32X4 m[MAX_LEN];
  MS_FLOAT32X4 vec_a[MAX_LEN];
  MS_FLOAT32X4 vec_at[MAX_LEN];
  int tmp_len = in_unit * out_unit;
  if (tmp_len > MAX_LEN) return;

  for (int i = 0; i < tmp_len; i++) {
    vec_a[i] = MS_MOVQ_F32(matrix_a[i]);
    vec_at[i] = MS_MOVQ_F32(matrix_at[i]);
  }
  for (int i = 0; i < src_len; i++) {
    src[i] = MS_LDQ_F32(src_data + i * src_step);
  }
  MatrixMultiplyVec(vec_at, src, t, NULL, out_unit, in_unit, in_unit);
  MatrixMultiplyVec(t, vec_a, m, bias_data, out_unit, in_unit, out_unit);

  for (int i = 0; i < out_unit; i++) {
    int dst_k_offset = i * dst_step * C4NUM;
    int m_k_offset = i * out_unit;
    for (int j = 0; j < out_unit; j++) {
      MS_STQ_F32(dst_data + dst_k_offset + j * C4NUM, m[m_k_offset + j]);
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

void InputTransform4x4Unit(const float *src_data, float *dst_data, int src_step, int dst_step, int real_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  if (real_c == 4) {
    MS_FLOAT32X4 src[16];
    MS_FLOAT32X4 t[16];
    MS_FLOAT32X4 m[16];
    Load16Data;
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = MS_SUBQ_F32(src[offset], src[2 + offset]);
      t[4 + l] = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
      t[8 + l] = MS_SUBQ_F32(src[2 + offset], src[1 + offset]);
      t[12 + l] = MS_SUBQ_F32(src[3 + offset], src[1 + offset]);
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      m[l] = MS_SUBQ_F32(t[offset], t[2 + offset]);
      m[4 + l] = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
      m[8 + l] = MS_SUBQ_F32(t[2 + offset], t[1 + offset]);
      m[12 + l] = MS_SUBQ_F32(t[3 + offset], t[1 + offset]);
    }
    for (int i = 0; i < 16; i++) {
      MS_STQ_F32(dst_data + i * dst_step, m[i]);
    }
  } else {
#endif
    float src[16];
    float t[16];
    float m[16];
    for (int i = 0; i < real_c; ++i) {
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
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  }
#endif
}

void InputTransform6x6Unit(const float *src_data, float *dst_data, int src_step, int dst_step, int real_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  if (real_c == 4) {
    MS_FLOAT32X4 src[36];
    MS_FLOAT32X4 t[36];
    MS_FLOAT32X4 m[36];
    Load36Data;
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      MS_FLOAT32X4 tmp1 = MS_SUBQ_F32(src[3 + offset], src[1 + offset]);
      MS_FLOAT32X4 tmp2 = MS_SUBQ_F32(src[4 + offset], src[2 + offset]);
      t[l] =
        MS_ADDQ_F32(MS_SUBQ_F32(MS_MULQ_N_F32(src[offset], 4), MS_MULQ_N_F32(src[2 + offset], 5)), src[4 + offset]);
      t[6 + l] = MS_ADDQ_F32(MS_MULQ_N_F32(MS_ADDQ_F32(src[1 + offset], src[2 + offset]), -4),
                             MS_ADDQ_F32(src[3 + offset], src[4 + offset]));
      t[12 + l] = MS_ADDQ_F32(MS_MULQ_N_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]), 4),
                              MS_SUBQ_F32(src[4 + offset], src[3 + offset]));
      t[18 + l] = MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 2), tmp2);
      t[24 + l] = MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, -2), tmp2);
      t[30 + l] =
        MS_ADDQ_F32(MS_SUBQ_F32(MS_MULQ_N_F32(src[1 + offset], 4), MS_MULQ_N_F32(src[3 + offset], 5)), src[5 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      MS_FLOAT32X4 tmp1 = MS_SUBQ_F32(t[3 + offset], t[1 + offset]);
      MS_FLOAT32X4 tmp2 = MS_SUBQ_F32(t[4 + offset], t[2 + offset]);
      m[l] = MS_ADDQ_F32(MS_SUBQ_F32(MS_MULQ_N_F32(t[offset], 4), MS_MULQ_N_F32(t[2 + offset], 5)), t[4 + offset]);
      m[6 + l] = MS_ADDQ_F32(MS_MULQ_N_F32(MS_ADDQ_F32(t[1 + offset], t[2 + offset]), -4),
                             MS_ADDQ_F32(t[3 + offset], t[4 + offset]));
      m[12 + l] = MS_ADDQ_F32(MS_MULQ_N_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]), 4),
                              MS_SUBQ_F32(t[4 + offset], t[3 + offset]));
      m[18 + l] = MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 2), tmp2);
      m[24 + l] = MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, -2), tmp2);
      m[30 + l] =
        MS_ADDQ_F32(MS_SUBQ_F32(MS_MULQ_N_F32(t[1 + offset], 4), MS_MULQ_N_F32(t[3 + offset], 5)), t[5 + offset]);
    }
    for (int i = 0; i < 36; i++) {
      MS_STQ_F32(dst_data + i * dst_step, m[i]);
    }
  } else {
#endif
    float src[36];
    float t[36];
    float m[36];
    for (int i = 0; i < real_c; ++i) {
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
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  }
#endif
}

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void InputTransform8x8Unit_block4(const float *src_data, float *dst_data, int src_step, int dst_step) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[64];
  MS_FLOAT32X4 m[64];
  Load64Data;
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    t[l] =
      MS_SUBQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(MS_MULQ_N_F32(src[offset], 0.5625), MS_MULQ_N_F32(src[2 + offset], 3.0625)),
                              MS_MULQ_N_F32(src[4 + offset], 3.5)),
                  src[6 + offset]);
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(MS_MULQ_N_F32(src[1 + offset], 1.125), MS_MULQ_N_F32(src[5 + offset], 0.5));
    MS_FLOAT32X4 tmp2 = MS_SUBQ_F32(MS_MULQ_N_F32(src[2 + offset], 2.25), MS_MULQ_N_F32(src[4 + offset], 3.25));
    t[8 + l] =
      MS_ADDQ_F32(MS_SUBQ_F32(MS_ADDQ_F32(tmp1, tmp2), MS_MULQ_N_F32(src[3 + offset], 1.625)), src[6 + offset]);
    t[16 + l] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(tmp2, tmp1), MS_MULQ_N_F32(src[3 + offset], 1.625)), src[6 + offset]);
    tmp1 = MS_ADDQ_F32(MS_MULQ_N_F32(src[1 + offset], 0.5625), src[5 + offset]);
    tmp2 = MS_SUBQ_F32(MS_MULQ_N_F32(src[2 + offset], 0.5625), MS_MULQ_N_F32(src[4 + offset], 2.5));
    t[24 + l] = MS_ADDQ_F32(MS_SUBQ_F32(MS_ADDQ_F32(tmp1, tmp2), MS_MULQ_N_F32(src[3 + offset], 2.5)), src[6 + offset]);
    t[32 + l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(tmp2, tmp1), MS_MULQ_N_F32(src[3 + offset], 2.5)), src[6 + offset]);
    tmp1 = MS_ADDQ_F32(MS_MULQ_N_F32(src[1 + offset], 0.375), MS_MULQ_N_F32(src[5 + offset], 1.5));
    tmp2 = MS_SUBQ_F32(MS_MULQ_N_F32(src[2 + offset], 0.25), MS_MULQ_N_F32(src[4 + offset], 1.25));
    t[40 + l] =
      MS_ADDQ_F32(MS_SUBQ_F32(MS_ADDQ_F32(tmp1, tmp2), MS_MULQ_N_F32(src[3 + offset], 1.875)), src[6 + offset]);
    t[48 + l] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(tmp2, tmp1), MS_MULQ_N_F32(src[3 + offset], 1.875)), src[6 + offset]);
    t[56 + l] = MS_ADDQ_F32(
      MS_SUBQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(src[1 + offset], -0.5625), MS_MULQ_N_F32(src[3 + offset], 3.0625)),
                  MS_MULQ_N_F32(src[5 + offset], 3.5)),
      src[7 + offset]);
  }
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    m[l] = MS_SUBQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(MS_MULQ_N_F32(t[offset], 0.5625), MS_MULQ_N_F32(t[2 + offset], 3.0625)),
                                   MS_MULQ_N_F32(t[4 + offset], 3.5)),
                       t[6 + offset]);
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(MS_MULQ_N_F32(t[1 + offset], 1.125), MS_MULQ_N_F32(t[5 + offset], 0.5));
    MS_FLOAT32X4 tmp2 = MS_SUBQ_F32(MS_MULQ_N_F32(t[2 + offset], 2.25), MS_MULQ_N_F32(t[4 + offset], 3.25));
    m[8 + l] = MS_ADDQ_F32(MS_SUBQ_F32(MS_ADDQ_F32(tmp1, tmp2), MS_MULQ_N_F32(t[3 + offset], 1.625)), t[6 + offset]);
    m[16 + l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(tmp2, tmp1), MS_MULQ_N_F32(t[3 + offset], 1.625)), t[6 + offset]);
    tmp1 = MS_ADDQ_F32(MS_MULQ_N_F32(t[1 + offset], 0.5625), t[5 + offset]);
    tmp2 = MS_SUBQ_F32(MS_MULQ_N_F32(t[2 + offset], 0.5625), MS_MULQ_N_F32(t[4 + offset], 2.5));
    m[24 + l] = MS_ADDQ_F32(MS_SUBQ_F32(MS_ADDQ_F32(tmp1, tmp2), MS_MULQ_N_F32(t[3 + offset], 2.5)), t[6 + offset]);
    m[32 + l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(tmp2, tmp1), MS_MULQ_N_F32(t[3 + offset], 2.5)), t[6 + offset]);
    tmp1 = MS_ADDQ_F32(MS_MULQ_N_F32(t[1 + offset], 0.375), MS_MULQ_N_F32(t[5 + offset], 1.5));
    tmp2 = MS_SUBQ_F32(MS_MULQ_N_F32(t[2 + offset], 0.25), MS_MULQ_N_F32(t[4 + offset], 1.25));
    m[40 + l] = MS_ADDQ_F32(MS_SUBQ_F32(MS_ADDQ_F32(tmp1, tmp2), MS_MULQ_N_F32(t[3 + offset], 1.875)), t[6 + offset]);
    m[48 + l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(tmp2, tmp1), MS_MULQ_N_F32(t[3 + offset], 1.875)), t[6 + offset]);
    m[56 + l] =
      MS_ADDQ_F32(MS_SUBQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(t[1 + offset], -0.5625), MS_MULQ_N_F32(t[3 + offset], 3.0625)),
                              MS_MULQ_N_F32(t[5 + offset], 3.5)),
                  t[7 + offset]);
  }
  for (int i = 0; i < 64; i++) {
    MS_STQ_F32(dst_data + i * dst_step, m[i]);
  }
}
#endif

void InputTransform8x8Unit(const float *src_data, float *dst_data, int src_step, int dst_step, int real_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  if (real_c == 4) {
    InputTransform8x8Unit_block4(src_data, dst_data, src_step, dst_step);
  } else {
#endif
    float src[64];
    float t[64];
    float m[64];
    for (int i = 0; i < real_c; ++i) {
      for (int j = 0; j < 64; ++j) {
        src[j] = src_data[i + j * src_step];
      }
      for (int l = 0; l < 8; ++l) {
        int offset = l * 8;
        t[l] = 0.5625f * src[offset] - 3.0625f * src[2 + offset] + 3.5f * src[4 + offset] - src[6 + offset];
        float tmp1 = 1.125f * src[1 + offset] + 0.5f * src[5 + offset];
        float tmp2 = 2.25f * src[2 + offset] - 3.25f * src[4 + offset];
        t[8 + l] = tmp1 + tmp2 - 1.625f * src[3 + offset] + src[6 + offset];
        t[16 + l] = tmp2 - tmp1 + 1.625f * src[3 + offset] + src[6 + offset];
        tmp1 = 0.5625f * src[1 + offset] + src[5 + offset];
        tmp2 = 0.5625f * src[2 + offset] - 2.5f * src[4 + offset];
        t[24 + l] = tmp1 + tmp2 - 2.5f * src[3 + offset] + src[6 + offset];
        t[32 + l] = tmp2 - tmp1 + 2.5f * src[3 + offset] + src[6 + offset];
        tmp1 = 0.375f * src[1 + offset] + 1.5f * src[5 + offset];
        tmp2 = 0.25f * src[2 + offset] - 1.25f * src[4 + offset];
        t[40 + l] = tmp1 + tmp2 - 1.875f * src[3 + offset] + src[6 + offset];
        t[48 + l] = tmp2 - tmp1 + 1.875f * src[3 + offset] + src[6 + offset];
        t[56 + l] = -0.5625f * src[1 + offset] + 3.0625f * src[3 + offset] - 3.5f * src[5 + offset] + src[7 + offset];
      }
      for (int l = 0; l < 8; ++l) {
        int offset = l * 8;
        m[l] = 0.5625f * t[offset] - 3.0625f * t[2 + offset] + 3.5f * t[4 + offset] - t[6 + offset];
        float tmp1 = 1.125f * t[1 + offset] + 0.5f * t[5 + offset];
        float tmp2 = 2.25f * t[2 + offset] - 3.25f * t[4 + offset];
        m[8 + l] = tmp1 + tmp2 - 1.625f * t[3 + offset] + t[6 + offset];
        m[16 + l] = tmp2 - tmp1 + 1.625f * t[3 + offset] + t[6 + offset];
        tmp1 = 0.5625f * t[1 + offset] + t[5 + offset];
        tmp2 = 0.5625f * t[2 + offset] - 2.5f * t[4 + offset];
        m[24 + l] = tmp1 + tmp2 - 2.5f * t[3 + offset] + t[6 + offset];
        m[32 + l] = tmp2 - tmp1 + 2.5f * t[3 + offset] + t[6 + offset];
        tmp1 = 0.375f * t[1 + offset] + 1.5f * t[5 + offset];
        tmp2 = 0.25f * t[2 + offset] - 1.25f * t[4 + offset];
        m[40 + l] = tmp1 + tmp2 - 1.875f * t[3 + offset] + t[6 + offset];
        m[48 + l] = tmp2 - tmp1 + 1.875f * t[3 + offset] + t[6 + offset];
        m[56 + l] = -0.5625f * t[1 + offset] + 3.0625f * t[3 + offset] - 3.5f * t[5 + offset] + t[7 + offset];
      }
      for (int k = 0; k < 64; ++k) {
        dst_data[i + k * dst_step] = m[k];
      }
    }
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  }
#endif
}

OutputTransFunc GetOutputTransFunc(int input_unit, int output_unit, ActType act_type) {
  if (input_unit == 4 && output_unit < 4) {
    if (act_type == ActType_Relu) {
      return OutputTransFuncReluList4[output_unit];
    } else if (act_type == ActType_Relu6) {
      return OutputTransFuncRelu6List4[output_unit];
    } else {
      return OutputTransFuncList4[output_unit];
    }
  } else if (input_unit == 6 && output_unit < 6) {
    if (act_type == ActType_Relu) {
      return OutputTransFuncReluList6[output_unit];
    } else if (act_type == ActType_Relu6) {
      return OutputTransFuncRelu6List6[output_unit];
    } else {
      return OutputTransFuncList6[output_unit];
    }
  } else if (input_unit == 8 && output_unit < 8) {
    if (act_type == ActType_Relu) {
      return OutputTransFuncReluList8[output_unit];
    } else if (act_type == ActType_Relu6) {
      return OutputTransFuncRelu6List8[output_unit];
    } else {
      return OutputTransFuncList8[output_unit];
    }
  } else {
    return NULL;
  }
}

void OutputTransform4x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[16];
  MS_FLOAT32X4 t[8];
  MS_FLOAT32X4 m[4];
  Load16Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = MS_ADDQ_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
  }
  if (r_c == C4NUM && r_h == 2 && r_w == 2) {
    Store4Data;
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
#else
  float src[16];
  float t[8];
  float m[4];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 2;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform4x2ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[16];
  MS_FLOAT32X4 t[8];
  MS_FLOAT32X4 m[4];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load16Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = MS_ADDQ_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 2] = MS_MAXQ_F32(zero, m[l + 2]);
  }
  if (r_c == C4NUM && r_h == 2 && r_w == 2) {
    Store4Data;
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
#else
  float src[16];
  float t[8];
  float m[4];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 2;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform4x2Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[16];
  MS_FLOAT32X4 t[8];
  MS_FLOAT32X4 m[4];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load16Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = MS_ADDQ_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 2] = MS_MAXQ_F32(zero, m[l + 2]);
    m[l + 2] = MS_MINQ_F32(six, m[l + 2]);
  }
  if (r_c == C4NUM && r_h == 2 && r_w == 2) {
    Store4Data;
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
#else
  float src[16];
  float t[8];
  float m[4];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 2;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform4x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[16];
  MS_FLOAT32X4 t[12];
  MS_FLOAT32X4 m[9];
  Load16Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    MS_FLOAT32X4 tmp = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    t[l] = MS_ADDQ_F32(src[offset], tmp);
    t[l + 4] = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    t[l + 8] = MS_ADDQ_F32(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    MS_FLOAT32X4 tmp = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp), bias_ptr);
    m[l + 3] = MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = MS_ADDQ_F32(MS_ADDQ_F32(tmp, t[3 + offset]), bias_ptr);
  }
  if (r_c == C4NUM && r_h == 3 && r_w == 3) {
    Store9Data;
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
#else
  float src[16];
  float t[12];
  float m[9];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 3;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform4x3ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[16];
  MS_FLOAT32X4 t[12];
  MS_FLOAT32X4 m[9];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load16Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    MS_FLOAT32X4 tmp = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    t[l] = MS_ADDQ_F32(src[offset], tmp);
    t[l + 4] = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    t[l + 8] = MS_ADDQ_F32(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    MS_FLOAT32X4 tmp = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp), bias_ptr);
    m[l + 3] = MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = MS_ADDQ_F32(MS_ADDQ_F32(tmp, t[3 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 3] = MS_MAXQ_F32(zero, m[l + 3]);
    m[l + 6] = MS_MAXQ_F32(zero, m[l + 6]);
  }
  if (r_c == C4NUM && r_h == 3 && r_w == 3) {
    Store9Data;
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
#else
  float src[16];
  float t[12];
  float m[9];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 3;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform4x3Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[16];
  MS_FLOAT32X4 t[12];
  MS_FLOAT32X4 m[9];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load16Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    MS_FLOAT32X4 tmp = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    t[l] = MS_ADDQ_F32(src[offset], tmp);
    t[l + 4] = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    t[l + 8] = MS_ADDQ_F32(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    MS_FLOAT32X4 tmp = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp), bias_ptr);
    m[l + 3] = MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = MS_ADDQ_F32(MS_ADDQ_F32(tmp, t[3 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 3] = MS_MAXQ_F32(zero, m[l + 3]);
    m[l + 3] = MS_MINQ_F32(six, m[l + 3]);
    m[l + 6] = MS_MAXQ_F32(zero, m[l + 6]);
    m[l + 6] = MS_MINQ_F32(six, m[l + 6]);
  }
  if (r_c == C4NUM && r_h == 3 && r_w == 3) {
    Store9Data;
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
#else
  float src[16];
  float t[12];
  float m[9];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 3;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform6x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[12];
  MS_FLOAT32X4 m[4];
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                  src[4 + offset]);
    t[l + 6] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]),
                                       MS_MULQ_N_F32(MS_SUBQ_F32(src[3 + offset], src[4 + offset]), 2)),
                           src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]),
                  t[4 + offset]),
      bias_ptr);
    m[l + 2] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]),
                                                   MS_MULQ_N_F32(MS_SUBQ_F32(t[3 + offset], t[4 + offset]), 2)),
                                       t[5 + offset]),
                           bias_ptr);
  }
  if (r_c == C4NUM && r_h == 2 && r_w == 2) {
    Store4Data;
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
#else
  float src[36];
  float t[12];
  float m[4];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 2;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform6x2ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[12];
  MS_FLOAT32X4 m[4];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                  src[4 + offset]);
    t[l + 6] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]),
                                       MS_MULQ_N_F32(MS_SUBQ_F32(src[3 + offset], src[4 + offset]), 2)),
                           src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]),
                  t[4 + offset]),
      bias_ptr);
    m[l + 2] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]),
                                                   MS_MULQ_N_F32(MS_SUBQ_F32(t[3 + offset], t[4 + offset]), 2)),
                                       t[5 + offset]),
                           bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 2] = MS_MAXQ_F32(zero, m[l + 2]);
  }
  if (r_c == C4NUM && r_h == 2 && r_w == 2) {
    Store4Data;
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
#else
  float src[36];
  float t[12];
  float m[4];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 2;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform6x2Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[12];
  MS_FLOAT32X4 m[4];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                  src[4 + offset]);
    t[l + 6] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]),
                                       MS_MULQ_N_F32(MS_SUBQ_F32(src[3 + offset], src[4 + offset]), 2)),
                           src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]),
                  t[4 + offset]),
      bias_ptr);
    m[l + 2] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]),
                                                   MS_MULQ_N_F32(MS_SUBQ_F32(t[3 + offset], t[4 + offset]), 2)),
                                       t[5 + offset]),
                           bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 2] = MS_MAXQ_F32(zero, m[l + 2]);
    m[l + 2] = MS_MINQ_F32(six, m[l + 2]);
  }
  if (r_c == C4NUM && r_h == 2 && r_w == 2) {
    Store4Data;
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
#else
  float src[36];
  float t[12];
  float m[4];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 2;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform6x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[18];
  MS_FLOAT32X4 m[9];
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADDQ_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]),
                           MS_MULQ_N_F32(MS_SUBQ_F32(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]),
                                       MS_MULQ_N_F32(MS_SUBQ_F32(t[3 + offset], t[4 + offset]), 2)),
                           bias_ptr);
    m[l + 6] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C4NUM && r_h == 3 && r_w == 3) {
    Store9Data;
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
#else
  float src[36];
  float t[18];
  float m[9];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 3;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform6x3ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[18];
  MS_FLOAT32X4 m[9];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADDQ_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]),
                           MS_MULQ_N_F32(MS_SUBQ_F32(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]),
                                       MS_MULQ_N_F32(MS_SUBQ_F32(t[3 + offset], t[4 + offset]), 2)),
                           bias_ptr);
    m[l + 6] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 3] = MS_MAXQ_F32(zero, m[l + 3]);
    m[l + 6] = MS_MAXQ_F32(zero, m[l + 6]);
  }
  if (r_c == C4NUM && r_h == 3 && r_w == 3) {
    Store9Data;
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
#else
  float src[36];
  float t[18];
  float m[9];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 3;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform6x3Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[18];
  MS_FLOAT32X4 m[9];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADDQ_F32(MS_SUBQ_F32(src[1 + offset], src[2 + offset]),
                           MS_MULQ_N_F32(MS_SUBQ_F32(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = MS_ADDQ_F32(MS_ADDQ_F32(MS_SUBQ_F32(t[1 + offset], t[2 + offset]),
                                       MS_MULQ_N_F32(MS_SUBQ_F32(t[3 + offset], t[4 + offset]), 2)),
                           bias_ptr);
    m[l + 6] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 3] = MS_MAXQ_F32(zero, m[l + 3]);
    m[l + 3] = MS_MINQ_F32(six, m[l + 3]);
    m[l + 6] = MS_MAXQ_F32(zero, m[l + 6]);
    m[l + 6] = MS_MINQ_F32(six, m[l + 6]);
  }
  if (r_c == C4NUM && r_h == 3 && r_w == 3) {
    Store9Data;
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
#else
  float src[36];
  float t[18];
  float m[9];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 3;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform6x4Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[24];
  MS_FLOAT32X4 m[16];
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2));
    t[l + 12] = MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4));
    t[l + 18] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2)), bias_ptr);
    m[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), bias_ptr);
    m[l + 12] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C4NUM && r_h == 4 && r_w == 4) {
    Store16Data;
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
#else
  float src[36];
  float t[24];
  float m[16];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 4;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform6x4ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[24];
  MS_FLOAT32X4 m[16];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2));
    t[l + 12] = MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4));
    t[l + 18] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2)), bias_ptr);
    m[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), bias_ptr);
    m[l + 12] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 4] = MS_MAXQ_F32(zero, m[l + 4]);
    m[l + 8] = MS_MAXQ_F32(zero, m[l + 8]);
    m[l + 12] = MS_MAXQ_F32(zero, m[l + 12]);
  }
  if (r_c == C4NUM && r_h == 4 && r_w == 4) {
    Store16Data;
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
#else
  float src[36];
  float t[24];
  float m[16];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 4;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform6x4Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[24];
  MS_FLOAT32X4 m[16];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2));
    t[l + 12] = MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4));
    t[l + 18] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2)), bias_ptr);
    m[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), bias_ptr);
    m[l + 12] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 4] = MS_MAXQ_F32(zero, m[l + 4]);
    m[l + 4] = MS_MINQ_F32(six, m[l + 4]);
    m[l + 8] = MS_MAXQ_F32(zero, m[l + 8]);
    m[l + 8] = MS_MINQ_F32(six, m[l + 8]);
    m[l + 12] = MS_MAXQ_F32(zero, m[l + 12]);
    m[l + 12] = MS_MINQ_F32(six, m[l + 12]);
  }
  if (r_c == C4NUM && r_h == 4 && r_w == 4) {
    Store16Data;
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
#else
  float src[36];
  float t[24];
  float m[16];
  for (int i = 0; i < r_c; ++i) {
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 4;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform6x5Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[30];
  MS_FLOAT32X4 m[25];
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2));
    t[l + 12] = MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4));
    t[l + 18] = MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2)), bias_ptr);
    m[l + 10] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), bias_ptr);
    m[l + 15] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8)), bias_ptr);
    m[l + 20] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 16)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C4NUM && r_h == 5 && r_w == 5) {
    Store25Data;
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 5;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform6x5ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[30];
  MS_FLOAT32X4 m[25];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2));
    t[l + 12] = MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4));
    t[l + 18] = MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2)), bias_ptr);
    m[l + 10] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), bias_ptr);
    m[l + 15] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8)), bias_ptr);
    m[l + 20] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 16)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 5] = MS_MAXQ_F32(zero, m[l + 5]);
    m[l + 10] = MS_MAXQ_F32(zero, m[l + 10]);
    m[l + 15] = MS_MAXQ_F32(zero, m[l + 15]);
    m[l + 20] = MS_MAXQ_F32(zero, m[l + 20]);
  }
  if (r_c == C4NUM && r_h == 5 && r_w == 5) {
    Store25Data;
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 5;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform6x5Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[36];
  MS_FLOAT32X4 t[30];
  MS_FLOAT32X4 m[25];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load36Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2));
    t[l + 12] = MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4));
    t[l + 18] = MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 2)), bias_ptr);
    m[l + 10] = MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 4)), bias_ptr);
    m[l + 15] = MS_ADDQ_F32(MS_ADDQ_F32(tmp3, MS_MULQ_N_F32(tmp4, 8)), bias_ptr);
    m[l + 20] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(tmp1, MS_MULQ_N_F32(tmp2, 16)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 5] = MS_MAXQ_F32(zero, m[l + 5]);
    m[l + 5] = MS_MINQ_F32(six, m[l + 5]);
    m[l + 10] = MS_MAXQ_F32(zero, m[l + 10]);
    m[l + 10] = MS_MINQ_F32(six, m[l + 10]);
    m[l + 15] = MS_MAXQ_F32(zero, m[l + 15]);
    m[l + 15] = MS_MINQ_F32(six, m[l + 15]);
    m[l + 20] = MS_MAXQ_F32(zero, m[l + 20]);
    m[l + 20] = MS_MINQ_F32(six, m[l + 20]);
  }
  if (r_c == C4NUM && r_h == 5 && r_w == 5) {
    Store25Data;
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
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 5;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform8x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[16];
  MS_FLOAT32X4 m[4];
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C4NUM && r_h == 2 && r_w == 2) {
    Store4Data;
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
#else
  float src[64];
  float t[16];
  float m[4];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 2] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 2;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform8x2ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[16];
  MS_FLOAT32X4 m[4];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 2] = MS_MAXQ_F32(zero, m[l + 2]);
  }
  if (r_c == C4NUM && r_h == 2 && r_w == 2) {
    Store4Data;
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
#else
  float src[64];
  float t[16];
  float m[4];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 2] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 2;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform8x2Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[16];
  MS_FLOAT32X4 m[4];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 2] = MS_MAXQ_F32(zero, m[l + 2]);
    m[l + 2] = MS_MINQ_F32(six, m[l + 2]);
  }
  if (r_c == C4NUM && r_h == 2 && r_w == 2) {
    Store4Data;
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
#else
  float src[64];
  float t[16];
  float m[4];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 2] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 2;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform8x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[24];
  MS_FLOAT32X4 m[9];
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 6] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C4NUM && r_h == 3 && r_w == 3) {
    Store9Data;
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
#else
  float src[64];
  float t[24];
  float m[9];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 3; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 3] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 6] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                 2.25f * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 3;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform8x3ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[24];
  MS_FLOAT32X4 m[9];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 6] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 3] = MS_MAXQ_F32(zero, m[l + 3]);
    m[l + 6] = MS_MAXQ_F32(zero, m[l + 6]);
  }
  if (r_c == C4NUM && r_h == 3 && r_w == 3) {
    Store9Data;
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
#else
  float src[64];
  float t[24];
  float m[9];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 3; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 3] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 6] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                 2.25f * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 3;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform8x3Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[24];
  MS_FLOAT32X4 m[9];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 6] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 3] = MS_MAXQ_F32(zero, m[l + 3]);
    m[l + 3] = MS_MINQ_F32(six, m[l + 3]);
    m[l + 6] = MS_MAXQ_F32(zero, m[l + 6]);
    m[l + 6] = MS_MINQ_F32(six, m[l + 6]);
  }
  if (r_c == C4NUM && r_h == 3 && r_w == 3) {
    Store9Data;
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
#else
  float src[64];
  float t[24];
  float m[9];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 3; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 3] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 6] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                 2.25f * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 3;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

void OutputTransform8x4Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[32];
  MS_FLOAT32X4 m[16];
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 8] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 12] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)),
                              t[7 + offset]),
                  bias_ptr);
  }
  if (r_c == C4NUM && r_h == 4 && r_w == 4) {
    Store16Data;
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
#else
  float src[64];
  float t[32];
  float m[16];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 4] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 8] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                 2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 12] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 4;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

void OutputTransform8x4ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[32];
  MS_FLOAT32X4 m[16];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 8] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 12] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)),
                              t[7 + offset]),
                  bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 4] = MS_MAXQ_F32(zero, m[l + 4]);
    m[l + 8] = MS_MAXQ_F32(zero, m[l + 8]);
    m[l + 12] = MS_MAXQ_F32(zero, m[l + 12]);
  }
  if (r_c == C4NUM && r_h == 4 && r_w == 4) {
    Store16Data;
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
#else
  float src[64];
  float t[32];
  float m[16];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 4] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 8] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                 2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 12] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 4;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
#endif
}

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void OutputTransform8x4Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[32];
  MS_FLOAT32X4 m[16];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 8] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 12] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)),
                              t[7 + offset]),
                  bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 4] = MS_MAXQ_F32(zero, m[l + 4]);
    m[l + 4] = MS_MINQ_F32(six, m[l + 4]);
    m[l + 8] = MS_MAXQ_F32(zero, m[l + 8]);
    m[l + 8] = MS_MINQ_F32(six, m[l + 8]);
    m[l + 12] = MS_MAXQ_F32(zero, m[l + 12]);
    m[l + 12] = MS_MINQ_F32(six, m[l + 12]);
  }
  if (r_c == C4NUM && r_h == 4 && r_w == 4) {
    Store16Data;
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
#else
void OutputTransform8x4Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float src[64];
  float t[32];
  float m[16];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 4] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 8] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                 2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 12] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 4;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
}
#endif

void OutputTransform8x5Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[40];
  MS_FLOAT32X4 m[25];
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 10] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 15] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 20] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)),
                              t[7 + offset]),
                  bias_ptr);
  }
  if (r_c == C4NUM && r_h == 5 && r_w == 5) {
    Store25Data;
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
#else
  float src[64];
  float t[40];
  float m[25];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = 0.0625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  5.0625f * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 5; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 5] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 10] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 15] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]);
      m[l + 20] = 0.0625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  5.0625f * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 5;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
#endif
}

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void OutputTransform8x5ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[40];
  MS_FLOAT32X4 m[25];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 10] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 15] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 20] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)),
                              t[7 + offset]),
                  bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 5] = MS_MAXQ_F32(zero, m[l + 5]);
    m[l + 10] = MS_MAXQ_F32(zero, m[l + 10]);
    m[l + 15] = MS_MAXQ_F32(zero, m[l + 15]);
    m[l + 20] = MS_MAXQ_F32(zero, m[l + 20]);
  }
  if (r_c == C4NUM && r_h == 5 && r_w == 5) {
    Store25Data;
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
#else
void OutputTransform8x5ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float src[64];
  float t[40];
  float m[25];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = 0.0625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  5.0625f * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 5; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 5] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 10] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 15] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]);
      m[l + 20] = 0.0625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  5.0625f * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 5;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
}
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void OutputTransform8x5Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[40];
  MS_FLOAT32X4 m[25];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 10] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 15] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 20] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)),
                              t[7 + offset]),
                  bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 5] = MS_MAXQ_F32(zero, m[l + 5]);
    m[l + 5] = MS_MINQ_F32(six, m[l + 5]);
    m[l + 10] = MS_MAXQ_F32(zero, m[l + 10]);
    m[l + 10] = MS_MINQ_F32(six, m[l + 10]);
    m[l + 15] = MS_MAXQ_F32(zero, m[l + 15]);
    m[l + 15] = MS_MINQ_F32(six, m[l + 15]);
    m[l + 20] = MS_MAXQ_F32(zero, m[l + 20]);
    m[l + 20] = MS_MINQ_F32(six, m[l + 20]);
  }
  if (r_c == C4NUM && r_h == 5 && r_w == 5) {
    Store25Data;
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
#else
void OutputTransform8x5Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float src[64];
  float t[40];
  float m[25];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = 0.0625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  5.0625f * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 5; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 5] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 10] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 15] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]);
      m[l + 20] = 0.0625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  5.0625f * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 5;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
}
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void OutputTransform8x6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[48];
  MS_FLOAT32X4 m[36];
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625));
    t[l + 40] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 12] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 18] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 24] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 30] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375)),
                  t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C4NUM && r_h == 6 && r_w == 6) {
    for (int i = 0; i < 6; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 6;
      MS_STQ_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_STQ_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_STQ_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_STQ_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_STQ_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_STQ_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
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
#else
void OutputTransform8x6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
  float src[64];
  float t[48];
  float m[36];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = 0.0625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  5.0625f * (src[5 + offset] + src[6 + offset]);
      t[l + 40] = 0.03125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  7.59375f * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 6] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 12] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 18] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]);
      m[l + 24] = 0.0625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  5.0625f * (t[5 + offset] + t[6 + offset]);
      m[l + 30] = 0.03125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  7.59375f * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 6;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
}
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void OutputTransform8x6ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[48];
  MS_FLOAT32X4 m[36];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625));
    t[l + 40] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 12] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 18] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 24] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 30] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375)),
                  t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 6] = MS_MAXQ_F32(zero, m[l + 6]);
    m[l + 12] = MS_MAXQ_F32(zero, m[l + 12]);
    m[l + 18] = MS_MAXQ_F32(zero, m[l + 18]);
    m[l + 24] = MS_MAXQ_F32(zero, m[l + 24]);
    m[l + 30] = MS_MAXQ_F32(zero, m[l + 30]);
  }
  if (r_c == C4NUM && r_h == 6 && r_w == 6) {
    for (int i = 0; i < 6; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 6;
      MS_STQ_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_STQ_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_STQ_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_STQ_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_STQ_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_STQ_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
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
#else
void OutputTransform8x6ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float src[64];
  float t[48];
  float m[36];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = 0.0625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  5.0625f * (src[5 + offset] + src[6 + offset]);
      t[l + 40] = 0.03125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  7.59375f * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 6] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 12] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 18] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]);
      m[l + 24] = 0.0625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  5.0625f * (t[5 + offset] + t[6 + offset]);
      m[l + 30] = 0.03125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  7.59375f * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 6;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
}
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void OutputTransform8x6Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[48];
  MS_FLOAT32X4 m[36];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625));
    t[l + 40] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375)),
                            src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 12] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 18] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 24] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 30] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375)),
                  t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 6] = MS_MAXQ_F32(zero, m[l + 6]);
    m[l + 6] = MS_MINQ_F32(six, m[l + 6]);
    m[l + 12] = MS_MAXQ_F32(zero, m[l + 12]);
    m[l + 12] = MS_MINQ_F32(six, m[l + 12]);
    m[l + 18] = MS_MAXQ_F32(zero, m[l + 18]);
    m[l + 18] = MS_MINQ_F32(six, m[l + 18]);
    m[l + 24] = MS_MAXQ_F32(zero, m[l + 24]);
    m[l + 24] = MS_MINQ_F32(six, m[l + 24]);
    m[l + 30] = MS_MAXQ_F32(zero, m[l + 30]);
    m[l + 30] = MS_MINQ_F32(six, m[l + 30]);
  }
  if (r_c == C4NUM && r_h == 6 && r_w == 6) {
    for (int i = 0; i < 6; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 6;
      MS_STQ_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_STQ_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_STQ_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_STQ_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_STQ_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_STQ_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
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
#else
void OutputTransform8x6Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float src[64];
  float t[48];
  float m[36];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = 0.0625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  5.0625f * (src[5 + offset] + src[6 + offset]);
      t[l + 40] = 0.03125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  7.59375f * (src[5 + offset] - src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 6] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 12] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 18] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]);
      m[l + 24] = 0.0625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  5.0625f * (t[5 + offset] + t[6 + offset]);
      m[l + 30] = 0.03125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  7.59375f * (t[5 + offset] - t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 6;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
}
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void OutputTransform8x7Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[56];
  MS_FLOAT32X4 m[49];
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625));
    t[l + 40] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375));
    t[l + 48] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.015625), tmp2), MS_MULQ_N_F32(tmp3, 11.390625)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 14] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 21] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 28] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 35] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.015625), tmp2), MS_MULQ_N_F32(tmp3, 11.390625)),
                  t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C4NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      MS_STQ_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_STQ_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_STQ_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_STQ_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_STQ_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_STQ_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      MS_STQ_F32(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
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
#else
void OutputTransform8x7Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c) {
  float src[64];
  float t[56];
  float m[49];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = 0.0625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  5.0625f * (src[5 + offset] + src[6 + offset]);
      t[l + 40] = 0.03125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  7.59375f * (src[5 + offset] - src[6 + offset]);
      t[l + 48] = 0.015625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  11.390625f * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 7; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 7] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 14] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 21] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]);
      m[l + 28] = 0.0625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  5.0625f * (t[5 + offset] + t[6 + offset]);
      m[l + 35] = 0.03125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  7.59375f * (t[5 + offset] - t[6 + offset]);
      m[l + 42] = 0.015625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  11.390625f * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 7;
      for (int j = 0; j < r_w; ++j) {
        dst_data[i + dst_k_offset + j * out_c] = m[j + m_k_offset] + bias_data[i];
      }
    }
  }
}
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void OutputTransform8x7ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[56];
  MS_FLOAT32X4 m[49];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625));
    t[l + 40] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375));
    t[l + 48] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.015625), tmp2), MS_MULQ_N_F32(tmp3, 11.390625)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 14] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 21] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 28] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 35] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.015625), tmp2), MS_MULQ_N_F32(tmp3, 11.390625)),
                  t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l + 7] = MS_MAXQ_F32(zero, m[l + 7]);
    m[l + 14] = MS_MAXQ_F32(zero, m[l + 14]);
    m[l + 21] = MS_MAXQ_F32(zero, m[l + 21]);
    m[l + 28] = MS_MAXQ_F32(zero, m[l + 28]);
    m[l + 35] = MS_MAXQ_F32(zero, m[l + 35]);
    m[l + 42] = MS_MAXQ_F32(zero, m[l + 42]);
  }
  if (r_c == C4NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      MS_STQ_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_STQ_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_STQ_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_STQ_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_STQ_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_STQ_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      MS_STQ_F32(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
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
#else
void OutputTransform8x7ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float src[64];
  float t[56];
  float m[49];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = 0.0625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  5.0625f * (src[5 + offset] + src[6 + offset]);
      t[l + 40] = 0.03125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  7.59375f * (src[5 + offset] - src[6 + offset]);
      t[l + 48] = 0.015625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  11.390625f * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 7; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 7] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 14] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 21] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]);
      m[l + 28] = 0.0625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  5.0625f * (t[5 + offset] + t[6 + offset]);
      m[l + 35] = 0.03125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  7.59375f * (t[5 + offset] - t[6 + offset]);
      m[l + 42] = 0.015625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  11.390625f * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 7;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
}
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void OutputTransform8x7Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X4 src[64];
  MS_FLOAT32X4 t[56];
  MS_FLOAT32X4 m[49];
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6);
  Load64Data;
  MS_FLOAT32X4 bias_ptr = MS_LDQ_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625));
    t[l + 40] = MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375));
    t[l + 48] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.015625), tmp2), MS_MULQ_N_F32(tmp3, 11.390625)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    MS_FLOAT32X4 tmp1 = MS_ADDQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp2 = MS_ADDQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp3 = MS_ADDQ_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X4 tmp4 = MS_SUBQ_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X4 tmp5 = MS_SUBQ_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X4 tmp6 = MS_SUBQ_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.5), tmp5), MS_MULQ_N_F32(tmp6, 1.5)), bias_ptr);
    m[l + 14] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.25), tmp2), MS_MULQ_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 21] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.125), tmp5), MS_MULQ_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 28] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.0625), tmp2), MS_MULQ_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 35] =
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp4, 0.03125), tmp5), MS_MULQ_N_F32(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = MS_ADDQ_F32(
      MS_ADDQ_F32(MS_ADDQ_F32(MS_ADDQ_F32(MS_MULQ_N_F32(tmp1, 0.015625), tmp2), MS_MULQ_N_F32(tmp3, 11.390625)),
                  t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAXQ_F32(zero, m[l]);
    m[l] = MS_MINQ_F32(six, m[l]);
    m[l + 7] = MS_MAXQ_F32(zero, m[l + 7]);
    m[l + 7] = MS_MINQ_F32(six, m[l + 7]);
    m[l + 14] = MS_MAXQ_F32(zero, m[l + 14]);
    m[l + 14] = MS_MINQ_F32(six, m[l + 14]);
    m[l + 21] = MS_MAXQ_F32(zero, m[l + 21]);
    m[l + 21] = MS_MINQ_F32(six, m[l + 21]);
    m[l + 28] = MS_MAXQ_F32(zero, m[l + 28]);
    m[l + 28] = MS_MINQ_F32(six, m[l + 28]);
    m[l + 35] = MS_MAXQ_F32(zero, m[l + 35]);
    m[l + 35] = MS_MINQ_F32(six, m[l + 35]);
    m[l + 42] = MS_MAXQ_F32(zero, m[l + 42]);
    m[l + 42] = MS_MINQ_F32(six, m[l + 42]);
  }
  if (r_c == C4NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      MS_STQ_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_STQ_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_STQ_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_STQ_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_STQ_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_STQ_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      MS_STQ_F32(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
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
#else
void OutputTransform8x7Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float src[64];
  float t[56];
  float m[49];
  for (int i = 0; i < r_c; ++i) {
    // load source data
    for (int j = 0; j < 64; ++j) {
      src[j] = src_data[i + j * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset] + src[3 + offset] + src[4 + offset] + src[5 + offset] +
             src[6 + offset];
      t[l + 8] = 0.5f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                 1.5f * (src[5 + offset] - src[6 + offset]);
      t[l + 16] = 0.25f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  2.25f * (src[5 + offset] + src[6 + offset]);
      t[l + 24] = 0.125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  3.375f * (src[5 + offset] - src[6 + offset]);
      t[l + 32] = 0.0625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  5.0625f * (src[5 + offset] + src[6 + offset]);
      t[l + 40] = 0.03125f * (src[1 + offset] - src[2 + offset]) + (src[3 + offset] - src[4 + offset]) +
                  7.59375f * (src[5 + offset] - src[6 + offset]);
      t[l + 48] = 0.015625f * (src[1 + offset] + src[2 + offset]) + (src[3 + offset] + src[4 + offset]) +
                  11.390625f * (src[5 + offset] + src[6 + offset]) + src[7 + offset];
    }
    for (int l = 0; l < 7; ++l) {
      int offset = l * 8;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + t[3 + offset] + t[4 + offset] + t[5 + offset] + t[6 + offset];
      m[l + 7] = 0.5f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                 1.5f * (t[5 + offset] - t[6 + offset]);
      m[l + 14] = 0.25f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  2.25f * (t[5 + offset] + t[6 + offset]);
      m[l + 21] = 0.125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  3.375f * (t[5 + offset] - t[6 + offset]);
      m[l + 28] = 0.0625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  5.0625f * (t[5 + offset] + t[6 + offset]);
      m[l + 35] = 0.03125f * (t[1 + offset] - t[2 + offset]) + (t[3 + offset] - t[4 + offset]) +
                  7.59375f * (t[5 + offset] - t[6 + offset]);
      m[l + 42] = 0.015625f * (t[1 + offset] + t[2 + offset]) + (t[3 + offset] + t[4 + offset]) +
                  11.390625f * (t[5 + offset] + t[6 + offset]) + t[7 + offset];
    }
    // store output
    for (int k = 0; k < r_h; ++k) {
      int dst_k_offset = k * dst_step * out_c;
      int m_k_offset = k * 7;
      for (int j = 0; j < r_w; ++j) {
        float out_value = m[j + m_k_offset] + bias_data[i];
        out_value = out_value > 0 ? out_value : 0;
        out_value = out_value < 6 ? out_value : 6;
        dst_data[i + dst_k_offset + j * out_c] = out_value;
      }
    }
  }
}
#endif

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
    if (!GetOutputTransFunc(input_unit, i, ActType_No)) {
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

bool CheckIfUseWinograd(int *output_unit, ConvParameter *conv_param) {
  if (conv_param->kernel_w_ == conv_param->kernel_h_ && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1 &&
      conv_param->stride_h_ == 1 && conv_param->stride_w_ == 1) {
    *output_unit = SelectOutputUnit(conv_param);
    if (*output_unit > 1) {
      return true;
    }
  }
  return false;
}
