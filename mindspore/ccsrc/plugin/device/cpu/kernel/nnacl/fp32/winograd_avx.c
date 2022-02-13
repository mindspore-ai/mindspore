/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless re256uired by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef ENABLE_AVX
#include "nnacl/fp32/winograd_avx.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"

void InputTransform4x4AvxUnit(const float *src_data, float *dst_data, const int src_step, const int dst_step,
                              const int real_c) {
  if (real_c == C8NUM) {
    MS_FLOAT32X8 src[16];
    MS_FLOAT32X8 t[16];
    MS_FLOAT32X8 m[16];
    LoadAvx16Data;
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = MS_SUB256_F32(src[offset], src[2 + offset]);
      t[4 + l] = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
      t[8 + l] = MS_SUB256_F32(src[2 + offset], src[1 + offset]);
      t[12 + l] = MS_SUB256_F32(src[3 + offset], src[1 + offset]);
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      m[l] = MS_SUB256_F32(t[offset], t[2 + offset]);
      m[4 + l] = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
      m[8 + l] = MS_SUB256_F32(t[2 + offset], t[1 + offset]);
      m[12 + l] = MS_SUB256_F32(t[3 + offset], t[1 + offset]);
    }
    for (int i = 0; i < 16; i++) {
      MS_ST256_F32(dst_data + i * dst_step, m[i]);
    }
  } else {
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
  }
}

void InputTransform6x6AvxUnit(const float *src_data, float *dst_data, const int src_step, const int dst_step,
                              const int real_c) {
  if (real_c == C8NUM) {
    MS_FLOAT32X8 src[36];
    MS_FLOAT32X8 t[36];
    MS_FLOAT32X8 m[36];
    LoadAvx36Data;
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      MS_FLOAT32X8 tmp1 = MS_SUB256_F32(src[3 + offset], src[1 + offset]);
      MS_FLOAT32X8 tmp2 = MS_SUB256_F32(src[4 + offset], src[2 + offset]);
      t[l] = MS_ADD256_F32(MS_SUB256_F32(MS_MUL256_N_F32(src[offset], 4), MS_MUL256_N_F32(src[2 + offset], 5)),
                           src[4 + offset]);
      t[6 + l] = MS_ADD256_F32(MS_MUL256_N_F32(MS_ADD256_F32(src[1 + offset], src[2 + offset]), -4),
                               MS_ADD256_F32(src[3 + offset], src[4 + offset]));
      t[12 + l] = MS_ADD256_F32(MS_MUL256_N_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]), 4),
                                MS_SUB256_F32(src[4 + offset], src[3 + offset]));
      t[18 + l] = MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 2), tmp2);
      t[24 + l] = MS_ADD256_F32(MS_MUL256_N_F32(tmp1, -2), tmp2);
      t[30 + l] = MS_ADD256_F32(MS_SUB256_F32(MS_MUL256_N_F32(src[1 + offset], 4), MS_MUL256_N_F32(src[3 + offset], 5)),
                                src[5 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      MS_FLOAT32X8 tmp1 = MS_SUB256_F32(t[3 + offset], t[1 + offset]);
      MS_FLOAT32X8 tmp2 = MS_SUB256_F32(t[4 + offset], t[2 + offset]);
      m[l] =
        MS_ADD256_F32(MS_SUB256_F32(MS_MUL256_N_F32(t[offset], 4), MS_MUL256_N_F32(t[2 + offset], 5)), t[4 + offset]);
      m[6 + l] = MS_ADD256_F32(MS_MUL256_N_F32(MS_ADD256_F32(t[1 + offset], t[2 + offset]), -4),
                               MS_ADD256_F32(t[3 + offset], t[4 + offset]));
      m[12 + l] = MS_ADD256_F32(MS_MUL256_N_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]), 4),
                                MS_SUB256_F32(t[4 + offset], t[3 + offset]));
      m[18 + l] = MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 2), tmp2);
      m[24 + l] = MS_ADD256_F32(MS_MUL256_N_F32(tmp1, -2), tmp2);
      m[30 + l] = MS_ADD256_F32(MS_SUB256_F32(MS_MUL256_N_F32(t[1 + offset], 4), MS_MUL256_N_F32(t[3 + offset], 5)),
                                t[5 + offset]);
    }
    for (int i = 0; i < 36; i++) {
      MS_ST256_F32(dst_data + i * dst_step, m[i]);
    }
  } else {
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
  }
}

void InputTransform8x8AvxUnit_block8(const float *src_data, float *dst_data, const int src_step, const int dst_step) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[64];
  MS_FLOAT32X8 m[64];
  LoadAvx64Data;
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    t[l] = MS_SUB256_F32(
      MS_ADD256_F32(MS_SUB256_F32(MS_MUL256_N_F32(src[offset], 0.5625), MS_MUL256_N_F32(src[2 + offset], 3.0625)),
                    MS_MUL256_N_F32(src[4 + offset], 3.5)),
      src[6 + offset]);
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(MS_MUL256_N_F32(src[1 + offset], 1.125), MS_MUL256_N_F32(src[5 + offset], 0.5));
    MS_FLOAT32X8 tmp2 = MS_SUB256_F32(MS_MUL256_N_F32(src[2 + offset], 2.25), MS_MUL256_N_F32(src[4 + offset], 3.25));
    t[8 + l] =
      MS_ADD256_F32(MS_SUB256_F32(MS_ADD256_F32(tmp1, tmp2), MS_MUL256_N_F32(src[3 + offset], 1.625)), src[6 + offset]);
    t[16 + l] =
      MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(tmp2, tmp1), MS_MUL256_N_F32(src[3 + offset], 1.625)), src[6 + offset]);
    tmp1 = MS_ADD256_F32(MS_MUL256_N_F32(src[1 + offset], 0.5625), src[5 + offset]);
    tmp2 = MS_SUB256_F32(MS_MUL256_N_F32(src[2 + offset], 0.5625), MS_MUL256_N_F32(src[4 + offset], 2.5));
    t[24 + l] =
      MS_ADD256_F32(MS_SUB256_F32(MS_ADD256_F32(tmp1, tmp2), MS_MUL256_N_F32(src[3 + offset], 2.5)), src[6 + offset]);
    t[32 + l] =
      MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(tmp2, tmp1), MS_MUL256_N_F32(src[3 + offset], 2.5)), src[6 + offset]);
    tmp1 = MS_ADD256_F32(MS_MUL256_N_F32(src[1 + offset], 0.375), MS_MUL256_N_F32(src[5 + offset], 1.5));
    tmp2 = MS_SUB256_F32(MS_MUL256_N_F32(src[2 + offset], 0.25), MS_MUL256_N_F32(src[4 + offset], 1.25));
    t[40 + l] =
      MS_ADD256_F32(MS_SUB256_F32(MS_ADD256_F32(tmp1, tmp2), MS_MUL256_N_F32(src[3 + offset], 1.875)), src[6 + offset]);
    t[48 + l] =
      MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(tmp2, tmp1), MS_MUL256_N_F32(src[3 + offset], 1.875)), src[6 + offset]);
    t[56 + l] = MS_ADD256_F32(
      MS_SUB256_F32(MS_ADD256_F32(MS_MUL256_N_F32(src[1 + offset], -0.5625), MS_MUL256_N_F32(src[3 + offset], 3.0625)),
                    MS_MUL256_N_F32(src[5 + offset], 3.5)),
      src[7 + offset]);
  }
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    m[l] = MS_SUB256_F32(
      MS_ADD256_F32(MS_SUB256_F32(MS_MUL256_N_F32(t[offset], 0.5625), MS_MUL256_N_F32(t[2 + offset], 3.0625)),
                    MS_MUL256_N_F32(t[4 + offset], 3.5)),
      t[6 + offset]);
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(MS_MUL256_N_F32(t[1 + offset], 1.125), MS_MUL256_N_F32(t[5 + offset], 0.5));
    MS_FLOAT32X8 tmp2 = MS_SUB256_F32(MS_MUL256_N_F32(t[2 + offset], 2.25), MS_MUL256_N_F32(t[4 + offset], 3.25));
    m[8 + l] =
      MS_ADD256_F32(MS_SUB256_F32(MS_ADD256_F32(tmp1, tmp2), MS_MUL256_N_F32(t[3 + offset], 1.625)), t[6 + offset]);
    m[16 + l] =
      MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(tmp2, tmp1), MS_MUL256_N_F32(t[3 + offset], 1.625)), t[6 + offset]);
    tmp1 = MS_ADD256_F32(MS_MUL256_N_F32(t[1 + offset], 0.5625), t[5 + offset]);
    tmp2 = MS_SUB256_F32(MS_MUL256_N_F32(t[2 + offset], 0.5625), MS_MUL256_N_F32(t[4 + offset], 2.5));
    m[24 + l] =
      MS_ADD256_F32(MS_SUB256_F32(MS_ADD256_F32(tmp1, tmp2), MS_MUL256_N_F32(t[3 + offset], 2.5)), t[6 + offset]);
    m[32 + l] =
      MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(tmp2, tmp1), MS_MUL256_N_F32(t[3 + offset], 2.5)), t[6 + offset]);
    tmp1 = MS_ADD256_F32(MS_MUL256_N_F32(t[1 + offset], 0.375), MS_MUL256_N_F32(t[5 + offset], 1.5));
    tmp2 = MS_SUB256_F32(MS_MUL256_N_F32(t[2 + offset], 0.25), MS_MUL256_N_F32(t[4 + offset], 1.25));
    m[40 + l] =
      MS_ADD256_F32(MS_SUB256_F32(MS_ADD256_F32(tmp1, tmp2), MS_MUL256_N_F32(t[3 + offset], 1.875)), t[6 + offset]);
    m[48 + l] =
      MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(tmp2, tmp1), MS_MUL256_N_F32(t[3 + offset], 1.875)), t[6 + offset]);
    m[56 + l] = MS_ADD256_F32(
      MS_SUB256_F32(MS_ADD256_F32(MS_MUL256_N_F32(t[1 + offset], -0.5625), MS_MUL256_N_F32(t[3 + offset], 3.0625)),
                    MS_MUL256_N_F32(t[5 + offset], 3.5)),
      t[7 + offset]);
  }
  for (int i = 0; i < 64; i++) {
    MS_ST256_F32(dst_data + i * dst_step, m[i]);
  }
}

void InputTransform8x8AvxUnit(const float *src_data, float *dst_data, int src_step, int dst_step, int real_c) {
  if (real_c == C8NUM) {
    InputTransform8x8AvxUnit_block8(src_data, dst_data, src_step, dst_step);
  } else {
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
  }
}

void OutputTransform4x2AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[16];
  MS_FLOAT32X8 t[8];
  MS_FLOAT32X8 m[4];
  LoadAvx16Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = MS_ADD256_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    StoreAvx4Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform4x2ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[16];
  MS_FLOAT32X8 t[8];
  MS_FLOAT32X8 m[4];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx16Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = MS_ADD256_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 2] = MS_MAX256_F32(zero, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    StoreAvx4Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform4x2Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[16];
  MS_FLOAT32X8 t[8];
  MS_FLOAT32X8 m[4];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx16Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], src[1 + offset]), src[2 + offset]);
    t[l + 4] = MS_ADD256_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]), src[3 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 4;
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
    m[l + 2] = MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 2] = MS_MAX256_F32(zero, m[l + 2]);
    m[l + 2] = MS_MIN256_F32(six, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    StoreAvx4Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform4x3AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[16];
  MS_FLOAT32X8 t[12];
  MS_FLOAT32X8 m[9];
  LoadAvx16Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    MS_FLOAT32X8 tmp = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    t[l] = MS_ADD256_F32(src[offset], tmp);
    t[l + 4] = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    t[l + 8] = MS_ADD256_F32(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    MS_FLOAT32X8 tmp = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp), bias_ptr);
    m[l + 3] = MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = MS_ADD256_F32(MS_ADD256_F32(tmp, t[3 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    StoreAvx9Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform4x3ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[16];
  MS_FLOAT32X8 t[12];
  MS_FLOAT32X8 m[9];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx16Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    MS_FLOAT32X8 tmp = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    t[l] = MS_ADD256_F32(src[offset], tmp);
    t[l + 4] = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    t[l + 8] = MS_ADD256_F32(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    MS_FLOAT32X8 tmp = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp), bias_ptr);
    m[l + 3] = MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = MS_ADD256_F32(MS_ADD256_F32(tmp, t[3 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 3] = MS_MAX256_F32(zero, m[l + 3]);
    m[l + 6] = MS_MAX256_F32(zero, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    StoreAvx9Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform4x3Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[16];
  MS_FLOAT32X8 t[12];
  MS_FLOAT32X8 m[9];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx16Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    MS_FLOAT32X8 tmp = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    t[l] = MS_ADD256_F32(src[offset], tmp);
    t[l + 4] = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    t[l + 8] = MS_ADD256_F32(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    MS_FLOAT32X8 tmp = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp), bias_ptr);
    m[l + 3] = MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = MS_ADD256_F32(MS_ADD256_F32(tmp, t[3 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 3] = MS_MAX256_F32(zero, m[l + 3]);
    m[l + 3] = MS_MIN256_F32(six, m[l + 3]);
    m[l + 6] = MS_MAX256_F32(zero, m[l + 6]);
    m[l + 6] = MS_MIN256_F32(six, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    StoreAvx9Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x2AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[12];
  MS_FLOAT32X8 m[4];
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
      src[4 + offset]);
    t[l + 6] = MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]),
                                           MS_MUL256_N_F32(MS_SUB256_F32(src[3 + offset], src[4 + offset]), 2)),
                             src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]),
                    t[4 + offset]),
      bias_ptr);
    m[l + 2] =
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]),
                                                MS_MUL256_N_F32(MS_SUB256_F32(t[3 + offset], t[4 + offset]), 2)),
                                  t[5 + offset]),
                    bias_ptr);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    StoreAvx4Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x2ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[12];
  MS_FLOAT32X8 m[4];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
      src[4 + offset]);
    t[l + 6] = MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]),
                                           MS_MUL256_N_F32(MS_SUB256_F32(src[3 + offset], src[4 + offset]), 2)),
                             src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]),
                    t[4 + offset]),
      bias_ptr);
    m[l + 2] =
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]),
                                                MS_MUL256_N_F32(MS_SUB256_F32(t[3 + offset], t[4 + offset]), 2)),
                                  t[5 + offset]),
                    bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 2] = MS_MAX256_F32(zero, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    StoreAvx4Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x2Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[12];
  MS_FLOAT32X8 m[4];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
      src[4 + offset]);
    t[l + 6] = MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]),
                                           MS_MUL256_N_F32(MS_SUB256_F32(src[3 + offset], src[4 + offset]), 2)),
                             src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]),
                    t[4 + offset]),
      bias_ptr);
    m[l + 2] =
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]),
                                                MS_MUL256_N_F32(MS_SUB256_F32(t[3 + offset], t[4 + offset]), 2)),
                                  t[5 + offset]),
                    bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 2] = MS_MAX256_F32(zero, m[l + 2]);
    m[l + 2] = MS_MIN256_F32(six, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    StoreAvx4Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}
void OutputTransform6x3AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[18];
  MS_FLOAT32X8 m[9];
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADD256_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]),
                             MS_MUL256_N_F32(MS_SUB256_F32(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]),
                                           MS_MUL256_N_F32(MS_SUB256_F32(t[3 + offset], t[4 + offset]), 2)),
                             bias_ptr);
    m[l + 6] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    StoreAvx9Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x3ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[18];
  MS_FLOAT32X8 m[9];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADD256_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]),
                             MS_MUL256_N_F32(MS_SUB256_F32(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]),
                                           MS_MUL256_N_F32(MS_SUB256_F32(t[3 + offset], t[4 + offset]), 2)),
                             bias_ptr);
    m[l + 6] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 3] = MS_MAX256_F32(zero, m[l + 3]);
    m[l + 6] = MS_MAX256_F32(zero, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    StoreAvx9Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x3Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[18];
  MS_FLOAT32X8 m[9];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADD256_F32(MS_SUB256_F32(src[1 + offset], src[2 + offset]),
                             MS_MUL256_N_F32(MS_SUB256_F32(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = MS_ADD256_F32(MS_ADD256_F32(MS_SUB256_F32(t[1 + offset], t[2 + offset]),
                                           MS_MUL256_N_F32(MS_SUB256_F32(t[3 + offset], t[4 + offset]), 2)),
                             bias_ptr);
    m[l + 6] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 3] = MS_MAX256_F32(zero, m[l + 3]);
    m[l + 3] = MS_MIN256_F32(six, m[l + 3]);
    m[l + 6] = MS_MAX256_F32(zero, m[l + 6]);
    m[l + 6] = MS_MIN256_F32(six, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    StoreAvx9Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x4AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[24];
  MS_FLOAT32X8 m[16];
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2));
    t[l + 12] = MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4));
    t[l + 18] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2)), bias_ptr);
    m[l + 8] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), bias_ptr);
    m[l + 12] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    StoreAvx16Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x4ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[24];
  MS_FLOAT32X8 m[16];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2));
    t[l + 12] = MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4));
    t[l + 18] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2)), bias_ptr);
    m[l + 8] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), bias_ptr);
    m[l + 12] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 4] = MS_MAX256_F32(zero, m[l + 4]);
    m[l + 8] = MS_MAX256_F32(zero, m[l + 8]);
    m[l + 12] = MS_MAX256_F32(zero, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    StoreAvx16Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x4Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[24];
  MS_FLOAT32X8 m[16];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2));
    t[l + 12] = MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4));
    t[l + 18] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2)), bias_ptr);
    m[l + 8] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), bias_ptr);
    m[l + 12] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 4] = MS_MAX256_F32(zero, m[l + 4]);
    m[l + 4] = MS_MIN256_F32(six, m[l + 4]);
    m[l + 8] = MS_MAX256_F32(zero, m[l + 8]);
    m[l + 8] = MS_MIN256_F32(six, m[l + 8]);
    m[l + 12] = MS_MAX256_F32(zero, m[l + 12]);
    m[l + 12] = MS_MIN256_F32(six, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    StoreAvx16Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x5AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[30];
  MS_FLOAT32X8 m[25];
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2));
    t[l + 12] = MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4));
    t[l + 18] = MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2)), bias_ptr);
    m[l + 10] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), bias_ptr);
    m[l + 15] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8)), bias_ptr);
    m[l + 20] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 16)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    StoreAvx25Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x5ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[30];
  MS_FLOAT32X8 m[25];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2));
    t[l + 12] = MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4));
    t[l + 18] = MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2)), bias_ptr);
    m[l + 10] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), bias_ptr);
    m[l + 15] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8)), bias_ptr);
    m[l + 20] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 16)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 5] = MS_MAX256_F32(zero, m[l + 5]);
    m[l + 10] = MS_MAX256_F32(zero, m[l + 10]);
    m[l + 15] = MS_MAX256_F32(zero, m[l + 15]);
    m[l + 20] = MS_MAX256_F32(zero, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    StoreAvx25Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform6x5Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[36];
  MS_FLOAT32X8 t[30];
  MS_FLOAT32X8 m[25];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx36Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2);
    t[l + 6] = MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2));
    t[l + 12] = MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4));
    t[l + 18] = MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 2)), bias_ptr);
    m[l + 10] = MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 4)), bias_ptr);
    m[l + 15] = MS_ADD256_F32(MS_ADD256_F32(tmp3, MS_MUL256_N_F32(tmp4, 8)), bias_ptr);
    m[l + 20] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(tmp1, MS_MUL256_N_F32(tmp2, 16)), t[5 + offset]), bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 5] = MS_MAX256_F32(zero, m[l + 5]);
    m[l + 5] = MS_MIN256_F32(six, m[l + 5]);
    m[l + 10] = MS_MAX256_F32(zero, m[l + 10]);
    m[l + 10] = MS_MIN256_F32(six, m[l + 10]);
    m[l + 15] = MS_MAX256_F32(zero, m[l + 15]);
    m[l + 15] = MS_MIN256_F32(six, m[l + 15]);
    m[l + 20] = MS_MAX256_F32(zero, m[l + 20]);
    m[l + 20] = MS_MIN256_F32(six, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    StoreAvx25Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x2AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[16];
  MS_FLOAT32X8 m[4];
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                    t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    StoreAvx4Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x2ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[16];
  MS_FLOAT32X8 m[4];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 2] = MS_MAX256_F32(zero, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    StoreAvx4Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x2Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[16];
  MS_FLOAT32X8 m[4];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 2] = MS_MAX256_F32(zero, m[l + 2]);
    m[l + 2] = MS_MIN256_F32(six, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    StoreAvx4Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x3AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[24];
  MS_FLOAT32X8 m[9];
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 6] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)),
                    t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    StoreAvx9Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x3ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[24];
  MS_FLOAT32X8 m[9];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 6] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 3] = MS_MAX256_F32(zero, m[l + 3]);
    m[l + 6] = MS_MAX256_F32(zero, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    StoreAvx9Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x3Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[24];
  MS_FLOAT32X8 m[9];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 6] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 3] = MS_MAX256_F32(zero, m[l + 3]);
    m[l + 3] = MS_MIN256_F32(six, m[l + 3]);
    m[l + 6] = MS_MAX256_F32(zero, m[l + 6]);
    m[l + 6] = MS_MIN256_F32(six, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    StoreAvx9Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x4AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[32];
  MS_FLOAT32X8 m[16];
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 8] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 12] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)),
                    t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    StoreAvx16Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x4ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[32];
  MS_FLOAT32X8 m[16];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 8] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 12] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 4] = MS_MAX256_F32(zero, m[l + 4]);
    m[l + 8] = MS_MAX256_F32(zero, m[l + 8]);
    m[l + 12] = MS_MAX256_F32(zero, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    StoreAvx16Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x4Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[32];
  MS_FLOAT32X8 m[16];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 8] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 12] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 4] = MS_MAX256_F32(zero, m[l + 4]);
    m[l + 4] = MS_MIN256_F32(six, m[l + 4]);
    m[l + 8] = MS_MAX256_F32(zero, m[l + 8]);
    m[l + 8] = MS_MIN256_F32(six, m[l + 8]);
    m[l + 12] = MS_MAX256_F32(zero, m[l + 12]);
    m[l + 12] = MS_MIN256_F32(six, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    StoreAvx16Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x5AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[40];
  MS_FLOAT32X8 m[25];
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375));
    t[l + 32] =
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)),
                    src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 10] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 15] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 20] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)),
                    t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    StoreAvx25Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x5ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[40];
  MS_FLOAT32X8 m[25];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375));
    t[l + 32] =
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)),
                    src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 10] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 15] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 20] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 5] = MS_MAX256_F32(zero, m[l + 5]);
    m[l + 10] = MS_MAX256_F32(zero, m[l + 10]);
    m[l + 15] = MS_MAX256_F32(zero, m[l + 15]);
    m[l + 20] = MS_MAX256_F32(zero, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    StoreAvx25Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x5Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[40];
  MS_FLOAT32X8 m[25];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375));
    t[l + 32] =
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)),
                    src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 10] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 15] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 20] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 5] = MS_MAX256_F32(zero, m[l + 5]);
    m[l + 5] = MS_MIN256_F32(six, m[l + 5]);
    m[l + 10] = MS_MAX256_F32(zero, m[l + 10]);
    m[l + 10] = MS_MIN256_F32(six, m[l + 10]);
    m[l + 15] = MS_MAX256_F32(zero, m[l + 15]);
    m[l + 15] = MS_MIN256_F32(six, m[l + 15]);
    m[l + 20] = MS_MAX256_F32(zero, m[l + 20]);
    m[l + 20] = MS_MIN256_F32(six, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    StoreAvx25Data;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[48];
  MS_FLOAT32X8 m[36];
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625));
    t[l + 40] =
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375)),
                    src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 12] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 18] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 24] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 30] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375)),
                    t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 6 && r_w == 6) {
    for (int i = 0; i < 6; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 6;
      MS_ST256_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_ST256_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_ST256_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_ST256_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_ST256_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_ST256_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x6ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[48];
  MS_FLOAT32X8 m[36];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625));
    t[l + 40] =
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375)),
                    src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 12] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 18] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 24] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 30] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 6] = MS_MAX256_F32(zero, m[l + 6]);
    m[l + 12] = MS_MAX256_F32(zero, m[l + 12]);
    m[l + 18] = MS_MAX256_F32(zero, m[l + 18]);
    m[l + 24] = MS_MAX256_F32(zero, m[l + 24]);
    m[l + 30] = MS_MAX256_F32(zero, m[l + 30]);
  }
  if (r_c == C8NUM && r_h == 6 && r_w == 6) {
    for (int i = 0; i < 6; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 6;
      MS_ST256_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_ST256_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_ST256_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_ST256_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_ST256_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_ST256_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x6Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[48];
  MS_FLOAT32X8 m[36];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625));
    t[l + 40] =
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375)),
                    src[7 + offset]);
  }
  for (int l = 0; l < 6; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 6] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 12] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 18] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 24] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 30] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375)),
                    t[7 + offset]),
      bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 6] = MS_MAX256_F32(zero, m[l + 6]);
    m[l + 6] = MS_MIN256_F32(six, m[l + 6]);
    m[l + 12] = MS_MAX256_F32(zero, m[l + 12]);
    m[l + 12] = MS_MIN256_F32(six, m[l + 12]);
    m[l + 18] = MS_MAX256_F32(zero, m[l + 18]);
    m[l + 18] = MS_MIN256_F32(six, m[l + 18]);
    m[l + 24] = MS_MAX256_F32(zero, m[l + 24]);
    m[l + 24] = MS_MIN256_F32(six, m[l + 24]);
    m[l + 30] = MS_MAX256_F32(zero, m[l + 30]);
    m[l + 30] = MS_MIN256_F32(six, m[l + 30]);
  }
  if (r_c == C8NUM && r_h == 6 && r_w == 6) {
    for (int i = 0; i < 6; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 6;
      MS_ST256_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_ST256_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_ST256_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_ST256_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_ST256_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_ST256_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x7AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[56];
  MS_FLOAT32X8 m[49];
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625));
    t[l + 40] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375));
    t[l + 48] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.015625), tmp2), MS_MUL256_N_F32(tmp3, 11.390625)),
      src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 14] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 21] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 28] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 35] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.015625), tmp2),
                                                          MS_MUL256_N_F32(tmp3, 11.390625)),
                                            t[7 + offset]),
                              bias_ptr);
  }
  if (r_c == C8NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      MS_ST256_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_ST256_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_ST256_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_ST256_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_ST256_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_ST256_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      MS_ST256_F32(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 7;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x7ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[56];
  MS_FLOAT32X8 m[49];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625));
    t[l + 40] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375));
    t[l + 48] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.015625), tmp2), MS_MUL256_N_F32(tmp3, 11.390625)),
      src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 14] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 21] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 28] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 35] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.015625), tmp2),
                                                          MS_MUL256_N_F32(tmp3, 11.390625)),
                                            t[7 + offset]),
                              bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l + 7] = MS_MAX256_F32(zero, m[l + 7]);
    m[l + 14] = MS_MAX256_F32(zero, m[l + 14]);
    m[l + 21] = MS_MAX256_F32(zero, m[l + 21]);
    m[l + 28] = MS_MAX256_F32(zero, m[l + 28]);
    m[l + 35] = MS_MAX256_F32(zero, m[l + 35]);
    m[l + 42] = MS_MAX256_F32(zero, m[l + 42]);
  }
  if (r_c == C8NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      MS_ST256_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_ST256_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_ST256_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_ST256_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_ST256_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_ST256_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      MS_ST256_F32(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 7;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}

void OutputTransform8x7Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c) {
  MS_FLOAT32X8 src[64];
  MS_FLOAT32X8 t[56];
  MS_FLOAT32X8 m[49];
  MS_FLOAT32X8 zero = MS_MOV256_F32(0);
  MS_FLOAT32X8 six = MS_MOV256_F32(6);
  LoadAvx64Data;
  MS_FLOAT32X8 bias_ptr = MS_LD256_F32(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(src[5 + offset], src[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(src[1 + offset], src[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(src[3 + offset], src[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(src[5 + offset], src[6 + offset]);
    t[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5));
    t[l + 16] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25));
    t[l + 24] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375));
    t[l + 32] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625));
    t[l + 40] = MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375));
    t[l + 48] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.015625), tmp2), MS_MUL256_N_F32(tmp3, 11.390625)),
      src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    MS_FLOAT32X8 tmp1 = MS_ADD256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp2 = MS_ADD256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp3 = MS_ADD256_F32(t[5 + offset], t[6 + offset]);
    MS_FLOAT32X8 tmp4 = MS_SUB256_F32(t[1 + offset], t[2 + offset]);
    MS_FLOAT32X8 tmp5 = MS_SUB256_F32(t[3 + offset], t[4 + offset]);
    MS_FLOAT32X8 tmp6 = MS_SUB256_F32(t[5 + offset], t[6 + offset]);
    m[l] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.5), tmp5), MS_MUL256_N_F32(tmp6, 1.5)),
                             bias_ptr);
    m[l + 14] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.25), tmp2), MS_MUL256_N_F32(tmp3, 2.25)), bias_ptr);
    m[l + 21] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.125), tmp5), MS_MUL256_N_F32(tmp6, 3.375)), bias_ptr);
    m[l + 28] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.0625), tmp2), MS_MUL256_N_F32(tmp3, 5.0625)), bias_ptr);
    m[l + 35] = MS_ADD256_F32(
      MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp4, 0.03125), tmp5), MS_MUL256_N_F32(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_ADD256_F32(MS_MUL256_N_F32(tmp1, 0.015625), tmp2),
                                                          MS_MUL256_N_F32(tmp3, 11.390625)),
                                            t[7 + offset]),
                              bias_ptr);
    m[l] = MS_MAX256_F32(zero, m[l]);
    m[l] = MS_MIN256_F32(six, m[l]);
    m[l + 7] = MS_MAX256_F32(zero, m[l + 7]);
    m[l + 7] = MS_MIN256_F32(six, m[l + 7]);
    m[l + 14] = MS_MAX256_F32(zero, m[l + 14]);
    m[l + 14] = MS_MIN256_F32(six, m[l + 14]);
    m[l + 21] = MS_MAX256_F32(zero, m[l + 21]);
    m[l + 21] = MS_MIN256_F32(six, m[l + 21]);
    m[l + 28] = MS_MAX256_F32(zero, m[l + 28]);
    m[l + 28] = MS_MIN256_F32(six, m[l + 28]);
    m[l + 35] = MS_MAX256_F32(zero, m[l + 35]);
    m[l + 35] = MS_MIN256_F32(six, m[l + 35]);
    m[l + 42] = MS_MAX256_F32(zero, m[l + 42]);
    m[l + 42] = MS_MIN256_F32(six, m[l + 42]);
  }
  if (r_c == C8NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      MS_ST256_F32(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      MS_ST256_F32(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      MS_ST256_F32(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      MS_ST256_F32(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      MS_ST256_F32(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      MS_ST256_F32(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      MS_ST256_F32(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 7;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = MS_F32X8_GETI(m[k + m_k_offset], i);
        }
      }
    }
  }
}
#endif
