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
