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

#include "src/runtime/kernel/arm/opclib/fp32/matmul.h"

void RowMajor2Row8Major(float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    float *src = src_ptr + r * col;
    for (int c = 0; c < col; c++) {
      int cd8 = c / 8;
      int cm8 = c % 8;
      dst_ptr[cd8 * 8 * row + r * 8 + cm8] = src[c];
    }
  }
  return;
}

void RowMajor2Col8Major(float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    int rd8 = r / 8;
    int rm8 = r % 8;
    for (int c = 0; c < col; c++) {
      dst_ptr[rd8 * col * 8 + c * 8 + rm8] = src_ptr[r * col + c];
    }
  }
  return;
}

void Row8x8Major2RowMajor(float *src_ptr, float *dst_ptr, int row, int col) {
  int row8 = UP_ROUND(row, 8);
  for (int c = 0; c < col; c++) {
    int cd8 = c / 8;
    int cm8 = c % 8;
    for (int r = 0; r < row; r++) {
      dst_ptr[r * col + c] = src_ptr[cd8 * row8 * 8 + r * 8 + cm8];
    }
  }
  return;
}

void MatMul8x8(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row_8_,
               int col_8_) {
  /*  col8-major * row8-major => col8x8-major  */
  for (int row = 0; row < row_8_; row++) {
    for (int col = 0; col < col_8_; col++) {
      int r8div = row / 8, r8mod = row % 8;
      int c8div = col / 8, c8mod = col % 8;
      size_t ci = c8div * row_8_ * 8 + row * 8 + c8mod;
      float value = 0;
      for (int d = 0; d < deep; d++) {
        size_t ai = r8div * deep * 8 + d * 8 + r8mod;
        size_t bi = c8div * deep * 8 + d * 8 + c8mod;
        value = value + a[ai] * b[bi];
      }
      if (bias != nullptr) value += bias[col];
      if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
      if (act_type != ActType_No) value = MSMAX(0.0f, value);
      c[ci] = value;
    }
  }
  return;
}

void MatMul(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row_8_,
            int col_8_) {
#ifdef __aarch64__
  float minf = (act_type == ActType_No) ? FLT_MIN : 0.f;
  float maxf = (act_type == ActType_Relu6) ? 6.0f : FLT_MAX;
  MatMulFloatNeon64(a, b, c, bias, maxf, minf, deep, row_8_, col_8_);
#else
  MatMul8x8(a, b, c, bias, act_type, deep, row_8_, col_8_);
#endif
  return;
}
