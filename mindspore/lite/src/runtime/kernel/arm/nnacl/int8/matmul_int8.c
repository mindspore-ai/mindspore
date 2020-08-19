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

#include "nnacl/int8/matmul_int8.h"
#include <limits.h>
#include "nnacl/quantization/fixed_point.h"
void RowMajor2Row8MajorInt8(int8_t *src_ptr, int8_t *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    int8_t *src = src_ptr + r * col;
    for (int c = 0; c < col; c++) {
      int cd8 = c / 8;
      int cm8 = c % 8;
      dst_ptr[cd8 * 8 * row + r * 8 + cm8] = src[c];
    }
  }
}

void RowMajor2Row16x4MajorInt8(void *src_ptr, void *dst_ptr, int row, int col) {
  /* Row-major to row16x4-major (block row-major) */
  int col16 = UP_ROUND(col, C16NUM);
  for (int r = 0; r < row; r++) {
    int r4div = r / C4NUM;
    int r4mod = r % C4NUM;
    for (int c = 0; c < col; c++) {
      int c16div = c / C16NUM;
      int c16mod = c % C16NUM;
      int src_index = r * col + c;
      int dst_index = r4div * C4NUM * col16 + c16div * C16NUM * C4NUM + r4mod * C16NUM + c16mod;
      ((int8_t *)dst_ptr)[dst_index] = ((int8_t *)src_ptr)[src_index];
    }
  }
  return;
}

void RowMajor2Col8MajorInt8(int8_t *src_ptr, int8_t *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    int rd8 = r / 8;
    int rm8 = r % 8;
    for (int c = 0; c < col; c++) {
      dst_ptr[rd8 * col * 8 + c * 8 + rm8] = src_ptr[r * col + c];
    }
  }
}

void MatMulInt8(const int8_t *a, const int8_t *b, int32_t *c, const int row8, const int col8, const int deep,
                const int32_t a_zp, const int32_t b_zp) {
  /*  col8-major * row8-major => row8x8-major  */
  for (int row = 0; row < row8; row++) {
    for (int col = 0; col < col8; col++) {
      int r8div = row / 8, r8mod = row % 8;
      int c8div = col / 8, c8mod = col % 8;
      size_t ci = c8div * row8 * 8 + row * 8 + c8mod;
      int32_t value = 0;
      for (int d = 0; d < deep; d++) {
        size_t ai = r8div * deep * 8 + d * 8 + r8mod;
        size_t bi = c8div * deep * 8 + d * 8 + c8mod;
        value = value + ((int32_t)a[ai] - a_zp) * ((int32_t)b[bi] - b_zp);
      }
      c[ci] = value;
    }
  }
}

void MatMulOptR4Int8(int32_t *dst, const int8_t *a, const int8_t *b, const int32_t *bias, const int32_t *input_sum,
                     size_t row_4, size_t col_4, size_t deep_16) {
  /*  row4x16-major * row16x4-major => row4x4-major  */
  for (int r = 0; r < row_4; r++) {
    for (int c = 0; c < col_4; c++) {
      int r4div = r / C4NUM, r4mod = r % C4NUM;
      int c4div = c / C4NUM, c4mod = c % C4NUM;
      size_t ci = c4div * row_4 * C4NUM + r * C4NUM + c4mod;
      int32_t value = 0;
      for (int d = 0; d < deep_16; d++) {
        int d16div = d / C16NUM, d16mod = d % C16NUM;
        size_t ai = r4div * deep_16 * C4NUM + d16div * C4NUM * C16NUM + r4mod * C16NUM + d16mod;
        size_t bi = c4div * deep_16 * C4NUM + d16div * C4NUM * C16NUM + c4mod * C16NUM + d16mod;
        value = value + a[ai] * b[bi];
      }
      value -= input_sum[r];
      value += bias[c];
      ((int32_t *)dst)[ci] = value;
    }
  }
  return;
}
