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

#include "src/runtime/kernel/arm/opclib/int8/matmul.h"
#include <limits.h>
#include "src/runtime/kernel/arm/opclib/quantization/fixed_point.h"

void RowMajor2Col8MajorInt8(int8_t *src_ptr, int8_t *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    int rd8 = r / 8;
    int rm8 = r % 8;
    for (int c = 0; c < col; c++) {
      dst_ptr[rd8 * col * 8 + c * 8 + rm8] = src_ptr[r * col + c];
    }
  }
  return;
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
  return;
}

// todo: need to delete, replace by above functions. z00445833
void GemmRowCol8x8Major2RowMajorInt8(int8_t *src_ptr, int8_t *dst_ptr, int row, int col) {
  int col8 = UP_ROUND(col, 8);
  for (int r = 0; r < row; r++) {
    int rd8 = r / 8;
    int rm8 = r % 8;
    for (int c = 0; c < col; c++) {
      dst_ptr[r * col + c] = src_ptr[rd8 * col8 * 8 + c * 8 + rm8];
    }
  }
}

void Gemm8x8Int8(const int8_t *lhs_data, const int8_t *rhs_data, const int8_t *bias_data, int8_t *output_data,
                 int depth, FcQuantArg *params) {
  int lhs_offset = params->input.zp_;
  int rhs_offset = params->weight.zp_;
  int output_offset = params->output.zp_;
  int output_multiplier = params->quant_multiplier;
  int output_shift = params->output_shift;

  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 8; ++col) {
      int c_index = col * 8 + row;
      int acc = 0;
      for (int d = 0; d < depth; ++d) {
        int a_index = d * 8 + row;
        int b_index = d * 8 + col;
        acc += (lhs_data[a_index] - lhs_offset) * (rhs_data[b_index] - rhs_offset);
      }
      acc += bias_data[col];
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift, output_shift) + output_offset;
      acc = MSMAX(CHAR_MIN, MSMIN(CHAR_MAX, acc));
      output_data[c_index] = (int8_t)acc;
    }
  }
}

void GemmInt8(const int8_t *input_data, const int8_t *weights_data, const int8_t *bias_data, int8_t *output_data,
              int row_8, int col_8, int depth, FcQuantArg *params) {
  for (int r = 0; r < row_8; r += 8) {
    int8_t *output = output_data + r * col_8;
    const int8_t *input = input_data + r * depth;
    for (int c = 0; c < col_8; c += 8) {
      const int8_t *bias = bias_data + c;
      const int8_t *weights = weights_data + c * depth;
      int8_t *dst = output + c * 8;
      Gemm8x8Int8(input, weights, bias, dst, depth, params);
    }
  }
}
