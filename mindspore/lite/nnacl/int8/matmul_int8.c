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

void RowMajor2Row4x16MajorInt8(int8_t *src_ptr, int8_t *dst_ptr, int row, int col) {
  int col16 = UP_ROUND(col, C16NUM);
  for (int r = 0; r < row; r++) {
    int rd4 = r / C4NUM;
    int rm4 = r % C4NUM;
    for (int c = 0; c < col; c++) {
      int cd16 = c / C16NUM;
      int cm16 = c % C16NUM;
      int dst_index = rd4 * col16 * C4NUM + cd16 * C4NUM * C16NUM + rm4 * C16NUM + cm16;
      int src_index = r * col + c;
      dst_ptr[dst_index] = src_ptr[src_index];
    }
  }
}

void RowMajor2Row8x4MajorInt8(const int8_t *src_ptr, int8_t *dst_ptr, int row, int col) {
  int col4 = UP_ROUND(col, C4NUM);
  for (int r = 0; r < row; r++) {
    int rd8 = r / C8NUM;
    int rm8 = r % C8NUM;
    for (int c = 0; c < col; c++) {
      int cd4 = c / C4NUM;
      int cm4 = c % C4NUM;
      int dst_index = rd8 * col4 * C8NUM + cd4 * C8NUM * C4NUM + rm8 * C4NUM + cm4;
      int src_index = r * col + c;
      dst_ptr[dst_index] = src_ptr[src_index];
    }
  }
}

void MatrixPack4x16UnitInt8(int8_t *src, int8_t *dst, int row, int col, int stride) {
  for (int r = 0; r < row; r++) {
    int8_t *src_r = src + r * stride;
    int8_t *dst_r = dst + r * C16NUM;
    memcpy(dst_r, src_r, col * sizeof(int8_t));
  }
  return;
}

void MatrixEmptyInt8(int8_t *dst, int row, int col) {
  for (int r = 0; r < row; r++) {
    int8_t *dst_r = dst + r * C16NUM;
    memset(dst_r, 0, col * sizeof(int8_t));
  }
  return;
}

void RowMajor2Row4x8MajorInt8(const int8_t *src_ptr, int8_t *dst_ptr, int row, int col) {
  /* Row-major to row16x4-major (block row-major) */
  int col4 = UP_ROUND(col, C4NUM);
  for (int r = 0; r < row; r++) {
    int rd8 = r / C8NUM, rm8 = r % C8NUM;
    for (int c = 0; c < col; c++) {
      int cd4 = c / C4NUM, cm4 = c % C4NUM;
      int src_index = r * col + c;
      int dst_index = rd8 * col4 * C8NUM + cd4 * C4NUM * C8NUM + rm8 * C4NUM + cm4;
      dst_ptr[dst_index] = src_ptr[src_index];
    }
  }
  return;
}

void RowMajor2Row16x4MajorInt8(void *src_ptr, void *dst_ptr, int row, int col) {
  /* Row-major to row16x4-major (block row-major) */
  int col16 = UP_ROUND(col, C16NUM);
  size_t row_4div = row / C4NUM * C4NUM;
  size_t row_4res = row - row_4div;
  size_t col_16div = col / C16NUM * C16NUM;
  size_t col_16res = col - col_16div;
  int8_t *src_r = (int8_t *)src_ptr;
  int8_t *dst_r = (int8_t *)dst_ptr;

  for (int ri = 0; ri < row_4div; ri += C4NUM) {
    for (int ci = 0; ci < col_16div; ci += C16NUM) {
#ifdef ENABLE_ARM64
      size_t col_offset = col;
      int8_t *src_c = src_r + ci;
      int8_t *dst_c = dst_r + ci * C4NUM;
      asm volatile(
        "mov x10, %[src_c] \n"
        "mov x11, %[dst_c] \n"

        "ld1 {v0.16b}, [x10], %[col_offset]\n"
        "ld1 {v1.16b}, [x10], %[col_offset]\n"
        "ld1 {v2.16b}, [x10], %[col_offset]\n"
        "ld1 {v3.16b}, [x10], %[col_offset]\n"

        "st1 {v0.16b}, [x11], #16\n"
        "st1 {v1.16b}, [x11], #16\n"
        "st1 {v2.16b}, [x11], #16\n"
        "st1 {v3.16b}, [x11], #16\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ col_offset ] "r"(col_offset)
        : "x10", "x11", "v0", "v1", "v2", "v3");
#else
      MatrixPack4x16UnitInt8(src_r + ci, dst_r + ci * C4NUM, C4NUM, C16NUM, col);
#endif
    }

    if (col != col_16div) {
      MatrixPack4x16UnitInt8(src_r + col_16div, dst_r + col_16div * C4NUM, C4NUM, col_16res, col);
      MatrixEmptyInt8(dst_r + col_16div * C4NUM + col_16res, C4NUM, C16NUM - col_16res);
    }
    src_r += C4NUM * col;
    dst_r += C4NUM * col16;
  }

  if (row != row_4div) {
    memset(dst_r, 0, C4NUM * col16);

    for (int ci = 0; ci < col_16div; ci += C16NUM) {
      MatrixPack4x16UnitInt8(src_r + ci, dst_r + ci * C4NUM, row_4res, C16NUM, col);
    }

    if (col != col_16div) {
      MatrixPack4x16UnitInt8(src_r + col_16div, dst_r + col_16div * C4NUM, row_4res, col_16res, col);
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

void MatMulInt8_16x4(const int8_t *a, const int8_t *b, int *dst, int row_4, int col_4, int deep_16,
                     const int *input_sum, const int *bias) {
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

void MatMulInt8_16x4_r(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_16,
                       size_t stride, const int32_t *input_sum, const int32_t *bias, int32_t *left_shift,
                       int32_t *right_shift, int32_t *multiplier, int32_t output_zp, int32_t mini, int32_t maxi,
                       bool per_channel) {
  /*  row4x16-major * row16x4-major => (int8)row-major  : per-channel */
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int r4div = r / C4NUM, r4mod = r % C4NUM;
      int c4div = c / C4NUM, c4mod = c % C4NUM;
      size_t ci = r * stride + c;
      int32_t value = 0;
      for (int d = 0; d < deep_16; d++) {
        int d16div = d / C16NUM, d16mod = d % C16NUM;
        size_t ai = r4div * deep_16 * C4NUM + d16div * C4NUM * C16NUM + r4mod * C16NUM + d16mod;
        size_t bi = c4div * deep_16 * C4NUM + d16div * C4NUM * C16NUM + c4mod * C16NUM + d16mod;
        value = value + a[ai] * b[bi];
      }
      int32_t cur_input_sum = per_channel ? input_sum[c4div * UP_ROUND(row, C4NUM) + r * C4NUM + c4mod] : input_sum[r];
      value -= cur_input_sum;
      value += bias[c];
      int32_t cur_left_shift = per_channel ? left_shift[c] : left_shift[0];
      int32_t cur_right_shift = per_channel ? right_shift[c] : right_shift[0];
      int32_t cur_multiplier = per_channel ? multiplier[c] : multiplier[0];
      value = MultiplyByQuantizedMultiplier(value, cur_multiplier, cur_left_shift, cur_right_shift) + output_zp;
      value = MSMIN(maxi, value);
      value = MSMAX(mini, value);
      dst[ci] = (int8_t)value;
    }
  }
  return;
}

void MatMulInt8_8x8_r(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_4,
                      size_t stride, const int32_t *input_sum, const int32_t *bias, int32_t *left_shift,
                      int32_t *right_shift, int32_t *multiplier, int32_t output_zp, int32_t mini, int32_t maxi,
                      bool per_channel) {
  /*  row8x4-major * row4x8-major => (int8)row-major  */
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int r8div = r / C8NUM, r8mod = r % C8NUM;
      int c8div = c / C8NUM, c8mod = c % C8NUM;
      size_t ci = r * stride + c;
      int32_t value = 0;
      for (int d = 0; d < deep_4; d++) {
        int d4div = d / C4NUM, d4mod = d % C4NUM;
        size_t ai = r8div * deep_4 * C8NUM + d4div * C8NUM * C4NUM + r8mod * C4NUM + d4mod;
        size_t bi = c8div * deep_4 * C8NUM + d4div * C8NUM * C4NUM + c8mod * C4NUM + d4mod;
        value = value + a[ai] * b[bi];
      }
      int32_t cur_input_sum = per_channel ? input_sum[c8div * UP_ROUND(row, C8NUM) + r * C8NUM + c8mod] : input_sum[r];
      value -= cur_input_sum;
      value += bias[c];
      int32_t cur_left_shift = per_channel ? left_shift[c] : left_shift[0];
      int32_t cur_right_shift = per_channel ? right_shift[c] : right_shift[0];
      int32_t cur_multiplier = per_channel ? multiplier[c] : multiplier[0];
      value = MultiplyByQuantizedMultiplier(value, cur_multiplier, cur_left_shift, cur_right_shift) + output_zp;
      value = MSMIN(maxi, value);
      value = MSMAX(mini, value);
      dst[ci] = (int8_t)value;
    }
  }
  return;
}

/*  row4x16-major * col16x4-major => row4x4-major  */
void MatmulInt8(const int8_t *a, const int8_t *b, int8_t *dst, const int *a_sums, const int *bias, int act_min,
                int act_max, int out_zp, int multiplier, int left_shift, int right_shift, int row, int col, int deep16,
                int stride) {
  int8_t *output = dst;
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int r4div = r / C4NUM;
      int r4mod = r % C4NUM;
      int c4div = c / C4NUM;
      int c4mod = c % C4NUM;
      int value = 0;
      for (int d = 0; d < deep16; d++) {
        int d16div = d / C16NUM;
        int d16mod = d % C16NUM;
        size_t ai = r4div * deep16 * C4NUM + d16div * C4NUM * C16NUM + r4mod * C16NUM + d16mod;
        size_t bi = c4div * deep16 * C4NUM + d16div * C4NUM * C16NUM + c4mod * C16NUM + d16mod;
        value += a[ai] * b[bi];
      }
      value -= a_sums[r];
      value += bias[c];
      value = MultiplyByQuantizedMultiplier(value, multiplier, left_shift, right_shift) + out_zp;
      value = MSMIN(INT8_MAX, value);
      value = MSMAX(INT8_MIN, value);
      output[c] = (int8_t)value;
    }
    output += stride;
  }
}

void RowMajor2Row4x16Major(int8_t *src, int row, int col, int8_t *dst, int col_16) {
  int stride = sizeof(int8_t) * 16 * 4;
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      int stride_n = r / 4 * (col_16 / 16) + c / 16;
      int src_idx = r * col + c;
      dst[stride * stride_n + r % 4 * 16 + c % 16] = src[src_idx];
    }
  }
}

void RowMajor2Col16x4Major(int8_t *src, int row, int col, int8_t *dst, int row_16) {
  int stride = sizeof(int8_t) * 16 * 4;
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      int stride_n = c / 4 * (row_16 / 16) + r / 16;
      int src_idx = r * col + c;
      dst[stride * stride_n + c % 4 * 16 + r % 16] = src[src_idx];
    }
  }
}

// dst: weight_zp * input_row_sums
void CalcInputSums(int8_t *input, int row, int col, int weight_zp, int *dst, DataOrder order) {
  for (int r = 0; r < row; ++r) {
    int sum = 0;
    for (int c = 0; c < col; ++c) {
      if (order == RowMajor) {
        sum += input[r * col + c];
      } else {
        sum += input[c * row + r];
      }
    }
    sum *= weight_zp;
    dst[r] = sum;
  }
}

// dst: bias + depth*input_zp*weight_zp - input_zp*weight_col_sums
void CalcWeightBiasSums(int8_t *weight, int row, int col, int input_zp, int weight_zp, int *bias, int *dst,
                        DataOrder order) {
  for (int c = 0; c < col; ++c) {
    int sum = 0;
    for (int r = 0; r < row; ++r) {
      if (order == RowMajor) {
        sum += weight[r * col + c];
      } else {
        sum += weight[c * row + r];
      }
    }
    dst[c] = row * input_zp * weight_zp - input_zp * sum;
    if (bias != NULL) {
      dst[c] += bias[c];
    }
  }
}

void Row4x4Major2RowMajor(int8_t *src, int row4, int8_t *dst, int row, int cow) {
  int stride = sizeof(int8_t) * 4 * 4;
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < cow; ++c) {
      int sride_n = c / 4 * (row4 / 4) + r / 4;
      int dst_idx = r * cow + c;
      dst[dst_idx] = src[stride * sride_n + r % 4 * 4 + c % 4];
    }
  }
}
