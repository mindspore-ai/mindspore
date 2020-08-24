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
      dst_ptr[cd16 * col16 * C4NUM + rd4 * C4NUM * C16NUM + rm4 * C16NUM + cm16] = src_ptr[r * col16 + c];
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
    }
    src_r += C4NUM * col;
    dst_r += C4NUM * col16;
  }

  if (row != row_4div) {
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

void RowMajor2Asums(int8_t *a, int row, int col, int b_zp, int *dst) {
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      int src_idx = r * col + c;
      dst[r] += a[src_idx];
    }
    dst[r] *= b_zp;
  }
}

void RowMajor2Bbias(int8_t *b, int row, int col, int a_zp, int b_zp, int *bias, int *dst) {
  for (int c = 0; c < col; ++c) {
    for (int r = 0; r < row; ++r) {
      int src_idx = r * col + c;
      dst[c] += b[src_idx];
    }
    dst[c] = row * a_zp * b_zp - a_zp * dst[c];
    if (bias) {
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
