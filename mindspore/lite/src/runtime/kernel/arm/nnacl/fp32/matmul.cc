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

#include "src/runtime/kernel/arm/nnacl/fp32/matmul.h"

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

void RowMajor2Col8Major(float *src_ptr, float *dst_ptr, size_t row, size_t col) {
  size_t row8 = row / C8NUM * C8NUM;
  size_t col4 = col / C4NUM * C4NUM;
  float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  size_t ri = 0;
  for (; ri < row8; ri += C8NUM) {
    size_t ci = 0;
    for (; ci < col4; ci += C4NUM) {
      float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C8NUM;

      /* 8x4 row-major to col-major */
#ifdef ENABLE_ARM64
      size_t stride = col * 4;
      asm volatile(
        "mov x10, %[src_c]\n"
        "mov x11, %[dst_c]\n"

        "ld1 {v0.4s}, [x10], %[stride]\n"
        "ld1 {v1.4s}, [x10], %[stride]\n"
        "ld1 {v2.4s}, [x10], %[stride]\n"
        "ld1 {v3.4s}, [x10], %[stride]\n"

        "zip1 v4.4s, v0.4s, v1.4s\n"
        "zip2 v5.4s, v0.4s, v1.4s\n"
        "zip1 v6.4s, v2.4s, v3.4s\n"
        "zip2 v7.4s, v2.4s, v3.4s\n"

        "ld1 {v8.4s},  [x10], %[stride]\n"
        "ld1 {v9.4s},  [x10], %[stride]\n"
        "ld1 {v10.4s}, [x10],  %[stride]\n"
        "ld1 {v11.4s}, [x10],  %[stride]\n"

        "trn1 v0.2d, v4.2d, v6.2d\n"
        "trn2 v1.2d, v4.2d, v6.2d\n"
        "trn1 v2.2d, v5.2d, v7.2d\n"
        "trn2 v3.2d, v5.2d, v7.2d\n"

        "zip1 v12.4s, v8.4s, v9.4s\n"
        "zip2 v13.4s, v8.4s, v9.4s\n"
        "zip1 v14.4s, v10.4s, v11.4s\n"
        "zip2 v15.4s, v10.4s, v11.4s\n"

        "trn1 v8.2d, v12.2d, v14.2d\n"
        "trn2 v9.2d, v12.2d, v14.2d\n"
        "trn1 v10.2d, v13.2d, v15.2d\n"
        "trn2 v11.2d, v13.2d, v15.2d\n"

        "st1 {v0.4s}, [x11],  #16\n"
        "st1 {v8.4s}, [x11],  #16\n"
        "st1 {v1.4s}, [x11],  #16\n"
        "st1 {v9.4s}, [x11],  #16\n"
        "st1 {v2.4s},  [x11],#16\n"
        "st1 {v10.4s}, [x11], #16\n"
        "st1 {v3.4s},  [x11],#16\n"
        "st1 {v11.4s}, [x11], #16\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
        : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15");
#else
      for (int tr = 0; tr < 8; tr++) {
        for (int tc = 0; tc < 4; tc++) {
          dst_c[tc * 8 + tr] = src_c[tr * col + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C8NUM;
      for (size_t i = 0; i < C8NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C8NUM * col;
    dst_r += C8NUM * col;
  }
  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C8NUM] = src_r[i];
    }
    src_r += col;
    dst_r += 1;
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
  MatmulFloatNeon64(a, b, c, bias, (int)act_type, deep, row_8_, col_8_);
#else
  MatMul8x8(a, b, c, bias, act_type, deep, row_8_, col_8_);
#endif
}
