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

#include "nnacl/fp16/matmul_fp16.h"

static void Col2Row8SrcFromFp16(const void *src_ptr, float16_t *dst_ptr, size_t row, size_t col) {
  int row_c8 = row / C8NUM * C8NUM;
  int col_c8 = col / C8NUM * C8NUM;
  const float16_t *src = (const float16_t *)src_ptr;
  int ci = 0;
  for (; ci < col_c8; ci += C8NUM) {
    int ri = 0;
    for (; ri < row_c8; ri += C8NUM) {
      const float16_t *src_ptr1 = src + ci * row + ri;
      float16_t *dst_ptr1 = dst_ptr + ci * row + ri * C8NUM;
#ifdef ENABLE_ARM64
      size_t strid_row = row * 2;
      asm volatile(
        "mov x10, %[src_ptr1]\n"
        "mov x11, %[dst_ptr1]\n"
        "mov x12, %[strid_row]\n"
        "ld1 {v0.8h}, [x10], x12\n"
        "ld1 {v1.8h}, [x10], x12\n"
        "ld1 {v2.8h}, [x10], x12\n"
        "ld1 {v3.8h}, [x10], x12\n"
        "ld1 {v4.8h}, [x10], x12\n"
        "ld1 {v5.8h}, [x10], x12\n"
        "ld1 {v6.8h}, [x10], x12\n"
        "ld1 {v7.8h}, [x10], x12\n"

        "zip1 v8.8h, v0.8h, v1.8h\n"
        "zip1 v9.8h, v2.8h, v3.8h\n"
        "zip1 v10.8h, v4.8h, v5.8h\n"
        "zip1 v11.8h, v6.8h, v7.8h\n"

        "trn1 v12.4s, v8.4s, v9.4s\n"
        "trn1 v14.4s, v10.4s, v11.4s\n"
        "trn2 v13.4s, v8.4s, v9.4s\n"
        "trn2 v15.4s, v10.4s, v11.4s\n"

        "trn1 v16.2d, v12.2d, v14.2d\n"
        "trn2 v18.2d, v12.2d, v14.2d\n"
        "trn1 v17.2d, v13.2d, v15.2d\n"
        "trn2 v19.2d, v13.2d, v15.2d\n"

        "zip2 v8.8h, v0.8h, v1.8h\n"
        "zip2 v9.8h, v2.8h, v3.8h\n"
        "zip2 v10.8h, v4.8h, v5.8h\n"
        "zip2 v11.8h, v6.8h, v7.8h\n"

        "trn1 v12.4s, v8.4s, v9.4s\n"
        "trn1 v14.4s, v10.4s, v11.4s\n"
        "trn2 v13.4s, v8.4s, v9.4s\n"
        "trn2 v15.4s, v10.4s, v11.4s\n"

        "trn1 v20.2d, v12.2d, v14.2d\n"
        "trn2 v22.2d, v12.2d, v14.2d\n"
        "trn1 v21.2d, v13.2d, v15.2d\n"
        "trn2 v23.2d, v13.2d, v15.2d\n"

        "st1 {v16.8h}, [x11], #16\n"
        "st1 {v17.8h}, [x11], #16\n"
        "st1 {v18.8h}, [x11], #16\n"
        "st1 {v19.8h}, [x11], #16\n"
        "st1 {v20.8h}, [x11], #16\n"
        "st1 {v21.8h}, [x11], #16\n"
        "st1 {v22.8h}, [x11], #16\n"
        "st1 {v23.8h}, [x11], #16\n"
        :
        : [ dst_ptr1 ] "r"(dst_ptr1), [ src_ptr1 ] "r"(src_ptr1), [ strid_row ] "r"(strid_row)
        : "x10", "x11", "x12", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
      for (int tr = 0; tr < C8NUM; ++tr) {
        for (int tc = 0; tc < C8NUM; ++tc) {
          dst_ptr1[tr * C8NUM + tc] = src_ptr1[tc * row + tr];
        }
      }
#endif
    }
    for (; ri < row; ++ri) {
      const float16_t *src_ptr1 = src + ci * row;
      float16_t *dst_ptr1 = dst_ptr + ci * row;
      for (int tc = 0; tc < C8NUM; ++tc) {
        dst_ptr1[ri * C8NUM + tc] = src_ptr1[tc * row + ri];
      }
    }
  }
  for (int r = 0; r < row; r++) {
    for (int tc = ci; tc < col; tc++) {
      int cd8 = tc / C8NUM;
      int cm8 = tc % C8NUM;
      dst_ptr[cd8 * C8NUM * row + r * C8NUM + cm8] = src[tc * row + r];
    }
  }
}

static void Col2Row8SrcFromFp32(const void *src_ptr, float16_t *dst_ptr, size_t row, size_t col) {
  int row_c8 = row / C8NUM * C8NUM;
  int col_c8 = col / C8NUM * C8NUM;
  int ci = 0;
  const float *src = (const float *)src_ptr;
  for (; ci < col_c8; ci += C8NUM) {
    int ri = 0;
    for (; ri < row_c8; ri += C8NUM) {
      const float *src_ptr1 = src + ci * row + ri;
      float16_t *dst_ptr1 = dst_ptr + ci * row + ri * C8NUM;
#ifdef ENABLE_ARM64
      size_t strid_row = row * 4;
      asm volatile(
        "mov x10, %[src_ptr1]\n"
        "mov x11, %[dst_ptr1]\n"
        "mov x12, %[strid_row]\n"
        "ld1 {v8.4s, v9.4s}, [x10], x12\n"
        "ld1 {v10.4s, v11.4s}, [x10], x12\n"
        "ld1 {v12.4s, v13.4s}, [x10], x12\n"
        "ld1 {v14.4s, v15.4s}, [x10], x12\n"
        "ld1 {v16.4s, v17.4s}, [x10], x12\n"
        "ld1 {v18.4s, v19.4s}, [x10], x12\n"
        "ld1 {v20.4s, v21.4s}, [x10], x12\n"
        "ld1 {v22.4s, v23.4s}, [x10], x12\n"

        "fcvtn v0.4h, v8.4s\n"
        "fcvtn2 v0.8h, v9.4s\n"
        "fcvtn v1.4h, v10.4s\n"
        "fcvtn2 v1.8h, v11.4s\n"
        "fcvtn v2.4h, v12.4s\n"
        "fcvtn2 v2.8h, v13.4s\n"
        "fcvtn v3.4h, v14.4s\n"
        "fcvtn2 v3.8h, v15.4s\n"
        "fcvtn v4.4h, v16.4s\n"
        "fcvtn2 v4.8h, v17.4s\n"
        "fcvtn v5.4h, v18.4s\n"
        "fcvtn2 v5.8h, v19.4s\n"
        "fcvtn v6.4h, v20.4s\n"
        "fcvtn2 v6.8h, v21.4s\n"
        "fcvtn v7.4h, v22.4s\n"
        "fcvtn2 v7.8h, v23.4s\n"

        "zip1 v8.8h, v0.8h, v1.8h\n"
        "zip1 v9.8h, v2.8h, v3.8h\n"
        "zip1 v10.8h, v4.8h, v5.8h\n"
        "zip1 v11.8h, v6.8h, v7.8h\n"

        "trn1 v12.4s, v8.4s, v9.4s\n"
        "trn1 v14.4s, v10.4s, v11.4s\n"
        "trn2 v13.4s, v8.4s, v9.4s\n"
        "trn2 v15.4s, v10.4s, v11.4s\n"

        "trn1 v16.2d, v12.2d, v14.2d\n"
        "trn2 v18.2d, v12.2d, v14.2d\n"
        "trn1 v17.2d, v13.2d, v15.2d\n"
        "trn2 v19.2d, v13.2d, v15.2d\n"

        "zip2 v8.8h, v0.8h, v1.8h\n"
        "zip2 v9.8h, v2.8h, v3.8h\n"
        "zip2 v10.8h, v4.8h, v5.8h\n"
        "zip2 v11.8h, v6.8h, v7.8h\n"

        "trn1 v12.4s, v8.4s, v9.4s\n"
        "trn1 v14.4s, v10.4s, v11.4s\n"
        "trn2 v13.4s, v8.4s, v9.4s\n"
        "trn2 v15.4s, v10.4s, v11.4s\n"

        "trn1 v20.2d, v12.2d, v14.2d\n"
        "trn2 v22.2d, v12.2d, v14.2d\n"
        "trn1 v21.2d, v13.2d, v15.2d\n"
        "trn2 v23.2d, v13.2d, v15.2d\n"

        "st1 {v16.8h}, [x11], #16\n"
        "st1 {v17.8h}, [x11], #16\n"
        "st1 {v18.8h}, [x11], #16\n"
        "st1 {v19.8h}, [x11], #16\n"
        "st1 {v20.8h}, [x11], #16\n"
        "st1 {v21.8h}, [x11], #16\n"
        "st1 {v22.8h}, [x11], #16\n"
        "st1 {v23.8h}, [x11], #16\n"
        :
        : [ dst_ptr1 ] "r"(dst_ptr1), [ src_ptr1 ] "r"(src_ptr1), [ strid_row ] "r"(strid_row)
        : "x10", "x11", "x12", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
      for (int tr = 0; tr < C8NUM; ++tr) {
        for (int tc = 0; tc < C8NUM; ++tc) {
          dst_ptr1[tr * C8NUM + tc] = (float16_t)(src_ptr1[tc * row + tr]);
        }
      }
#endif
    }
    for (; ri < row; ++ri) {
      const float *src_ptr1 = src + ci * row;
      float16_t *dst_ptr1 = dst_ptr + ci * row;
      for (int tc = 0; tc < C8NUM; ++tc) {
        dst_ptr1[ri * C8NUM + tc] = (float16_t)(src_ptr1[tc * row + ri]);
      }
    }
  }
  for (int r = 0; r < row; r++) {
    for (int tc = ci; tc < col; tc++) {
      int cd8 = tc / C8NUM;
      int cm8 = tc % C8NUM;
      dst_ptr[cd8 * C8NUM * row + r * C8NUM + cm8] = (float16_t)(src[tc * row + r]);
    }
  }
}

void ColMajor2Row8MajorFp16(const void *src_ptr, float16_t *dst_ptr, size_t row, size_t col, bool src_float16) {
  if (src_float16) {
    Col2Row8SrcFromFp16(src_ptr, dst_ptr, row, col);
  } else {
    Col2Row8SrcFromFp32(src_ptr, dst_ptr, row, col);
  }
  return;
}

void MatMul16x8(const float16_t *a, const float16_t *b, float16_t *dst, const float16_t *bias, ActType act_type,
                int deep, int row, int col, int stride, bool write_nhwc) {
  if (write_nhwc) {
    /*  col16-major * row8-major => col-major  */
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        int r16div = r / C16NUM, r16mod = r % C16NUM;
        int c8div = c / C8NUM, c8mod = c % C8NUM;
        size_t ci = r * stride + c;
        float value = 0;
        for (int d = 0; d < deep; d++) {
          size_t ai = r16div * deep * C16NUM + d * C16NUM + r16mod;
          size_t bi = c8div * deep * C8NUM + d * C8NUM + c8mod;
          value = value + a[ai] * b[bi];
        }
        if (bias != NULL) value += bias[c];
        if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
        if (act_type != ActType_No) value = MSMAX(0.0f, value);
        dst[ci] = value;
      }
    }
  } else {
    for (int i = 0; i < row; ++i) {
      int src_r_offset = i;
      int dst_r_offset = i * col * stride;
      for (int j = 0; j < col; ++j) {
        int c8div = j / 8, c8mod = j % 8;
        size_t ci = dst_r_offset + c8div * 8 * stride + c8mod;
        float value = 0;
        for (int d = 0; d < deep; ++d) {
          size_t ai = src_r_offset + d * C16NUM;
          size_t bi = c8div * deep * 8 + d * 8 + c8mod;
          value = value + a[ai] * b[bi];
        }
        if (bias != NULL) value += bias[j];
        if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
        if (act_type != ActType_No) value = MSMAX(0.0f, value);
        dst[ci] = value;
      }
    }
  }
  return;
}

void MatMulFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, ActType act_type,
                int depth, int row, int col, int stride, int out_type) {
  if (out_type == OutType_C8) {
    MatmulFp16Neon64(a, b, c, bias, (int)act_type, depth, row, col, stride, false);
  } else {
    MatmulFp16Neon64Opt(a, b, c, bias, (int)act_type, depth, row, col, stride, out_type);
  }
  return;
}

void MatVecMulFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, ActType act_type,
                   int depth, int col) {
  MatVecMulFp16Neon64(a, b, c, bias, (int)act_type, depth, col);
}

static void Row2Col16Block16(const float16_t *src_ptr, float16_t *dst_ptr, size_t col) {
  size_t stride = col * 2;
  asm volatile(
    "mov x10, %[src_c]\n"
    "mov x11, %[dst_c]\n"

    "ld1 {v0.8h}, [x10], %[stride]\n"
    "ld1 {v1.8h}, [x10], %[stride]\n"
    "ld1 {v2.8h}, [x10], %[stride]\n"
    "ld1 {v3.8h}, [x10], %[stride]\n"
    "ld1 {v4.8h}, [x10], %[stride]\n"
    "ld1 {v5.8h}, [x10], %[stride]\n"
    "ld1 {v6.8h}, [x10], %[stride]\n"
    "ld1 {v7.8h}, [x10], %[stride]\n"

    "zip1 v16.8h, v0.8h, v1.8h\n"
    "zip1 v17.8h, v2.8h, v3.8h\n"
    "zip1 v18.8h, v4.8h, v5.8h\n"
    "zip1 v19.8h, v6.8h, v7.8h\n"

    "ld1 {v8.8h}, [x10], %[stride]\n"
    "ld1 {v9.8h}, [x10], %[stride]\n"
    "ld1 {v10.8h}, [x10], %[stride]\n"
    "ld1 {v11.8h}, [x10], %[stride]\n"
    "ld1 {v12.8h}, [x10], %[stride]\n"
    "ld1 {v13.8h}, [x10], %[stride]\n"
    "ld1 {v14.8h}, [x10], %[stride]\n"
    "ld1 {v15.8h}, [x10], %[stride]\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v24.2d, v20.2d, v22.2d\n"
    "trn2 v25.2d, v20.2d, v22.2d\n"
    "trn1 v26.2d, v21.2d, v23.2d\n"
    "trn2 v27.2d, v21.2d, v23.2d\n"

    "zip1 v16.8h, v8.8h, v9.8h\n"
    "zip1 v17.8h, v10.8h, v11.8h\n"
    "zip1 v18.8h, v12.8h, v13.8h\n"
    "zip1 v19.8h, v14.8h, v15.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v28.2d, v20.2d, v22.2d\n"
    "trn2 v29.2d, v20.2d, v22.2d\n"
    "trn1 v30.2d, v21.2d, v23.2d\n"
    "trn2 v31.2d, v21.2d, v23.2d\n"

    "st1 {v24.8h}, [x11], #16\n"
    "st1 {v28.8h}, [x11], #16\n"
    "st1 {v26.8h}, [x11], #16\n"
    "st1 {v30.8h}, [x11], #16\n"
    "st1 {v25.8h}, [x11], #16\n"
    "st1 {v29.8h}, [x11], #16\n"
    "st1 {v27.8h}, [x11], #16\n"
    "st1 {v31.8h}, [x11], #16\n"

    "zip2 v16.8h, v0.8h, v1.8h\n"
    "zip2 v17.8h, v2.8h, v3.8h\n"
    "zip2 v18.8h, v4.8h, v5.8h\n"
    "zip2 v19.8h, v6.8h, v7.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v24.2d, v20.2d, v22.2d\n"
    "trn2 v25.2d, v20.2d, v22.2d\n"
    "trn1 v26.2d, v21.2d, v23.2d\n"
    "trn2 v27.2d, v21.2d, v23.2d\n"

    "zip2 v16.8h, v8.8h, v9.8h\n"
    "zip2 v17.8h, v10.8h, v11.8h\n"
    "zip2 v18.8h, v12.8h, v13.8h\n"
    "zip2 v19.8h, v14.8h, v15.8h\n"

    "trn1 v20.4s, v16.4s, v17.4s\n"
    "trn2 v21.4s, v16.4s, v17.4s\n"
    "trn1 v22.4s, v18.4s, v19.4s\n"
    "trn2 v23.4s, v18.4s, v19.4s\n"

    "trn1 v28.2d, v20.2d, v22.2d\n"
    "trn2 v29.2d, v20.2d, v22.2d\n"
    "trn1 v30.2d, v21.2d, v23.2d\n"
    "trn2 v31.2d, v21.2d, v23.2d\n"

    "st1 {v24.8h}, [x11], #16\n"
    "st1 {v28.8h}, [x11], #16\n"
    "st1 {v26.8h}, [x11], #16\n"
    "st1 {v30.8h}, [x11], #16\n"
    "st1 {v25.8h}, [x11], #16\n"
    "st1 {v29.8h}, [x11], #16\n"
    "st1 {v27.8h}, [x11], #16\n"
    "st1 {v31.8h}, [x11], #16\n"
    :
    : [ dst_c ] "r"(dst_ptr), [ src_c ] "r"(src_ptr), [ stride ] "r"(stride)
    : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
      "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
}

void RowMajor2Col16MajorFp16Opt(const float16_t *src_ptr, float16_t *dst_ptr, size_t row, size_t col) {
  size_t row_up_16 = UP_ROUND(row, C16NUM);
  size_t row16 = row / C16NUM * C16NUM;
  size_t col8 = col / C8NUM * C8NUM;
  const float16_t *src_r = src_ptr;
  float16_t *dst_r = dst_ptr;
  size_t ri = 0;
  // find 16 block unit
  for (; ri < row16; ri += C16NUM) {
    size_t ci = 0;
    for (; ci < col8; ci += C8NUM) {
      const float16_t *src_c = src_r + ci;
      float16_t *dst_c = dst_r + ci * C16NUM;
#ifdef ENABLE_ARM64
      Row2Col16Block16(src_c, dst_c, col);
#else
      for (int tr = 0; tr < C16NUM; tr++) {
        for (int tc = 0; tc < C8NUM; tc++) {
          dst_c[tc * C16NUM + tr] = src_c[tr * col + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float16_t *src_c = src_r + ci;
      float16_t *dst_c = dst_r + ci * C16NUM;
      for (size_t i = 0; i < C16NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C16NUM * col;
    dst_r += C16NUM * col;
  }
  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; ++i) {
      dst_r[i * C16NUM] = src_r[i];
    }
    src_r += col;
    dst_r += 1;
  }
  for (; ri < row_up_16; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C16NUM] = 0;
    }
    dst_r += 1;
  }
  return;
}

void RowMajor2Col16MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src) {
  if (is_fp32_src) {
    const float *fp32_src = (const float *)src;
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        int r_div16 = r / 16;
        int r_mod16 = r % 16;
        dst[r_div16 * 16 * col + c * 16 + r_mod16] = (float16_t)(fp32_src[r * col + c]);
      }
    }
  } else {
    const float16_t *fp16_src = (const float16_t *)src;
    RowMajor2Col16MajorFp16Opt(fp16_src, dst, row, col);
  }
  return;
}

void RowMajor2Row16MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src) {
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int c_div16 = c / 16;
      int c_mod16 = c % 16;
      if (is_fp32_src) {
        dst[c_div16 * 16 * row + r * 16 + c_mod16] = (float16_t)(((const float *)src)[r * col + c]);
      } else {
        dst[c_div16 * 16 * row + r * 16 + c_mod16] = ((const float16_t *)src)[r * col + c];
      }
    }
  }
}

void RowMajor2Row8MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src) {
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int c_div8 = c / 8;
      int c_mod8 = c % 8;
      if (is_fp32_src) {
        dst[c_div8 * 8 * row + r * 8 + c_mod8] = (float16_t)(((const float *)src)[r * col + c]);
      } else {
        dst[c_div8 * 8 * row + r * 8 + c_mod8] = ((const float16_t *)src)[r * col + c];
      }
    }
  }
}

void RowMajor2ColMajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src) {
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      if (is_fp32_src) {
        dst[c * row + r] = (float16_t)(((const float *)src)[r * col + c]);
      } else {
        dst[c * row + r] = ((const float16_t *)src)[r * col + c];
      }
    }
  }
}

void RowMajor2Col8MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src) {
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int r_div8 = r / 8;
      int r_mod8 = r % 8;
      if (is_fp32_src) {
        dst[r_div8 * 8 * col + c * 8 + r_mod8] = (float16_t)(((const float *)src)[r * col + c]);
      } else {
        dst[r_div8 * 8 * col + c * 8 + r_mod8] = ((const float16_t *)src)[r * col + c];
      }
    }
  }
}
