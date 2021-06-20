/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp16_grad/gemm_fp16.h"
#include <string.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "nnacl/fp16/matmul_fp16.h"
#include "nnacl/fp16/pack_fp16.h"

#ifdef ENABLE_ARM64
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
#endif

void AddMatrixFp16(const float16_t *restrict v1, float16_t *restrict v2, float16_t beta, int row, int col, int stride) {
  const float16_t *src_ptr = v1;
  float16_t *dst_ptr = v2;
#ifdef ENABLE_NEON
  float16x8_t beta_0 = vdupq_n_f16(beta);
#endif
  for (int r = 0; r < row; r++) {
    int c = 0;
#ifdef ENABLE_NEON
    for (; c <= (col - C8NUM); c += C8NUM) {
      float16x8_t dst_0 = vld1q_f16(dst_ptr + c);
      float16x8_t src_0 = vld1q_f16(src_ptr + c);
      float16x8_t sum_0 = vfmaq_f16(dst_0, beta_0, src_0);
      vst1q_f16(dst_ptr + c, sum_0);
    }
#endif
    for (; c < col; c++) {
      dst_ptr[c] += beta * src_ptr[c];
    }
    src_ptr += stride;
    dst_ptr += stride;
  }
}

int MatSizeFp16(int row, int col, int round) {
  int res = UP_ROUND(row, round) * col;
  return res;
}

int MatSizeTotalFp16(int row, int col, int deep, int stride) {
#ifdef ENABLE_ARM64
  const int num = C16NUM;
#else
  const int num = C12NUM;
#endif
  int res = MatSizeFp16(row, deep, num) + MatSizeFp16(col, deep, C8NUM);
  if (stride > 0) res += row * stride;
  return res;
}

#ifdef ENABLE_ARM64
static void RowMajor2Col16MajorStrideFp16(const float16_t *src, float16_t *dst, int row, int col, int stride) {
  size_t row_up_16 = UP_ROUND(row, C16NUM);
  size_t row16 = row / C16NUM * C16NUM;
  size_t col8 = col / C8NUM * C8NUM;
  const float16_t *src_r = src;
  float16_t *dst_r = dst;
  size_t ri = 0;
  // find 16 block unit
  for (; ri < row16; ri += C16NUM) {
    size_t ci = 0;
    for (; ci < col8; ci += C8NUM) {
      const float16_t *src_c = src_r + ci;
      float16_t *dst_c = dst_r + ci * C16NUM;
#ifdef ENABLE_ARM64
      Row2Col16Block16(src_c, dst_c, stride);
#else
      for (int tr = 0; tr < C16NUM; tr++) {
        for (int tc = 0; tc < C8NUM; tc++) {
          dst_c[tc * C16NUM + tr] = src_c[tr * stride + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float16_t *src_c = src_r + ci;
      float16_t *dst_c = dst_r + ci * C16NUM;
      for (size_t i = 0; i < C16NUM; i++) {
        dst_c[i] = src_c[i * stride];
      }
    }
    src_r += C16NUM * stride;
    dst_r += C16NUM * col;
  }
  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; ++i) {
      dst_r[i * C16NUM] = src_r[i];
    }
    src_r += stride;
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
#endif

void RowMajor2Row16MajorStrideFp16(const float16_t *src, float16_t *dst, int row, int col, int stride) {
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int c_div16 = c / 16;
      int c_mod16 = c % 16;
      dst[c_div16 * 16 * row + r * 16 + c_mod16] = src[r * stride + c];
    }
  }
}

void RowMajor2Col12MajorStrideFp16(const float16_t *src, float16_t *dst, size_t row, size_t col, int stride) {
  size_t row_up_12 = UP_ROUND(row, C12NUM);
  size_t row12 = row / C12NUM * C12NUM;
  size_t col8 = col / C8NUM * C8NUM;
  const float16_t *src_r = src;
  float16_t *dst_r = dst;
  size_t ri = 0;
  // transpose 12x8
  for (; ri < row12; ri += C12NUM) {
    size_t ci = 0;
    for (; ci < col8; ci += C8NUM) {
      const float16_t *src_c = src_r + ci;
      float16_t *dst_c = dst_r + ci * C12NUM;
#ifdef ENABLE_ARM82_A32
      Transpose12x8A32Fp16(src_c, dst_c, stride * sizeof(float16_t), 24);
#else
      for (int tr = 0; tr < C12NUM; tr++) {
        for (int tc = 0; tc < C8NUM; tc++) {
          dst_c[tc * C12NUM + tr] = src_c[tr * stride + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float16_t *src_c = src_r + ci;
      float16_t *dst_c = dst_r + ci * C12NUM;
      for (size_t i = 0; i < C12NUM; i++) {
        dst_c[i] = src_c[i * stride];
      }
    }
    src_r += C12NUM * stride;
    dst_r += C12NUM * col;
  }
  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; ++i) {
      dst_r[i * C12NUM] = src_r[i];
    }
    src_r += stride;
    dst_r += 1;
  }
  for (; ri < row_up_12; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C12NUM] = 0;
    }
    dst_r += 1;
  }
}

void RowMajor2Row12MajorStrideFp16(const float16_t *src, float16_t *dst, int row, int col, int stride) {
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int c_div12 = c / 12;
      int c_mod12 = c % 12;
      dst[c_div12 * 12 * row + r * 12 + c_mod12] = src[r * stride + c];
    }
  }
}

static void RowMajor2Col8MajorStrideFp16(const float16_t *src, float16_t *dst, int row, int col, int stride) {
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int r_div8 = r / 8;
      int r_mod8 = r % 8;
      dst[r_div8 * 8 * col + c * 8 + r_mod8] = src[r * stride + c];
    }
  }
}

static void RowMajor2Row8MajorStrideFp16(const float16_t *src, float16_t *dst, int row, int col, int stride) {
  for (int r = 0; r < row; r++) {
    const float16_t *src_ptr = src + r * stride;
    int c = 0;
    for (; c < col; c++) {
      int cd8 = c / C8NUM;
      int cm8 = c % C8NUM;
      dst[cd8 * C8NUM * row + r * C8NUM + cm8] = src_ptr[c];
    }
    for (; c < UP_ROUND(col, C8NUM); c++) {
      int cd8 = c / C8NUM;
      int cm8 = c % C8NUM;
      dst[cd8 * C8NUM * row + r * C8NUM + cm8] = 0;
    }
  }
  return;
}

static void RowMajor2ColXMajorStrideFp16(const float16_t *src, float16_t *dst, int row, int col, int stride) {
#ifdef ENABLE_ARM64
  RowMajor2Col16MajorStrideFp16(src, dst, row, col, stride);
#else
  RowMajor2Col12MajorStrideFp16(src, dst, row, col, stride);
#endif
}

static void RowMajor2RowXMajorStrideFp16(const float16_t *src, float16_t *dst, int row, int col, int stride) {
#ifdef ENABLE_ARM64
  RowMajor2Row16MajorStrideFp16(src, dst, row, col, stride);
#else
  RowMajor2Row12MajorStrideFp16(src, dst, row, col, stride);
#endif
}

void GemmMatmulFp16(int ta, int tb, int M, int N, int K, float16_t alpha, const float16_t *mat_a, int lda,
                    const float16_t *mat_b, int ldb, float16_t beta, float16_t *mat_c, int ldc, float16_t *workspace) {
  GemmCbFp16 gcb;
  gcb.atype = ActType_No;
  gcb.ca = 0;
  gcb.cb = 0;
  gcb.bias = NULL;
  GemmMatmulPlusFp16(ta, tb, M, N, K, alpha, mat_a, lda, mat_b, ldb, beta, mat_c, ldc, workspace, &gcb);
}

void GemmMatmulPlusFp16(int ta, int tb, int M, int N, int K, float16_t alpha, const float16_t *mat_a, int lda,
                        const float16_t *mat_b, int ldb, float16_t beta, float16_t *mat_c, int ldc,
                        float16_t *workspace, GemmCbFp16 *gcb) {
#ifdef ENABLE_ARM64
  const int num = C16NUM;
#else
  const int num = C12NUM;
#endif
  float16_t *output = mat_c;
  float16_t *fworkspace = workspace;
  int incremental = (beta < 0.f) || (beta > 0.f);
  float16_t *mat_a_input = (float16_t *)mat_a;
  float16_t *mat_b_input = (float16_t *)mat_b;

  if (!gcb->ca) {
    mat_a_input = fworkspace;
    fworkspace += MatSizeFp16(M, K, num);
    if (ta) {
      RowMajor2RowXMajorStrideFp16(mat_a, mat_a_input, K, M, lda);
    } else {
      RowMajor2ColXMajorStrideFp16(mat_a, mat_a_input, M, K, lda);
    }
  }
  if (!gcb->cb) {
    mat_b_input = fworkspace;
    fworkspace += MatSizeFp16(N, K, C8NUM);
    if (tb) {
      RowMajor2Col8MajorStrideFp16(mat_b, mat_b_input, N, K, ldb);
    } else {
      RowMajor2Row8MajorStrideFp16(mat_b, mat_b_input, K, N, ldb);
    }
  }
  if (incremental) output = fworkspace;
  MatMulFp16(mat_a_input, mat_b_input, output, gcb->bias, gcb->atype, K, M, N, ldc, OutType_Nhwc);
  if (incremental) AddMatrixFp16(output, mat_c, beta, M, N, ldc);
  gcb->mat_a = mat_a_input;
  gcb->mat_b = mat_b_input;
}
