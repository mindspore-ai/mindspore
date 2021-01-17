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

#include "nnacl/fp32_grad/gemm.h"
#include <string.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "nnacl/fp32/matmul_fp32.h"

void AddMatrix(const float *restrict v1, float *restrict v2, float beta, int row, int col, int stride) {
  const float *src_ptr = v1;
  float *dst_ptr = v2;
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      dst_ptr[c] += beta * src_ptr[c];
    }
    src_ptr += stride;
    dst_ptr += stride;
  }
}

int MatSize(int row, int col, int round) {
  int res = UP_ROUND(row, round) * col;
  return res;
}

int MatSizeTotal(int row, int col, int deep, int stride) {
#ifdef ENABLE_ARM32
  const int num = C4NUM;
#else
  const int num = C12NUM;
#endif
  int res = MatSize(row, deep, num) + MatSize(col, deep, C8NUM);
  if (stride > 0) res += row * stride;
  return res;
}
#ifdef ENABLE_ARM32
static void RowMajor2Row4MajorStride(const float *src_ptr, float *dst_ptr, int row, int col, int lead) {
  for (int r = 0; r < row; r++) {
    const float *src = src_ptr + r * lead;
    for (int c = 0; c < col; c++) {
      int cd8 = c / 4;
      int cm8 = c % 4;
      dst_ptr[cd8 * 4 * row + r * 4 + cm8] = src[c];
    }
  }
}
#endif

static void RowMajor2Row8MajorStride(const float *src_ptr, float *dst_ptr, int row, int col, int lead) {
  for (int r = 0; r < row; r++) {
    const float *src = src_ptr + r * lead;
    for (int c = 0; c < col; c++) {
      int cd8 = c / 8;
      int cm8 = c % 8;
      dst_ptr[cd8 * 8 * row + r * 8 + cm8] = src[c];
    }
  }
  return;
}

#ifndef ENABLE_ARM32
static void RowMajor2Row12MajorStride(const float *src_ptr, float *dst_ptr, int row, int col, int lead) {
  for (int r = 0; r < row; r++) {
    const float *src = src_ptr + r * lead;
    for (int c = 0; c < col; c++) {
      int cd8 = c / C12NUM;
      int cm8 = c % C12NUM;
      dst_ptr[cd8 * C12NUM * row + r * C12NUM + cm8] = src[c];
    }
  }
  return;
}

static void RowMajor2Col12MajorStride(const float *restrict src_ptr, float *restrict dst_ptr, size_t row, size_t col,
                                      int lead) {
  size_t row_up_12 = UP_ROUND(row, C12NUM);
  size_t row12 = row / C12NUM * C12NUM;
  size_t col4 = col / C4NUM * C4NUM;
  const float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  size_t ri = 0;
  for (; ri < row12; ri += C12NUM) {
    size_t ci = 0;
    for (; ci < col4; ci += C4NUM) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C12NUM;

      /* 12x4 row-major to col-major */
#ifdef ENABLE_ARM64
      size_t stride = lead * sizeof(float);
      asm volatile(
        "mov x10, %[src_c]\n"
        "mov x11, %[dst_c]\n"

        "ld1 {v0.4s}, [x10], %[stride]\n"
        "ld1 {v1.4s}, [x10], %[stride]\n"
        "ld1 {v2.4s}, [x10], %[stride]\n"
        "ld1 {v3.4s}, [x10], %[stride]\n"

        "ld1 {v4.4s}, [x10], %[stride]\n"
        "ld1 {v5.4s}, [x10], %[stride]\n"
        "ld1 {v6.4s}, [x10], %[stride]\n"
        "ld1 {v7.4s}, [x10], %[stride]\n"

        "zip1 v12.4s, v0.4s, v1.4s\n"
        "zip2 v13.4s, v0.4s, v1.4s\n"
        "zip1 v14.4s, v2.4s, v3.4s\n"
        "zip2 v15.4s, v2.4s, v3.4s\n"

        "ld1 {v8.4s}, [x10], %[stride]\n"
        "ld1 {v9.4s}, [x10], %[stride]\n"
        "ld1 {v10.4s}, [x10], %[stride]\n"
        "ld1 {v11.4s}, [x10], %[stride]\n"

        "zip1 v16.4s, v4.4s, v5.4s\n"
        "zip2 v17.4s, v4.4s, v5.4s\n"
        "zip1 v18.4s, v6.4s, v7.4s\n"
        "zip2 v19.4s, v6.4s, v7.4s\n"

        "trn1 v20.2d, v12.2d, v14.2d\n"
        "trn2 v23.2d, v12.2d, v14.2d\n"
        "trn1 v26.2d, v13.2d, v15.2d\n"
        "trn2 v29.2d, v13.2d, v15.2d\n"

        "trn1 v21.2d, v16.2d, v18.2d\n"
        "trn2 v24.2d, v16.2d, v18.2d\n"
        "trn1 v27.2d, v17.2d, v19.2d\n"
        "trn2 v30.2d, v17.2d, v19.2d\n"

        "zip1 v12.4s, v8.4s, v9.4s\n"
        "zip2 v13.4s, v8.4s, v9.4s\n"
        "zip1 v14.4s, v10.4s, v11.4s\n"
        "zip2 v15.4s, v10.4s, v11.4s\n"

        "trn1 v22.2d, v12.2d, v14.2d\n"
        "trn2 v25.2d, v12.2d, v14.2d\n"
        "trn1 v28.2d, v13.2d, v15.2d\n"
        "trn2 v31.2d, v13.2d, v15.2d\n"

        "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x11], #64\n"
        "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x11], #64\n"
        "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x11], #64\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
        : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31");
#elif ENABLE_ARM32
      size_t stride = lead * sizeof(float);
      asm volatile(
        "mov r10, %[src_c]\n"
        "mov r12, %[dst_c]\n"

        "vld1.32 {q0}, [r10], %[stride]\n"
        "vld1.32 {q3}, [r10], %[stride]\n"
        "vld1.32 {q10}, [r10], %[stride]\n"
        "vld1.32 {q13}, [r10], %[stride]\n"

        "vtrn.32 d0, d6\n"
        "vtrn.32 d1, d7\n"
        "vtrn.32 d20, d26\n"
        "vtrn.32 d21, d27\n"

        "vld1.32 {q1}, [r10], %[stride]\n"
        "vld1.32 {q8}, [r10], %[stride]\n"
        "vld1.32 {q11}, [r10], %[stride]\n"
        "vld1.32 {q14}, [r10], %[stride]\n"

        "vswp d1, d20\n"
        "vswp d7, d26\n"

        "vld1.32 {q2}, [r10], %[stride]\n"
        "vld1.32 {q9}, [r10], %[stride]\n"
        "vld1.32 {q12}, [r10], %[stride]\n"
        "vld1.32 {q15}, [r10], %[stride]\n"

        "vtrn.32 d2, d16\n"
        "vtrn.32 d3, d17\n"
        "vtrn.32 d22, d28\n"
        "vtrn.32 d23, d29\n"

        "vswp d3, d22\n"
        "vswp d17, d28\n"

        "vtrn.32 d4, d18\n"
        "vtrn.32 d5, d19\n"
        "vtrn.32 d24, d30\n"
        "vtrn.32 d25, d31\n"

        "vswp d5, d24\n"
        "vswp d19, d30\n"

        "vst1.32 {q0, q1}, [r12]!\n"
        "vst1.32 {q2, q3}, [r12]!\n"
        "vst1.32 {q8, q9}, [r12]!\n"
        "vst1.32 {q10, q11}, [r12]!\n"
        "vst1.32 {q12, q13}, [r12]!\n"
        "vst1.32 {q14, q15}, [r12]!\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
        : "r10", "r12", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#else
      for (int tr = 0; tr < C12NUM; tr++) {
        for (int tc = 0; tc < C4NUM; tc++) {
          dst_c[tc * C12NUM + tr] = src_c[tr * lead + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C12NUM;
      for (size_t i = 0; i < C12NUM; i++) {
        dst_c[i] = src_c[i * lead];
      }
    }
    src_r += C12NUM * lead;
    dst_r += C12NUM * col;
  }

  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C12NUM] = src_r[i];
    }
    src_r += lead;
    dst_r += 1;
  }

  for (; ri < row_up_12; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C12NUM] = 0;
    }
    dst_r += 1;
  }
  return;
}
#endif

static void RowMajor2Col8MajorStride(const float *src_ptr, float *dst_ptr, size_t row, size_t col, int lead) {
  size_t row8 = row / C8NUM * C8NUM;
#ifdef ENABLE_ARM64
  size_t col_skip = col / C8NUM * C8NUM;
  int skip_size = C8NUM;
#else
  size_t col_skip = col / C4NUM * C4NUM;
  int skip_size = C4NUM;
#endif
  const float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  size_t ri = 0;
  for (; ri < row8; ri += C8NUM) {
    size_t ci = 0;
    for (; ci < col_skip; ci += skip_size) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C8NUM;

#ifdef ENABLE_ARM64
      /* 8x8 row-major to col-major */
      size_t stride = lead * sizeof(float);
      asm volatile(
        "mov x10, %[src_c]\n"
        "mov x11, %[dst_c]\n"

        "ld1 {v0.4s, v1.4s}, [x10], %[stride]\n"
        "ld1 {v2.4s, v3.4s}, [x10], %[stride]\n"
        "ld1 {v4.4s, v5.4s}, [x10], %[stride]\n"
        "ld1 {v6.4s, v7.4s}, [x10], %[stride]\n"

        "zip1 v8.4s, v0.4s, v2.4s\n"
        "zip2 v9.4s, v0.4s, v2.4s\n"
        "zip1 v10.4s, v4.4s, v6.4s\n"
        "zip2 v11.4s, v4.4s, v6.4s\n"

        "ld1 {v16.4s, v17.4s}, [x10], %[stride]\n"
        "ld1 {v18.4s, v19.4s}, [x10], %[stride]\n"
        "ld1 {v20.4s, v21.4s}, [x10], %[stride]\n"
        "ld1 {v22.4s, v23.4s}, [x10], %[stride]\n"

        "zip1 v12.4s, v1.4s, v3.4s\n"
        "zip2 v13.4s, v1.4s, v3.4s\n"
        "zip1 v14.4s, v5.4s, v7.4s\n"
        "zip2 v15.4s, v5.4s, v7.4s\n"

        "trn1 v0.2d, v8.2d, v10.2d\n"
        "trn2 v1.2d, v8.2d, v10.2d\n"
        "trn1 v2.2d, v9.2d, v11.2d\n"
        "trn2 v3.2d, v9.2d, v11.2d\n"

        "zip1 v24.4s, v16.4s, v18.4s\n"
        "zip2 v25.4s, v16.4s, v18.4s\n"
        "zip1 v26.4s, v20.4s, v22.4s\n"
        "zip2 v27.4s, v20.4s, v22.4s\n"

        "trn1 v4.2d, v12.2d, v14.2d\n"
        "trn2 v5.2d, v12.2d, v14.2d\n"
        "trn1 v6.2d, v13.2d, v15.2d\n"
        "trn2 v7.2d, v13.2d, v15.2d\n"

        "zip1 v28.4s, v17.4s, v19.4s\n"
        "zip2 v29.4s, v17.4s, v19.4s\n"
        "zip1 v30.4s, v21.4s, v23.4s\n"
        "zip2 v31.4s, v21.4s, v23.4s\n"

        "trn1 v16.2d, v24.2d, v26.2d\n"
        "trn2 v17.2d, v24.2d, v26.2d\n"
        "trn1 v18.2d, v25.2d, v27.2d\n"
        "trn2 v19.2d, v25.2d, v27.2d\n"

        "trn1 v20.2d, v28.2d, v30.2d\n"
        "trn2 v21.2d, v28.2d, v30.2d\n"
        "trn1 v22.2d, v29.2d, v31.2d\n"
        "trn2 v23.2d, v29.2d, v31.2d\n"

        "st1 {v0.4s}, [x11], #16\n"
        "st1 {v16.4s}, [x11], #16\n"
        "st1 {v1.4s}, [x11], #16\n"
        "st1 {v17.4s}, [x11], #16\n"
        "st1 {v2.4s}, [x11], #16\n"
        "st1 {v18.4s}, [x11], #16\n"
        "st1 {v3.4s}, [x11], #16\n"
        "st1 {v19.4s}, [x11], #16\n"
        "st1 {v4.4s}, [x11], #16\n"
        "st1 {v20.4s}, [x11], #16\n"
        "st1 {v5.4s}, [x11], #16\n"
        "st1 {v21.4s}, [x11], #16\n"
        "st1 {v6.4s}, [x11], #16\n"
        "st1 {v22.4s}, [x11], #16\n"
        "st1 {v7.4s}, [x11], #16\n"
        "st1 {v23.4s}, [x11], #16\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
        : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31");
#elif ENABLE_ARM32
      /* 8x4 row-major to col-major */
      size_t stride = col * sizeof(float);
      asm volatile(
        "mov r10, %[src_c]\n"
        "mov r11, %[dst_c]\n"

        "vld1.32 {q0}, [r10], %[stride]\n"
        "vld1.32 {q2}, [r10], %[stride]\n"
        "vld1.32 {q4}, [r10], %[stride]\n"
        "vld1.32 {q6}, [r10], %[stride]\n"

        "vtrn.32 d0, d4\n"
        "vtrn.32 d1, d5\n"
        "vtrn.32 d8, d12\n"
        "vtrn.32 d9, d13\n"

        "vld1.32 {q1}, [r10], %[stride]\n"
        "vld1.32 {q3}, [r10], %[stride]\n"
        "vld1.32 {q5}, [r10], %[stride]\n"
        "vld1.32 {q7}, [r10], %[stride]\n"

        "vswp d1, d8\n"
        "vswp d5, d12\n"

        "vtrn.32 d2, d6\n"
        "vtrn.32 d3, d7\n"
        "vtrn.32 d10, d14\n"
        "vtrn.32 d11, d15\n"

        "vswp d3, d10\n"
        "vswp d7, d14\n"

        "vst1.32 {q0, q1}, [r11]!\n"
        "vst1.32 {q2, q3}, [r11]!\n"
        "vst1.32 {q4, q5}, [r11]!\n"
        "vst1.32 {q6, q7}, [r11]!\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
        : "r10", "r11", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#else
      for (int tr = 0; tr < 8; tr++) {
        for (int tc = 0; tc < 4; tc++) {
          dst_c[tc * 8 + tr] = src_c[tr * lead + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C8NUM;
      for (size_t i = 0; i < C8NUM; i++) {
        dst_c[i] = src_c[i * lead];
      }
    }
    src_r += C8NUM * lead;
    dst_r += C8NUM * col;
  }
  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C8NUM] = src_r[i];
    }
    src_r += lead;
    dst_r += 1;
  }
  return;
}
#ifdef ENABLE_ARM32
static void RowMajor2Col4MajorStride(const float *src_ptr, float *dst_ptr, size_t row, size_t col, int lead) {
  size_t row8 = row / C4NUM * C4NUM;
  size_t col4 = col / C4NUM * C4NUM;
  const float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  size_t ri = 0;
  for (; ri < row8; ri += C4NUM) {
    size_t ci = 0;
    for (; ci < col4; ci += C4NUM) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C4NUM;

      /* 4x4 row-major to col-major */
#ifdef ENABLE_ARM32
      size_t stride = col * 4;
      asm volatile(
        "mov r10, %[src_c]\n"
        "mov r12, %[dst_c]\n"

        "vld1.32 {q0}, [r10], %[stride]\n"
        "vld1.32 {q1}, [r10], %[stride]\n"
        "vld1.32 {q2}, [r10], %[stride]\n"
        "vld1.32 {q3}, [r10], %[stride]\n"

        "vtrn.32 d0, d2\n"
        "vtrn.32 d1, d3\n"
        "vtrn.32 d4, d6\n"
        "vtrn.32 d5, d7\n"

        "vswp d1, d4\n"
        "vswp d3, d6\n"

        "vst1.32 {q0}, [r12]!\n"
        "vst1.32 {q1}, [r12]!\n"
        "vst1.32 {q2}, [r12]!\n"
        "vst1.32 {q3}, [r12]!\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
        : "r10", "r12", "q0", "q1", "q2", "q3");
#else
      for (int tr = 0; tr < C4NUM; tr++) {
        for (int tc = 0; tc < C4NUM; tc++) {
          dst_c[tc * C4NUM + tr] = src_c[tr * lead + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C4NUM;
      for (size_t i = 0; i < C4NUM; i++) {
        dst_c[i] = src_c[i * lead];
      }
    }
    src_r += C4NUM * col;
    dst_r += C4NUM * col;
  }
  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C4NUM] = src_r[i];
    }
    src_r += lead;
    dst_r += 1;
  }
  return;
}
#endif

void GemmMatmul(int ta, int tb, int M, int N, int K, float alpha, const float *mat_a, int lda, const float *mat_b,
                int ldb, float beta, float *mat_c, int ldc, float *workspace) {
  GemmCb gcb;
  gcb.atype = ActType_No;
  gcb.ca = 0;
  gcb.cb = 0;
  gcb.bias = NULL;
  GemmMatmulPlus(ta, tb, M, N, K, alpha, mat_a, lda, mat_b, ldb, beta, mat_c, ldc, workspace, &gcb);
}

void GemmMatmulPlus(int ta, int tb, int M, int N, int K, float alpha, const float *mat_a, int lda, const float *mat_b,
                    int ldb, float beta, float *mat_c, int ldc, float *workspace, GemmCb *gcb) {
#ifdef ENABLE_ARM32
  const int num = C4NUM;
#else
  const int num = C12NUM;
#endif
  float *output = mat_c;
  float *fworkspace = workspace;
  int incremental = (beta < 0.f) || (beta > 0.f);
  float *mat_a_input = (float *)mat_a;
  float *mat_b_input = (float *)mat_b;

#ifdef ENABLE_ARM32
  if (!gcb->ca) {
    mat_a_input = fworkspace;
    fworkspace += MatSize(M, K, num);
    if (ta) {
      RowMajor2Row4MajorStride(mat_a, mat_a_input, K, M, lda);
    } else {
      RowMajor2Col4MajorStride(mat_a, mat_a_input, M, K, lda);
    }
  }
#else
  if (!gcb->ca) {
    mat_a_input = fworkspace;
    fworkspace += MatSize(M, K, num);
    if (ta) {
      RowMajor2Row12MajorStride(mat_a, mat_a_input, K, M, lda);
    } else {
      RowMajor2Col12MajorStride(mat_a, mat_a_input, M, K, lda);
    }
  }
#endif
  if (!gcb->cb) {
    mat_b_input = fworkspace;
    fworkspace += MatSize(N, K, C8NUM);
    if (tb) {
      RowMajor2Col8MajorStride(mat_b, mat_b_input, N, K, ldb);
    } else {
      RowMajor2Row8MajorStride(mat_b, mat_b_input, K, N, ldb);
    }
  }
  if (incremental) output = fworkspace;
#ifdef ENABLE_ARM32
  MatmulFloatNeon32Opt(mat_a_input, mat_b_input, output, gcb->bias, (int)gcb->atype, K, M, N, ldc, 1);
#else
  MatMulOpt(mat_a_input, mat_b_input, output, gcb->bias, gcb->atype, K, M, N, ldc, OutType_Nhwc);
#endif
  if (incremental) AddMatrix(output, mat_c, beta, M, N, ldc);
  gcb->mat_a = mat_a_input;
  gcb->mat_b = mat_b_input;
}
