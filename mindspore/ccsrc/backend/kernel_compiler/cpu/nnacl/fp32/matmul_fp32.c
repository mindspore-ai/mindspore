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

#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#ifdef ENABLE_SSE
#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif
#endif
#ifdef ENABLE_ARM64
#include <arm_neon.h>
#endif
void RowMajor2ColMajor(const float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      dst_ptr[c * row + r] = src_ptr[r * col + c];
    }
  }
}

void RowMajor2Row4Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    const float *src = src_ptr + r * col;
    int c = 0;
    for (; c < col; c++) {
      int cd4 = c / C4NUM;
      int cm4 = c % C4NUM;
      dst_ptr[cd4 * C4NUM * row + r * C4NUM + cm4] = src[c];
    }
    for (; c < UP_ROUND(col, C4NUM); c++) {
      int cd4 = c / C4NUM;
      int cm4 = c % C4NUM;
      dst_ptr[cd4 * C4NUM * row + r * C4NUM + cm4] = 0;
    }
  }
  return;
}

void RowMajor2Row6Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    const float *src = src_ptr + r * col;
    int c = 0;
    for (; c < col; c++) {
      int cd6 = c / C6NUM;
      int cm6 = c % C6NUM;
      dst_ptr[cd6 * C6NUM * row + r * C6NUM + cm6] = src[c];
    }
    for (; c < UP_ROUND(col, C6NUM); c++) {
      int cd6 = c / C6NUM;
      int cm6 = c % C6NUM;
      dst_ptr[cd6 * C6NUM * row + r * C6NUM + cm6] = 0;
    }
  }
  return;
}

void RowMajor2Row8Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    const float *src = src_ptr + r * col;
    int c = 0;
    for (; c < col; c++) {
      int cd8 = c / C8NUM;
      int cm8 = c % C8NUM;
      dst_ptr[cd8 * C8NUM * row + r * C8NUM + cm8] = src[c];
    }
    for (; c < UP_ROUND(col, C8NUM); c++) {
      int cd8 = c / C8NUM;
      int cm8 = c % C8NUM;
      dst_ptr[cd8 * C8NUM * row + r * C8NUM + cm8] = 0;
    }
  }
  return;
}

void RowMajor2Row12Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    const float *src = src_ptr + r * col;
    int c = 0;
    for (; c < col; c++) {
      int cd12 = c / C12NUM;
      int cm12 = c % C12NUM;
      dst_ptr[cd12 * C12NUM * row + r * C12NUM + cm12] = src[c];
    }
    for (; c < UP_ROUND(col, C12NUM); c++) {
      int cd12 = c / C12NUM;
      int cm12 = c % C12NUM;
      dst_ptr[cd12 * C12NUM * row + r * C12NUM + cm12] = 0;
    }
  }
  return;
}

void RowMajor2Row16Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    const float *src = src_ptr + r * col;
    int c = 0;
    for (; c < col; c++) {
      int cd16 = c / C16NUM;
      int cm16 = c % C16NUM;
      dst_ptr[cd16 * C16NUM * row + r * C16NUM + cm16] = src[c];
    }
    for (; c < UP_ROUND(col, C16NUM); c++) {
      int cd16 = c / C16NUM;
      int cm16 = c % C16NUM;
      dst_ptr[cd16 * C16NUM * row + r * C16NUM + cm16] = 0;
    }
  }
  return;
}

void RowMajor2Row32Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  // Not exactly aligned to 32, but aligned to 24 or 16 or 8 If 32 is not met.
  int row_block_num = UP_DIV(row, C8NUM);
  int row_block = C4NUM;
  for (int i = 0; i < row_block_num; i += row_block) {
    row_block = MSMIN(C4NUM, row_block_num - i);  // max_tile = 4
    int row_remainder = MSMIN(row_block * C8NUM, row - i * C8NUM);
    for (int oc = 0; oc < col; ++oc) {
      memcpy(dst_ptr, src_ptr + oc * row + i * C8NUM, row_remainder * sizeof(float));
      dst_ptr += row_block * C8NUM;
    }
  }
}

#ifdef ENABLE_ARM64
void RowMajor2Col12Major_arm64(const float *src_c, float *dst_c, size_t col) {
  size_t stride = col * sizeof(float);
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
      "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
  return;
}
#endif
#ifdef ENABLE_ARM32
void RowMajor2Col12Major_arm32(const float *src_c, float *dst_c, size_t col) {
  size_t stride = col * sizeof(float);
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
  return;
}
#endif
void RowMajor2Col12Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  const float *src_r = src_ptr;
  float *dst_r = dst_ptr;
  int ri = 0;
  for (; ri < (row / C12NUM * C12NUM); ri += C12NUM) {
    int ci = 0;
    for (; ci < (col / C4NUM * C4NUM); ci += C4NUM) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C12NUM;
#ifdef ENABLE_ARM64
      RowMajor2Col12Major_arm64(src_c, dst_c, col);
#elif ENABLE_ARM32
      RowMajor2Col12Major_arm32(src_c, dst_c, col);
#elif ENABLE_SSE
      __m128 src1 = _mm_loadu_ps(src_c);
      __m128 src2 = _mm_loadu_ps(src_c + col);
      __m128 src3 = _mm_loadu_ps(src_c + 2 * col);
      __m128 src4 = _mm_loadu_ps(src_c + 3 * col);
      src_c += 4 * col;
      __m128 src12L = _mm_unpacklo_ps(src1, src2);
      __m128 src12H = _mm_unpackhi_ps(src1, src2);
      __m128 src34L = _mm_unpacklo_ps(src3, src4);
      __m128 src34H = _mm_unpackhi_ps(src3, src4);

      __m128 dst0 = _mm_movelh_ps(src12L, src34L);
      __m128 dst3 = _mm_movehl_ps(src34L, src12L);
      __m128 dst6 = _mm_movelh_ps(src12H, src34H);
      __m128 dst9 = _mm_movehl_ps(src34H, src12H);

      __m128 src5 = _mm_loadu_ps(src_c);
      __m128 src6 = _mm_loadu_ps(src_c + col);
      __m128 src7 = _mm_loadu_ps(src_c + 2 * col);
      __m128 src8 = _mm_loadu_ps(src_c + 3 * col);
      src_c += 4 * col;
      __m128 src56L = _mm_unpacklo_ps(src5, src6);
      __m128 src56H = _mm_unpackhi_ps(src5, src6);
      __m128 src78L = _mm_unpacklo_ps(src7, src8);
      __m128 src78H = _mm_unpackhi_ps(src7, src8);
      __m128 dst1 = _mm_movelh_ps(src56L, src78L);
      __m128 dst4 = _mm_movehl_ps(src78L, src56L);
      __m128 dst7 = _mm_movelh_ps(src56H, src78H);
      __m128 dst10 = _mm_movehl_ps(src78H, src56H);

      __m128 src9 = _mm_loadu_ps(src_c);
      __m128 src10 = _mm_loadu_ps(src_c + col);
      __m128 src11 = _mm_loadu_ps(src_c + 2 * col);
      __m128 src12 = _mm_loadu_ps(src_c + 3 * col);
      src_c += 4 * col;
      __m128 src910L = _mm_unpacklo_ps(src9, src10);
      __m128 src910H = _mm_unpackhi_ps(src9, src10);
      __m128 src1112L = _mm_unpacklo_ps(src11, src12);
      __m128 src1112H = _mm_unpackhi_ps(src11, src12);
      __m128 dst2 = _mm_movelh_ps(src910L, src1112L);
      __m128 dst5 = _mm_movehl_ps(src1112L, src910L);
      __m128 dst8 = _mm_movelh_ps(src910H, src1112H);
      __m128 dst11 = _mm_movehl_ps(src1112H, src910H);

      _mm_storeu_ps(dst_c, dst0);
      _mm_storeu_ps(dst_c + 4, dst1);
      _mm_storeu_ps(dst_c + 8, dst2);
      _mm_storeu_ps(dst_c + 12, dst3);
      _mm_storeu_ps(dst_c + 16, dst4);
      _mm_storeu_ps(dst_c + 20, dst5);
      _mm_storeu_ps(dst_c + 24, dst6);
      _mm_storeu_ps(dst_c + 28, dst7);
      _mm_storeu_ps(dst_c + 32, dst8);
      _mm_storeu_ps(dst_c + 36, dst9);
      _mm_storeu_ps(dst_c + 40, dst10);
      _mm_storeu_ps(dst_c + 44, dst11);
#else
      for (int tr = 0; tr < C12NUM; tr++) {
        for (int tc = 0; tc < C4NUM; tc++) {
          dst_c[tc * C12NUM + tr] = src_c[tr * col + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C12NUM;
      for (int i = 0; i < C12NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C12NUM * col;
    dst_r += C12NUM * col;
  }
  for (; ri < row; ri++, dst_r++, src_r += col) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C12NUM] = src_r[i];
    }
  }
  for (; ri < UP_ROUND(row, C12NUM); ri++, dst_r++) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C12NUM] = 0;
    }
  }
}

#ifdef ENABLE_ARM64
void RowMajor2Col8Major_arm64(const float *src_c, float *dst_c, size_t col) {
  size_t stride = col * sizeof(float);
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
      "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
  return;
}
#endif
#ifdef ENABLE_ARM32
#ifndef SUPPORT_NNIE
void RowMajor2Col8Major_arm32(const float *src_c, float *dst_c, size_t col) {
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
  return;
}
#else
void RowMajor2Col8Major_arm32(const float *src_c, float *dst_c, size_t col) {
  size_t stride = col * sizeof(float);
  asm volatile(
    "mov r10, %[src_c]\n"
    "mov r7, %[dst_c]\n"

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

    "vst1.32 {q0, q1}, [r7]!\n"
    "vst1.32 {q2, q3}, [r7]!\n"
    "vst1.32 {q4, q5}, [r7]!\n"
    "vst1.32 {q6, q7}, [r7]!\n"

    :
    : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
    : "r10", "r7", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
  return;
}
#endif
#endif
void RowMajor2Col8Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  int row8 = row / C8NUM * C8NUM;
#ifdef ENABLE_ARM64
  int col_skip = col / C8NUM * C8NUM;
  int skip_size = C8NUM;
#else
  int col_skip = col / C4NUM * C4NUM;
  int skip_size = C4NUM;
#endif
  const float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  int ri = 0;
  for (; ri < row8; ri += C8NUM) {
    int ci = 0;
    for (; ci < col_skip; ci += skip_size) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C8NUM;

#ifdef ENABLE_ARM64
      RowMajor2Col8Major_arm64(src_c, dst_c, col);
#elif ENABLE_ARM32
      RowMajor2Col8Major_arm32(src_c, dst_c, col);
#elif ENABLE_SSE
      __m128 src1 = _mm_loadu_ps(src_c);
      __m128 src2 = _mm_loadu_ps(src_c + col);
      __m128 src3 = _mm_loadu_ps(src_c + 2 * col);
      __m128 src4 = _mm_loadu_ps(src_c + 3 * col);
      src_c += 4 * col;
      __m128 src12L = _mm_unpacklo_ps(src1, src2);  // x5
      __m128 src12H = _mm_unpackhi_ps(src1, src2);  // x1
      __m128 src34L = _mm_unpacklo_ps(src3, src4);  // x
      __m128 src34H = _mm_unpackhi_ps(src3, src4);
      _mm_storeu_ps(dst_c, _mm_movelh_ps(src12L, src34L));
      _mm_storeu_ps(dst_c + 8, _mm_movehl_ps(src34L, src12L));
      _mm_storeu_ps(dst_c + 16, _mm_movelh_ps(src12H, src34H));
      _mm_storeu_ps(dst_c + 24, _mm_movehl_ps(src34H, src12H));

      __m128 src5 = _mm_loadu_ps(src_c);
      __m128 src6 = _mm_loadu_ps(src_c + col);
      __m128 src7 = _mm_loadu_ps(src_c + 2 * col);
      __m128 src8 = _mm_loadu_ps(src_c + 3 * col);
      src_c += 4 * col;
      __m128 src56L = _mm_unpacklo_ps(src5, src6);
      __m128 src56H = _mm_unpackhi_ps(src5, src6);
      __m128 src78L = _mm_unpacklo_ps(src7, src8);
      __m128 src78H = _mm_unpackhi_ps(src7, src8);
      _mm_storeu_ps(dst_c + 4, _mm_movelh_ps(src56L, src78L));
      _mm_storeu_ps(dst_c + 12, _mm_movehl_ps(src78L, src56L));
      _mm_storeu_ps(dst_c + 20, _mm_movelh_ps(src56H, src78H));
      _mm_storeu_ps(dst_c + 28, _mm_movehl_ps(src78H, src56H));
#else
      for (int tr = 0; tr < 8; tr++) {
        for (int tc = 0; tc < 4; tc++) {
          dst_c[tc * 8 + tr] = src_c[tr * col + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C8NUM;
      for (int i = 0; i < C8NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C8NUM * col;
    dst_r += C8NUM * col;
  }
  for (; ri < row; ri++, src_r += col, dst_r++) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C8NUM] = src_r[i];
    }
  }

  for (; ri < UP_ROUND(row, C8NUM); ri++, dst_r++) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C8NUM] = 0;
    }
  }
}

void RowMajor2Col16Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  int row16 = row / C16NUM * C16NUM;
  int col8 = col / C8NUM * C8NUM;
  const float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  int ri = 0;
  for (; ri < row16; ri += C16NUM) {
    int ci = 0;
    for (; ci < col8; ci += C8NUM) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C16NUM;
#ifdef ENABLE_AVX
      Transpose8X8Fp32Avx(src_c, dst_c, col, C16NUM);
      Transpose8X8Fp32Avx(src_c + C8NUM * col, dst_c + C8NUM, col, C16NUM);
#endif
      for (int tr = 0; tr < C16NUM; tr++) {
        for (int tc = 0; tc < C8NUM; tc++) {
          dst_c[tc * C16NUM + tr] = src_c[tr * col + tc];
        }
      }
    }
    for (; ci < col; ci++) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C16NUM;
      for (int i = 0; i < C16NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C16NUM * col;
    dst_r += C16NUM * col;
  }
  for (; ri < row; ri++) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C16NUM] = src_r[i];
    }
    src_r += col;
    dst_r += 1;
  }

  int total_row = UP_ROUND(row, C16NUM);
  for (; ri < total_row; ri++) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C16NUM] = 0;
    }
    dst_r += 1;
  }
}

void RowMajor2Col32Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  // Not exactly aligned to 32, but aligned to 24 or 16 or 8 If 32 is not met.
  int col_block_num = UP_DIV(col, C8NUM);
  int col_block = C4NUM;
  for (int i = 0; i < col_block_num; i += col_block) {
    col_block = MSMIN(C4NUM, col_block_num - i);  // max_tile = 4
    int index = i * row * C8NUM;
    int col_remainder = MSMIN(C8NUM * col_block, col - i * C8NUM);
    for (int ir = 0; ir < row; ++ir) {
      for (int oc = 0; oc < col_remainder; ++oc) {
        int oc_index = oc * row + ir + index;
        dst_ptr[oc] = src_ptr[oc_index];
      }
      dst_ptr += col_block * C8NUM;
    }
  }
}

void RowMajor2Col6Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  int totalRow = UP_ROUND(row, C6NUM);
  int row6 = row / C6NUM * C6NUM;
  int col8 = col / C8NUM * C8NUM;
  const float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  int ri = 0;
  for (; ri < row6; ri += C6NUM) {
    int ci = 0;
    for (; ci < col8; ci += C8NUM) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C6NUM;

#ifdef ENABLE_AVX
      __m256 src0 = _mm256_loadu_ps(src_c);
      __m256 src1 = _mm256_loadu_ps(src_c + col);
      __m256 src2 = _mm256_loadu_ps(src_c + 2 * col);
      __m256 src3 = _mm256_loadu_ps(src_c + 3 * col);
      __m256 src4 = _mm256_loadu_ps(src_c + 4 * col);
      __m256 src5 = _mm256_loadu_ps(src_c + 5 * col);
      __m256 trans0 = _mm256_unpacklo_ps(src0, src1);
      __m256 trans1 = _mm256_unpacklo_ps(src2, src3);
      __m256 trans2 = _mm256_unpacklo_ps(src4, src5);
      __m256 trans3 = _mm256_unpackhi_ps(src0, src1);
      __m256 trans4 = _mm256_unpackhi_ps(src2, src3);
      __m256 trans5 = _mm256_unpackhi_ps(src4, src5);
      __m128 lo0 = _mm256_castps256_ps128(trans0);
      __m128 lo1 = _mm256_castps256_ps128(trans1);
      __m128 lo2 = _mm256_castps256_ps128(trans2);
      __m128 lo3 = _mm256_castps256_ps128(trans3);
      __m128 lo4 = _mm256_castps256_ps128(trans4);
      __m128 lo5 = _mm256_castps256_ps128(trans5);
      __m128 hi0 = _mm256_extractf128_ps(trans0, 1);
      __m128 hi1 = _mm256_extractf128_ps(trans1, 1);
      __m128 hi2 = _mm256_extractf128_ps(trans2, 1);
      __m128 hi3 = _mm256_extractf128_ps(trans3, 1);
      __m128 hi4 = _mm256_extractf128_ps(trans4, 1);
      __m128 hi5 = _mm256_extractf128_ps(trans5, 1);
      __m128 res0 = _mm_shuffle_ps(lo0, lo1, _MM_SHUFFLE(1, 0, 1, 0));
      __m128 res1 = _mm_shuffle_ps(lo2, lo0, _MM_SHUFFLE(3, 2, 1, 0));
      __m128 res2 = _mm_shuffle_ps(lo1, lo2, _MM_SHUFFLE(3, 2, 3, 2));
      __m128 res3 = _mm_shuffle_ps(lo3, lo4, _MM_SHUFFLE(1, 0, 1, 0));
      __m128 res4 = _mm_shuffle_ps(lo5, lo3, _MM_SHUFFLE(3, 2, 1, 0));
      __m128 res5 = _mm_shuffle_ps(lo4, lo5, _MM_SHUFFLE(3, 2, 3, 2));
      __m128 res6 = _mm_shuffle_ps(hi0, hi1, _MM_SHUFFLE(1, 0, 1, 0));
      __m128 res7 = _mm_shuffle_ps(hi2, hi0, _MM_SHUFFLE(3, 2, 1, 0));
      __m128 res8 = _mm_shuffle_ps(hi1, hi2, _MM_SHUFFLE(3, 2, 3, 2));
      __m128 res9 = _mm_shuffle_ps(hi3, hi4, _MM_SHUFFLE(1, 0, 1, 0));
      __m128 res10 = _mm_shuffle_ps(hi5, hi3, _MM_SHUFFLE(3, 2, 1, 0));
      __m128 res11 = _mm_shuffle_ps(hi4, hi5, _MM_SHUFFLE(3, 2, 3, 2));
      _mm_storeu_ps(dst_c, res0);
      _mm_storeu_ps(dst_c + 4, res1);
      _mm_storeu_ps(dst_c + 8, res2);
      _mm_storeu_ps(dst_c + 12, res3);
      _mm_storeu_ps(dst_c + 16, res4);
      _mm_storeu_ps(dst_c + 20, res5);
      _mm_storeu_ps(dst_c + 24, res6);
      _mm_storeu_ps(dst_c + 28, res7);
      _mm_storeu_ps(dst_c + 32, res8);
      _mm_storeu_ps(dst_c + 36, res9);
      _mm_storeu_ps(dst_c + 40, res10);
      _mm_storeu_ps(dst_c + 44, res11);
#else
      for (int tr = 0; tr < C6NUM; tr++) {
        for (int tc = 0; tc < C8NUM; tc++) {
          dst_c[tc * C6NUM + tr] = src_c[tr * col + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C6NUM;
      for (int i = 0; i < C6NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C6NUM * col;
    dst_r += C6NUM * col;
  }

  for (; ri < row; ri++) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C6NUM] = src_r[i];
    }
    src_r += col;
    dst_r += 1;
  }

  for (; ri < totalRow; ri++) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C6NUM] = 0;
    }
    dst_r += 1;
  }
}

void RowMajor2Col4Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  int total_row = UP_ROUND(row, C4NUM);
  int row4 = row / C4NUM * C4NUM;
  int col4 = col / C4NUM * C4NUM;
  const float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  int ri = 0;
  for (; ri < row4; ri += C4NUM) {
    int ci = 0;
    for (; ci < col4; ci += C4NUM) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C4NUM;

#ifdef ENABLE_ARM32
      int stride = col * 4;
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
#elif ENABLE_SSE
      __m128 src1 = _mm_loadu_ps(src_c);
      __m128 src2 = _mm_loadu_ps(src_c + col);
      __m128 src3 = _mm_loadu_ps(src_c + 2 * col);
      __m128 src4 = _mm_loadu_ps(src_c + 3 * col);
      src_c += 4 * col;
      __m128 src12L = _mm_unpacklo_ps(src1, src2);
      __m128 src12H = _mm_unpackhi_ps(src1, src2);
      __m128 src34L = _mm_unpacklo_ps(src3, src4);
      __m128 src34H = _mm_unpackhi_ps(src3, src4);

      __m128 dst0 = _mm_movelh_ps(src12L, src34L);
      __m128 dst1 = _mm_movehl_ps(src34L, src12L);
      __m128 dst2 = _mm_movelh_ps(src12H, src34H);
      __m128 dst3 = _mm_movehl_ps(src34H, src12H);

      _mm_storeu_ps(dst_c, dst0);
      _mm_storeu_ps(dst_c + 4, dst1);
      _mm_storeu_ps(dst_c + 8, dst2);
      _mm_storeu_ps(dst_c + 12, dst3);
#else
      for (size_t tr = 0; tr < C4NUM; tr++) {
        for (size_t tc = 0; tc < C4NUM; tc++) {
          dst_c[tc * C4NUM + tr] = src_c[tr * col + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C4NUM;
      for (int i = 0; i < C4NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C4NUM * col;
    dst_r += C4NUM * col;
  }
  for (; ri < row; ri++) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C4NUM] = src_r[i];
    }
    src_r += col;
    dst_r += 1;
  }

  for (; ri < total_row; ri++) {
    for (int i = 0; i < col; i++) {
      dst_r[i * C4NUM] = 0;
    }
    dst_r += 1;
  }
}

#ifndef ENABLE_ARM
void MatVecMulFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col) {
  for (int ci = 0; ci < col; ci++) {
    float value = 0;
    for (int di = 0; di < depth; di++) {
      value += a[di] * b[ci * depth + di];
    }
    if (bias != NULL) value += bias[ci];
    if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
    if (act_type == ActType_Relu || act_type == ActType_Relu6) value = MSMAX(0.0f, value);
    c[ci] = value;
  }
}
#endif

#ifdef ENABLE_ARM64
// 4x8
void MatVecMulFp32Neon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col,
                         int align_col) {
  int ci = 0;
  for (; ci < align_col - C8NUM + 1; ci += C8NUM) {
    float32x4_t acc_0;
    float32x4_t acc_1;
    if (bias != NULL) {
      acc_0 = vld1q_f32(bias + ci);
      acc_1 = vld1q_f32(bias + ci + C4NUM);
    } else {
      acc_0 = vdupq_n_f32(0.0f);
      acc_1 = vdupq_n_f32(0.0f);
    }
    const float *bv_base = b + ci * depth;
    int di = 0;
    for (; di < depth - C4NUM + 1; di += C4NUM) {
      float32x4_t av = vld1q_f32(a + di);
      float32x4_t bv_00 = vld1q_f32(bv_base);
      float32x4_t bv_10 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      float32x4_t bv_01 = vld1q_f32(bv_base);
      float32x4_t bv_11 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      float32x4_t bv_02 = vld1q_f32(bv_base);
      float32x4_t bv_12 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      float32x4_t bv_03 = vld1q_f32(bv_base);
      float32x4_t bv_13 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      acc_0 = vmlaq_n_f32(acc_0, bv_00, av[0]);
      acc_1 = vmlaq_n_f32(acc_1, bv_10, av[0]);
      acc_0 = vmlaq_n_f32(acc_0, bv_01, av[1]);
      acc_1 = vmlaq_n_f32(acc_1, bv_11, av[1]);
      acc_0 = vmlaq_n_f32(acc_0, bv_02, av[2]);
      acc_1 = vmlaq_n_f32(acc_1, bv_12, av[2]);
      acc_0 = vmlaq_n_f32(acc_0, bv_03, av[3]);
      acc_1 = vmlaq_n_f32(acc_1, bv_13, av[3]);
    }
    if (di < depth) {
      for (; di < depth; ++di) {
        float ai = a[di];
        float32x4_t bv0 = vld1q_f32(bv_base);
        float32x4_t bv1 = vld1q_f32(bv_base + C4NUM);
        acc_0 = vmlaq_n_f32(acc_0, bv0, ai);
        acc_1 = vmlaq_n_f32(acc_1, bv1, ai);
        bv_base += C8NUM;
      }
    }  // only save actual col num data
    if (ci + C4NUM - 1 >= col) {
      int c_remain = col - ci;
      for (int i = 0; i < c_remain; ++i) {
        if (act_type == ActType_Relu) {
          c[i] = MSMAX(acc_0[i], 0.0f);
        } else if (act_type == ActType_Relu6) {
          c[i] = MSMIN(MSMAX(acc_0[i], 0.0f), 6.0f);
        } else {
          c[i] = acc_0[i];
        }
      }
      return;
    }
    if (act_type == ActType_Relu) {
      acc_0 = vmaxq_f32(acc_0, vdupq_n_f32(0.0f));
    } else if (act_type == ActType_Relu6) {
      acc_0 = vminq_f32(vmaxq_f32(acc_0, vdupq_n_f32(0.0f)), vdupq_n_f32(6.0f));
    }
    vst1q_f32(c, acc_0);
    if (ci + C8NUM - 1 >= col) {
      int c_remain = col - ci - C4NUM;
      for (int i = 0; i < c_remain; ++i) {
        if (act_type == ActType_Relu) {
          c[C4NUM + i] = MSMAX(acc_1[i], 0.0f);
        } else if (act_type == ActType_Relu6) {
          c[C4NUM + i] = MSMIN(MSMAX(acc_1[i], 0.0f), 6.0f);
        } else {
          c[C4NUM + i] = acc_1[i];
        }
      }
      return;
    }
    if (act_type == ActType_Relu) {
      acc_1 = vmaxq_f32(acc_1, vdupq_n_f32(0.0f));
    } else if (act_type == ActType_Relu6) {
      acc_1 = vminq_f32(vmaxq_f32(acc_1, vdupq_n_f32(0.0f)), vdupq_n_f32(6.0f));
    }
    vst1q_f32(c + C4NUM, acc_1);
    c += C8NUM;
  }
}
#endif

void MatMul12x8(const float *a, const float *b, float *dst, const float *bias, ActType act_type, int deep, int row,
                int col, int stride, int out_type) {
  if (out_type == OutType_Nhwc) {
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        int r12div = r / 12, r12mod = r % 12;
        int c8div = c / 8, c8mod = c % 8;
        size_t ci = r * stride + c;
        float value = 0;
        for (int d = 0; d < deep; d++) {
          size_t ai = r12div * deep * 12 + d * 12 + r12mod;
          size_t bi = c8div * deep * 8 + d * 8 + c8mod;
          value = value + a[ai] * b[bi];
        }
        ADD_BIAS(value, bias, c)
        DO_RELU(value, act_type)
        DO_RELU6(value, act_type)
        dst[ci] = value;
      }
    }
  } else if (out_type == OutType_C8) {
    int col_8 = UP_ROUND(col, C8NUM);
    int row_12 = UP_ROUND(row, C12NUM);
    for (int r = 0; r < row_12; r++) {
      for (int c = 0; c < col_8; c++) {
        int r12div = r / C12NUM, r12mod = r % C12NUM;
        int c8div = c / C8NUM, c8mod = c % C8NUM;
        size_t ci = (c8div * C8NUM * row_12 + r * C8NUM + c8mod);
        float value = 0;
        for (int d = 0; d < deep; d++) {
          size_t ai = r12div * deep * C12NUM + d * C12NUM + r12mod;
          size_t bi = c8div * deep * C8NUM + d * C8NUM + c8mod;
          value = value + a[ai] * b[bi];
        }
        ADD_BIAS(value, bias, c)
        DO_RELU(value, act_type)
        DO_RELU6(value, act_type)
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
          size_t ai = src_r_offset + d * C12NUM;
          size_t bi = c8div * deep * 8 + d * 8 + c8mod;
          value = value + a[ai] * b[bi];
        }
        ADD_BIAS(value, bias, j)
        DO_RELU(value, act_type)
        DO_RELU6(value, act_type)
        dst[ci] = value;
      }
    }
  }
}

void MatMulOpt(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row,
               int col, size_t stride, int out_type) {
#ifdef ENABLE_ARM64
  if (out_type == OutType_C8) {
    MatmulFloatNeon64(a, b, c, bias, (int)act_type, deep, row, col, stride, 0, 0);
  } else {
    MatmulFloatNeon64Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type));
  }
#elif ENABLE_ARM32
  if (out_type == OutType_C8) {
    MatmulFloatNeon32(a, b, c, bias, (int)act_type, deep, row, col, stride, 0, 0);
  } else if (out_type == OutType_Nhwc) {
    MatmulFloatNeon32Opt12x4(a, b, c, bias, (int)act_type, deep, row, col, stride, 1);
  } else {
    MatmulFloatNeon32Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type));
  }
#elif ENABLE_AVX
  MatmulFloatAvxOpt(a, b, c, bias, (size_t)act_type, deep, row, col, stride, (size_t)(out_type));
#elif ENABLE_SSE
  MatmulFloatSse64Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type));
#else
  MatMul12x8(a, b, c, bias, act_type, deep, row, col, stride, out_type);
#endif
}

#ifdef ENABLE_AVX
void MatVecMulAvxFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int cur_col,
                      int col_align) {
  // one time process 32 out_channel
  int col_block = C32NUM;
  int act_flag = 0;
  if (act_type == ActType_Relu6) {
    act_flag += 1;
  }
  if (act_type == ActType_Relu || act_type == ActType_Relu6) {
    act_flag += 2;
  }
  MatVecMulKernel kernel[4] = {MatVecMul1x8Kernel, MatVecMul1x16Kernel, MatVecMul1x24Kernel, MatVecMul1x32Kernel};
  const float *bias_data = bias;
  for (int col_index = 0; col_index < cur_col; col_index += col_block) {
    col_block = cur_col - col_index < col_block ? cur_col - col_index : col_block;
    kernel[(col_block >> 3) - 1](c + col_index, a, b + col_index * depth, bias_data, act_flag, 1, col_block >> 3,
                                 col_align, depth);
    if (bias_data != NULL) {
      bias_data += col_block;
    }
  }
}

void MatVecMul1x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "vmovups 0x60(%2), %%ymm3\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "1:\n"  // deep_c8
    "movq %3, %%rcx\n"
    "shr $3, %%ecx\n"
    "je 3f\n"
    "2:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 0x60(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 4(%0), %%ymm4\n"
    "vfmadd231ps 128(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 160(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 192(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 224(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 8(%0), %%ymm4\n"
    "vfmadd231ps 256(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 288(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 320(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 352(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 12(%0), %%ymm4\n"
    "vfmadd231ps 384(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 416(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 448(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 480(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 16(%0), %%ymm4\n"
    "vfmadd231ps 512(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 544(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 576(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 608(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 20(%0), %%ymm4\n"
    "vfmadd231ps 640(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 672(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 704(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 736(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 24(%0), %%ymm4\n"
    "vfmadd231ps 768(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 800(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 832(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 864(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 28(%0), %%ymm4\n"
    "vfmadd231ps 896(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 928(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 960(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 992(%1), %%ymm4, %%ymm3\n"
    "addq $1024, %1\n"
    "addq $32, %0\n"
    "dec %%ecx\n"
    "jg 2b\n"

    "3:\n"
    "and $7, %3\n"  // deep_remainder
    "je 5f\n"
    "4:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 0x60(%1), %%ymm4, %%ymm3\n"
    "addq $128, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 4b\n"

    "5:\n"
    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%5)\n"
    "vmovups %%ymm2, 0x40(%5)\n"
    "vmovups %%ymm3, 0x60(%5)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst)  // 5
    : "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm12", "%ymm4", "%ymm14");
}

void MatVecMul1x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"

    "1:\n"  // deep
    "movq %3, %%rcx\n"
    "shr $3, %%ecx\n"
    "je 3f\n"
    "2:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 4(%0), %%ymm4\n"
    "vfmadd231ps 96(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 128(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 160(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 8(%0), %%ymm4\n"
    "vfmadd231ps 192(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 224(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 256(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 12(%0), %%ymm4\n"
    "vfmadd231ps 288(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 320(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 352(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 16(%0), %%ymm4\n"
    "vfmadd231ps 384(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 416(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 448(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 20(%0), %%ymm4\n"
    "vfmadd231ps 480(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 512(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 544(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 24(%0), %%ymm4\n"
    "vfmadd231ps 576(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 608(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 640(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 28(%0), %%ymm4\n"
    "vfmadd231ps 672(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 704(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 736(%1), %%ymm4, %%ymm2\n"
    "addq $768, %1\n"
    "addq $32, %0\n"
    "dec %%ecx\n"
    "jg 2b\n"

    "3:\n"
    "and $7, %3\n"  // deep_remainder
    "je 5f\n"
    "4:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm4, %%ymm2\n"
    "addq $96, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 4b\n"

    "5:\n"
    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"

    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"

    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%5)\n"
    "vmovups %%ymm2, 0x40(%5)\n"

    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst)  // 5
    : "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm12", "%ymm4", "%ymm14");
}

void MatVecMul1x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "1:\n"
    "movq %3, %%rcx\n"
    "shr $3, %%ecx\n"
    "je 3f\n"
    "2:\n"  // deep_c8
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 4(%0), %%ymm4\n"
    "vfmadd231ps 64(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 96(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 8(%0), %%ymm4\n"
    "vfmadd231ps 128(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 160(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 12(%0), %%ymm4\n"
    "vfmadd231ps 192(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 224(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 16(%0), %%ymm4\n"
    "vfmadd231ps 256(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 288(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 20(%0), %%ymm4\n"
    "vfmadd231ps 320(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 352(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 24(%0), %%ymm4\n"
    "vfmadd231ps 384(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 416(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 28(%0), %%ymm4\n"
    "vfmadd231ps 448(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 480(%1), %%ymm4, %%ymm1\n"
    "addq $512, %1\n"
    "addq $32, %0\n"
    "dec %%ecx\n"
    "jg 2b\n"

    "3:\n"
    "and $7, %3\n"
    "je 5f\n"
    "4:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "addq $64, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 4b\n"

    "5:\n"
    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"

    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"

    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%5)\n"

    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst)  // 5
    : "%rcx", "%ymm0", "%ymm1", "%ymm12", "%ymm4", "%ymm14");
}

void MatVecMul1x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                        size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "1:\n"
    "movq %3, %%rcx\n"
    "shr $3, %%ecx\n"
    "je 3f\n"
    "2:\n"  // deep_c8
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 4(%0), %%ymm4\n"
    "vfmadd231ps 32(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 8(%0), %%ymm4\n"
    "vfmadd231ps 64(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 12(%0), %%ymm4\n"
    "vfmadd231ps 96(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 16(%0), %%ymm4\n"
    "vfmadd231ps 128(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 20(%0), %%ymm4\n"
    "vfmadd231ps 160(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 24(%0), %%ymm4\n"
    "vfmadd231ps 192(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 28(%0), %%ymm4\n"
    "vfmadd231ps 224(%1), %%ymm4, %%ymm0\n"
    "addq $256, %1\n"
    "addq $32, %0\n"
    "dec %%ecx\n"
    "jg 2b\n"

    "3:\n"
    "and $7, %3\n"
    "je 5f\n"
    "4:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "addq $32, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 4b\n"

    "5:\n"
    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"

    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"

    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0

    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst)  // 5
    : "%rcx", "%ymm0", "%ymm1", "%ymm12", "%ymm4", "%ymm14");
}

#ifdef ENABLE_DEBUG
void MatVecMulRowxColKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  __m256 dst_data[12];
  const float *src_sw[12];
  __m256 weight_data[4];
  for (int i = 0; i < 4; ++i) {
    weight_data[i] = _mm256_set1_ps(0.0f);
  }
  for (int i = 0; i < row_block; ++i) {
    if (bias != NULL) {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] = _mm256_loadu_ps(bias + j * 8);
      }
    } else {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] = _mm256_set1_ps(0.0f);
      }
    }
    src_sw[i] = src + i * deep;
  }
  const float *weight_kernel = weight;
  for (int ic = 0; ic < deep; ++ic) {
    for (int j = 0; j < col_block; ++j) {
      weight_data[j] = _mm256_loadu_ps(weight_kernel + j * C8NUM);
    }
    for (int i = 0; i < row_block; ++i) {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] =
          _mm256_fmadd_ps(_mm256_set1_ps(src_sw[i][ic]), weight_data[j], dst_data[i * col_block + j]);
      }
    }
    weight_kernel += C8NUM * col_block;
  }  // ic loop
  // add bias and relu
  for (int i = 0; i < row_block; ++i) {
    for (int j = 0; j < col_block; ++j) {
      if (0x1 & act_flag) {  // relu6
        dst_data[i * col_block + j] = _mm256_min_ps(dst_data[i * col_block + j], _mm256_set1_ps(6.0f));
      }
      if (0x2 & act_flag) {  // relu
        dst_data[i * col_block + j] = _mm256_max_ps(dst_data[i * col_block + j], _mm256_set1_ps(0.0f));
      }
      _mm256_storeu_ps(dst + i * col_algin + j * C8NUM, dst_data[i * col_block + j]);
    }
  }
}
#endif
#endif
