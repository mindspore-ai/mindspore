/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"

void PackWeightKHWToHWKFp32(const void *src, void *dst, int plane, int channel) {
  PackNCHWToNHWCFp32(src, dst, 1, plane, channel, 0, 0);
}

void PackHWCToWHC(const float *src, float *dst, int height, int width, int channel) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      memcpy(dst + (j * height + i) * channel, src + (i * width + j) * channel, channel * sizeof(float));
    }
  }
}

void PackNHWCToNC4HW4NotAlignedFp32(const float *src, float *dst, const int batch, const int plane, const int channel) {
  if (channel <= C4NUM) {
    memcpy(dst, src, batch * plane * channel * sizeof(float));
    return;
  }
  int tmp = DOWN_DIV(channel, C4NUM);
  int c_res = channel - tmp * C4NUM;
  int c4_block = tmp * plane * C4NUM;
  for (int b = 0; b < batch; b++) {
    int batch_oc_offset = b * plane * channel;
    for (int k = 0; k < plane; k++) {
      int src_kernel_offset = batch_oc_offset + k * channel;
      int dst_kernel_offset = batch_oc_offset + k * C4NUM;
      int c = 0;
      for (; c <= channel - C4NUM; c += C4NUM) {
#if defined(ENABLE_SSE) || defined(ENABLE_ARM)
        MS_FLOAT32X4 src_data = MS_LDQ_F32(src + src_kernel_offset + c);
        MS_STQ_F32(dst + dst_kernel_offset + c * plane, src_data);
#else
        for (int k1 = 0; k1 < C4NUM; ++k1) {
          (dst + dst_kernel_offset + c * plane)[k1] = (src + src_kernel_offset + c)[k1];
        }
#endif
      }
      for (; c < channel; ++c) {
        dst[batch_oc_offset + c4_block + k * c_res + c - tmp * C4NUM] = src[src_kernel_offset + c];
      }
    }
  }
}

void PackNHWCToNC8HW8NotAlignedFp32(const float *src, float *dst, const int batch, const int plane, const int channel) {
  if (channel <= C8NUM) {
    memcpy(dst, src, batch * plane * channel * sizeof(float));
    return;
  }
  int tmp = DOWN_DIV(channel, C8NUM);
  int c_res = channel - tmp * C8NUM;
  int c8_block = tmp * plane * C8NUM;
  for (int b = 0; b < batch; b++) {
    int batch_oc_offset = b * plane * channel;
    for (int k = 0; k < plane; k++) {
      int src_kernel_offset = batch_oc_offset + k * channel;
      int dst_kernel_offset = batch_oc_offset + k * C8NUM;
      int c = 0;
      for (; c <= channel - C8NUM; c += C8NUM) {
#ifdef ENABLE_AVX
        MS_FLOAT32X8 src_data = MS_LD256_F32(src + src_kernel_offset + c);
        MS_ST256_F32(dst + dst_kernel_offset + c * plane, src_data);
#else
        for (int k1 = 0; k1 < C8NUM; ++k1) {
          (dst + dst_kernel_offset + c * plane)[k1] = (src + src_kernel_offset + c)[k1];
        }
#endif
      }
      for (; c < channel; ++c) {
        dst[batch_oc_offset + c8_block + k * c_res + c - tmp * C8NUM] = src[src_kernel_offset + c];
      }
    }
  }
}

void RowMajor2ColMajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  for (int r = row_start; r < row_end; ++r) {
    for (int c = 0; c < col; ++c) {
      dst_ptr[c * row + r] = src_ptr[r * col + c];
    }
  }
}

void RowMajor2RowMajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  if (row_end > row_start) {
    src_ptr += row_start * col;
    dst_ptr += row_start * col;
    memcpy(dst_ptr, src_ptr, (row_end - row_start) * col * (int)(sizeof(float)));
  }
}

void RowMajor2Row4MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  for (int r = row_start; r < row_end; r++) {
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

void RowMajor2Row6MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  for (int r = row_start; r < row_end; r++) {
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

void RowMajor2Row8MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  for (int r = row_start; r < row_end; r++) {
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

void RowMajor2Row12MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  for (int r = row_start; r < row_end; r++) {
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

void RowMajor2Row16MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  for (int r = row_start; r < row_end; r++) {
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

void RowMajor2Row32MajorParallel(const float *src_ptr, float *dst_ptr, int col, int row, int col_start, int col_end) {
  // Not exactly aligned to 32, but aligned to 24 or 16 or 8 If 32 is not met.
  int row_block_num = UP_DIV(row, C8NUM);
  int row_block = C4NUM;
  for (int i = 0; i < row_block_num; i += row_block) {
    row_block = MSMIN(C4NUM, row_block_num - i);  // max_tile = 4
    int row_remainder = MSMIN(row_block * C8NUM, row - i * C8NUM);
    dst_ptr += col_start * row_block * C8NUM;
    for (int oc = col_start; oc < col_end; ++oc) {
      memcpy(dst_ptr, src_ptr + oc * row + i * C8NUM, row_remainder * sizeof(float));
      dst_ptr += row_block * C8NUM;
    }
    dst_ptr += (col - col_end) * row_block * C8NUM;
  }
}

void RowMajor2Row64MajorParallel(const float *src_ptr, float *dst_ptr, int col, int row, int col_start, int col_end) {
  // Not exactly aligned to 32, but aligned to 24 or 16 or 8 If 32 is not met.
  int row_block_num = UP_DIV(row, C16NUM);
  int row_block = C4NUM;
  for (int i = 0; i < row_block_num; i += row_block) {
    row_block = MSMIN(C4NUM, row_block_num - i);  // max_tile = 4
    int row_remainder = MSMIN(row_block * C16NUM, row - i * C16NUM);
    dst_ptr += col_start * row_block * C16NUM;
    for (int oc = col_start; oc < col_end; ++oc) {
      memcpy(dst_ptr, src_ptr + oc * row + i * C16NUM, row_remainder * sizeof(float));
      dst_ptr += row_block * C16NUM;
    }
    dst_ptr += (col - col_end) * row_block * C16NUM;
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
void RowMajor2Col12MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  int ri = (row_start / C12NUM * C12NUM);
  float *dst_r = dst_ptr + ri * col;
  const float *src_r = src_ptr + ri * col;
  for (; ri < (row_end / C12NUM * C12NUM); ri += C12NUM) {
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
      src_c += C4NUM * col;
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
      src_c += C4NUM * col;
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
      src_c += C4NUM * col;
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
  if (row_end == row) {
    for (; ri < row_end; ri++, dst_r++, src_r += col) {
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
void RowMajor2Col8MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  int row8 = row_end / C8NUM * C8NUM;
#ifdef ENABLE_ARM64
  int col_skip = col / C8NUM * C8NUM;
  int skip_size = C8NUM;
#else
  int col_skip = col / C4NUM * C4NUM;
  int skip_size = C4NUM;
#endif
  int ri = (row_start / C8NUM * C8NUM);
  const float *src_r = src_ptr + ri * col;
  float *dst_r = dst_ptr + ri * col;

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
      src_c += C4NUM * col;
      __m128 src12L = _mm_unpacklo_ps(src1, src2);  // x5
      __m128 src12H = _mm_unpackhi_ps(src1, src2);  // x1
      __m128 src34L = _mm_unpacklo_ps(src3, src4);  // x
      __m128 src34H = _mm_unpackhi_ps(src3, src4);
      _mm_storeu_ps(dst_c, _mm_movelh_ps(src12L, src34L));
      _mm_storeu_ps(dst_c + C8NUM, _mm_movehl_ps(src34L, src12L));
      _mm_storeu_ps(dst_c + C16NUM, _mm_movelh_ps(src12H, src34H));
      _mm_storeu_ps(dst_c + C24NUM, _mm_movehl_ps(src34H, src12H));

      __m128 src5 = _mm_loadu_ps(src_c);
      __m128 src6 = _mm_loadu_ps(src_c + col);
      __m128 src7 = _mm_loadu_ps(src_c + 2 * col);
      __m128 src8 = _mm_loadu_ps(src_c + 3 * col);
      src_c += C4NUM * col;
      __m128 src56L = _mm_unpacklo_ps(src5, src6);
      __m128 src56H = _mm_unpackhi_ps(src5, src6);
      __m128 src78L = _mm_unpacklo_ps(src7, src8);
      __m128 src78H = _mm_unpackhi_ps(src7, src8);
      _mm_storeu_ps(dst_c + C4NUM, _mm_movelh_ps(src56L, src78L));
      _mm_storeu_ps(dst_c + C12NUM, _mm_movehl_ps(src78L, src56L));
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
  if (row_end == row) {
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
}

void RowMajor2Col16MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  int row16 = row_end / C16NUM * C16NUM;
  int ri = row_start / C16NUM * C16NUM;
  int col8 = col / C8NUM * C8NUM;
  const float *src_r = src_ptr + ri * col;
  float *dst_r = dst_ptr + ri * col;

  for (; ri < row16; ri += C16NUM) {
    int ci = 0;
    for (; ci < col8; ci += C8NUM) {
      const float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C16NUM;
#ifdef ENABLE_AVX
      Transpose8X8Fp32Avx(src_c, dst_c, col, C16NUM);
      Transpose8X8Fp32Avx(src_c + C8NUM * col, dst_c + C8NUM, col, C16NUM);
#else
      for (int tr = 0; tr < C16NUM; tr++) {
        for (int tc = 0; tc < C8NUM; tc++) {
          dst_c[tc * C16NUM + tr] = src_c[tr * col + tc];
        }
      }
#endif
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
  if (row_end == row) {
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
}

void RowMajor2Col32MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  // Not exactly aligned to 32, but aligned to 24 or 16 or 8 If 32 is not met.
#ifdef ENABLE_AVX
  int col8 = col / C8NUM * C8NUM;
#endif
  int all_block_num = UP_DIV(row, C8NUM);
  int cur_block = C4NUM;
  row_start = UP_DIV(row_start, C8NUM);
  row_end = UP_DIV(row_end, C8NUM);
  for (int i = UP_ROUND(row_start, C4NUM); i < row_end; i += cur_block) {
    cur_block = MSMIN(C4NUM, all_block_num - i);  // max_tile = 4
    int dst_stride = cur_block * C8NUM;
    int row_num = MSMIN(dst_stride, row - i * C8NUM);
#ifdef ENABLE_AVX
    int row8_num = row_num / C8NUM * C8NUM;
#endif
    const float *src = src_ptr + i * C8NUM * col;
    float *dst = dst_ptr + i * C8NUM * col;
    int r = 0;
#ifdef ENABLE_AVX
    for (; r < row8_num; r += C8NUM) {
      int c = 0;
      for (; c < col8; c += C8NUM) {
        Transpose8X8Fp32Avx(src + r * col + c, dst + c * dst_stride + r, col, dst_stride);
      }
      for (; c < col; ++c) {
        for (int k = 0; k < C8NUM; ++k) {
          dst[c * dst_stride + r + k] = src[r * col + c + k * col];
        }
      }
    }
#endif
    for (; r < row_num; r++) {
      for (int c = 0; c < col; ++c) {
        dst[c * dst_stride + r] = src[r * col + c];
      }
    }
  }
}

void RowMajor2Col64MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  // Not exactly aligned to 64, but aligned to 48 or 32 or 16 If 64 is not met.
  int all_block_num = UP_DIV(row, C16NUM);
  int cur_block = C4NUM;
  row_start = UP_DIV(row_start, C16NUM);
  row_end = UP_DIV(row_end, C16NUM);
  for (int i = UP_ROUND(row_start, C4NUM); i < row_end; i += cur_block) {
    cur_block = MSMIN(C4NUM, all_block_num - i);  // max_tile = 4
    int dst_stride = cur_block * C16NUM;
    int row_num = MSMIN(dst_stride, row - i * C16NUM);
    const float *src = src_ptr + i * C16NUM * col;
    float *dst = dst_ptr + i * C16NUM * col;
    int r = 0;
    for (; r < row_num; r++) {
      for (int c = 0; c < col; ++c) {
        dst[c * dst_stride + r] = src[r * col + c];
      }
    }
  }
}

void RowMajor2Col6MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  int row6 = row_end / C6NUM * C6NUM;
  int ri = row_start / C6NUM * C6NUM;
  int col8 = col / C8NUM * C8NUM;
  const float *src_r = src_ptr + ri * col;
  float *dst_r = dst_ptr + ri * col;

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

  if (row_end == row) {
    for (; ri < row_end; ri++) {
      for (int i = 0; i < col; i++) {
        dst_r[i * C6NUM] = src_r[i];
      }
      src_r += col;
      dst_r += 1;
    }

    int totalRow = UP_ROUND(row, C6NUM);
    for (; ri < totalRow; ri++) {
      for (int i = 0; i < col; i++) {
        dst_r[i * C6NUM] = 0;
      }
      dst_r += 1;
    }
  }
}

void RowMajor2Col4MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end) {
  int row4 = row_end / C4NUM * C4NUM;
  int ri = row_start / C4NUM * C4NUM;
  int col4 = col / C4NUM * C4NUM;
  const float *src_r = src_ptr + ri * col;
  float *dst_r = dst_ptr + ri * col;

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
      src_c += C4NUM * col;
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
  if (row_end == row) {
    for (; ri < row; ri++) {
      for (int i = 0; i < col; i++) {
        dst_r[i * C4NUM] = src_r[i];
      }
      src_r += col;
      dst_r += 1;
    }

    int total_row = UP_ROUND(row, C4NUM);
    for (; ri < total_row; ri++) {
      for (int i = 0; i < col; i++) {
        dst_r[i * C4NUM] = 0;
      }
      dst_r += 1;
    }
  }
}

void RowMajor2ColMajor(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2ColMajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2RowMajor(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2RowMajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Row4Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Row4MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Row6Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Row6MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Row8Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Row8MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Row12Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Row12MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Row16Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Row16MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Row32Major(const float *src_ptr, float *dst_ptr, int col, int row) {
  RowMajor2Row32MajorParallel(src_ptr, dst_ptr, col, row, 0, col);
}
void RowMajor2Row64Major(const float *src_ptr, float *dst_ptr, int col, int row) {
  RowMajor2Row64MajorParallel(src_ptr, dst_ptr, col, row, 0, col);
}
void RowMajor2Col12Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Col12MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Col8Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Col8MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Col16Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Col16MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Col32Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Col32MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Col64Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Col64MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Col6Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Col6MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}
void RowMajor2Col4Major(const float *src_ptr, float *dst_ptr, int row, int col) {
  RowMajor2Col4MajorParallel(src_ptr, dst_ptr, row, col, 0, row);
}

void PackNHWCToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  int c4_minus = c4 - 1;
  for (int b = 0; b < batch; b++) {
    int src_oc_offset = b * plane * channel;
    int dst_oc_offset = b * plane * c4 * C4NUM;
    for (int k = 0; k < plane; k++) {
      int src_kernel_offset = src_oc_offset + k * channel;
      int dst_kernel_offset = dst_oc_offset + k * C4NUM;
      for (int j = 0; j < c4_minus; ++j) {
        int src_ic_offset = src_kernel_offset + j * C4NUM;
        int dst_ic_offset = dst_kernel_offset + j * plane * C4NUM;
#ifdef ENABLE_ARM
        vst1q_f32((float *)dst + dst_ic_offset, vld1q_f32((float *)src + src_ic_offset));
#else
        for (int i = 0; i < C4NUM; ++i) {
          ((float *)dst + dst_ic_offset)[i] = ((float *)src + src_ic_offset)[i];
        }
#endif
      }
      int tmp_c = c4_minus * C4NUM;
      int tmp_c_offset = tmp_c * plane;
      int res_c = channel - tmp_c;
      if (res_c > channel) {
        return;
      }
      for (int l = 0; l < res_c; ++l) {
        int src_ic_offset = src_kernel_offset + tmp_c + l;
        int dst_ic_offset = dst_kernel_offset + tmp_c_offset + l;
        ((float *)dst + dst_ic_offset)[0] = ((float *)src + src_ic_offset)[0];
      }
    }
  }
}

void PackNCHWToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * channel;
    int dst_offset = b * plane * c4 * C4NUM;
    RowMajor2Col4Major((const float *)src + src_offset, (float *)dst + dst_offset, channel, plane);
  }
}

void PackNHWCToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int oc_block = UP_DIV(channel, C4NUM);
  int oc_block_channel = oc_block * C4NUM;
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    for (int b = 0; b < batch; b++) {
      int dst_batch_offset = b * oc_block_channel * plane;
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        float *dst_per_plane = (float *)dst + dst_batch_offset + i * oc_block_channel;
        memcpy(dst_per_plane, (float *)src + batch_offset + i * channel, channel * sizeof(float));
        memset(dst_per_plane + channel, 0, (oc_block_channel - channel) * sizeof(float));
      }
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float);
    memcpy((float *)dst, (float *)src, ori_input_size);
  }
}

void PackNHWCToNHWCXFp32(const void *src, void *dst, int batch, int plane, int channel, int oc_tile) {
  int oc_block = UP_DIV(channel, oc_tile);
  int oc_block_channel = oc_block * oc_tile;
  int ic_remainder_ = channel % oc_tile;
  if (ic_remainder_ != 0) {
    for (int b = 0; b < batch; b++) {
      int dst_batch_offset = b * oc_block_channel * plane;
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        float *dst_per_plane = (float *)dst + dst_batch_offset + i * oc_block_channel;
        memcpy(dst_per_plane, (float *)src + batch_offset + i * channel, channel * sizeof(float));
        memset(dst_per_plane + channel, 0, (oc_block_channel - channel) * sizeof(float));
      }
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float);
    memcpy((float *)dst, (float *)src, ori_input_size);
  }
}

#ifdef ENABLE_AVX
// PackNHWCToNXHWCXFp32 is SWPackNHWCToNXHWCXFp32 asm optimize
void PackNHWCToNXHWCXFp32(int kernel_h, int kernel_w, int output_channel, int oc_block_num, int input_channel,
                          float *tmp_weight, const float *src) {
  // pack weight NHWC to N32HWC32 N24HWC24 N16HWC16 N8HWC8
  // output_channel: batch
  int ic8 = DOWN_ROUND(input_channel, C8NUM);
  int oc_block8 = DOWN_DIV(output_channel, C8NUM);
  int oc_block = 0;
  int oc = 0;
  int oc_remainder_step = 0;
  if (oc_block8 != oc_block_num) {
    oc_block8 = oc_block8 / C4NUM * C4NUM;
    oc_remainder_step = (oc_block_num - oc_block8) * C8NUM;
  }
  int plane = kernel_w * kernel_h;
  if (plane == 1) {  // conv 1x1 weight pack
    for (; oc < oc_block8; oc += (oc_block / C8NUM)) {
      oc_block = MSMIN(C4NUM, oc_block8 - oc) * C8NUM;  // max_tile = 32 ==> 24 ==> 16 ==> 8
      for (int oc_tmp = 0; oc_tmp < oc_block; oc_tmp += C8NUM) {
        int ic = 0;
        for (; ic < ic8; ic += C8NUM) {
          Transpose8X8Fp32Avx(src + ic, tmp_weight + ic * oc_block + oc_tmp, input_channel, oc_block);
        }
        for (; ic < input_channel; ++ic) {
          for (int j = 0; j < C8NUM; ++j) {
            tmp_weight[ic * oc_block + oc_tmp + j] = src[ic + input_channel * j];
          }
        }
        src += C8NUM * input_channel;
      }
      tmp_weight += oc_block * input_channel;
    }
    oc = output_channel - oc_block8 * C8NUM;
    for (int oc_remainder = 0; oc_remainder < oc; ++oc_remainder) {
      for (int ic = 0; ic < input_channel; ++ic) {
        tmp_weight[oc_remainder + oc_remainder_step * ic] = src[ic + oc_remainder * input_channel];
      }
    }
    return;
  }

  for (; oc < oc_block8; oc += (oc_block / C8NUM)) {
    oc_block = MSMIN(C4NUM, oc_block8 - oc) * C8NUM;  // max_tile = 32 ==> 24 ==> 16 ==> 8
    for (int oc_tmp = 0; oc_tmp < oc_block; oc_tmp += C8NUM) {
      for (int hw = 0; hw < plane; ++hw) {
        int ic = 0;
        for (; ic < ic8; ic += C8NUM) {
          Transpose8X8Fp32Avx(src + hw * input_channel + ic,
                              tmp_weight + hw * oc_block * input_channel + ic * oc_block + oc_tmp,
                              input_channel * plane, oc_block);
        }
        for (; ic < input_channel; ++ic) {
          for (int j = 0; j < C8NUM; ++j) {
            tmp_weight[ic * oc_block + oc_tmp + j + hw * oc_block * input_channel] =
              src[ic + input_channel * j * plane + hw * input_channel];
          }
        }
      }
      src += C8NUM * plane * input_channel;
    }
    tmp_weight += oc_block * input_channel * plane;
  }
  oc = output_channel - oc_block8 * C8NUM;
  for (int oc_remainder = 0; oc_remainder < oc; ++oc_remainder) {
    for (int hw = 0; hw < plane; ++hw) {
      for (int ic = 0; ic < input_channel; ++ic) {
        tmp_weight[oc_remainder + oc_remainder_step * ic + hw * input_channel * oc_remainder_step] =
          src[ic + (oc_remainder * plane + hw) * input_channel];
      }
    }
  }
}

#ifdef ENABLE_DEBUG
void SWPackNHWCToNXHWCXFp32(int kernel_h, int kernel_w, int output_channel, int oc_block_num, int input_channel,
                            float *tmp_weight, const float *src) {
  // pack weight NHWC to N32HWC32 N24HWC24 N16HWC16 N8HWC8
  int oc_block = 0;
  for (int i = 0; i < oc_block_num; i += oc_block) {
    oc_block = MSMIN(C4NUM, oc_block_num - i);  // max_tile = 4
    int index = i * C8NUM * kernel_h * kernel_w * input_channel;
    int oc_remainder = MSMIN(C8NUM * oc_block, output_channel - i * C8NUM);
    for (int h = 0; h < kernel_h; ++h) {
      for (int w = 0; w < kernel_w; ++w) {
        int w_index = (h * kernel_w + w) * input_channel + index;
        for (int ic = 0; ic < input_channel; ++ic) {
          int ic_index = ic + w_index;
          for (int oc = 0; oc < oc_remainder; ++oc) {
            int oc_index = oc * kernel_w * kernel_h * input_channel + ic_index;
            tmp_weight[oc] = src[oc_index];
          }
          tmp_weight += oc_block * C8NUM;
        }
      }
    }
  }
}
#endif
#endif

void PackNHWCToNHWC8Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c8 = UP_DIV(channel, C8NUM);
  int c8_channel = c8 * C8NUM;
  int nhwc8_batch_unit_offset = c8 * C8NUM * plane;
  int ic_remainder_ = channel % C8NUM;
  if (ic_remainder_ != 0) {
    int nhwc8_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; i++) {
        float *dst_per_plane = (float *)dst + nhwc8_batch_offset + i * c8_channel;
        memcpy(dst_per_plane, (float *)src + batch_offset + i * channel, channel * sizeof(float));
        for (int j = channel; j < c8_channel; ++j) {
          dst_per_plane[j] = 0;
        }
      }
      nhwc8_batch_offset += nhwc8_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float);
    memcpy((float *)dst, (float *)src, ori_input_size);
  }
}

void PackNHWCXToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel, int cx_num) {
  int c_algin = UP_DIV(channel, cx_num);
  int ic_remainder_ = channel % cx_num;
  if (ic_remainder_ != 0) {
    int nhwc_batch_unit_offset = channel * plane;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * c_algin * cx_num * plane;
      for (int i = 0; i < plane; i++) {
        memcpy((float *)dst + b * nhwc_batch_unit_offset + i * channel,
               (float *)src + batch_offset + i * c_algin * cx_num, channel * sizeof(float));
      }
    }
  } else {
    size_t ori_input_size = batch * plane * channel * sizeof(float);
    memcpy((float *)dst, (float *)src, ori_input_size);
  }
}

void PackNC4HW4ToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * c4 * C4NUM;
    int dst_offset = b * plane * channel;
    for (int c = 0; c < channel; c++) {
      int c4_block_num = c / C4NUM;
      int c4_block_res = c % C4NUM;
      int src_c_offset = src_offset + c4_block_num * plane * C4NUM + c4_block_res;
      int dst_c_offset = dst_offset + c4_block_num * C4NUM + c4_block_res;
      for (int k = 0; k < plane; k++) {
        int src_kernel_offset = src_c_offset + k * C4NUM;
        int dst_kernel_offset = dst_c_offset + k * c4 * C4NUM;
        ((float *)dst + dst_kernel_offset)[0] = ((float *)src + src_kernel_offset)[0];
      }
    }
  }
}

void UnPackC4Uint(const void *src, void *dst, size_t plane, size_t channel) {
  const float *fp32_src = (const float *)src;
  float *fp32_dst = (float *)dst;
  for (size_t c = 0; c < channel; c++) {
    size_t c_div = c / C4NUM;
    size_t c_mod = c % C4NUM;
    for (size_t p = 0; p < plane; p++) {
      int src_offset = c_div * plane * C4NUM + p * C4NUM + c_mod;
      int dst_offset = c * plane + p;
      fp32_dst[dst_offset] = fp32_src[src_offset];
    }
  }
}

void PackNC4HW4ToNCHWFp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_ROUND(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * c4;
    int dst_offset = b * plane * channel;
    UnPackC4Uint((const float *)src + src_offset, (float *)dst + dst_offset, plane, channel);
  }
}

void PackNC4HW4ToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * c4 * C4NUM;
    int dst_offset = b * plane * channel;
    for (int k = 0; k < plane; k++) {
      int src_kernel_offset = src_offset + k * C4NUM;
      int dst_kernel_offset = dst_offset + k * channel;
      for (int c = 0; c < c4 - 1; c++) {
        int src_c_offset = src_kernel_offset + c * plane * C4NUM;
        int dst_c_offset = dst_kernel_offset + c * C4NUM;
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
        MS_STQ_F32((float *)dst + dst_c_offset, MS_LDQ_F32((float *)src + src_c_offset));
#else
        ((float *)dst + dst_c_offset)[0] = ((float *)src + src_c_offset)[0];
        ((float *)dst + dst_c_offset)[1] = ((float *)src + src_c_offset)[1];
        ((float *)dst + dst_c_offset)[2] = ((float *)src + src_c_offset)[2];
        ((float *)dst + dst_c_offset)[3] = ((float *)src + src_c_offset)[3];
#endif
      }
      // res part
      int res_c = channel - (c4 - 1) * C4NUM;
      for (int i = 0; i < res_c; i++) {
        int src_res_c_offset = src_kernel_offset + (c4 - 1) * C4NUM * plane + i;
        int dst_res_c_offset = dst_kernel_offset + (c4 - 1) * C4NUM + i;
        ((float *)dst + dst_res_c_offset)[0] = ((float *)src + src_res_c_offset)[0];
      }
    }
  }
}

void PackNC8HW8AlignedToNC8HW8NotAlignedFp32(const void *src, void *dst, const int batch, const int plane,
                                             const int channel) {
  int down_channel_8 = DOWN_ROUND(channel, C8NUM);
  int up_channel_16 = UP_ROUND(channel, C16NUM);
  size_t dst_batch_offset = (size_t)(plane * channel) * sizeof(float);
  size_t src_batch_offset = (size_t)(plane * up_channel_16) * sizeof(float);
  size_t unaligned_channel_size = (size_t)(channel - down_channel_8) * sizeof(float);
  size_t aligned_channel_size = (size_t)(down_channel_8 * plane) * sizeof(float);
  size_t src_p_offset = C8NUM * sizeof(float);
  for (size_t b = 0; b < (size_t)(batch); ++b) {
    const char *src_batch = (char *)(src) + b * src_batch_offset;
    char *dst_bacth = (char *)(dst) + b * dst_batch_offset;
    memcpy(dst_bacth, src_batch, aligned_channel_size);
    src_batch += aligned_channel_size;
    dst_bacth += aligned_channel_size;
    for (int p = 0; p < plane; ++p) {
      memcpy(dst_bacth + p * unaligned_channel_size, src_batch + p * src_p_offset, unaligned_channel_size);
    }
  }
}

void PackNHWCToC8HWN8Fp32(const void *src, void *dst, int batch, int plane, int channel) {
  int channel_up8 = UP_ROUND(channel, C8NUM);
  for (int n = 0; n < batch; n++) {
    for (int hw = 0; hw < plane; hw++) {
      int c = 0;
      for (; c < channel; c++) {
        int c8div = c / C8NUM;
        int c8mod = c % C8NUM;
        int src_index = n * plane * channel + hw * channel + c;
        int dst_index = c8div * batch * plane * C8NUM + hw * batch * C8NUM + n * C8NUM + c8mod;
        ((float *)dst)[dst_index] = ((float *)src)[src_index];
      }
      for (; c < channel_up8; c++) {
        int c8div = c / C8NUM;
        int c8mod = c % C8NUM;
        int dst_index = c8div * batch * plane * C8NUM + hw * batch * C8NUM + n * C8NUM + c8mod;
        ((float *)dst)[dst_index] = 0;
      }
    }
  }
}

void PackNHWCToCXHWNXFp32(const float *src, float *dst, int batch, int plane, int channel) {
  // pack weight NHWC to C24HWN24 (Priority 24)=>C16HWN16 (Not satisfied 24)=>C8HWN8 (Not satisfied 16)
#ifdef ENABLE_AVX
  int oc_block_num = UP_DIV(channel, C8NUM);
  int plane16 = plane / C16NUM * C16NUM;
  for (int i = 0, oc_block = 0; i < oc_block_num; i += oc_block) {
    oc_block = MSMIN(C3NUM, oc_block_num - i);
    int oc_remainder = MSMIN(C8NUM * oc_block, channel - i * C8NUM);
    int oc_remainder_c8 = oc_remainder / C8NUM * C8NUM;
    int p = 0;
    for (; p < plane16; p += C16NUM) {
      int index_plane = i * C8NUM + p * channel;
      for (int b = 0; b < batch; ++b) {
        int index_batch = index_plane + b * plane * channel;
        int oc = 0;
        int stride = oc_block * C8NUM * batch;
        for (; oc < oc_remainder_c8; oc += C8NUM) {
          const float *cur_src = src + index_batch + oc;
          float *cur_dst = dst + oc;
          MS_LOAD256X16_F32(r, cur_src, channel);
          STORE256X16_F32(cur_dst, stride, r);
        }
        for (; oc < oc_remainder; ++oc) {
          for (int k = 0; k < C16NUM; ++k) {
            dst[oc + stride * k] = src[index_batch + oc + channel * k];
          }
        }
        for (; oc < C8NUM; ++oc) {
          for (int k = 0; k < C16NUM; ++k) {
            dst[oc + stride * k] = 0;
          }
        }
        dst += oc_block * C8NUM;
      }
      dst += (C16NUM - 1) * oc_block * C8NUM * batch;
    }
    for (; p < plane; ++p) {
      int index_plane = i * C8NUM + p * channel;
      for (int b = 0; b < batch; ++b) {
        int index_batch = index_plane + b * plane * channel;
        int oc = 0;
        for (; oc < oc_remainder; ++oc) {
          dst[oc] = src[index_batch + oc];
        }
        for (; oc < C8NUM; ++oc) {
          dst[oc] = 0;
        }
        dst += oc_block * C8NUM;
      }
    }
  }
#else
  int oc_block = 0;
  int oc_block_num = UP_DIV(channel, C8NUM);
  for (int i = 0; i < oc_block_num; i += oc_block) {
    oc_block = MSMIN(C3NUM, oc_block_num - i);  // max_tile = 4
    int oc_remainder = MSMIN(C8NUM * oc_block, channel - i * C8NUM);
    for (int p = 0; p < plane; ++p) {
      int index_plane = i * C8NUM + p * channel;
      for (int b = 0; b < batch; ++b) {
        int index_batch = index_plane + b * plane * channel;
        for (int oc = 0; oc < oc_remainder; ++oc) {
          dst[oc] = src[index_batch + oc];
        }
        dst += oc_block * C8NUM;
      }
    }
  }
#endif
}

void PackDepthwiseIndirectWeightC4Fp32(const void *src, void *dst, int height, int width, int channel) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int c = 0; c < c4; c++) {
    int dst_off_c = c * C4NUM * height * width;
    for (int i = 0; i < C4NUM; i++) {
      int src_off_c = (c * C4NUM + i) * height * width;
      for (int kh = 0; kh < height; kh++) {
        int src_off_kh = src_off_c + kh * width;
        for (int kw = 0; kw < width; kw++) {
          int dst_off = dst_off_c + kw * height * C4NUM + kh * C4NUM + i;
          ((float *)dst)[dst_off] = ((float *)src)[src_off_kh + kw];
        }
      }
    }
  }
}

void PackDepthwiseIndirectWeightC8Fp32(const void *src, void *dst, int height, int width, int channel) {
  int c8 = UP_DIV(channel, C8NUM);
  for (int c = 0; c < c8; c++) {
    int dst_off_c = c * C8NUM * height * width;
    for (int i = 0; i < C8NUM; i++) {
      int src_off_c = (c * C8NUM + i) * height * width;
      for (int kh = 0; kh < height; kh++) {
        int src_off_kh = src_off_c + kh * width;
        for (int kw = 0; kw < width; kw++) {
          int dst_off = dst_off_c + kw * height * C8NUM + kh * C8NUM + i;
          ((float *)dst)[dst_off] = ((float *)src)[src_off_kh + kw];
        }
      }
    }
  }
}

void PackNHWCToNCHWFp32(const void *src, void *dst, int batches, int plane, int channel, int task_id,
                        int thread_count) {
#ifdef ENABLE_ARM64
  Transpose8X8Fp32Func Transpose8X8Fp32Func_ = Transpose8X8Fp32Arm64;
#elif defined(ENABLE_ARM32)
  Transpose8X8Fp32Func Transpose8X8Fp32Func_ = Transpose8X8Fp32Arm32;
#elif defined(ENABLE_AVX)
  Transpose8X8Fp32Func Transpose8X8Fp32Func_ = Transpose8X8Fp32Avx;
#elif defined(ENABLE_SSE) && !defined(ENABLE_AVX)
  Transpose8X8Fp32Func Transpose8X8Fp32Func_ = Transpose8X8Fp32Sse;
#endif
  int hw8 = plane / C8NUM;
  int task_start = 0;
  int task_end = plane;
  if (thread_count > 0) {
    int offset_hw = UP_DIV(hw8, thread_count) * C8NUM;
    task_start = offset_hw * task_id;
    int count = plane - task_start;
    if (count <= 0) {
      return;
    }
    task_end = (task_id + 1) == thread_count ? plane : MSMIN(plane, task_start + offset_hw);
    hw8 = task_start + ((task_end - task_start) >= offset_hw ? offset_hw : 0);
  } else {
    hw8 *= C8NUM;
  }
  int c8 = channel / C8NUM * C8NUM;
  int batch = plane * channel;
  for (int n = 0; n < batches; n++) {
    const float *src_batch = (const float *)src + n * batch;
    float *dst_batch = (float *)dst + n * batch;
    int hw = task_start;
    for (; hw < hw8; hw += C8NUM) {
      int c = 0;
      for (; c < c8; c += C8NUM) {
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;
#if defined(ENABLE_ARM64) || defined(ENABLE_AVX) || defined(ENABLE_SSE) || defined(ENABLE_ARM32)
        Transpose8X8Fp32Func_(src_ptr, dst_ptr, channel, plane);
#else
        for (int tr = 0; tr < C8NUM; tr++) {
          for (int tc = 0; tc < C8NUM; tc++) {
            dst_ptr[tc * plane + tr] = src_ptr[tr * channel + tc];
          }
        }
#endif
      }
      for (; c < channel; c++) {
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;
        for (size_t i = 0; i < C8NUM; i++) {
          dst_ptr[i] = src_ptr[i * channel];
        }
      }
    }
    for (; hw < task_end; hw++) {
      const float *src_ptr = src_batch + hw * channel;
      float *dst_ptr = dst_batch + hw;
      for (size_t i = 0; i < channel; i++) {
        dst_ptr[i * plane] = src_ptr[i];
      }
    }
  }
}

void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel, int task_id, int thread_count) {
  PackNHWCToNCHWFp32(src, dst, batch, channel, plane, task_id, thread_count);
}

#ifdef ENABLE_ARM64
inline void Transpose8X8Fp32Arm64(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride) {
  size_t srcStride = src_stride * sizeof(float);
  size_t dstStride = dst_stride * sizeof(float);
  asm volatile(
    "mov x10, %[src_ptr]\n"
    "mov x11, %[dst_ptr]\n"

    "ld1 {v0.4s, v1.4s}, [x10], %[srcStride]\n"
    "ld1 {v2.4s, v3.4s}, [x10], %[srcStride]\n"

    "zip1 v8.4s, v0.4s, v2.4s\n"
    "zip2 v9.4s, v0.4s, v2.4s\n"
    "zip1 v12.4s, v1.4s, v3.4s\n"
    "zip2 v13.4s, v1.4s, v3.4s\n"

    "ld1 {v4.4s, v5.4s}, [x10], %[srcStride]\n"
    "ld1 {v6.4s, v7.4s}, [x10], %[srcStride]\n"

    "zip1 v10.4s, v4.4s, v6.4s\n"
    "zip2 v11.4s, v4.4s, v6.4s\n"
    "zip1 v14.4s, v5.4s, v7.4s\n"
    "zip2 v15.4s, v5.4s, v7.4s\n"

    "ld1 {v0.4s, v1.4s}, [x10], %[srcStride]\n"
    "ld1 {v2.4s, v3.4s}, [x10], %[srcStride]\n"

    "trn1 v16.2d, v8.2d, v10.2d\n"
    "trn2 v18.2d, v8.2d, v10.2d\n"
    "trn1 v20.2d, v9.2d, v11.2d\n"
    "trn2 v22.2d, v9.2d, v11.2d\n"

    "ld1 {v4.4s, v5.4s}, [x10], %[srcStride]\n"
    "ld1 {v6.4s, v7.4s}, [x10], %[srcStride]\n"

    "trn1 v24.2d, v12.2d, v14.2d\n"
    "trn2 v26.2d, v12.2d, v14.2d\n"
    "trn1 v28.2d, v13.2d, v15.2d\n"
    "trn2 v30.2d, v13.2d, v15.2d\n"

    "zip1 v8.4s, v0.4s, v2.4s\n"
    "zip2 v9.4s, v0.4s, v2.4s\n"
    "zip1 v12.4s, v1.4s, v3.4s\n"
    "zip2 v13.4s, v1.4s, v3.4s\n"

    "zip1 v10.4s, v4.4s, v6.4s\n"
    "zip2 v11.4s, v4.4s, v6.4s\n"
    "zip1 v14.4s, v5.4s, v7.4s\n"
    "zip2 v15.4s, v5.4s, v7.4s\n"

    "trn1 v17.2d, v8.2d, v10.2d\n"
    "trn2 v19.2d, v8.2d, v10.2d\n"
    "trn1 v21.2d, v9.2d, v11.2d\n"
    "trn2 v23.2d, v9.2d, v11.2d\n"

    "trn1 v25.2d, v12.2d, v14.2d\n"
    "trn2 v27.2d, v12.2d, v14.2d\n"
    "trn1 v29.2d, v13.2d, v15.2d\n"
    "trn2 v31.2d, v13.2d, v15.2d\n"

    "st1 {v16.4s, v17.4s}, [x11], %[dstStride]\n"
    "st1 {v18.4s, v19.4s}, [x11], %[dstStride]\n"
    "st1 {v20.4s, v21.4s}, [x11], %[dstStride]\n"
    "st1 {v22.4s, v23.4s}, [x11], %[dstStride]\n"
    "st1 {v24.4s, v25.4s}, [x11], %[dstStride]\n"
    "st1 {v26.4s, v27.4s}, [x11], %[dstStride]\n"
    "st1 {v28.4s, v29.4s}, [x11], %[dstStride]\n"
    "st1 {v30.4s, v31.4s}, [x11], %[dstStride]\n"

    :
    : [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ srcStride ] "r"(srcStride), [ dstStride ] "r"(dstStride)
    : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
      "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
}
#endif

#ifdef ENABLE_ARM32
inline void Transpose8X8Fp32Arm32(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride) {
  size_t srcStride = src_stride * sizeof(float);
  size_t dstStride = dst_stride * sizeof(float);
  asm volatile(
    "mov r10, %[src_ptr]\n"
    "mov r12, %[dst_ptr]\n"

    "vld1.32 {q0, q1}, [r10], %[srcStride]\n"
    "vld1.32 {q2, q3}, [r10], %[srcStride]\n"

    "vtrn.32 d0, d4\n"
    "vtrn.32 d1, d5\n"
    "vtrn.32 d2, d6\n"
    "vtrn.32 d3, d7\n"

    "vld1.32 {q4, q5}, [r10], %[srcStride]\n"
    "vld1.32 {q6, q7}, [r10], %[srcStride]\n"

    "vtrn.32 d8, d12\n"
    "vtrn.32 d9, d13\n"
    "vtrn.32 d10, d14\n"
    "vtrn.32 d11, d15\n"

    "vld1.32 {q8, q9}, [r10], %[srcStride]\n"
    "vld1.32 {q10, q11}, [r10], %[srcStride]\n"

    "vswp d1, d8\n"
    "vswp d3, d10\n"
    "vswp d5, d12\n"
    "vswp d7, d14\n"

    "vtrn.32 d16, d20\n"
    "vtrn.32 d17, d21\n"
    "vtrn.32 d18, d22\n"
    "vtrn.32 d19, d23\n"

    "vld1.32 {q12, q13}, [r10], %[srcStride]\n"
    "vld1.32 {q14, q15}, [r10], %[srcStride]\n"

    "vtrn.32 d24, d28\n"
    "vtrn.32 d25, d29\n"
    "vtrn.32 d26, d30\n"
    "vtrn.32 d27, d31\n"

    "vswp d17, d24\n"
    "vswp d19, d26\n"
    "vswp d21, d28\n"
    "vswp d23, d30\n"

    "add r10, r12, #16\n"
    "vst1.32 {q0}, [r12], %[dstStride]\n"
    "vst1.32 {q8}, [r10], %[dstStride]\n"
    "vst1.32 {q2}, [r12], %[dstStride]\n"
    "vst1.32 {q10}, [r10], %[dstStride]\n"
    "vst1.32 {q4}, [r12], %[dstStride]\n"
    "vst1.32 {q12}, [r10], %[dstStride]\n"
    "vst1.32 {q6}, [r12], %[dstStride]\n"
    "vst1.32 {q14}, [r10], %[dstStride]\n"
    "vst1.32 {q1}, [r12], %[dstStride]\n"
    "vst1.32 {q9}, [r10], %[dstStride]\n"
    "vst1.32 {q3}, [r12], %[dstStride]\n"
    "vst1.32 {q11}, [r10], %[dstStride]\n"
    "vst1.32 {q5}, [r12], %[dstStride]\n"
    "vst1.32 {q13}, [r10], %[dstStride]\n"
    "vst1.32 {q7}, [r12], %[dstStride]\n"
    "vst1.32 {q15}, [r10], %[dstStride]\n"

    :
    : [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ srcStride ] "r"(srcStride), [ dstStride ] "r"(dstStride)
    : "r10", "r12", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14",
      "q15");
}
#endif

#ifdef ENABLE_AVX
inline void Transpose8X8Fp32Avx(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride) {
  MS_LOAD256X8_F32(src, src_ptr, src_stride)
  __m256 r1 = _mm256_unpacklo_ps(src1, src2);
  __m256 r2 = _mm256_unpackhi_ps(src1, src2);
  __m256 r3 = _mm256_unpacklo_ps(src3, src4);
  __m256 r4 = _mm256_unpackhi_ps(src3, src4);
  __m256 r5 = _mm256_unpacklo_ps(src5, src6);
  __m256 r6 = _mm256_unpackhi_ps(src5, src6);
  __m256 r7 = _mm256_unpacklo_ps(src7, src8);
  __m256 r8 = _mm256_unpackhi_ps(src7, src8);

  __m256 v;
  v = _mm256_shuffle_ps(r1, r3, 0x4E);
  src1 = _mm256_blend_ps(r1, v, 0xCC);
  src2 = _mm256_blend_ps(r3, v, 0x33);

  v = _mm256_shuffle_ps(r2, r4, 0x4E);
  src3 = _mm256_blend_ps(r2, v, 0xCC);
  src4 = _mm256_blend_ps(r4, v, 0x33);

  v = _mm256_shuffle_ps(r5, r7, 0x4E);
  src5 = _mm256_blend_ps(r5, v, 0xCC);
  src6 = _mm256_blend_ps(r7, v, 0x33);

  v = _mm256_shuffle_ps(r6, r8, 0x4E);
  src7 = _mm256_blend_ps(r6, v, 0xCC);
  src8 = _mm256_blend_ps(r8, v, 0x33);

  r1 = _mm256_permute2f128_ps(src1, src5, 0x20);
  r2 = _mm256_permute2f128_ps(src2, src6, 0x20);
  r3 = _mm256_permute2f128_ps(src3, src7, 0x20);
  r4 = _mm256_permute2f128_ps(src4, src8, 0x20);
  r5 = _mm256_permute2f128_ps(src1, src5, 0x31);
  r6 = _mm256_permute2f128_ps(src2, src6, 0x31);
  r7 = _mm256_permute2f128_ps(src3, src7, 0x31);
  r8 = _mm256_permute2f128_ps(src4, src8, 0x31);

  STORE256X8_F32(dst_ptr, dst_stride, r);
}
#endif

#if defined(ENABLE_SSE) && !defined(ENABLE_AVX)
inline void Transpose8X8Fp32Sse(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride) {
  __m128 v0_ma = _mm_loadu_ps(src_ptr);
  __m128 v1_ma = _mm_loadu_ps(src_ptr + src_stride);
  __m128 v2_ma = _mm_loadu_ps(src_ptr + 2 * src_stride);
  __m128 v3_ma = _mm_loadu_ps(src_ptr + 3 * src_stride);

  __m128 v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
  __m128 v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
  __m128 v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
  __m128 v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

  __m128 v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
  __m128 v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
  __m128 v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
  __m128 v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

  _mm_storeu_ps(dst_ptr, v8_ma);
  _mm_storeu_ps(dst_ptr + dst_stride, v9_ma);
  _mm_storeu_ps(dst_ptr + 2 * dst_stride, v10_ma);
  _mm_storeu_ps(dst_ptr + 3 * dst_stride, v11_ma);

  v0_ma = _mm_loadu_ps(src_ptr + C4NUM);
  v1_ma = _mm_loadu_ps(src_ptr + src_stride + C4NUM);
  v2_ma = _mm_loadu_ps(src_ptr + 2 * src_stride + C4NUM);
  v3_ma = _mm_loadu_ps(src_ptr + 3 * src_stride + C4NUM);

  v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
  v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
  v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
  v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

  v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
  v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
  v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
  v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

  _mm_storeu_ps(dst_ptr + C4NUM * dst_stride, v8_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 1) * dst_stride, v9_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 2) * dst_stride, v10_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 3) * dst_stride, v11_ma);

  v0_ma = _mm_loadu_ps(src_ptr + C4NUM * src_stride);
  v1_ma = _mm_loadu_ps(src_ptr + (C4NUM + 1) * src_stride);
  v2_ma = _mm_loadu_ps(src_ptr + (C4NUM + 2) * src_stride);
  v3_ma = _mm_loadu_ps(src_ptr + (C4NUM + 3) * src_stride);

  v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
  v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
  v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
  v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

  v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
  v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
  v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
  v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

  _mm_storeu_ps(dst_ptr + C4NUM, v8_ma);
  _mm_storeu_ps(dst_ptr + dst_stride + C4NUM, v9_ma);
  _mm_storeu_ps(dst_ptr + 2 * dst_stride + C4NUM, v10_ma);
  _mm_storeu_ps(dst_ptr + 3 * dst_stride + C4NUM, v11_ma);

  v0_ma = _mm_loadu_ps(src_ptr + C4NUM * src_stride + C4NUM);
  v1_ma = _mm_loadu_ps(src_ptr + (C4NUM + 1) * src_stride + C4NUM);
  v2_ma = _mm_loadu_ps(src_ptr + (C4NUM + 2) * src_stride + C4NUM);
  v3_ma = _mm_loadu_ps(src_ptr + (C4NUM + 3) * src_stride + C4NUM);

  v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
  v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
  v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
  v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

  v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
  v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
  v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
  v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

  _mm_storeu_ps(dst_ptr + C4NUM * dst_stride + C4NUM, v8_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 1) * dst_stride + C4NUM, v9_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 2) * dst_stride + C4NUM, v10_ma);
  _mm_storeu_ps(dst_ptr + (C4NUM + 3) * dst_stride + C4NUM, v11_ma);
}
#endif

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
void PackWeightConvDw3x3Fp32(const void *src, void *dst, int channel) {
  // nchw to nc4hw4 with 1D F(2,3)
  for (int i = 0; i < channel; i++) {
    float *src_kernel = (float *)src + i * 9;
    float *dst_kernel = (float *)dst + (i / 4) * 48 + i % 4;
    for (int y = 0; y < 3; y++) {
      float g0 = src_kernel[3 * y];
      float g1 = src_kernel[3 * y + 1];
      float g2 = src_kernel[3 * y + 2];

      dst_kernel[16 * y] = g0;
      dst_kernel[16 * y + 4] = 0.5f * (g0 + g1 + g2);
      dst_kernel[16 * y + 8] = 0.5f * (g0 - g1 + g2);
      dst_kernel[16 * y + 12] = g2;
    }
  }
}
#endif
