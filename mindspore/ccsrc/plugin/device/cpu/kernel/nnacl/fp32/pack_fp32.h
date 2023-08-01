/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_NNACL_FP32_PACK_H_
#define MINDSPORE_NNACL_FP32_PACK_H_

#include <stdint.h>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
static inline void transpose_tail(const float *from, float *to, int j_start, int j_end, int i_start, int i_end,
                                  int j_stride, int i_stride) {
  // write consecutively
  for (int j = j_start; j < j_end; j++) {
    for (int i = i_start; i < i_end; i++) {
      to[j * j_stride + i] = from[i * i_stride + j];
    }
  }
}
void TransposeFp32(const void *src, void *dst, int batches, int channel, int plane, int start, int end);
void PackHWCToWHC(const float *src, float *dst, int height, int width, int channel);
void PackNHWCToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNCHWToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToNHWCXFp32(const void *src, void *dst, int batch, int plane, int channel, int oc_tile);
void PackNHWCToNHWC8Fp32(const void *src, void *dst, int batch, int plane, int channel);
// Note: If not multithreaded, please set task_id = 0 and thread_count = 0;
void PackNHWCToNCHWFp32(const void *src, void *dst, int batch, int plane, int channel, int task_id, int thread_count);
void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel, int task_id, int thread_count);
void PackNHWCXToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel, int cx_num);
void PackNC4HW4ToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNC4HW4ToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNC4HW4ToNCHWFp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNC8HW8ToNCHWFp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToNC8HW8Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNC8HW8ToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel);
void UnPackC4Uint(const void *src, void *dst, size_t plane, size_t channel);
void PackNC8HW8AlignedToNC8HW8NotAlignedFp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToC8HWN8Fp32(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToCXHWNXFp32(const float *src, float *dst, int batch, int plane, int channel);
void PackNHWCToNC4HW4NotAlignedFp32(const float *src, float *dst, const int batch, const int plane, const int channel);
void PackNHWCToNC8HW8NotAlignedFp32(const float *src, float *dst, const int batch, const int plane, const int channel);

void RowMajor2ColMajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2RowMajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Row4MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Row6MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Row8MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Row12MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Row16MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Row32MajorParallel(const float *src_ptr, float *dst_ptr, int col, int row, int col_start, int col_end);
void RowMajor2Row64MajorParallel(const float *src_ptr, float *dst_ptr, int col, int row, int col_start, int col_end);
void RowMajor2Col4MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Col6MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Col8MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Col12MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Col16MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Col32MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);
void RowMajor2Col64MajorParallel(const float *src_ptr, float *dst_ptr, int row, int col, int row_start, int row_end);

void RowMajor2ColMajor(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2RowMajor(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row4Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row6Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row8Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row12Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row16Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row32Major(const float *src_ptr, float *dst_ptr, int col, int row);
void RowMajor2Row64Major(const float *src_ptr, float *dst_ptr, int col, int row);
void RowMajor2Col4Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col6Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col8Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col12Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col16Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col32Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col64Major(const float *src_ptr, float *dst_ptr, int row, int col);

void PackWeightKHWToHWKFp32(const void *src, void *dst, int plane, int channel);
void PackDepthwiseIndirectWeightC4Fp32(const void *src, void *dst, int height, int width, int channel);
void PackDepthwiseIndirectWeightC8Fp32(const void *src, void *dst, int height, int width, int channel);

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
void PackWeightConvDw3x3Fp32(const void *src, void *dst, int channel);
#endif

// Transpose 8X8 Fp32 block data
typedef void (*Transpose8X8Fp32Func)(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride);
#ifdef ENABLE_ARM64
void Transpose8X8Fp32Arm64(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride);
#endif
#ifdef ENABLE_ARM32
void Transpose8X8Fp32Arm32(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride);
#endif
#if defined(ENABLE_AVX) || defined(ENABLE_ARM64)
void PackNHWCToNXHWCXFp32(int kernel_h, int kernel_w, int output_channel, int oc_block_num, int input_channel,
                          float *tmp_weight, const float *src);
#endif
#ifdef ENABLE_AVX
#ifdef ENABLE_DEBUG
void SWPackNHWCToNXHWCXFp32(int kernel_h, int kernel_w, int output_channel, int oc_block_num, int input_channel,
                            float *tmp_weight, const float *src);
#endif
void Transpose8X8Fp32Avx(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride);
#endif
#if defined(ENABLE_SSE) && !defined(ENABLE_AVX)
void Transpose8X8Fp32Sse(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride);
#endif

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_PAD_H_
