/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_NNACL_FP32_MATMUL_MASK_AVX512_H_
#define MINDSPORE_NNACL_FP32_MATMUL_MASK_AVX512_H_
#include <stdint.h>
#include <sys/types.h>
#include "nnacl/op_base.h"
typedef void (*GemmAvx512MaskKernel)(float *dst, const float *src, const float *weight, const float *bias,
                                     const size_t act_flag, const size_t row_block, const size_t col_block,
                                     const size_t deep, const size_t src_stride, const size_t dst_stride,
                                     const size_t inc_flag, const u_int16_t *mask);
#ifdef __cplusplus
extern "C" {
#endif

void GemmRowxColMaskKernelFp32(float *dst, const float *src, const float *weight, const float *bias,
                               const size_t act_flag, const size_t row_block, const size_t col_block,
                               const size_t depth, const size_t src_stride, const size_t dst_stride,
                               const size_t inc_flag, const u_int16_t *mask);

void MatVecMulMaskAvx512Fp32(const float *a, const float *b, float *c, const float *bias, const int act_type,
                             const int depth, const int cur_col, const int col_);

void MatMulMaskAvx512Fp32(const float *a, const float *b, float *c, const float *bias, const int act_type,
                          const int depth, const int cur_col, const int col_, const int row);

// 64 block
void nnacl_gemm_avx512_6x64_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_5x64_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_4x64_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_3x64_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_2x64_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_1x64_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);

// 48 block
void nnacl_gemm_avx512_8x48_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_7x48_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_6x48_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_5x48_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_4x48_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_3x48_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_2x48_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_1x48_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);

// 32 block
void nnacl_gemm_avx512_12x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                   const size_t act_flag, const size_t row_block,
                                                   const size_t col_block, const size_t depth, const size_t src_stride,
                                                   const size_t dst_stride, const size_t inc_flag,
                                                   const u_int16_t *mask);
void nnacl_gemm_avx512_11x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                   const size_t act_flag, const size_t row_block,
                                                   const size_t col_block, const size_t depth, const size_t src_stride,
                                                   const size_t dst_stride, const size_t inc_flag,
                                                   const u_int16_t *mask);
void nnacl_gemm_avx512_10x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                   const size_t act_flag, const size_t row_block,
                                                   const size_t col_block, const size_t depth, const size_t src_stride,
                                                   const size_t dst_stride, const size_t inc_flag,
                                                   const u_int16_t *mask);
void nnacl_gemm_avx512_9x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_8x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_7x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_6x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_5x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_4x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_3x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_2x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_1x32_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);

// 16 block
void nnacl_gemm_avx512_12x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                   const size_t act_flag, const size_t row_block,
                                                   const size_t col_block, const size_t depth, const size_t src_stride,
                                                   const size_t dst_stride, const size_t inc_flag,
                                                   const u_int16_t *mask);
void nnacl_gemm_avx512_11x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                   const size_t act_flag, const size_t row_block,
                                                   const size_t col_block, const size_t depth, const size_t src_stride,
                                                   const size_t dst_stride, const size_t inc_flag,
                                                   const u_int16_t *mask);
void nnacl_gemm_avx512_10x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                   const size_t act_flag, const size_t row_block,
                                                   const size_t col_block, const size_t depth, const size_t src_stride,
                                                   const size_t dst_stride, const size_t inc_flag,
                                                   const u_int16_t *mask);
void nnacl_gemm_avx512_9x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_8x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_7x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_6x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_5x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_4x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_3x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_2x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
void nnacl_gemm_avx512_1x16_mask_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                                  const size_t act_flag, const size_t row_block, const size_t col_block,
                                                  const size_t depth, const size_t src_stride, const size_t dst_stride,
                                                  const size_t inc_flag, const u_int16_t *mask);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_MATMUL_AVX512_H_
