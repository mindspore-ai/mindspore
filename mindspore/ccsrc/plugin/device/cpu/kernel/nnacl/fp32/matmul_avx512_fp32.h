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
#ifndef MINDSPORE_NNACL_FP32_MATMUL_AVX512_H_
#define MINDSPORE_NNACL_FP32_MATMUL_AVX512_H_
#include "nnacl/op_base.h"
typedef void (*GemmAvx512Kernel)(float *dst, const float *src, const float *weight, const float *bias,
                                 const size_t act_flag, const size_t row_block, const size_t col_block,
                                 const size_t deep, const size_t src_stride, const size_t dst_stride,
                                 const size_t inc_flag);
#ifdef __cplusplus
extern "C" {
#endif
void MatVecMulAvx512Fp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                         int cur_col, int col_align);

void MatMulAvx512Fp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int cur_col,
                      int col_align, int row);

int64_t GemmIsNotPackOptimizeAVX512(int64_t m_index, const float *a, const float *b, float *c, const float *bias, int m,
                                    int k, int act_type);

// 64 block
void nnacl_gemm_avx512_6x64_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_5x64_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_4x64_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_3x64_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_2x64_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_1x64_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);

// 48 block
void nnacl_gemm_avx512_8x48_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_7x48_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_6x48_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_5x48_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_4x48_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_3x48_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_2x48_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_1x48_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);

// 32 block
void nnacl_gemm_avx512_12x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                              const size_t act_flag, const size_t row_block, const size_t col_block,
                                              const size_t depth, const size_t src_stride, const size_t dst_stride,
                                              const size_t inc_flag);
void nnacl_gemm_avx512_11x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                              const size_t act_flag, const size_t row_block, const size_t col_block,
                                              const size_t depth, const size_t src_stride, const size_t dst_stride,
                                              const size_t inc_flag);
void nnacl_gemm_avx512_10x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                              const size_t act_flag, const size_t row_block, const size_t col_block,
                                              const size_t depth, const size_t src_stride, const size_t dst_stride,
                                              const size_t inc_flag);
void nnacl_gemm_avx512_9x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_8x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_7x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_6x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_5x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_4x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_3x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_2x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_1x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);

// 16 block
void nnacl_gemm_avx512_12x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                              const size_t act_flag, const size_t row_block, const size_t col_block,
                                              const size_t depth, const size_t src_stride, const size_t dst_stride,
                                              const size_t inc_flag);
void nnacl_gemm_avx512_11x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                              const size_t act_flag, const size_t row_block, const size_t col_block,
                                              const size_t depth, const size_t src_stride, const size_t dst_stride,
                                              const size_t inc_flag);
void nnacl_gemm_avx512_10x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                              const size_t act_flag, const size_t row_block, const size_t col_block,
                                              const size_t depth, const size_t src_stride, const size_t dst_stride,
                                              const size_t inc_flag);
void nnacl_gemm_avx512_9x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_8x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_7x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_6x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_5x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_4x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_3x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_2x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
void nnacl_gemm_avx512_1x16_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_MATMUL_AVX512_H_
