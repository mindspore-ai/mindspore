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

#ifndef MINDSPORE_NNACL_FP32_MATMUL_AVX_H_
#define MINDSPORE_NNACL_FP32_MATMUL_AVX_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*DeconvAvxKernel)(const float *src, const float *weight, float *dst, int col, int row, int depth,
                                int stride);
void DeconvMatmulAvx(const float *a, const float *b, float *c, int depth, int row, int col, int kernel_plane);
void MatmulFloatAvxOpt(const float *a, const float *b, float *c, const float *bias, size_t act_type, size_t depth,
                       size_t row, size_t col, size_t stride, size_t write_mode);
typedef void (*MatVecMulKernel)(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                                size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatVecMulAvxFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int cur_col,
                      int col_align);
void MatMulAvxFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int cur_col,
                   int col_align, int row);
void MatVecMul1x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatVecMul1x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatVecMul1x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatVecMul1x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                        size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatMul3x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                      size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatMul4x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                      size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatMul6x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                      size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatMul8x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                     size_t row_block, size_t col_block, size_t col_algin, size_t deep);
#ifdef ENABLE_DEBUG
void DeconvColXRowAvxKernel(const float *src, const float *weight, float *dst, int col, int row, int depth, int stride);

void MatVecMulRowxColKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t row_block, size_t col_block, size_t col_algin, size_t deep);
#endif

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_FP32_MATMUL_H_
