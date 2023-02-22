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

#ifndef MINDSPORE_NNACL_FP32_MATMUL_H_
#define MINDSPORE_NNACL_FP32_MATMUL_H_

#include <float.h>
#include <string.h>
#include "nnacl/errorcode.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/fp32/matmul_avx_fp32.h"

#define ADD_BIAS(value, bias, c) \
  if (bias != NULL) value = value + bias[c];

#define DO_RELU(value, act_type) \
  if (act_type == ActType_Relu) value = MSMAX(0.0f, value);

#define DO_RELU6(value, act_type)                            \
  if (act_type == ActType_Relu6) value = MSMIN(6.0f, value); \
  if (act_type == ActType_Relu6) value = MSMAX(0.0f, value);

#ifdef __cplusplus
extern "C" {
#endif
void MatMulOpt(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row,
               int col, size_t stride, int out_type);
void MatVecMulFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col);
void MatVecMulFp32Block8(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col);
void MatVecMulFp32Block4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col);

#ifdef ENABLE_ARM64
void MatmulFloatNeon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                       int col, size_t stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatNeon64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                          int col, size_t stride, size_t write_mode);
void BigMatmulFloatNeon64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                             int row, int col, size_t stride);
void MatmulFloatNeon64OptRow8(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                              int row, int col, size_t stride, size_t write_mode);
void MatmulFloatNeon64OptRow4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                              int row, int col, size_t stride, size_t write_mode);
void MatmulFloatNeon64OptRow12(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                               int row, int col, size_t stride, size_t write_mode);
void MatVecMulPackFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col);
void MatVecMulFp32Neon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col,
                         int align_col);

#elif defined(ENABLE_ARM32)
void MatmulFloatNeon32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                       int col, int stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatNeon32Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                          int col, int stride, int write_mode);
void MatmulFloatNeon32Opt12x4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                              int row, int col, int stride, int write_mode);

#elif defined(ENABLE_SSE)
void DeconvMatmulFloatSse(const float *a, const float *b, float *c, int depth, int row, int col);
void MatmulFloatSse64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                         int col, int stride, int write_mode);
#endif

void MatMul12x8(const float *a, const float *b, float *dst, const float *bias, ActType act_type, int deep, int row,
                int col, int stride, int out_type);

void GemmIsNotPack(const float *a, const float *b, float *c, const float *bias, int row, int deep, int act_type);

void Row1Deep1GemmIsNotPack(const float *a, const float *b, float *c, const float *bias, int col, int deep,
                            int act_type);

void Row1Deep1NoBiasGemmIsNotPack(const float *a, const float *b, float *c, const float *bias, int col, int deep,
                                  int act_type);

void GemmIsNotPackOptimize(const float *a, const float *b, float *c, const float *bias, int m, int k, int act_type);

void MatVecMulNoPackFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int64_t depth,
                         int64_t cur_col, int64_t col);
#ifdef ENABLE_ARM64
void GemmIsNotPackByRow(const float *a, const float *b, float *c, const float *bias, int start_row, int end_row,
                        int deep, int act_type);
#endif
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_FP32_MATMUL_H_
