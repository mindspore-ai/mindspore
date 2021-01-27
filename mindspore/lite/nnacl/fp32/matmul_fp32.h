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

#ifndef MINDSPORE_LITE_NNACL_FP32_MATMUL_H_
#define MINDSPORE_LITE_NNACL_FP32_MATMUL_H_

#include <float.h>
#include <string.h>
#include "nnacl/errorcode.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/op_base.h"

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

void RowMajor2ColMajor(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row4Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row6Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row8Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row12Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row16Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col4Major(const float *src_ptr, float *dst_ptr, size_t row, size_t col);
void RowMajor2Col6Major(const float *src_ptr, float *dst_ptr, size_t row, size_t col);
void RowMajor2Col8Major(const float *src_ptr, float *dst_ptr, size_t row, size_t col);
void RowMajor2Col12Major(const float *src_ptr, float *dst_ptr, size_t row, size_t col);
void RowMajor2Col16Major(const float *src_ptr, float *dst_ptr, size_t row, size_t col);

#ifdef ENABLE_ARM64
void MatmulFloatNeon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                       int col, size_t stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatNeon64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                          int col, size_t stride, size_t write_mode);
#elif ENABLE_ARM32
void MatmulFloatNeon32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                       int col, int stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatNeon32Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                          int col, int stride, int write_mode);
void MatmulFloatNeon32Opt12x4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                              int row, int col, int stride, int write_mode);
#elif ENABLE_SSE
#include <x86intrin.h>
void MatmulFloatSse64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                      int col, int stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatSse64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                         int col, int stride, int write_mode);
#ifdef ENABLE_AVX
void MatmulFloatAvxOpt(const float *a, const float *b, float *c, const float *bias, size_t act_type, size_t depth,
                       size_t row, size_t col, size_t stride, size_t write_mode);
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_MATMUL_H_
