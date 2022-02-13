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

#ifndef MINDSPORE_NNACL_FP16_MATMUL_H_
#define MINDSPORE_NNACL_FP16_MATMUL_H_

#include <float.h>
#include <string.h>
#include "nnacl/errorcode.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"
#include "nnacl/fp16/pack_fp16.h"

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
void MatMul16x8Fp16(const float16_t *a, const float16_t *b, float16_t *dst, const float16_t *bias, ActType act_type,
                    int deep, int row, int col, int stride, int write_mode);

void MatMul12x8Fp16(const float16_t *a, const float16_t *b, float16_t *dst, const float16_t *bias, ActType act_type,
                    int deep, int row, int col, int stride, int write_mode);

#ifdef ENABLE_ARM64
void RowMajor2ColNMajorFp16(const float16_t *src, float16_t *dst_ptr, int row, int col);

void RowMajor2RowNMajorFp16(const float16_t *src, float16_t *dst, int row, int col);

void MatMul12x16Fp16Opt(const float16_t *a, const float16_t *b, float16_t *dst, const float16_t *bias, ActType act_type,
                        int deep, int row, int col, size_t stride, size_t out_type);
void MatmulFp16Neon64(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                      size_t depth, size_t row, size_t col, size_t stride, bool write_nhwc);

void MatmulFp16Neon64Opt(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                         size_t depth, size_t row, size_t col, size_t stride, size_t write_nhwc);

void MatmulBaseFp16Neon(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                        size_t depth, size_t row, size_t col, size_t stride, size_t write_nhwc);

#ifdef ENABLE_DEBUG
void MatmulBaseFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                    size_t depth, size_t row, size_t col, size_t stride, size_t write_nhwc);
#endif

void MatVecMulFp16Neon64(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                         int depth, int col);

void VecMatmulFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type, int depth,
                   int col);
void VecMatmulFp16Neon64(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                         int depth, int col);
#elif ENABLE_ARM82_A32
void MatMul12x8A32Fp16(const float16_t *a, const float16_t *b, float16_t *dst, const float16_t *bias, ActType act_type,
                       int deep, int row, int col, int stride, int write_mode);

void MatVecMulA32Fp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                      int depth, int col);

void MatVecMulA32NeonFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                          int depth, int col);
#endif

void MatMul12x16Fp16(const float16_t *a, const float16_t *b, float16_t *dst, const float16_t *bias, ActType act_type,
                     int deep, int row, int col, size_t stride, size_t out_type);

void MatMulFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, ActType act_type,
                int depth, int row, int col, int stride, int out_type);

void MatVecMulFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, ActType act_type,
                   int depth, int col);

void ColMajor2Row8MajorFp16(const void *src_ptr, float16_t *dst_ptr, size_t row, size_t col, bool src_float16);

void RowMajor2Col16MajorFp16Opt(const float16_t *src_ptr, float16_t *dst_ptr, size_t row, size_t col);

void RowMajor2Col12MajorFp16Opt(const float16_t *src_ptr, float16_t *dst_ptr, size_t row, size_t col);

void RowMajor2Col16MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2Col12MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2Row16MajorFp16Opt(const float16_t *src, float16_t *dst, int row, int col);

void RowMajor2Row16MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2Row12MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2Row8MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2Col8MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2ColMajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_MATMUL_H_
