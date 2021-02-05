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

#ifndef MINDSPORE_LITE_NNACL_FP16_MATMUL_H_
#define MINDSPORE_LITE_NNACL_FP16_MATMUL_H_

#include <float.h>
#include <string.h>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/errorcode.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
void MatMul16x8(const float16_t *a, const float16_t *b, float16_t *dst, const float16_t *bias, ActType act_type,
                int deep, int row, int col, int stride, bool write_nhwc);

void MatMulFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, ActType act_type,
                int depth, int row, int col, int stride, int out_type);

void MatVecMulFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, ActType act_type,
                   int depth, int col);

void ColMajor2Row8MajorFp16(const void *src_ptr, float16_t *dst_ptr, size_t row, size_t col, bool src_float16);

void RowMajor2Col16MajorFp16Opt(const float16_t *src_ptr, float16_t *dst_ptr, size_t row, size_t col);

void MatmulFp16Neon64(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                      size_t depth, size_t row, size_t col, size_t stride, bool write_nhwc);

void MatmulFp16Neon64Opt(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                         size_t depth, size_t row, size_t col, size_t stride, size_t write_nhwc);

void MatVecMulFp16Neon64(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
                         int depth, int col);

void RowMajor2Col16MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2Row16MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2Row8MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2Col8MajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

void RowMajor2ColMajorFp16(const void *src, float16_t *dst, int row, int col, bool is_fp32_src);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_MATMUL_H_
