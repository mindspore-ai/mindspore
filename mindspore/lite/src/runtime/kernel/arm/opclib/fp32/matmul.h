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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_FP32_MATMUL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_FP32_MATMUL_H_

#include <float.h>
#include "src/runtime/kernel/arm/opclib/errorcode.h"
#include "src/runtime/kernel/arm/opclib/op_base.h"
#include "src/runtime/kernel/arm/opclib/matmul.h"

void MatMul(const float *a, const float *b, float *c, const float *bias, ActType act_type, int depth, int row, int col);
void RowMajor2Row8Major(float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col8Major(float *src_ptr, float *dst_ptr, int row, int col);
void Row8x8Major2RowMajor(float *src_ptr, float *dst_ptr, int row, int col);
void MatMul8x8(const float *a, const float *b, float *c, const float *bias, float maxf, float minf, int deep,
               int row_8_, int col_8_);
#ifdef __cplusplus
extern "C" {
#endif
#ifdef __aarch64__
void MatMulFloatNeon64(const float *a, const float *b, float *c, const float *bias, float maxf, float minf, int depth,
                       int row, int col);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_FP32_MATMUL_H_
