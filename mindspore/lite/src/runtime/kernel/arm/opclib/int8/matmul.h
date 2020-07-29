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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_INT8_MATMUL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_INT8_MATMUL_H_

#include "src/runtime/kernel/arm/opclib/op_base.h"
#include "src/runtime/kernel/arm/opclib/matmul.h"

#ifdef __cplusplus
extern "C" {
#endif

void MatMulInt8(const int8_t *a, const int8_t *b, int32_t *c, const int row8, const int col8, const int deep,
                const int32_t a_zp, const int32_t b_zp);
void RowMajor2Col8MajorInt8(int8_t *src_ptr, int8_t *dst_ptr, int row, int col);

void GemmRowCol8x8Major2RowMajorInt8(int8_t *src_ptr, int8_t *dst_ptr, int row, int col);
void Gemm8x8Int8(const int8_t *lhs_data, const int8_t *rhs_data, const int8_t *bias_data, int8_t *output_data,
                 int depth, FcQuantArg *params);
void GemmInt8(const int8_t *input_data, const int8_t *weights_data, const int8_t *bias_data, int8_t *output_data,
              int row_8, int col_8, int depth, FcQuantArg *params);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_INT8_MATMUL_H_

