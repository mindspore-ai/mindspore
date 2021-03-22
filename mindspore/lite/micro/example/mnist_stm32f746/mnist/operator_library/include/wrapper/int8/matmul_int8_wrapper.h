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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_MATMUL_INT8_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_MATMUL_INT8_WRAPPER_H_
#include <string.h>
#include "nnacl/int8/matmul_int8.h"
#ifdef __cplusplus
extern "C" {
#endif

void InitInt8MatrixA(int8_t *src_ptr, int32_t *input_sums, int8_t *dst_ptr, int batch, int row, int deep, int input_zp,
                     const int *weight_zp, bool a_transpose);

void InitInt8MatrixB(int8_t *weight_ptr, int32_t *weight_bias_sums_batch_, int8_t *dst_ptr, int batch, int deep,
                     int col, int col_align, int deep_16, int input_zp, int *weight_zp, const int *bias_ptr,
                     bool b_transpose, bool filter_per_channel);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_MATMUL_INT8_WRAPPER_H_
