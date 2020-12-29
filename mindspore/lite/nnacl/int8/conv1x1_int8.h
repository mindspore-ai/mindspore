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
#ifndef MINDSPORE_LITE_NNACL_INT8_CONV1X1_INT8_H_
#define MINDSPORE_LITE_NNACL_INT8_CONV1X1_INT8_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/pack.h"
#include "nnacl/op_base.h"
#include "nnacl/common_func.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/int8/matmul_int8.h"

#ifdef __cplusplus
extern "C" {
#endif

void Conv1x1Int8(const int8_t *packed_input, const int8_t *packed_weight, int8_t *dst, const int32_t *input_sum,
                 const int32_t *bias, int row, int col, int deep16, int32_t *left_shift, int32_t *right_shift,
                 int32_t *multiplier, ConvParameter *conv_param, int32_t *filter_zp);
void Conv1x1Int8Opt(const int8_t *packed_input, const int8_t *packed_weight, int8_t *dst, const int32_t *input_sum,
                    const int32_t *bias, int row, int col, int deep4, int32_t *left_shift, int32_t *right_shift,
                    int32_t *multiplier, ConvParameter *conv_param, MATMUL_OPT_DP_FUNC matmul_func, int32_t *filter_zp);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_INT8_CONV1X1_INT8_H_
