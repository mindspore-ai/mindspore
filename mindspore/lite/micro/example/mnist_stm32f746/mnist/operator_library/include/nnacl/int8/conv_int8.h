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
#ifndef MINDSPORE_LITE_NNACL_INT8_CONV_INT8_H_
#define MINDSPORE_LITE_NNACL_INT8_CONV_INT8_H_

#include <string.h>
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
#include "nnacl/int8/common_func_int8.h"

#ifdef __cplusplus
extern "C" {
#endif
// int8 conv common
void ConvInt8(int8_t *input_data, int8_t *packed_input, int8_t *matmul_input, int8_t *packed_weight,
              const int32_t *bias_data, int8_t *output_data, int32_t *filter_zp, int32_t *input_sum, int task_id,
              ConvParameter *conv_param, MATMUL_OPT_R_FUNC matmul_func, bool is_optimize);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_INT8_CONV_INT8_H_
