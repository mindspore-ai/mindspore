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
#include "nnacl/int8/fixed_point.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/int8/common_func_int8.h"

#ifdef __cplusplus
extern "C" {
#endif

void Conv3x3Int8FilterTransform(const int16_t *weight_data, int16_t *trans_weight, int iC8, int output_channel,
                                int kernel_plane);

void Conv3x3Int8(int16_t *input_data, int16_t *transed_weight, const int32_t *bias_data, int8_t *output_data,
                 int16_t *tile_buffer, int16_t *block_unit_buffer, int32_t *tmp_dst_buffer, int8_t *tmp_out,
                 int task_id, ConvParameter *conv_param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_INT8_CONV_INT8_H_
