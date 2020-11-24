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

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/pack.h"
#include "nnacl/op_base.h"
#include "nnacl/common_func.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/winograd_utils.h"
#include "nnacl/quantization/quantize.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/int8/matmul_int8.h"

#ifdef __cplusplus
extern "C" {
#endif
// int8 conv common
void ConvInt8(int8_t *input_data, int8_t *packed_input, int8_t *matmul_input, int8_t *packed_weight,
              const int32_t *bias_data, int8_t *output_data, int32_t *filter_zp, int32_t *input_sum, int task_id,
              ConvParameter *conv_param, MATMUL_OPT_R_FUNC matmul_func, bool is_optimize);

// int8 convolution 1x1
void Conv1x1PreOptPeroc(const int8_t *src_input, int8_t *packed_input, int32_t *input_sum, size_t input_channel,
                        size_t output_channel, size_t plane_size, int32_t *filter_zp, size_t inputsum_stride);
void Conv1x1PreOptPert(const int8_t *src_input, int8_t *packed_input, int32_t *input_sum, size_t input_channel,
                       size_t plane_size, ConvParameter *conv_param);
void Conv1x1Int8(const int8_t *packed_input, const int8_t *packed_weight, int8_t *dst, const int32_t *input_sum,
                 const int32_t *bias, int row, int col, int deep16, int32_t *left_shift, int32_t *right_shift,
                 int32_t *multiplier, ConvParameter *conv_param, int32_t *filter_zp);
void Conv1x1Int8Opt(const int8_t *packed_input, const int8_t *packed_weight, int8_t *dst, const int32_t *input_sum,
                    const int32_t *bias, int row, int col, int deep4, int32_t *left_shift, int32_t *right_shift,
                    int32_t *multiplier, ConvParameter *conv_param, MATMUL_OPT_DP_FUNC matmul_func, int32_t *filter_zp);

// int8 convolution 3x3
void Conv3x3Int8(int16_t *input_data, int16_t *transed_weight, const int32_t *bias_data, int8_t *output_data,
                 int16_t *tile_buffer, int16_t *block_unit_buffer, int32_t *tmp_dst_buffer, int8_t *tmp_out,
                 int task_id, ConvParameter *conv_param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_INT8_CONV_INT8_H_
