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

#ifndef MINDSPORE_NNACL_INT8_CONV_DEPTHWISE_H_
#define MINDSPORE_NNACL_INT8_CONV_DEPTHWISE_H_

#include "nnacl/conv_parameter.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"

#ifdef __cplusplus
extern "C" {
#endif

void ConvDwInt8(int8_t *output_data, int32_t *output_row, const int8_t *input_data, const int16_t *weight_data,
                const int32_t *bias_data, const ConvParameter *conv_param, int task_id);

void ConvDw3x3Int8Pad(int8_t *output_data, const int8_t *input_data, const int16_t *weight_data,
                      const int32_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding);

void ConvDw3x3Int8(int8_t *output_data, int8_t *buffer, const int8_t *input_data, const int16_t *weight_data,
                   const int32_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                   int task_id);

void ConvDwInt8SW(int8_t *output_data, const int8_t *input_data, const int16_t *weight_data, const int32_t *bias_data,
                  const int8_t *input_zp, const int32_t *output_zp, const ConvParameter *conv_param,
                  const SlidingWindowParam *sliding, int task_id);

void DeconvDwInt8(int8_t *output_data, int32_t *output_buffer, const int16_t *input_data, const int16_t *weight_data,
                  const int32_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                  int task_id);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_INT8_CONV_DEPTHWISE_H_
