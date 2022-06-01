/*
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_INT8_CONV1X1_INIT_INT8_WRAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_INT8_CONV1X1_INIT_INT8_WRAPPER_H_

#include <stdint.h>
#include <stdbool.h>
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int Conv1x1Init(int8_t *src_weight, int32_t *src_bias, int32_t *filter_zps, int32_t input_channel,
                int32_t output_channel, int32_t input_zp, bool support_optimize, bool filter_peroc,
                int8_t **packed_weight, int32_t **bias_data, uint8_t *buf, size_t *offset, size_t buf_size);
size_t Conv1x1PackWeightSize(int32_t input_channel, int32_t output_channel, bool support_optimize);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_INT8_CONV1X1_INIT_INT8_WRAPPER_H_
