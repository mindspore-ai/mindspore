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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_CONV1X1_INIT_INT8_H_
#define MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_CONV1X1_INIT_INT8_H_

#include <stdint.h>
#include <stdbool.h>
#include "nnacl/conv_parameter.h"

int Conv1x1Init(int8_t *src_weight, int32_t *src_bias, int32_t *filter_zps, int32_t input_channel,
                int32_t output_channel, int32_t input_zp, bool support_optimize, bool filter_peroc,
                int8_t **packed_weight, int32_t **bias_data);

#endif  // MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_CONV1X1_INIT_INT8_H_
