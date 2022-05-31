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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_INT8_CONV_INIT_INT8_WRAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_INT8_CONV_INIT_INT8_WRAPPER_H_

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
int ConvInit(int8_t *origin_weight, const int32_t *ori_bias, const int32_t *filter_quant_zps, int kernel_h,
             int kernel_w, int input_channel, int output_channel, int32_t input_zp, bool filter_peroc,
             bool support_optimize, int8_t **packed_weight, int32_t **bias_data, uint8_t *buf, size_t *offset,
             size_t buf_size);
size_t ConvPackWeightSize(int input_channel, int output_channel, int kernel_plane, bool support_optimize);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_INT8_CONV_INIT_INT8_WRAPPER_H_
