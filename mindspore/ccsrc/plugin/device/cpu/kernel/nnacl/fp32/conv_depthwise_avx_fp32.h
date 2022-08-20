/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_CONV_DEPTHWISE_AVX_H_
#define MINDSPORE_NNACL_FP32_CONV_DEPTHWISE_AVX_H_

#include "nnacl/conv_parameter.h"
#include "nnacl/base/conv_common_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int ConvDwAVX(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
              const ConvParameter *conv_param, int task_id, ConvDwCalcParam *conv_dw_calc_param_);

void ConvDwAVXFp32Row(float *output_ptr, const float *input_ptr, const float *weight_ptr, size_t num_pixels,
                      size_t output_channel, size_t input_step, bool first_calc_flag, const float *bias);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_CONV_DEPTHWISE_H_
