/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef NNACL_FP32_CONV_SW_ARM64_FP32_H_
#define NNACL_FP32_CONV_SW_ARM64_FP32_H_
#include "nnacl/pack.h"
#include "nnacl/op_base.h"
#include "nnacl/common_func.h"
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
bool CheckArm64UseSWConv(const ConvParameter *conv_param);
void ConvSWArm64Fp32(const float *input_data, const float *packed_weight, const float *bias_data, float *output_data,
                     int task_id, ConvParameter *conv_param, SlidingWindowParam *sw_param);
#ifdef __cplusplus
}
#endif
#endif  // NNACL_FP32_CONV_SW_ARM64_FP32_H_
