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

#ifndef MINDSPORE_NNACL_BASE_CONV_DEPTHWISE_BASE_H_
#define MINDSPORE_NNACL_BASE_CONV_DEPTHWISE_BASE_H_

#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
bool CheckConvDw1DWinograd(const ConvParameter *conv_param, int thread_num);
#endif

bool CheckWinogradInputOutputUnit(int input_unit, int output_unit);

bool CheckIfUseWinograd(int *output_unit, const ConvParameter *conv_param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_BASE_CONV_DEPTHWISE_BASE_H_
