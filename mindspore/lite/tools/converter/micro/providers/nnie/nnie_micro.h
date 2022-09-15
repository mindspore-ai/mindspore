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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_PROVIDERS_NNIE_NNIE_MICRO_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_PROVIDERS_NNIE_NNIE_MICRO_H_

#include "nnacl/custom_parameter.h"
#include "nnacl/tensor_c.h"

#ifdef __cplusplus
extern "C" {
#endif

int CustomKernel(TensorC *inputs, int input_num, TensorC *outputs, int output_num, CustomParameter *param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_PROVIDERS_NNIE_NNIE_MICRO_H_
