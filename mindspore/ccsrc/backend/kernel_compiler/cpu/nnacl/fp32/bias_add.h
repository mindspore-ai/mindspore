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

#ifndef MINDSPORE_NNACL_FP32_BIAS_ADD_H_
#define MINDSPORE_NNACL_FP32_BIAS_ADD_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void BiasAddOpt(const float *input, const float *bias, float *output, int64_t start, int64_t end, int64_t inner_num,
                bool batch_priority);

#ifdef __cplusplus
};
#endif

#endif  // MINDSPORE_NNACL_FP32_BIAS_ADD_H_
