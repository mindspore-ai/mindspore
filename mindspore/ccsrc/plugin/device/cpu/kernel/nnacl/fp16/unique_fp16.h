/**
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
#ifndef NNACL_FP16_UNIQUE_FP16_H
#define NNACL_FP16_UNIQUE_FP16_H

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
void UniqueFp16(const float16_t *input, int input_len, float16_t *output0, int *output0_len, int *output1);
#ifdef __cplusplus
}
#endif

#endif  //  NNACL_FP16_UNIQUE_FP16_H
