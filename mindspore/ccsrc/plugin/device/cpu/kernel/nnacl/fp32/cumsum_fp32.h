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
#ifndef MINDSPORE_NNACL_FP32_CUMSUM_H_
#define MINDSPORE_NNACL_FP32_CUMSUM_H_
#include "nnacl/op_base.h"
#include "nnacl/cumsum_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
void Cumsum(const float *input, float *output, int out_dim, int axis_dim, int inner_dim, bool exclusive);
void CumsumReverse(const float *input, float *output, int out_dim, int axis_dim, int inner_dim, bool exclusive);
void CumsumInt(const int *input, int *output, int out_dim, int axis_dim, int inner_dim, bool exclusive);
void CumsumReverseInt(const int *input, int *output, int out_dim, int axis_dim, int inner_dim, bool exclusive);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_CUMSUM_H_
