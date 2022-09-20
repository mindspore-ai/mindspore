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
#ifndef MINDSPORE_NNACL_RMDPROP_FP32_H
#define MINDSPORE_NNACL_RMDPROP_FP32_H

#include <math.h>
#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
int RMSPropUnuseCenterFp32(float *variable, float *mean_square, float *moment, float *gradients, float momentum,
                           float learning_rate, float decay, float epsilon, size_t start, size_t end);

int RMSPropUseCenterFp32(float *variable, float *mean_square, float *moment, float *gradients, float *mean_gradients,
                         float momentum, float learning_rate, float decay, float epsilon, size_t start, size_t end);

#ifdef __cplusplus
}
#endif
#endif  //  MINDSPORE_NNACL_RMDPROP_FP32_H
