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

#ifndef NNACL_FP32_LOG_SOFTMAX_FP32_H_
#define NNACL_FP32_LOG_SOFTMAX_FP32_H_

#include "nnacl/op_base.h"
#include "nnacl/softmax_parameter.h"
#ifdef __cplusplus
extern "C" {
#endif
void LogSoftmax(const float *input_ptr, float *output_ptr, float *sum_data, int32_t *input_shape, int n_dim, int axis);
void LogSoftmaxLastAxis(const float *src, float *dst, float *exp_data, int batch, int channel);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_FP32_LOG_SOFTMAX_FP32_H_
