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

#ifndef NNACL_FP32_GRAD_MAXPOOL_GRAD_GARD_H_
#define NNACL_FP32_GRAD_MAXPOOL_GRAD_GARD_H_

#include "nnacl/op_base.h"
#include "nnacl/pooling_parameter.h"
#include "nnacl/kernel/pooling.h"

#ifdef __cplusplus
extern "C" {
#endif
int MaxPoolGradGrad(const float *input, const float *grad, float *output, size_t start, size_t end,
                    PoolingParameter *param, PoolingComputeParam *args);

int MaxPool3DGradGrad(const float *input, const float *grad, float *output, size_t start, size_t end,
                      Pooling3DParameter *param, PoolingComputeParam *args);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_FP32_GRAD_MAXPOOL_GRAD_GARD_H_
