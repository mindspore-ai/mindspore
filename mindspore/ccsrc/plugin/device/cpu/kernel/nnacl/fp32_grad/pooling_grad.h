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

#ifndef MINDSPORE_NNACL_FP32_GRAD_POOLING_GRAD_H_
#define MINDSPORE_NNACL_FP32_GRAD_POOLING_GRAD_H_

#include "nnacl/fp32/pooling_fp32.h"

#ifdef __cplusplus
extern "C" {
#endif
void AvgPoolingGrad(const float *input_ptr, float *output_ptr, int count, const PoolingParameter *pooling_param);
void MaxPoolingGrad(const float *input_ptr, const float *dy_ptr, float *output_ptr, int output_batch,
                    const PoolingParameter *pooling_param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_GRAD_POOLING_GRAD_H_
