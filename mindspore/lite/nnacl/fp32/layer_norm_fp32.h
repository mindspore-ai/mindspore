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
#ifndef MINDSPORE_LITE_NNACL_FP32_LAYER_NORM_H_
#define MINDSPORE_LITE_NNACL_FP32_LAYER_NORM_H_

#include "nnacl/op_base.h"
#include "nnacl/layer_norm_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

int LayerNorm(size_t outer_size, size_t inner_size, const float *src_data, const float *gamma_data,
              const float *beta_data, enum ElementwiseMode elementwise_mode, float epsilon, float *dst_data,
              size_t task_id, size_t thread_num);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_LAYER_NORM_H_
