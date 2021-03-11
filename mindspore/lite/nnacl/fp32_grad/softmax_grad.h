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

#ifndef MINDSPORE_LITE_NNACL_FP32_GRAD_SOFTMAX_GRAD_H_
#define MINDSPORE_LITE_NNACL_FP32_GRAD_SOFTMAX_GRAD_H_

#include "nnacl/op_base.h"
#include "nnacl/fp32/softmax_fp32.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SoftmaxCrossEntropyParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int n_dim_;

  // shape correlative
  int input_shape_[5];

  // other parameter
  int32_t batch_size_;
  unsigned int number_of_classes_;
  bool is_grad_;
} SoftmaxCrossEntropyParameter;

void SoftmaxGrad(const float *input_ptr, const float *yt_ptr, float *output_ptr, float *sum_data, float *sum_mul,
                 SoftmaxParameter *parameter);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_GRAD_SOFTMAX_GRAD_H_
