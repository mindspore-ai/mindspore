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

#ifndef MINDSPORE_LITE_NNACL_UNSQUEEZE_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_UNSQUEEZE_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct UnSqueezeQuantArg {
  int *output_shape_;
  float alpha;
  int axis_;
  size_t input_num_;
  QuantArg in_quant_args_;
  QuantArg out_quant_args_;
} UnSqueezeQuantArg;

typedef struct UnSqueezeParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int dims_[COMM_SHAPE_SIZE];
  int num_dim_;

  // shape correlative
  const int *in_shape_;
  const int *out_shape_;
  int64_t offset_[COMM_SHAPE_SIZE];
  int64_t axis_;

  // other parameter
  UnSqueezeQuantArg quant_arg;
  int thread_count_;
} UnSqueezeParameter;

#endif  // MINDSPORE_LITE_NNACL_UNSQUEEZE_PARAMETER_H_
