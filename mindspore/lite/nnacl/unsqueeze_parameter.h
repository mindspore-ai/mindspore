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

#include <string.h>
#include <math.h>
#include "nnacl/op_base.h"

#define UNSQUEEZE_OFFSET_MAX_SIZE 4

typedef struct UnSqueezeQuantArg {
  int *input_sizes_;
  int output_size_;
  int **input_shapes_;
  int *output_shape_;
  float alpha;
  int axis_;
  size_t input_num_;
  size_t output_dim_;
  QuantArg in_quant_args_;
  QuantArg out_quant_args_;
} UnSqueezeQuantArg;

typedef struct UnSqueezeParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int64_t axis_;

  // shape correlative
  const int *in_shape_;
  const int *out_shape_;
  int input_dim_;
  int64_t offset_[UNSQUEEZE_OFFSET_MAX_SIZE];
  int64_t in_offset_[UNSQUEEZE_OFFSET_MAX_SIZE];

  // other parameter
  UnSqueezeQuantArg quant_arg;
  int thread_count_;
  int thread_id_;
  int offset_size_;
} UnSqueezeParameter;

#endif  // MINDSPORE_LITE_NNACL_UNSQUEEZE_PARAMETER_H_
