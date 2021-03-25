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

#ifndef MINDSPORE_LITE_NNACL_SPLIT_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_SPLIT_PARAMETER_H_

#include "nnacl/op_base.h"

#define SPLIT_STRIDES_SIZE 32

typedef struct SplitQuantArg {
  QuantArg in_args_;
  QuantArg out_args_[20];
  int output_activation_min_;
  int output_activation_max_;
} SplitQuantArg;

typedef struct SplitParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int num_split_;
  int *split_sizes_;
  int split_dim_;

  // shape correlative
  int strides_[SPLIT_STRIDES_SIZE];

  // other parameter
  SplitQuantArg quant_arg_;
  int n_dims_;
  int split_count_;
} SplitParameter;

#endif  // MINDSPORE_LITE_NNACL_SPLIT_PARAMETER_H_
