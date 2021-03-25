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

#ifndef MINDSPORE_LITE_NNACL_SLICE_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_SLICE_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct SliceQuantArg {
  QuantArg in_args_;
  QuantArg out_args_;
  int output_activation_min_;
  int output_activation_max_;
} SliceQuantArg;

typedef struct SliceParameter {
  // primitive parameter
  OpParameter op_parameter_;

  // shape correlative
  int32_t shape_[COMM_SHAPE_SIZE];
  int32_t begin_[COMM_SHAPE_SIZE];
  int32_t end_[COMM_SHAPE_SIZE];
  int32_t size_[COMM_SHAPE_SIZE];
  int32_t axis_[COMM_SHAPE_SIZE];

  // other parameter
  SliceQuantArg quant_arg_;
  int32_t param_length_;
} SliceParameter;

#endif  // MINDSPORE_LITE_NNACL_SLICE_PARAMETER_H_
