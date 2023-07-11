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

#ifndef NNACL_SLICE_PARAMETER_H_
#define NNACL_SLICE_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct SliceQuantArg {
  QuantArg in_args_;
  QuantArg out_args_;
  int output_activation_min_;
  int output_activation_max_;
  QuantMulArg multiplier_;
} SliceQuantArg;

typedef struct SliceParameter {
  OpParameter op_parameter_;
  int32_t axis_[DIMENSION_8D];
} SliceParameter;

#endif  // NNACL_SLICE_PARAMETER_H_
