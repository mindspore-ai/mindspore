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

#ifndef MINDSPORE_LITE_NNACL_MUL_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_MUL_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct MulQuantArg {
  QuantArg in_quant_args_[2];
  QuantArg out_quant_arg_;
  int output_multiplier_;
  int output_activation_min_;
  int output_activation_max_;
  int shift_left_;
  int shift_right_;
} MulQuantArg;

typedef struct MulParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  // other parameter
  int thread_count_;
  MulQuantArg mul_quant_arg_;
} MulParameter;

#endif  // MINDSPORE_LITE_NNACL_MUL_PARAMETER_H_
