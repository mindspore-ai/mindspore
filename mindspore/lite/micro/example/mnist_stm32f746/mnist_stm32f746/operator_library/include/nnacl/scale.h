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

#ifndef MINDSPORE_LITE_NNACL_SCALE_H_
#define MINDSPORE_LITE_NNACL_SCALE_H_

#include "nnacl/op_base.h"

typedef struct ScaleParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int axis_;
  int activation_type_;

  // shape correlative
  int outer_size_;
  int axis_size_;
  int inner_size_;

  // other parameter
  bool const_scale_;
  bool const_offset_;
  QuantMulArg scale_mul_arg_;
  QuantMulArg offset_mul_arg_;
  int input_zp_;
  int scale_zp_;
  int offset_zp_;
  int output_zp_;
  int output_activation_min_;
  int output_activation_max_;
} ScaleParameter;

#endif  // MINDSPORE_LITE_NNACL_SCALE_H_
