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

#ifndef MINDSPORE_LITE_NNACL_ARTITHMETIC_H_
#define MINDSPORE_LITE_NNACL_ARTITHMETIC_H_

#include "nnacl/op_base.h"
#include "nnacl/common_func.h"
#include "nnacl/nnacl_utils.h"

typedef struct ArithmeticParameter {
  OpParameter op_parameter_;
  bool broadcasting_;
  size_t ndim_;
  int activation_type_;
  int in_shape0_[10];
  int in_elements_num0_;
  int in_shape1_[10];
  int in_elements_num1_;

  int out_shape_[10];
  int out_elements_num_;

  int in_strides0_[10];
  int in_strides1_[10];
  int out_strides_[10];

  int multiples0_[10];
  int multiples1_[10];
  int eltwise_mode_;  // eltwise need
} ArithmeticParameter;

#endif  // MINDSPORE_LITE_NNACL_ARTITHMETIC_H_
