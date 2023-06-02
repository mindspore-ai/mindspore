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

#ifndef NNACL_ARTITHMETIC_PARAMETER_H_
#define NNACL_ARTITHMETIC_PARAMETER_H_

#include "nnacl/op_base.h"
#include "nnacl/common_func.h"
#include "nnacl/nnacl_utils.h"

#define ARITHMETIC_SUPPORT_DIMS_NUM 10

typedef struct ArithmeticParameter {
  OpParameter op_parameter_;
  bool broadcasting_;
  size_t ndim_;
  int activation_type_;
  int in_shape0_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int64_t in_elements_num0_;
  int in_shape1_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int64_t in_elements_num1_;

  int out_shape_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int out_elements_num_;

  int in_strides0_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int in_strides1_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int out_strides_[ARITHMETIC_SUPPORT_DIMS_NUM];

  int multiples0_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int multiples1_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int eltwise_mode_;  // eltwise need
} ArithmeticParameter;

#endif  // NNACL_ARTITHMETIC_PARAMETER_H_
