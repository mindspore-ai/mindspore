/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef NNACL_AFFINE_PARAMETER_H_
#define NNACL_AFFINE_PARAMETER_H_
#include "nnacl/op_base.h"
#include "nnacl/matmul_parameter.h"
typedef struct AffineParameter {
  OpParameter op_parameter_;
  // parameters from splice op
  int context_size_;
  int *context_;
  int output_dim_;
  // parameters from activation op
  int activation_type_;
  // parameters from matmul op
  MatMulParameter *matmul_parameter_;
} AffineParameter;
#endif  // NNACL_AFFINE_PARAMETER_H_
