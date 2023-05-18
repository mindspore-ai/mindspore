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

#ifndef NNACL_TRANSPOSE_PARAMETER_H_
#define NNACL_TRANSPOSE_PARAMETER_H_

#include "nnacl/op_base.h"

// MAX_TRANSPOSE_SERIAL_SIZE = 64 * 3 * 512 * 512
#define MAX_TRANSPOSE_SERIAL_SIZE 50331648
#define MAX_TRANSPOSE_DIM_SIZE 20
#define PERM_NUM_THREE 3
#define PERM_NUM_FOUR 4

typedef struct TransposeParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int perm_[MAX_TRANSPOSE_DIM_SIZE];
  size_t perm_size_;
  bool conjugate_;

  // shape correlative
  int strides_[MAX_TRANSPOSE_DIM_SIZE];
  int out_strides_[MAX_TRANSPOSE_DIM_SIZE];

  // other parameter
  int num_axes_;
  int data_num_;
} TransposeParameter;

#endif  // NNACL_TRANSPOSE_PARAMETER_H_
