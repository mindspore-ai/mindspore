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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_MATMUL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_MATMUL_H_

#include "nnacl/op_base.h"

typedef void (*MATMUL_OPT_R4_FUNC)(const int8_t *a, const int8_t *b, int *dst, int row_4, int col_4, int deep_16,
                                   const int *input_sum, const int *bias);

typedef void (*MAT_TRANS_FUNC)(void *dst, void *a, int row, int col);

typedef enum ActType { ActType_No, ActType_Relu, ActType_Relu6 } ActType;

typedef struct MatMulParameter {
  OpParameter op_parameter_;
  int row_;
  int col_;
  int row_8_;
  int row_16_;
  int col_8_;
  int deep_;
  bool has_bias_;
  int batch;
  bool a_transpose_; /* false :  row-major  */
  bool b_transpose_; /* true  :  col-major  */
  bool a_const_;
  bool b_const_;
  ActType act_type_;
} MatMulParameter;

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_MATMUL_H_
