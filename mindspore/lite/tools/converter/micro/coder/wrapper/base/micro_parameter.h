/*
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_BASE_MICRO_PARAMETER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_BASE_MICRO_PARAMETER_H_

#include "nnacl/matmul_parameter.h"

typedef struct {
  ActType act_type_;
  int thread_num_;
  int row_;
  int col_;
  int row_4_;
  int row_6_;
  int row_12_;
  int row_16_;
  int row_align_;
  int col_4_;
  int col_8_;
  int col_align_;
  int deep_;
  int deep_4_;
  int deep_16_;
  int deep_align_;
  int a_batch_;
  int b_batch_;
  int batch;
  bool a_transpose_; /* false :  row-major  */
  bool b_transpose_; /* true  :  col-major  */
  bool a_const_;
  bool b_const_;
} MicroMatmulParameter;

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_BASE_MICRO_PARAMETER_H_
