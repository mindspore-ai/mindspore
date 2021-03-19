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
#ifndef MINDSPORE_LITE_NNACL_LSTM_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_LSTM_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct LstmParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  // shape correlative
  int input_size_;
  int hidden_size_;  // output_size
  int seq_len_;
  int batch_;
  // other parameter
  int output_step_;
  bool bidirectional_;
  float zoneout_cell_;
  float zoneout_hidden_;
  int input_row_align_;
  int input_col_align_;
  int state_row_align_;
  int state_col_align_;
} LstmParameter;

#endif  // MINDSPORE_LITE_NNACL_LSTM_PARAMETER_H_
