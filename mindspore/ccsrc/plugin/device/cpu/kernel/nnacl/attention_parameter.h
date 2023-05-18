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
#ifndef NNACL_ATTENTION_PARAMETER_H_
#define NNACL_ATTENTION_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct AttentionParameter {
  OpParameter op_parameter_;
  int head_num_;
  int head_size_;
  bool cross_;
} AttentionParameter;

typedef struct RelativePositionAttentionParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  // multi-head-attention args
  int num_heads_;  // number of heads of multi-head-attention
  int k_seq_;      // length of sequence of key of attention
  int v_seq_;      // length of sequence of value of attention
  bool use_bias_;  // if matmul in attention has bias
  // relative-position-attention args
  int p_seq_;  // length of sequence of position of attention
  // args for compute
  int batch_;      // batch of query/key/value/position
  int d_model_;    // d_model of multi-head-attention
  int q_seq_;      // length of sequence of query of attention
  int row_tile_;   // row tile for matrix pack
  int col_tile_;   // col tile for matrix pack
  int bias_tile_;  // tile for bias pack
} RelativePositionAttentionParameter;

#endif  // NNACL_ATTENTION_PARAMETER_H_
