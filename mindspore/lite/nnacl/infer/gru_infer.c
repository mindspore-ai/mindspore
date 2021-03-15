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

#include "nnacl/infer/gru_infer.h"
#include "nnacl/infer/infer_register.h"

int GruInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                  OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSizeInputTwo(inputs, inputs_size, outputs, outputs_size, parameter, 5, 6, 2);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  const TensorC *weight_gate = inputs[1];
  const TensorC *weight_recurrence = inputs[2];
  const TensorC *bias = inputs[3];
  TensorC *output = outputs[0];
  for (int i = 0; i < 2; i++) {
    SetDataTypeFormat(outputs[i], input);
  }
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  const int *in_shape = input->shape_;                  // seq_len, batch, input_size
  const int *w_gate_shape = weight_gate->shape_;        // num_direction, hidden_size * 3, input_size
  const int *w_recu_shape = weight_recurrence->shape_;  // num_direction, hidden_size * 3, hidden_size
  const int *bias_shape = bias->shape_;                 // num_direction, hidden_size * 6
  if (input->shape_size_ != 3 || weight_gate->shape_size_ != 3 || weight_recurrence->shape_size_ != 3) {
    return NNACL_ERR;
  }
  if (w_gate_shape[1] != w_recu_shape[1] || w_recu_shape[1] * 2 != bias_shape[1]) {
    return NNACL_ERR;
  }
  if (inputs_size == 6) {
    const int *seq_len_shape = inputs[5]->shape_;
    if (seq_len_shape[0] > 1) {
      return NNACL_ERR;
    }
    if (inputs[5]->shape_size_ != 1 && seq_len_shape[0] != in_shape[1]) {
      return NNACL_ERR;
    }
  }

  int hidden_size = w_gate_shape[1] / 3;
  // set output
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  ShapeSet(out_shape, &out_shape_size, in_shape, input->shape_size_);
  out_shape[2] = hidden_size;

  GruParameter *param = (GruParameter *)parameter;
  if (param->bidirectional_) {
    ShapeInsert(out_shape, &out_shape_size, 1, 2);
  } else {
    ShapeInsert(out_shape, &out_shape_size, 1, 1);
  }
  SetShapeArray(output, out_shape, out_shape_size);
  // set hidden state
  int state_shape[MAX_SHAPE_SIZE];
  size_t state_shape_size = 0;
  ShapeSet(state_shape, &state_shape_size, in_shape, input->shape_size_);
  state_shape[0] = param->bidirectional_ ? 2 : 1;
  state_shape[2] = hidden_size;
  SetShapeArray(outputs[1], state_shape, state_shape_size);
  return NNACL_OK;
}

REG_INFER(GRU, PrimType_GRU, GruInferShape)
