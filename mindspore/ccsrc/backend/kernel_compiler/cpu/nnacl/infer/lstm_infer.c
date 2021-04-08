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

#include "nnacl/infer/lstm_infer.h"
#include "nnacl/infer/infer_register.h"

int LstmInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 6, 3);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  const TensorC *weight_i = inputs[1];
  TensorC *output = outputs[0];
  for (int i = 0; i < 3; i++) {
    SetDataTypeFormat(outputs[i], input);
  }

  LstmParameter *param = (LstmParameter *)parameter;
  if (!param->op_parameter_.infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  if (input->shape_size_ != 3 || weight_i->shape_size_ != 3) {
    return NNACL_ERR;
  }

  // int hidden_size = w_shape[1] / 4;
  int hidden_size = weight_i->shape_[1] / 4;
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  ShapeSet(out_shape, &out_shape_size, input->shape_, input->shape_size_);
  out_shape[2] = hidden_size;
  if (param->bidirectional_) {
    ShapeInsert(out_shape, &out_shape_size, 1, 2);
  } else {
    ShapeInsert(out_shape, &out_shape_size, 1, 1);
  }
  SetShapeArray(output, out_shape, out_shape_size);
  int state_shape[MAX_SHAPE_SIZE];
  size_t state_shape_size = 0;
  ShapeSet(state_shape, &state_shape_size, input->shape_, input->shape_size_);
  state_shape[0] = param->bidirectional_ ? 2 : 1;
  state_shape[2] = hidden_size;
  SetShapeArray(outputs[1], state_shape, state_shape_size);
  SetShapeArray(outputs[2], state_shape, state_shape_size);

  return NNACL_OK;
}

REG_INFER(LSTM, PrimType_LSTM, LstmInferShape)
