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

static const int no_of_recorde_values = 5;

int CheckInputShapeValid(const TensorC *const *inputs, size_t inputs_size, const LstmParameter *parameter) {
  if (inputs_size < C6NUM) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  const TensorC *input = inputs[FIRST_INPUT];
  const TensorC *weight_i = inputs[SECOND_INPUT];
  const TensorC *weight_g = inputs[THIRD_INPUT];
  const TensorC *bias = inputs[FOURTH_INPUT];
  const TensorC *hidden_init = inputs[FIFTH_INPUT];
  const TensorC *cell_init = inputs[SIXTH_INPUT];
  NNACL_CHECK_TRUE_RET(input->shape_size_ == DIMENSION_3D && weight_i->shape_size_ == DIMENSION_3D &&
                         weight_g->shape_size_ == DIMENSION_3D && bias->shape_size_ == DIMENSION_2D,
                       NNACL_ERR);
  int batch = input->shape_[kNHWC_H];
  int input_size = input->shape_[kNHWC_W];
  int hidden_size = weight_i->shape_[kNHWC_H] / C4NUM;
  int out_size = hidden_size;
  if (inputs_size == C7NUM) {
    NNACL_CHECK_TRUE_RET(inputs[SEVENTH_INPUT]->shape_size_ == DIMENSION_3D, NNACL_INPUT_TENSOR_ERROR);
    out_size = inputs[SEVENTH_INPUT]->shape_[kNHWC_H];
  }
  bool bidirectional = parameter->bidirectional_;
  int bidirection = bidirectional ? C2NUM : C1NUM;
  NNACL_CHECK_TRUE_RET(weight_i->shape_[kNHWC_N] == bidirection && weight_i->shape_[kNHWC_H] == hidden_size * C4NUM &&
                         weight_i->shape_[kNHWC_W] == input_size,
                       NNACL_ERR);
  NNACL_CHECK_TRUE_RET(weight_g->shape_[kNHWC_N] == bidirection && weight_g->shape_[kNHWC_H] == hidden_size * C4NUM &&
                         weight_g->shape_[kNHWC_W] == out_size,
                       NNACL_ERR);
  NNACL_CHECK_TRUE_RET(bias->shape_[kNHWC_N] == bidirection && bias->shape_[kNHWC_H] == hidden_size * C8NUM, NNACL_ERR);
  if (!bidirectional && hidden_init->shape_size_ == DIMENSION_2D) {
    NNACL_CHECK_TRUE_RET(hidden_init->shape_[kNHWC_N] == batch && hidden_init->shape_[kNHWC_H] == out_size, NNACL_ERR);
  } else {
    NNACL_CHECK_TRUE_RET(hidden_init->shape_size_ == DIMENSION_3D && hidden_init->shape_[kNHWC_N] == bidirection &&
                           hidden_init->shape_[kNHWC_H] == batch && hidden_init->shape_[kNHWC_W] == out_size,
                         NNACL_ERR);
  }
  if (!bidirectional && cell_init->shape_size_ == DIMENSION_2D) {
    NNACL_CHECK_TRUE_RET(cell_init->shape_[kNHWC_N] == batch && cell_init->shape_[kNHWC_H] == hidden_size, NNACL_ERR);
  } else {
    NNACL_CHECK_TRUE_RET(cell_init->shape_size_ == DIMENSION_3D && cell_init->shape_[kNHWC_N] == bidirection &&
                           cell_init->shape_[kNHWC_H] == batch && cell_init->shape_[kNHWC_W] == hidden_size,
                         NNACL_ERR);
  }
  return NNACL_OK;
}

int InferFirstOutputMindir(const TensorC *const *inputs, size_t inputs_size, TensorC *output, LstmParameter *param) {
  for (size_t i = 0; i < inputs_size; ++i) {
    if (inputs[i]->shape_size_ != C3NUM) {
      return NNACL_INPUT_TENSOR_ERROR;
    }
  }
  ShapeSet(output->shape_, &output->shape_size_, inputs[0]->shape_, inputs[0]->shape_size_);
  int out_size = inputs[SECOND_INPUT]->shape_[THIRD_INPUT];
  output->shape_[THIRD_INPUT] = (param->bidirectional_ ? C2NUM : 1) * out_size;
  return NNACL_OK;
}

int InferFirstOutputNonMindir(const TensorC *const *inputs, size_t inputs_size, TensorC *output, LstmParameter *param) {
  if (CheckInputShapeValid(inputs, inputs_size, param) != NNACL_OK) {
    return NNACL_ERR;
  }
  ShapeSet(output->shape_, &output->shape_size_, inputs[0]->shape_, inputs[0]->shape_size_);
  const TensorC *hidden_init = inputs[FIFTH_INPUT];
  int out_size = hidden_init->shape_[hidden_init->shape_size_ - 1];
  output->shape_[THIRD_INPUT] = out_size;
  int direction = param->bidirectional_ ? C2NUM : C1NUM;
  int ret = ShapeInsert(output->shape_, &output->shape_size_, 1, direction);
  return ret;
}

int LstmInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 4, 3);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  for (int i = 0; i < outputs_size; i++) {
    SetDataTypeFormat(outputs[i], input);
  }

  LstmParameter *param = (LstmParameter *)parameter;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  int hidden_size = 0;
  int out_size = 0;
  if (inputs_size == C4NUM) {
    int ret = InferFirstOutputMindir(inputs, inputs_size, output, param);
    if (ret != NNACL_OK) {
      return ret;
    }
    hidden_size = inputs[THIRD_INPUT]->shape_[THIRD_INPUT];
    out_size = inputs[SECOND_INPUT]->shape_[THIRD_INPUT];
  } else {
    int ret = InferFirstOutputNonMindir(inputs, inputs_size, output, param);
    if (ret != NNACL_OK) {
      return ret;
    }
    hidden_size = inputs[SIXTH_INPUT]->shape_[inputs[SIXTH_INPUT]->shape_size_ - 1];
    out_size = inputs[FIFTH_INPUT]->shape_[inputs[FIFTH_INPUT]->shape_size_ - 1];
  }

  int dir_multiplier = param->bidirectional_ ? C2NUM : C1NUM;
  int state_shape[MAX_SHAPE_SIZE];
  size_t state_shape_size = 0;

  ShapeSet(state_shape, &state_shape_size, input->shape_, input->shape_size_);
  state_shape[FIRST_INPUT] = dir_multiplier;
  state_shape[THIRD_INPUT] = out_size;
  SetShapeArray(outputs[SECOND_INPUT], state_shape, state_shape_size);
  state_shape[THIRD_INPUT] = hidden_size;
  SetShapeArray(outputs[THIRD_INPUT], state_shape, state_shape_size);

  if (outputs_size > DIMENSION_4D) {
    int intermediate_states_shape[MAX_SHAPE_SIZE];
    const size_t intermediate_states_shape_size = 1;
    int batch_size = input->shape_[SECOND_INPUT];
    int seq_len = input->shape_[FIRST_INPUT];
    intermediate_states_shape[FIRST_INPUT] =
      batch_size * seq_len * dir_multiplier * (out_size + no_of_recorde_values * hidden_size);
    SetShapeArray(outputs[FOURTH_INPUT], intermediate_states_shape, intermediate_states_shape_size);
    SetShapeArray(outputs[FIFTH_INPUT], state_shape, state_shape_size);
  }

  return NNACL_OK;
}

REG_INFER(LSTM, PrimType_LSTM, LstmInferShape)
