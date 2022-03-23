/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/lstm_grad_weight_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/infer/common_infer.h"
#include "nnacl/fp32_grad/lstm_grad_fp32.h"

int LstmGradWeightInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                             OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 5, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[FIRST_INPUT];
  const TensorC *H = inputs[SECOND_INPUT];
  const TensorC *Y = inputs[THIRD_INPUT];

  TensorC *output = outputs[FIRST_INPUT];
  for (int i = 0; i < outputs_size; i++) {
    SetDataTypeFormat(outputs[i], input);
  }

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (input->shape_size_ != C3NUM || H->shape_size_ != C3NUM || Y->shape_size_ != C3NUM) {
    return NNACL_ERR;
  }
  LstmGradParameter *param = (LstmGradParameter *)parameter;
  int has_bias = param->has_bias_;
  int output_shape[3] = {0, 1, 1};
  int gate_size = 4 * param->hidden_size_;
  output_shape[0] += gate_size * param->input_size_;
  output_shape[0] += gate_size * param->hidden_size_;
  if (has_bias) {
    output_shape[0] += C2NUM * gate_size;
  }
  int dir_mul = (param->bidirectional_) ? C2NUM : C1NUM;
  output_shape[0] *= dir_mul;
  SetShapeArray(output, output_shape, C3NUM);

  return NNACL_OK;
}

REG_INFER(LSTMGradWeight, PrimType_LSTMGradWeight, LstmGradWeightInferShape)
