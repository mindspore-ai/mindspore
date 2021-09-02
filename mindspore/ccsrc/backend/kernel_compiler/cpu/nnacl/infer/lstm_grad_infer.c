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

#include "nnacl/infer/infer_register.h"
#include "nnacl/infer/common_infer.h"
#include "nnacl/fp32/lstm_fp32.h"

int LstmGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                       OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 11, 4);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  const TensorC *H = inputs[1];
  const TensorC *C = inputs[2];
  const TensorC *weight = inputs[3];
  TensorC *output = outputs[0];
  for (int i = 0; i < outputs_size; i++) {
    SetDataTypeFormat(outputs[i], input);
  }

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (input->shape_size_ != 3 || weight->shape_size_ != 3) {
    return NNACL_ERR;
  }

  SetShapeArray(output, input->shape_, input->shape_size_);
  SetShapeArray(outputs[1], H->shape_, H->shape_size_);
  SetShapeArray(outputs[2], C->shape_, C->shape_size_);
  SetShapeArray(outputs[3], weight->shape_, weight->shape_size_);

  return NNACL_OK;
}

REG_INFER(LSTMGrad, PrimType_LSTMGrad, LstmGradInferShape)
