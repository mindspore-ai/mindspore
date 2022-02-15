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

#include "nnacl/infer/lstm_grad_data_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/infer/common_infer.h"
#include "nnacl/fp32/lstm_fp32.h"

int LstmGradDataInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 9, 3);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *dY = inputs[SECOND_INPUT];
  const TensorC *H = inputs[THIRD_INPUT];
  const TensorC *C = inputs[FOURTH_INPUT];
  const TensorC *weight = inputs[FIFTH_INPUT];
  TensorC *dX = outputs[FIRST_INPUT];
  for (int i = 0; i < outputs_size; i++) {
    SetDataTypeFormat(outputs[i], dY);
  }

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (dY->shape_size_ != C3NUM || weight->shape_size_ != C3NUM) {
    return NNACL_ERR;
  }

  SetShapeArray(dX, dY->shape_, dY->shape_size_);
  SetShapeArray(outputs[SECOND_INPUT], H->shape_, H->shape_size_);
  SetShapeArray(outputs[THIRD_INPUT], C->shape_, C->shape_size_);

  return NNACL_OK;
}

REG_INFER(LSTMGradData, PrimType_LSTMGradData, LstmGradDataInferShape)
