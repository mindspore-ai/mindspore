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
#include "nnacl/fp32_grad/lstm_grad_fp32.h"

int LstmGradDataInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 9, 3);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  LstmGradParameter *p = (LstmGradParameter *)parameter;
  const TensorC *Y = inputs[SECOND_INPUT];
  const TensorC *H = inputs[THIRD_INPUT];
  const TensorC *C = inputs[FOURTH_INPUT];
  const TensorC *weight = inputs[FIFTH_INPUT];

  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;

  for (int i = 0; i < outputs_size; i++) {
    SetDataTypeFormat(outputs[i], Y);
  }

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (Y->shape_size_ != C3NUM || weight->shape_size_ != C3NUM) {
    return NNACL_ERR;
  }
  ShapePush(out_shape, &out_shape_size, Y->shape_[out_shape_size]);
  ShapePush(out_shape, &out_shape_size, Y->shape_[out_shape_size]);
  ShapePush(out_shape, &out_shape_size, p->input_size_);

  SetShapeArray(outputs[FIRST_INPUT], out_shape, C3NUM);
  SetShapeArray(outputs[SECOND_INPUT], H->shape_, H->shape_size_);
  SetShapeArray(outputs[THIRD_INPUT], C->shape_, C->shape_size_);

  return NNACL_OK;
}

REG_INFER(LSTMGradData, PrimType_LSTMGradData, LstmGradDataInferShape)
