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

#include "nnacl/infer/apply_momentum_infer.h"
#include "nnacl/infer/infer_register.h"

int ApplyMomentumInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                            OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullInputSize(inputs, inputs_size, outputs, outputs_size, parameter, 5);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  if (GetElementNum(inputs[0]) != GetElementNum(inputs[1]) || GetElementNum(inputs[0]) != GetElementNum(inputs[3]) ||
      GetElementNum(inputs[2]) != 1 || GetElementNum(inputs[4]) != 1) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  if (outputs_size != 0) {
    TensorC *out = outputs[0];
    if (out == NULL) {
      return NNACL_NULL_PTR;
    }
    out->data_type_ = inputs[0]->data_type_;
    out->format_ = inputs[0]->format_;
    out->shape_size_ = 1;
    out->shape_[0] = 1;
  }

  return NNACL_OK;
}

REG_INFER(ApplyMomentum, PrimType_ApplyMomentum, ApplyMomentumInferShape)
