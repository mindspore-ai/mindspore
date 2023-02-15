/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/adam_weight_decay_infer.h"
#include "nnacl/infer/infer_register.h"

int AdamWeightDecayInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
  const size_t expected_inputs_size = 9;
  const int var_idx = 0;
  const int m_idx = 1;
  const int v_idx = 2;
  const int lr_idx = 3;
  const int beta1_idx = 4;
  const int beta2_idx = 5;
  const int epsilon = 6;
  const int decay_idx = 7;
  const int grad_idx = 8;
  int check_ret =
    CheckAugmentNullInputSize(inputs, inputs_size, outputs, outputs_size, parameter, expected_inputs_size);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  if (GetElementNum(inputs[var_idx]) != GetElementNum(inputs[m_idx]) ||
      GetElementNum(inputs[var_idx]) != GetElementNum(inputs[v_idx]) ||
      GetElementNum(inputs[var_idx]) != GetElementNum(inputs[grad_idx]) || GetElementNum(inputs[lr_idx]) != 1 ||
      GetElementNum(inputs[beta1_idx]) != 1 || GetElementNum(inputs[beta2_idx]) != 1 ||
      GetElementNum(inputs[epsilon]) != 1 || GetElementNum(inputs[decay_idx]) != 1) {
    return NNACL_ERR;
  }
  if (outputs_size != 0) {
    TensorC *out = outputs[0];
    SetDataTypeFormat(out, inputs[0]);
    out->shape_size_ = 1;
    out->shape_[0] = 1;
  }
  return NNACL_OK;
}

REG_INFER(AdamWeightDecay, PrimType_AdamWeightDecay, AdamWeightDecayInferShape)
