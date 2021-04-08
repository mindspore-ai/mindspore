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

#include "nnacl/infer/mfcc_infer.h"
#include "nnacl/infer/infer_register.h"

int MfccInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ != 3) {
    return NNACL_ERR;
  }
  if (GetElementNum(inputs[1]) != 1) {
    return NNACL_ERR;
  }
  output->shape_size_ = 3;
  output->shape_[0] = input->shape_[0];
  output->shape_[1] = input->shape_[1];
  MfccParameter *param = (MfccParameter *)parameter;
  output->shape_[2] = param->dct_coeff_num_;
  return NNACL_OK;
}

REG_INFER(Mfcc, PrimType_Mfcc, MfccInferShape)
