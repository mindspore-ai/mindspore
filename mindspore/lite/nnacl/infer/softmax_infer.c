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

#include "nnacl/infer/softmax_infer.h"
#include "nnacl/infer/infer_register.h"

int SoftMaxInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  output->data_type_ = input->data_type_;
  output->format_ = input->format_;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ > 5) {
    return NNACL_ERR;
  }
  SetShapeTensor(output, input);
  return NNACL_OK;
}

REG_INFER(Softmax, PrimType_Softmax, SoftMaxInferShape)
