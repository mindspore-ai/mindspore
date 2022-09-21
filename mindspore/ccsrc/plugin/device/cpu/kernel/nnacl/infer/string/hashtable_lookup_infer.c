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

#include "nnacl/infer/string/hashtable_lookup_infer.h"
#include "nnacl/infer/infer_register.h"

int HashtableLoopupInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 2);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  const TensorC *values = inputs[2];
  if (input == NULL || values == NULL) {
    return NNACL_NULL_PTR;
  }

  TensorC *output = outputs[0];
  TensorC *hits = outputs[1];

  output->data_type_ = values->data_type_;
  output->format_ = input->format_;
  hits->shape_size_ = 1;
  hits->shape_[0] = GetDimensionSize(input, 0);
  hits->data_type_ = kNumberTypeUInt8;
  hits->format_ = input->format_;

  if (input->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  return NNACL_OK;
}

REG_INFER(HashtableLookup, PrimType_HashtableLookup, HashtableLoopupInferShape)
