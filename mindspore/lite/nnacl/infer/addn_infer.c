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

#include "nnacl/infer/addn_infer.h"
#include "nnacl/infer/infer_register.h"

int AddnInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  if (inputs_size < 2) {
    return NNACL_ERR;
  }
  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  size_t max_dims = input->shape_size_;
  size_t max_dims_idx = 0;

  // determine max_dims
  for (size_t i = 1; i < inputs_size; ++i) {
    if (inputs[i]->shape_size_ > max_dims) {
      max_dims = inputs[i]->shape_size_;
      max_dims_idx = i;
    }
  }
  ShapeSet(output->shape_, &output->shape_size_, inputs[max_dims_idx]->shape_, inputs[max_dims_idx]->shape_size_);

  // make sure all elements have the same size or 1 (broadcasting) in all dimensions
  for (size_t i = 1; i < inputs_size; ++i) {
    if ((inputs[i]->shape_size_ != max_dims) && (GetElementNum(inputs[i]) != GetElementNum(inputs[max_dims_idx]))) {
      return NNACL_ERR;
    }
    if (inputs[i]->data_type_ != inputs[0]->data_type_) {
      return NNACL_ERR;
    }
  }

  for (size_t d = 0; d < input->shape_size_; ++d) {
    size_t max_dim = 0;
    for (size_t i = 0; i < inputs_size; ++i) {
      size_t shift = max_dims - inputs[i]->shape_size_;
      size_t dim = (i < shift) ? 1 : inputs[i]->shape_[d];
      if (dim > max_dim) {
        max_dim = dim;
      }
    }
    output->shape_[d] = max_dim;  // set the biggest dimension in the output tensor
  }

  return NNACL_OK;
}

REG_INFER(AddN, PrimType_AddN, AddnInferShape)
