/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/scatter_nd_update_infer.h"
#include "nnacl/infer/infer_register.h"

int ScatterNdUpdateInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input_x = inputs[0];
  const TensorC *indices = inputs[1];
  const TensorC *updates = inputs[2];
  TensorC *output = outputs[0];
  if (updates->data_type_ != input_x->data_type_ ||
      (indices->data_type_ != kNumberTypeInt32 && indices->data_type_ != kNumberTypeInt64)) {
    return NNACL_ERR;
  }
  SetDataTypeFormat(output, input_x);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (indices->shape_size_ < 2 || indices->shape_[indices->shape_size_ - 1] > input_x->shape_size_) {
    return NNACL_ERR;
  }
  if (updates->shape_size_ !=
      (indices->shape_size_ - 1) + input_x->shape_size_ - indices->shape_[indices->shape_size_ - 1]) {
    return NNACL_ERR;
  }
  for (int i = 0; i < updates->shape_size_; i++) {
    if ((i < indices->shape_size_ - 1 && updates->shape_[i] != indices->shape_[i]) ||
        (i >= indices->shape_size_ - 1 &&
         updates->shape_[i] !=
           input_x->shape_[indices->shape_[indices->shape_size_ - 1] + i - indices->shape_size_ + 1])) {
      return NNACL_ERR;
    }
  }
  SetShapeArray(output, input_x->shape_, input_x->shape_size_);
  return NNACL_OK;
}

REG_INFER(ScatterNdUpdate, PrimType_ScatterNdUpdate, ScatterNdUpdateInferShape)
REG_INFER(TensorScatterAdd, PrimType_TensorScatterAdd, ScatterNdUpdateInferShape)
