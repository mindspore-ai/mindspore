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

#include "nnacl/infer/sparse_reshape_infer.h"
#include "nnacl/infer/infer_register.h"

int SparseReshapeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                            OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, C3NUM, C2NUM);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *in_indices_tensor = inputs[0];
  TensorC *out_indices_tensor = outputs[0];
  SetDataTypeFormat(out_indices_tensor, in_indices_tensor);

  const TensorC *in_out_shape_tensor = inputs[C2NUM];
  TensorC *out_shape_tensor = outputs[C1NUM];
  SetDataTypeFormat(out_shape_tensor, in_out_shape_tensor);

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  SetShapeArray(out_shape_tensor, in_out_shape_tensor->shape_, in_out_shape_tensor->shape_size_);

  int out_indices_shape[MAX_SHAPE_SIZE] = {0};
  out_indices_shape[0] = in_indices_tensor->shape_[0];
  size_t out_indices_shape_size = 1;

  for (int i = 0; i < in_out_shape_tensor->shape_size_; ++i) {
    out_indices_shape[i + 1] = in_out_shape_tensor->shape_[i];
    out_indices_shape_size++;
  }
  SetShapeArray(out_indices_tensor, out_indices_shape, out_indices_shape_size);
  return NNACL_OK;
}

REG_INFER(SparseReshape, PrimType_SparseReshape, SparseReshapeInferShape)
