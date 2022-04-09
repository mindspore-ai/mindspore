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

#include "nnacl/infer/all_gather_infer.h"
#include "nnacl/infer/infer_register.h"

int AllGatherInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
  if (inputs_size != 1 || outputs_size != 1) {
    return NNACL_NULL_PTR;
  }
  if (parameter == NULL || inputs[0] == NULL || outputs[0] == NULL) {
    return NNACL_NULL_PTR;
  }
  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  AllGatherParameter *param = (AllGatherParameter *)parameter;
  if (param->rank_size_ <= 0) {
    return NNACL_INFER_INVALID;
  }

  const TensorC *input_tensor = inputs[0];
  const int *in_shape = input_tensor->shape_;
  TensorC *out_tensor = outputs[0];

  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  out_shape[0] = in_shape[0] * param->rank_size_;
  out_shape_size++;
  for (int i = 1; i < input_tensor->shape_size_; i++) {
    out_shape[i] = in_shape[i];
    out_shape_size++;
  }
  SetShapeArray(out_tensor, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(AllGather, PrimType_AllGather, AllGatherInferShape)
