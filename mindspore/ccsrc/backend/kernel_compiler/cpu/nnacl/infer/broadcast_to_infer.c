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

#include "nnacl/infer/broadcast_to_infer.h"
#include "nnacl/infer/infer_register.h"

int BroadcastToInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  if (inputs_size != 1 && inputs_size != 2) {
    return NNACL_ERR;
  }
  if (outputs_size != 1) {
    return NNACL_ERR;
  }

  const TensorC *input = inputs[0];
  SetDataTypeFormat(outputs[0], input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  BroadcastToParameter *param = (BroadcastToParameter *)parameter;
  int32_t dst_shape[MAX_SHAPE_SIZE];
  size_t dst_shape_size = param->shape_size_;
  for (size_t i = 0; i < dst_shape_size; i++) {
    dst_shape[i] = param->shape_[i];
  }
  for (size_t i = 0; i < dst_shape_size; ++i) {
    if (dst_shape[i] == -1) {
      dst_shape[i] = inputs[0]->shape_[i];
    }
  }
  const int *input_shape = input->shape_;
  size_t input_shape_size = input->shape_size_;
  int shape[MAX_SHAPE_SIZE];
  size_t shape_size = dst_shape_size;
  int input_shape_index = input_shape_size - 1;
  if (input_shape_size > dst_shape_size) {
    return NNACL_ERR;
  }

  for (int i = dst_shape_size - 1; i >= 0; --i) {
    if (dst_shape[i] < 0) {
      return NNACL_ERR;
    }
    if (input_shape_index >= 0) {
      int dim = input_shape[input_shape_index];
      if (dim != dst_shape[i] && dim != 1) {
        return NNACL_ERR;
      }
    }
    shape[i] = dst_shape[i];
    --input_shape_index;
  }
  SetShapeArray(outputs[0], shape, shape_size);
  return NNACL_OK;
}

REG_INFER(BroadcastTo, PrimType_BroadcastTo, BroadcastToInferShape)
