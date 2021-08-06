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

int GetShapeByType(const TensorC *shape_tensor, size_t shape_size, int32_t *dst_shape) {
  if (shape_tensor == NULL || dst_shape == NULL) {
    return NNACL_ERR;
  }
  if (shape_size == 0) {
    return NNACL_INFER_INVALID;
  }
  switch (shape_tensor->data_type_) {
    case kNumberTypeInt8: {
      int8_t *data = (int8_t *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    case kNumberTypeInt32: {
      int32_t *data = (int32_t *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    case kNumberTypeInt64: {
      int64_t *data = (int64_t *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    case kNumberTypeFloat: {
      float *data = (float *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    case kNumberTypeUInt32: {
      uint32_t *data = (uint32_t *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    default: {
      return NNACL_ERR;
    }
  }
  return NNACL_OK;
}

int BroadcastToInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  int ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (ret != NNACL_OK) {
    return ret;
  }
  if (inputs_size != 1 && inputs_size != 2) {
    return NNACL_ERR;
  }
  if (outputs_size != 1) {
    return NNACL_ERR;
  }

  const TensorC *input = inputs[0];
  SetDataTypeFormat(outputs[0], input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  int32_t dst_shape[MAX_SHAPE_SIZE] = {0};
  size_t dst_shape_size;
  if (inputs_size == 1) {
    BroadcastToParameter *param = (BroadcastToParameter *)parameter;
    dst_shape_size = param->shape_size_;
    if (dst_shape_size > MAX_SHAPE_SIZE) {
      return NNACL_PARAM_INVALID;
    }
    for (size_t i = 0; i < dst_shape_size; i++) {
      dst_shape[i] = param->shape_[i];
    }
  } else {
    const TensorC *shape_tensor = inputs[1];
    dst_shape_size = GetElementNum(shape_tensor);
    if (dst_shape_size > MAX_SHAPE_SIZE) {
      return NNACL_INPUT_TENSOR_ERROR;
    }
    ret = GetShapeByType(shape_tensor, dst_shape_size, dst_shape);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  for (size_t i = 0; i < dst_shape_size; ++i) {
    if (dst_shape[i] == -1) {
      dst_shape[i] = inputs[0]->shape_[i];
    }
  }
  const int *input_shape = input->shape_;
  size_t input_shape_size = input->shape_size_;
  int shape[MAX_SHAPE_SIZE];
  int input_shape_index = (int)(input_shape_size)-1;
  if (input_shape_size > dst_shape_size) {
    return NNACL_ERR;
  }

  for (int i = (int)(dst_shape_size)-1; i >= 0; --i) {
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
  SetShapeArray(outputs[0], shape, dst_shape_size);
  return NNACL_OK;
}

REG_INFER(BroadcastTo, PrimType_BroadcastTo, BroadcastToInferShape)
