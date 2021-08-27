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

int GetShapeByType(const TensorC *shape_tensor, int shape_size, int *dst_shape) {
  if (shape_tensor == NULL || dst_shape == NULL) {
    return NNACL_ERR;
  }
  if (shape_size == 0) {
    return NNACL_INFER_INVALID;
  }
  NNACL_CHECK_NULL_RETURN_ERR(shape_tensor->data_);
  switch (shape_tensor->data_type_) {
    case kNumberTypeInt8: {
      int8_t *data = (int8_t *)(shape_tensor->data_);
      for (int i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    case kNumberTypeInt32: {
      int32_t *data = (int32_t *)(shape_tensor->data_);
      for (int i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    case kNumberTypeInt64: {
      int64_t *data = (int64_t *)(shape_tensor->data_);
      for (int i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    case kNumberTypeFloat: {
      float *data = (float *)(shape_tensor->data_);
      for (int i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    case kNumberTypeUInt32: {
      uint32_t *data = (uint32_t *)(shape_tensor->data_);
      for (int i = 0; i < shape_size; i++) {
        dst_shape[i] = data[i];
      }
    } break;
    default: {
      return NNACL_ERR;
    }
  }
  return NNACL_OK;
}

void MakeUpInputShapes(const int input_shape0_size, const int input_shape1_size, const int *input_shape0,
                       const int *input_shape1, int *ndim, int *in_shape0, int *in_shape1) {
  if (input_shape0_size < input_shape1_size) {
    *ndim = input_shape1_size;
    int fill_dim_num = input_shape1_size - input_shape0_size;
    int j = 0;
    for (size_t i = 0; i < input_shape1_size; i++) {
      if (i < fill_dim_num) {
        in_shape0[i] = 1;
      } else {
        in_shape0[i] = input_shape0[j++];
      }
      in_shape1[i] = input_shape1[i];
    }
  } else if (input_shape0_size > input_shape1_size) {
    *ndim = input_shape0_size;
    int fill_dim_num = input_shape0_size - input_shape1_size;
    int j = 0;
    for (size_t i = 0; i < input_shape0_size; i++) {
      if (i < fill_dim_num) {
        in_shape1[i] = 1;
      } else {
        in_shape1[i] = input_shape1[j++];
      }
      in_shape0[i] = input_shape0[i];
    }
  } else {
    for (size_t i = 0; i < input_shape0_size; i++) {
      in_shape1[i] = input_shape1[i];
      in_shape0[i] = input_shape0[i];
    }
  }
}

int BroadCastOutputShape(const int *in_shape0, const int *in_shape1, const int ndim, int *out_shape,
                         bool *has_broad_cast) {
  for (int i = 0; i < ndim; i++) {
    if (in_shape0[i] != in_shape1[i]) {
      if (in_shape0[i] == 1) {
        out_shape[i] = in_shape1[i];
      } else if (in_shape1[i] == 1) {
        out_shape[i] = in_shape0[i];
      } else {
        return NNACL_ERR;
      }
      *has_broad_cast = true;
    } else {
      out_shape[i] = in_shape0[i];
    }
  }
  return NNACL_OK;
}

int BroadCastToShape(const int input_shape0_size, const int input_shape1_size, const int *input_shape0,
                     const int *input_shape1, int *ndim, int *out_shape, bool *has_broad_cast) {
  if (input_shape0_size > MAX_SHAPE_SIZE || input_shape1_size > MAX_SHAPE_SIZE) {
    return NNACL_ERR;
  }

  int in_shape0[MAX_SHAPE_SIZE] = {0};
  int in_shape1[MAX_SHAPE_SIZE] = {0};

  MakeUpInputShapes(input_shape0_size, input_shape1_size, input_shape0, input_shape1, ndim, in_shape0, in_shape1);
  if (*ndim >= MAX_SHAPE_SIZE) {
    return NNACL_INFER_INVALID;
  }

  return BroadCastOutputShape(in_shape0, in_shape1, *ndim, out_shape, has_broad_cast);
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
  int dst_shape[MAX_SHAPE_SIZE] = {0};
  int dst_shape_size;
  const int *input_shape = input->shape_;
  int input_shape_size = input->shape_size_;
  int output_shape[MAX_SHAPE_SIZE] = {0};
  int ndim = input_shape_size;
  bool has_broad_cast = false;
  if (inputs_size == 1) {
    BroadcastToParameter *param = (BroadcastToParameter *)parameter;
    dst_shape_size = param->shape_size_;
    if (dst_shape_size > MAX_SHAPE_SIZE) {
      return NNACL_PARAM_INVALID;
    }
    for (int i = 0; i < dst_shape_size; i++) {
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
    for (size_t i = 0; i < dst_shape_size; ++i) {
      if (dst_shape[i] == -1) {
        dst_shape[i] = inputs[0]->shape_[i];
      }
    }

    if (BroadCastToShape(input_shape_size, dst_shape_size, input_shape, dst_shape, &ndim, output_shape,
                         &has_broad_cast) != NNACL_OK) {
      return NNACL_ERR;
    }
  }

  SetShapeArray(outputs[0], output_shape, ndim);
  return NNACL_OK;
}

REG_INFER(BroadcastTo, PrimType_BroadcastTo, BroadcastToInferShape)
