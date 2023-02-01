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

#include "nnacl/infer/reshape_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/op_base.h"

int CalShape(const int *data, const TensorC *const *inputs, int *out_shape, size_t *out_shape_size, int shape_size) {
  int input_count = GetElementNum(inputs[0]);
  int index = 0;
  int size = 1;
  for (int i = 0; i < shape_size; i++) {
    if ((int)(data[i]) == -1) {
      index = i;
    } else if ((int)(data[i]) == 0) {
      size *= inputs[0]->shape_[i];
    } else {
      size *= data[i];
    }
    ShapePush(out_shape, out_shape_size, data[i]);
  }

  if ((int)(data[index]) == -1) {
    if (index >= MAX_SHAPE_SIZE) {
      return NNACL_ERR;
    }
    out_shape[index] = size == 0 ? 0 : input_count / size;
  }
  return NNACL_OK;
}

int CalNewShape(const TensorC *in_tensor, int *out_shape, size_t out_shape_size) {
  int in_shape_size = 1;
  for (size_t i = 0; i < in_tensor->shape_size_; i++) {
    in_shape_size *= in_tensor->shape_[i];
  }
  int64_t infer_index = -1;
  int out_shape_size_new = 1;
  for (size_t i = 0; i < out_shape_size; i++) {
    if (out_shape[i] == -1) {
      if (infer_index == -1) {
        infer_index = (int64_t)(i);
      } else {
        return NNACL_ERR;
      }
    } else if (out_shape[i] < 0) {
      return NNACL_ERR;
    } else if (out_shape[i] == 0) {
      if (GetElementNum(in_tensor) != 0) {
        out_shape[i] = in_tensor->shape_[i];
        out_shape_size_new *= out_shape[i];
      } else {
        out_shape_size_new = 0;
        break;
      }
    } else {
      out_shape_size_new *= out_shape[i];
    }
  }
  if (infer_index == -1 && out_shape_size_new != in_shape_size) {
    return NNACL_ERR;
  }
  if (infer_index != -1) {
    if (out_shape_size_new == 0) {
      return NNACL_ERR;
    }
    if (infer_index >= MAX_SHAPE_SIZE) {
      return NNACL_ERR;
    }
    out_shape[infer_index] = in_shape_size / out_shape_size_new;
  }
  return NNACL_OK;
}

int CalShapeByType(const TensorC *const *inputs, size_t shape_size, int *out_shape, size_t *out_shape_size) {
  const TensorC *shape_tensor = inputs[1];
  if (shape_size == 0) {
    return NNACL_ERR;
  }
  MS_CHECK_FALSE(INT_MUL_OVERFLOW((sizeof(int)), shape_size), NNACL_ERR);
  int *data_int = (int *)malloc(sizeof(int) * shape_size);
  if (data_int == NULL) {
    return NNACL_ERR;
  }
  switch (shape_tensor->data_type_) {
    case kNumberTypeInt8: {
      int8_t *data = (int8_t *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        data_int[i] = data[i];
      }
      int cal_ret = CalShape(data_int, inputs, out_shape, out_shape_size, shape_size);
      if (cal_ret != NNACL_OK) {
        free(data_int);
        return NNACL_ERR;
      }
    } break;
    case kNumberTypeInt32: {
      int32_t *data = (int32_t *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        data_int[i] = data[i];
      }
      int cal_ret = CalShape(data_int, inputs, out_shape, out_shape_size, shape_size);
      if (cal_ret != NNACL_OK) {
        free(data_int);
        return NNACL_ERR;
      }
    } break;
    case kNumberTypeInt64: {
      int64_t *data = (int64_t *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        data_int[i] = data[i];
      }
      int cal_ret = CalShape(data_int, inputs, out_shape, out_shape_size, shape_size);
      if (cal_ret != NNACL_OK) {
        free(data_int);
        return NNACL_ERR;
      }
    } break;
    case kNumberTypeFloat: {
      float *data = (float *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        data_int[i] = data[i];
      }
      int cal_ret = CalShape(data_int, inputs, out_shape, out_shape_size, shape_size);
      if (cal_ret != NNACL_OK) {
        free(data_int);
        return NNACL_ERR;
      }
    } break;
    case kNumberTypeUInt32: {
      uint32_t *data = (uint32_t *)(shape_tensor->data_);
      for (size_t i = 0; i < shape_size; i++) {
        data_int[i] = (int)data[i];
      }
      int cal_ret = CalShape(data_int, inputs, out_shape, out_shape_size, shape_size);
      if (cal_ret != NNACL_OK) {
        free(data_int);
        return NNACL_ERR;
      }
    } break;
    default: {
      free(data_int);
      return NNACL_ERR;
    }
  }
  free(data_int);
  return NNACL_OK;
}

int ReshapeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  ReshapeParameter *param = (ReshapeParameter *)parameter;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  int out_shape[MAX_SHAPE_SIZE] = {0};
  size_t out_shape_size = 0;
  if (inputs_size == 2) {
    const TensorC *shape_tensor = inputs[1];
    if (GetElementNum(input) == 1 && input->shape_size_ == 0) {
      if (shape_tensor->data_ == NULL || (shape_tensor->shape_size_ == 1 && shape_tensor->shape_[0] == 0)) {
        SetShapeArray(output, out_shape, out_shape_size);
        return NNACL_OK;
      }
    }

    if (shape_tensor->data_ == NULL) {
      return NNACL_INFER_INVALID;
    }
    int shape_size = GetElementNum(shape_tensor);
    if (shape_size > MAX_SHAPE_SIZE) {
      return NNACL_ERR;
    }
    int calRet = CalShapeByType(inputs, shape_size, out_shape, &out_shape_size);
    if (calRet != NNACL_OK) {
      return calRet;
    }
  } else if (inputs_size == 1) {
    if (param->shape_dim_ > MAX_SHAPE_SIZE) {
      return NNACL_PARAM_INVALID;
    }
    for (int i = 0; i < param->shape_dim_; ++i) {
      ShapePush(out_shape, &out_shape_size, param->shape_[i]);
    }
  } else {
    return NNACL_ERR;
  }
  int ret = CalNewShape(inputs[0], out_shape, out_shape_size);
  if (ret != NNACL_OK) {
    return ret;
  }
  SetShapeArray(output, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(Reshape, PrimType_Reshape, ReshapeInferShape)
