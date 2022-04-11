/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use tensor file except in compliance with the License.
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

#include "nnacl/tensor_c_utils.h"

int CheckAugmentNull(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     const OpParameter *parameter) {
  NNACL_CHECK_NULL_RETURN_ERR(inputs);
  NNACL_CHECK_NULL_RETURN_ERR(outputs);
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] == NULL) {
      return NNACL_NULL_PTR;
    }
  }
  for (size_t i = 0; i < outputs_size; i++) {
    if (outputs[i] == NULL) {
      return NNACL_NULL_PTR;
    }
  }
  if (parameter == NULL) {
    return NNACL_NULL_PTR;
  }
  return NNACL_OK;
}

int CheckAugmentNullSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         const OpParameter *parameter, size_t inputs_size_obj, size_t outputs_size_obj) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret == NNACL_NULL_PTR) {
    return NNACL_NULL_PTR;
  }
  if (inputs_size != inputs_size_obj || outputs_size != outputs_size_obj) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  return NNACL_OK;
}

int CheckAugmentNullSizeInputTwo(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                 size_t outputs_size, const OpParameter *parameter, size_t inputs_size_obj_0,
                                 size_t inputs_size_obj_1, size_t outputs_size_obj) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret == NNACL_NULL_PTR) {
    return NNACL_NULL_PTR;
  }
  if ((inputs_size != inputs_size_obj_0 && inputs_size != inputs_size_obj_1) || outputs_size != outputs_size_obj) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  return NNACL_OK;
}

int CheckAugmentNullInputSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              const OpParameter *parameter, size_t inputs_size_obj) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret == NNACL_NULL_PTR) {
    return NNACL_NULL_PTR;
  }
  if (inputs_size != inputs_size_obj) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  return NNACL_OK;
}

int CheckAugmentNullOutputSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                               const OpParameter *parameter, size_t outputs_size_obj) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret == NNACL_NULL_PTR) {
    return NNACL_NULL_PTR;
  }
  if (outputs_size != outputs_size_obj) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  return NNACL_OK;
}

int CheckAugmentWithMinSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                            const OpParameter *parameter, size_t inputs_size_obj, size_t outputs_size_obj) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret == NNACL_NULL_PTR) {
    return NNACL_NULL_PTR;
  }
  if (inputs_size < inputs_size_obj || outputs_size < outputs_size_obj) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  return NNACL_OK;
}

void SetShapeTensor(TensorC *dst, const TensorC *src) {
  for (size_t i = 0; i < src->shape_size_; i++) {
    dst->shape_[i] = src->shape_[i];
  }
  dst->shape_size_ = src->shape_size_;
}

void SetShapeArray(TensorC *dst, const int *src, size_t src_size) {
  for (size_t i = 0; i < src_size && i < MAX_SHAPE_SIZE; i++) {
    dst->shape_[i] = src[i];
  }
  dst->shape_size_ = src_size;
}

void SetDataTypeFormat(TensorC *dst, const TensorC *src) {
  dst->format_ = src->format_;
  dst->data_type_ = src->data_type_;
}

int GetBatch(const TensorC *tensor) {
  if (tensor->shape_size_ != 4 && tensor->shape_size_ != 2) {
    return -1;
  }
  switch (tensor->format_) {
    case Format_NHWC:
    case Format_NHWC4:
    case Format_NCHW:
    case Format_NC4HW4:
    case Format_KCHW:
    case Format_KHWC:
    case Format_NC:
    case Format_NC4:
      return tensor->shape_[0];
    case Format_HWCK:
    case Format_CHWK:
      return tensor->shape_[3];
    case Format_HWKC:
      return tensor->shape_[2];
    case Format_CKHW:
      return tensor->shape_[1];
    default:
      return -1;
  }
}
int GetHeight(const TensorC *tensor) {
  if (tensor->shape_size_ != 4 && tensor->shape_size_ != 2) {
    return -1;
  }
  switch (tensor->format_) {
    case Format_NCHW:
    case Format_KCHW:
    case Format_CKHW:
      return tensor->shape_[2];
    case Format_NHWC:
    case Format_NHWC4:
    case Format_NC4HW4:
    case Format_KHWC:
    case Format_CHWK:
      return tensor->shape_[1];
    case Format_HWCK:
    case Format_HWKC:
    case Format_HW:
    case Format_HW4:
      return tensor->shape_[0];
    default:
      return -1;
  }
}
int GetWidth(const TensorC *tensor) {
  if (tensor->shape_size_ != 4 && tensor->shape_size_ != 2) {
    return -1;
  }
  switch (tensor->format_) {
    case Format_NCHW:
    case Format_KCHW:
    case Format_CKHW:
      return tensor->shape_[3];
    case Format_KHWC:
    case Format_NHWC:
    case Format_NHWC4:
    case Format_NC4HW4:
    case Format_CHWK:
      return tensor->shape_[2];
    case Format_HWCK:
    case Format_HWKC:
    case Format_HW:
    case Format_HW4:
      return tensor->shape_[1];
    default:
      return -1;
  }
}
int GetChannel(const TensorC *tensor) {
  if (tensor->shape_size_ != 4 && tensor->shape_size_ != 2) {
    return -1;
  }
  switch (tensor->format_) {
    case Format_NCHW:
    case Format_KCHW:
    case Format_NC:
    case Format_NC4:
      return tensor->shape_[1];
    case Format_HWCK:
      return tensor->shape_[2];
    case Format_HWKC:
    case Format_NHWC:
    case Format_NHWC4:
    case Format_NC4HW4:
    case Format_KHWC:
      return tensor->shape_[3];
    case Format_CKHW:
    case Format_CHWK:
      return tensor->shape_[0];
    default:
      return -1;
  }
}

int GetElementNum(const TensorC *tensor) {
  if (tensor == NULL) {
    return -1;
  }
  if (tensor->shape_size_ == 0) {
    return 1;  // scalar mode
  }
  int res = 1;
  for (size_t i = 0; i < tensor->shape_size_; i++) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(res, tensor->shape_[i], NNACL_ERRCODE_MUL_OVERFLOW);
    res = res * tensor->shape_[i];
  }
  return res;
}
int GetDimensionSize(const TensorC *tensor, const size_t index) {
  int dim_size = -1;
  if (index < tensor->shape_size_) {
    dim_size = tensor->shape_[index];
  }
  return dim_size;
}
