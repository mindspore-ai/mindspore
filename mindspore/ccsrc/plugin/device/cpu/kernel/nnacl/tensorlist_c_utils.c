/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "nnacl/tensorlist_c_utils.h"

int MallocTensorListData(TensorListC *tensor_list, TypeIdC dtype, const vvector *tensor_shape) {
  // This function will create a new tensors_
  // Your must to set shape(param2: tensor_shape) and data_type_(tensors_data_type_ = param1: dtype) of each tensor in
  // tensors_. After that, you need to call function:MallocData to malloc data buf of each tensor in tensors_.

  if (tensor_list->element_num_ == 0) {
    return NNACL_OK;
  }
  if (((size_t)(tensor_list->element_num_)) != tensor_shape->size_) {
    return NNACL_ERR;
  }
  tensor_list->tensors_data_type_ = dtype;
  void *addr = malloc(tensor_list->element_num_ * sizeof(void *) +
                      tensor_list->element_num_ * sizeof(TensorC));  // free in infer_manager
  if (addr == NULL) {
    free(tensor_list->tensors_);
    return NNACL_NULL_PTR;
  }
  memset(addr, 0, tensor_list->element_num_ * sizeof(void *) + tensor_list->element_num_ * sizeof(TensorC));
  tensor_list->tensors_ = (TensorC **)addr;
  TensorC *tensors = (TensorC *)(tensor_list->tensors_ + tensor_list->element_num_);
  for (size_t i = 0; i < tensor_list->element_num_; ++i) {
    TensorC *tensor = tensors + i;
    tensor_list->tensors_[i] = tensor;
    tensor->format_ = Format_NHWC;
    tensor->data_type_ = dtype;
    ShapeSet(tensor->shape_, &(tensor->shape_size_), tensor_shape->shape_[i], (size_t)tensor_shape->shape_size_[i]);
  }
  return NNACL_OK;
}

int TensorListMergeShape(int *element_shape, size_t *element_shape_size, const int *tmp, size_t tmp_size) {
  if (*element_shape_size >= 255 || element_shape[0] == -1) {
    ShapeSet(element_shape, element_shape_size, tmp, tmp_size);
    return NNACL_OK;
  }
  if (*element_shape_size != tmp_size) {
    return NNACL_ERR;
  }
  for (size_t j = 0; j < tmp_size; ++j) {
    if (element_shape[j] >= 0 && tmp[j] >= 0 && element_shape[j] != tmp[j]) {
      return NNACL_ERR;
    }
    element_shape[j] = element_shape[j] >= 0 ? element_shape[j] : tmp[j];
  }
  return NNACL_OK;
}

bool TensorListIsFullyDefined(const int *shape, size_t shape_size) {
  for (size_t i = 0; i < shape_size; ++i) {
    if (shape[i] < 0) {
      return false;
    }
  }
  return true;
}

bool InferFlagTensorList(TensorC *tensorc) {
  TensorListC *input_tensor_list = (TensorListC *)tensorc;
  if (input_tensor_list->shape_value_ == -1) {
    return false;
  }
  return true;
}
