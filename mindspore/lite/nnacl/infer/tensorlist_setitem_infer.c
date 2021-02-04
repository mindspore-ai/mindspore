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

#include "nnacl/infer/tensorlist_setitem_infer.h"

int TensorListSetItemInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                size_t outputs_size, OpParameter *parameter) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  TensorListC *input0 = (TensorListC *)(inputs[0]);
  const TensorC *get_index = inputs[1];
  const TensorC *value_tensor = inputs[2];
  TensorListC *output0 = (TensorListC *)(outputs[0]);
  output0->data_type_ = input0->data_type_;
  output0->format_ = input0->format_;

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if (get_index->data_ == NULL || value_tensor->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }

  if (get_index->data_type_ != kNumberTypeInt && get_index->data_type_ != kNumberTypeInt32) {
    return NNACL_ERR;
  }
  if (GetElementNum(get_index) != 1) {
    return NNACL_ERR;
  }
  if (get_index->data_ == NULL) {
    return NNACL_NULL_PTR;
  }
  int index = ((int *)(get_index->data_))[0];
  if (index < 0 || (index >= ((int)(input0->element_num_)) && index != 0)) {
    return NNACL_ERR;
  }

  output0->max_elements_num_ = input0->max_elements_num_;
  ShapeSet(output0->element_shape_, &(output0->element_shape_size_), input0->element_shape_,
           input0->element_shape_size_);

  vvector *out_shape = (vvector *)malloc(sizeof(vvector));
  if (out_shape == NULL) {
    return NNACL_NULL_PTR;
  }
  out_shape->size_ = 0;
  out_shape->shape_ = (int **)malloc((input0->element_num_ + 1) * sizeof(int *));
  if (out_shape->shape_ == NULL) {
    free(out_shape);
    return NNACL_NULL_PTR;
  }
  out_shape->shape_size_ = (int *)malloc((input0->element_num_ + 1) * sizeof(int));
  if (out_shape->shape_size_ == NULL) {
    free(out_shape->shape_);
    free(out_shape);
    return NNACL_NULL_PTR;
  }

  if (index == 0 && input0->element_num_ == 0) {  // uninitialized tensorlist
    out_shape->shape_[out_shape->size_] = (int *)(value_tensor->shape_);
    out_shape->shape_size_[out_shape->size_] = value_tensor->shape_size_;
    out_shape->size_++;
    output0->element_num_ = 1;  // note: maybe error
  } else {
    output0->element_num_ = input0->element_num_;  // note: maybe error
    for (int i = 0; i < input0->element_num_; ++i) {
      TensorC *src_ptr = input0->tensors_[i];
      if (src_ptr == NULL) {
        free(out_shape->shape_);
        free(out_shape->shape_size_);
        free(out_shape);
        return NNACL_ERR;
      }
      if (src_ptr->data_type_ != kTypeUnknown) {
        out_shape->shape_[out_shape->size_] = src_ptr->shape_;
        out_shape->shape_size_[out_shape->size_] = src_ptr->shape_size_;
        out_shape->size_++;
      } else {
        out_shape->shape_[out_shape->size_] = NULL;
        out_shape->shape_size_[out_shape->size_] = 0;
        out_shape->size_++;
      }
    }
  }

  out_shape->shape_[index] = (int *)(value_tensor->shape_);
  out_shape->shape_size_[index] = value_tensor->shape_size_;
  MallocTensorListData(output0, input0->tensors_data_type_, out_shape);
  free(out_shape->shape_);
  free(out_shape->shape_size_);
  free(out_shape);
  return NNACL_OK;
}
