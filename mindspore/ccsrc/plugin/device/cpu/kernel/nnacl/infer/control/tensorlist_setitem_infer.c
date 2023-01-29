/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/control/tensorlist_setitem_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/tensorlist_c_utils.h"

int PreJudge(const TensorC *get_index, TensorListC *input0, const TensorC *value_tensor) {
  if (get_index->data_ == NULL) {
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
  return NNACL_OK;
}

int TensorListSetItemInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                size_t outputs_size, OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  TensorListC *input0 = (TensorListC *)(inputs[0]);
  const TensorC *get_index = inputs[1];
  const TensorC *value_tensor = inputs[2];
  TensorListC *output0 = (TensorListC *)(outputs[0]);
  output0->data_type_ = input0->data_type_;
  output0->format_ = input0->format_;
  output0->tensors_data_type_ = value_tensor->data_type_;

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  int judge_ret = PreJudge(get_index, input0, value_tensor);
  if (judge_ret != NNACL_OK) {
    return judge_ret;
  }

  int index = ((int *)(get_index->data_))[0];
  output0->max_elements_num_ = input0->max_elements_num_;

  if (input0->element_num_ == 0 && input0->element_shape_size_ == 0 && index == 0) {
    ShapeSet(input0->element_shape_, &(input0->element_shape_size_), value_tensor->shape_, value_tensor->shape_size_);
    ShapeSet(output0->element_shape_, &(output0->element_shape_size_), value_tensor->shape_, value_tensor->shape_size_);
  } else {
    ShapeSet(output0->element_shape_, &(output0->element_shape_size_), input0->element_shape_,
             input0->element_shape_size_);
  }

  vvector out_shape;
  out_shape.size_ = 0;
  out_shape.shape_ = (int **)malloc((input0->element_num_ + 1) * sizeof(int *));
  if (out_shape.shape_ == NULL) {
    return NNACL_NULL_PTR;
  }
  out_shape.shape_size_ = (int *)malloc((input0->element_num_ + 1) * sizeof(int));
  if (out_shape.shape_size_ == NULL) {
    free(out_shape.shape_);
    return NNACL_NULL_PTR;
  }

  if (index == 0 && input0->element_num_ == 0) {  // uninitialized tensorlist
    out_shape.shape_[out_shape.size_] = (int *)(value_tensor->shape_);
    out_shape.shape_size_[out_shape.size_] = value_tensor->shape_size_;
    out_shape.size_++;
    output0->element_num_ = 1;
  } else {
    output0->element_num_ = input0->element_num_;
    for (size_t i = 0; i < input0->element_num_; ++i) {
      TensorC *src_ptr = input0->tensors_[i];
      if (src_ptr == NULL) {
        free(out_shape.shape_);
        free(out_shape.shape_size_);
        return NNACL_NULL_PTR;
      }
      if (src_ptr->data_type_ != kTypeUnknown) {
        out_shape.shape_[out_shape.size_] = src_ptr->shape_;
        out_shape.shape_size_[out_shape.size_] = (int)(src_ptr->shape_size_);
        out_shape.size_++;
      } else {
        out_shape.shape_[out_shape.size_] = NULL;
        out_shape.shape_size_[out_shape.size_] = 0;
        out_shape.size_++;
      }
    }
  }

  if (input0->tensors_data_type_ == kTypeUnknown) {
    input0->tensors_data_type_ = value_tensor->data_type_;
  }

  out_shape.shape_[index] = (int *)(value_tensor->shape_);
  out_shape.shape_size_[index] = (int)value_tensor->shape_size_;
  int ret = MallocTensorListData(output0, input0->tensors_data_type_, &out_shape);
  if (ret != NNACL_OK) {
    free(out_shape.shape_);
    free(out_shape.shape_size_);
    return NNACL_ERR;
  }
  free(out_shape.shape_);
  free(out_shape.shape_size_);
  return NNACL_OK;
}

REG_INFER(TensorListSetItem, PrimType_TensorListSetItem, TensorListSetItemInferShape)
