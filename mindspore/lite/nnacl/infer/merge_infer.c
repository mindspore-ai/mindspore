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

#include "nnacl/infer/merge_infer.h"
#include <string.h>

int MergeAbleToInfer(const TensorC *const *inputs, size_t inputs_size) {
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i]->shape_size_ == 0) {
      return HasZeroShape;
    }
    if (inputs[i]->data_ == NULL) {
      return NotAble;
    }
  }
  return Able;
}

int MergeInfer(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size) {
  for (size_t i = 0; i < inputs_size; i++) {
    SetDataTypeFormat(outputs[i], inputs[i]);
    if (((TensorListC *)inputs[i])->data_type_ == kObjectTypeTensorType) {
      TensorListC *input_tensorlist = (TensorListC *)inputs[i];
      TensorListC *output_tensorlist = (TensorListC *)outputs[i];
      ShapeSet(output_tensorlist->element_shape_, &output_tensorlist->element_shape_size_,
               input_tensorlist->element_shape_, input_tensorlist->element_shape_size_);
      output_tensorlist->max_elements_num_ = input_tensorlist->max_elements_num_;
      output_tensorlist->tensors_data_type_ = input_tensorlist->tensors_data_type_;

      output_tensorlist->element_num_ = input_tensorlist->element_num_;
      for (size_t j = 0; j < output_tensorlist->element_num_; j++) {
        memcpy(&output_tensorlist->tensors_[j], &input_tensorlist->tensors_[j], sizeof(TensorC));
      }
    } else {
      SetShapeTensor(outputs[i], inputs[i]);
    }
  }
  return NNACL_OK;
}

int MergeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (inputs_size != 2 * outputs_size) {
    return NNACL_ERR;
  }
#endif

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  for (size_t i = 0; i < outputs_size; ++i) {
    outputs[i]->data_type_ = inputs[i]->data_type_;
  }
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  const TensorC *const *left_part_inputs = inputs;
  size_t left_part_inputs_size = inputs_size / 2;

  const TensorC *const *right_part_inputs = inputs + left_part_inputs_size;
  size_t right_part_inputs_size = inputs_size / 2;

  if (MergeAbleToInfer(left_part_inputs, left_part_inputs_size) == Able) {
    return MergeInfer(left_part_inputs, left_part_inputs_size, outputs, outputs_size);
  }

  if (MergeAbleToInfer(right_part_inputs, right_part_inputs_size) == Able) {
    return MergeInfer(right_part_inputs, right_part_inputs_size, outputs, outputs_size);
  }

  if (MergeAbleToInfer(left_part_inputs, left_part_inputs_size) == HasZeroShape &&
      MergeAbleToInfer(right_part_inputs, right_part_inputs_size) == HasZeroShape) {
    return MergeInfer(left_part_inputs, left_part_inputs_size, outputs, outputs_size);
  }

  return NNACL_INFER_INVALID;
}
