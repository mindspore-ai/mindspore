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

#include "nnacl/infer/switch_infer.h"
#include <string.h>

int SwitchInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (2 * (inputs_size - 1) != outputs_size) {
    return NNACL_ERR;
  }

  for (size_t i = 0; i < outputs_size / 2; i++) {
    const TensorC *input = inputs[i + 1];
    TensorC *output_true = outputs[i];
    TensorC *output_false = outputs[i + outputs_size / 2];

    SetDataTypeFormat(output_false, input);
    SetDataTypeFormat(output_true, input);

    if (input->data_type_ == kObjectTypeTensorType) {
      TensorListC *input_tensorlist = (TensorListC *)(input);
      TensorListC *output_true_tensorlist = (TensorListC *)(output_true);
      TensorListC *output_false_tensorlist = (TensorListC *)(output_false);

      ShapeSet(output_true_tensorlist->element_shape_, &output_true_tensorlist->element_shape_size_,
               input_tensorlist->element_shape_, input_tensorlist->element_shape_size_);
      ShapeSet(output_false_tensorlist->element_shape_, &output_false_tensorlist->element_shape_size_,
               input_tensorlist->element_shape_, input_tensorlist->element_shape_size_);
      output_true_tensorlist->max_elements_num_ = input_tensorlist->max_elements_num_;
      output_false_tensorlist->max_elements_num_ = input_tensorlist->max_elements_num_;
      output_true_tensorlist->tensors_data_type_ = input_tensorlist->tensors_data_type_;
      output_false_tensorlist->tensors_data_type_ = input_tensorlist->tensors_data_type_;

      // note: need delete below?
      for (size_t j = 0; j < output_false_tensorlist->element_num_; j++) {
        memcpy(output_true_tensorlist->tensors_[j], input_tensorlist->tensors_[j], sizeof(TensorC));
        memcpy(output_false_tensorlist->tensors_[j], input_tensorlist->tensors_[j], sizeof(TensorC));
      }

    } else {
    }
  }

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  for (size_t i = 0; i < outputs_size / 2; i++) {
    const TensorC *input = inputs[i + 1];
    TensorC *output_true = outputs[i];
    TensorC *output_false = outputs[i + outputs_size / 2];

    SetDataTypeFormat(output_false, input);
    SetDataTypeFormat(output_true, input);

    if (input->data_type_ == kObjectTypeTensorType) {
      TensorListC *input_tensorlist = (TensorListC *)(input);
      TensorListC *output_true_tensorlist = (TensorListC *)(output_true);
      TensorListC *output_false_tensorlist = (TensorListC *)(output_false);

      ShapeSet(output_true_tensorlist->element_shape_, &output_true_tensorlist->element_shape_size_,
               input_tensorlist->element_shape_, input_tensorlist->element_shape_size_);
      ShapeSet(output_false_tensorlist->element_shape_, &output_false_tensorlist->element_shape_size_,
               input_tensorlist->element_shape_, input_tensorlist->element_shape_size_);
      output_true_tensorlist->max_elements_num_ = input_tensorlist->max_elements_num_;
      output_false_tensorlist->max_elements_num_ = input_tensorlist->max_elements_num_;
      output_true_tensorlist->tensors_data_type_ = input_tensorlist->tensors_data_type_;
      output_false_tensorlist->tensors_data_type_ = input_tensorlist->tensors_data_type_;

      output_false_tensorlist->element_num_ = input_tensorlist->element_num_;
      output_true_tensorlist->element_num_ = input_tensorlist->element_num_;

      for (size_t j = 0; j < output_false_tensorlist->element_num_; j++) {
        memcpy(output_true_tensorlist->tensors_[j], input_tensorlist->tensors_[j], sizeof(TensorC));
        memcpy(output_false_tensorlist->tensors_[j], input_tensorlist->tensors_[j], sizeof(TensorC));
      }

    } else {
      SetShapeTensor(output_true, input);
      SetShapeTensor(output_false, input);
    }
  }

  return NNACL_OK;
}
