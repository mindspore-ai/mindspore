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

#include "nnacl/infer/select_infer.h"
#include <string.h>
#include "nnacl/infer/infer_register.h"
#include "nnacl/tensorlist_c_utils.h"

int SelectInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int check_ret =
    CheckAugmentNullInputSize(inputs, inputs_size, outputs, outputs_size, parameter, 2 * outputs_size + 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  for (size_t i = 0; i < outputs_size; i++) {
    const TensorC *input = inputs[i + 1];
    TensorC *output = outputs[i];
    SetDataTypeFormat(output, input);
  }

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  for (size_t i = 0; i < outputs_size; i++) {
    const TensorC *input = inputs[i + 1];
    TensorC *output = outputs[i];
    if (input->data_type_ == kObjectTypeTensorType) {
      TensorListC *input_tensorlist = (TensorListC *)(input);
      TensorListC *output_tensorlist = (TensorListC *)(output);
      output_tensorlist->element_shape_size_ = input_tensorlist->element_shape_size_;
      for (size_t j = 0; j < input_tensorlist->element_shape_size_; j++) {
        output_tensorlist->element_shape_[j] = input_tensorlist->element_shape_[j];
      }
      output_tensorlist->max_elements_num_ = input_tensorlist->max_elements_num_;
      output_tensorlist->tensors_data_type_ = input_tensorlist->tensors_data_type_;
      output_tensorlist->element_num_ = input_tensorlist->element_num_;

      for (size_t j = 0; j < output_tensorlist->element_num_; j++) {
        memcpy(&output_tensorlist->tensors_[j], &input_tensorlist->tensors_[j], sizeof(TensorC));
      }
    } else {
      SetShapeTensor(output, input);
    }
  }
  return NNACL_OK;
}

REG_INFER(Select, PrimType_Select, SelectInferShape)
