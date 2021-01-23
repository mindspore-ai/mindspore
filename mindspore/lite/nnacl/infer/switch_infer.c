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
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  for (size_t i = 0; i < outputs_size / 2; i++) {
    if (((TensorListC *)inputs[i + 1])->data_type_ == kObjectTypeTensorType) {
      TensorListC *input_tensorlist = (TensorListC *)inputs[i + 1];
      TensorListC *output_tensorlist1 = (TensorListC *)outputs[i];
      memcpy(output_tensorlist1, input_tensorlist, sizeof(TensorListC));
      outputs[i] = (TensorC *)output_tensorlist1;

      TensorListC *output_tensorlist2 = (TensorListC *)outputs[i + outputs_size / 2];
      memcpy(output_tensorlist2, input_tensorlist, sizeof(TensorListC));
      outputs[i + outputs_size / 2] = (TensorC *)output_tensorlist2;
      continue;
    }

    outputs[i]->data_type_ = (inputs[i + 1]->data_type_);
    outputs[i + outputs_size / 2]->data_type_ = inputs[i + 1]->data_type_;
    SetShapeTensor(outputs[i], inputs[i + 1]);
    SetShapeTensor(outputs[i + outputs_size / 2], inputs[i + 1]);
    outputs[i]->format_ = inputs[i + 1]->format_;
    outputs[i + outputs_size / 2]->format_ = inputs[i + 1]->format_;
  }
  return NNACL_OK;
}
