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

int MergeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (inputs_size != 2 * outputs_size) {
    return NNACL_ERR;
  }
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  for (size_t i = 0; i < inputs_size / 2; i++) {
    if (((TensorListC *)inputs[i])->data_type_ == kObjectTypeTensorType) {
      TensorListC *input_tensorlist = (TensorListC *)inputs[i];
      free(outputs[i]);
      TensorListC *output_tensorlist = (TensorListC *)malloc(sizeof(TensorListC));
      memcpy(output_tensorlist, input_tensorlist, sizeof(TensorListC));
      outputs[i] = (TensorC *)output_tensorlist;
      continue;
    }
    outputs[i]->data_type_ = inputs[i]->data_type_;
    SetShapeTensor(outputs[i], inputs[i]);
    SetDataTypeFormat(outputs[i], inputs[i]);
  }
  return NNACL_OK;
}
