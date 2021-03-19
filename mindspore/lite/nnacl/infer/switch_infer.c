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
#include "nnacl/infer/infer_register.h"

int SwitchInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
#ifdef Debug
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] == NULL) {
      return NNACL_NULL_PTR;
    }
  }
  if (2 * (inputs_size - 1) != outputs_size) {
    return NNACL_ERR;
  }
#endif

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  for (size_t i = 0; i < outputs_size / 2; i++) {
    outputs[i] = (TensorC *)inputs[i + 1];
    if (inputs[i + 1]->data_type_ == kObjectTypeTensorType) {
      TensorListC *input = (TensorListC *)inputs[i + 1];
      TensorListC *mirror_tensorlist = (TensorListC *)malloc(sizeof(TensorListC));  // free in infer_manager
      if (mirror_tensorlist == NULL) {
        return NNACL_ERR;  // memory that has been applied will be free in infer_manager
      }
      memcpy(mirror_tensorlist, input, sizeof(TensorListC));

      TensorC *tensor_buffer = (TensorC *)malloc(input->element_num_ * sizeof(TensorC));
      if (tensor_buffer == NULL) {
        free(mirror_tensorlist);
        return NNACL_ERR;
      }
      memcpy(tensor_buffer, input->tensors_, input->element_num_ * sizeof(TensorC));
      mirror_tensorlist->tensors_ = tensor_buffer;
      outputs[i + outputs_size / 2] = (TensorC *)(mirror_tensorlist);
    } else {
      TensorC *mirror_tensor = (TensorC *)malloc(sizeof(TensorC));
      if (mirror_tensor == NULL) {
        return NNACL_ERR;
      }
      memcpy(mirror_tensor, inputs[i + 1], sizeof(TensorC));
      outputs[i + outputs_size / 2] = mirror_tensor;
    }
    *((const TensorC **)inputs + i + 1) = NULL;
  }

  return NNACL_OK;
}

REG_INFER(Switch, PrimType_Switch, SwitchInferShape)
