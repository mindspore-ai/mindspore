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

#include "tools/converter/ops/exit.h"
#include "src/tensorlist.h"

namespace mindspore {
namespace lite {

int Exit::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto *input = inputs_[i];
    auto *output = outputs_[i];
    if (input == nullptr) {
      MS_LOG(ERROR) << "input tensor is nullptr";
      return RET_ERROR;
    }
    if (output == nullptr) {
      MS_LOG(ERROR) << "output tensor is nullptr";
      return RET_ERROR;
    }
    output->set_data_type(input->data_type());
    output->set_shape(input->shape());
    output->set_format(input->format());
    auto data_type = input->data_type();
    if (data_type != kObjectTypeTensorType) {
      continue;
    } else {
      auto input_tensorlist = reinterpret_cast<TensorList *>(input);
      auto output_tensorlist = reinterpret_cast<TensorList *>(output);
      output_tensorlist->set_element_shape(input_tensorlist->element_shape());
      output_tensorlist->set_max_elements_num(input_tensorlist->max_elements_num());
      output_tensorlist->set_tensors_data_type(input_tensorlist->tensors_data_type());
    }
  }
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
