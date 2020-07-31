/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/ops.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
namespace {
constexpr int kLeastInputNum = 2;
}
int AddN::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs.front();
  MS_ASSERT(output != nullptr);
  if (inputs.size() < kLeastInputNum) {
    MS_LOG(ERROR) << "input size" << inputs.size() << " is error!";
    return RET_INPUT_TENSOR_ERROR;
  }
  for (int i = 1; i < inputs.size(); ++i) {
    if (inputs.at(i)->shape() != inputs.at(0)->shape()) {
      MS_LOG(ERROR) << "AddN inputs shape is not equal!";
      return RET_INPUT_TENSOR_ERROR;
    }
    if (inputs.at(i)->data_type() != inputs.at(0)->data_type()) {
      MS_LOG(ERROR) << "AddN all input data type should be the same!";
      return RET_INPUT_TENSOR_ERROR;
    }
  }
  output->SetFormat(input->GetFormat());
  output->set_shape(input->shape());
  output->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite

