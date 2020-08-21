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

#include "src/ops/addn.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int AddN::GetN() const { return this->primitive_->value.AsAddN()->N; }

void AddN::SetN(int n) { this->primitive_->value.AsAddN()->N = n; }

#else

int AddN::GetN() const { return this->primitive_->value_as_AddN()->N(); }

void AddN::SetN(int n) {}
#endif

namespace {
constexpr int kLeastInputNum = 2;
}
int AddN::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs.front();
  MS_ASSERT(output != nullptr);
  if (inputs.size() < kLeastInputNum) {
    MS_LOG(ERROR) << "input size" << inputs.size() << " is error!";
    return RET_INPUT_TENSOR_ERROR;
  }
  output->SetFormat(input->GetFormat());
  output->set_data_type(input->data_type());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs.at(i)->shape() != inputs.at(0)->shape()) {
      MS_LOG(ERROR) << "AddN inputs shape is not equal!";
      return RET_INPUT_TENSOR_ERROR;
    }
    if (inputs.at(i)->data_type() != inputs.at(0)->data_type()) {
      MS_LOG(ERROR) << "AddN all input data type should be the same!";
      return RET_INPUT_TENSOR_ERROR;
    }
  }

  output->set_shape(input->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
