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

#include "src/ops/softmax.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int SoftMax::GetAxis() const { return this->primitive->value.AsSoftMax()->axis; }

void SoftMax::SetAxis(int axis) { this->primitive->value.AsSoftMax()->axis = axis; }

#else

int SoftMax::GetAxis() const { return this->primitive->value_as_SoftMax()->axis(); }

void SoftMax::SetAxis(int axis) {}
#endif

int SoftMax::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  output->set_shape(input->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
