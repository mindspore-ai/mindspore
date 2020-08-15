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

#include "c_ops/unstack.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int Unstack::GetNum() const { return this->primitive->value.AsUnstack()->num; }
int Unstack::GetAxis() const { return this->primitive->value.AsUnstack()->axis; }

void Unstack::SetNum(int num) { this->primitive->value.AsUnstack()->num = num; }
void Unstack::SetAxis(int axis) { this->primitive->value.AsUnstack()->axis = axis; }

#else

int Unstack::GetNum() const { return this->primitive->value_as_Unstack()->num(); }
int Unstack::GetAxis() const { return this->primitive->value_as_Unstack()->axis(); }

void Unstack::SetNum(int num) {}
void Unstack::SetAxis(int axis) {}
#endif
int Unstack::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  auto input = inputs.at(0);
  MS_ASSERT(input != nullptr);
  auto input_shape = input->shape();
  int axis = GetAxis() < 0 ? GetAxis() + input_shape.size() : GetAxis();
  if (axis < 0 || axis >= input_shape.size()) {
    MS_LOG(ERROR) << "Invalid axis " << GetAxis();
    return 1;
  }

  std::vector<int> output_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i != axis) {
      output_shape.push_back(input_shape.at(i));
    }
  }
  for (auto &out : outputs) {
    MS_ASSERT(out != nullptr);
    out->set_shape(output_shape);
    out->set_data_type(input->data_type());
    out->SetFormat(input->GetFormat());
  }
  return 0;
}
}  // namespace mindspore
