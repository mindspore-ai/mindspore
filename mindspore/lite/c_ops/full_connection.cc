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

#include "c_ops/full_connection.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
bool FullConnection::GetHasBias() const { return this->primitive->value.AsFullConnection()->hasBias; }
int FullConnection::GetAxis() const { return this->primitive->value.AsFullConnection()->axis; }
bool FullConnection::GetUseAxis() const { return this->primitive->value.AsFullConnection()->useAxis; }

void FullConnection::SetHasBias(bool has_bias) { this->primitive->value.AsFullConnection()->hasBias = has_bias; }
void FullConnection::SetAxis(int axis) { this->primitive->value.AsFullConnection()->axis = axis; }
void FullConnection::SetUseAxis(bool use_axis) { this->primitive->value.AsFullConnection()->useAxis = use_axis; }

#else

bool FullConnection::GetHasBias() const { return this->primitive->value_as_FullConnection()->hasBias(); }
int FullConnection::GetAxis() const { return this->primitive->value_as_FullConnection()->axis(); }
bool FullConnection::GetUseAxis() const { return this->primitive->value_as_FullConnection()->useAxis(); }

void FullConnection::SetHasBias(bool has_bias) {}
void FullConnection::SetAxis(int axis) {}
void FullConnection::SetUseAxis(bool use_axis) {}
#endif
int FullConnection::InferShape(std::vector<lite::tensor::Tensor *> inputs_,
                               std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto input1 = inputs_.at(1);
  MS_ASSERT(input1 != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  if ((GetHasBias() && inputs_.size() != kMultiNum) || (!GetHasBias() && inputs_.size() != kDoubleNum)) {
    MS_LOG(ERROR) << "Input tensors num error";
    return 1;
  }
  if (GetAxis() < 1 || GetAxis() > input0->shape().size()) {
    MS_LOG(ERROR) << "FullConnection axis invalid";
    return 1;
  }
  int new_k = 1;
  for (size_t i = GetAxis(); i < input0->shape().size(); ++i) {
    new_k *= input0->shape().at(i);
  }
  if (new_k != input1->shape().at(1)) {
    MS_LOG(ERROR) << "Input1 size invalid";
    return 1;
  }
  if (GetHasBias()) {
    if (inputs_.at(2)->shape()[0] != input1->shape()[0]) {
      MS_LOG(ERROR) << "bias size invalid";
      return 1;
    }
  }
  std::vector<int> out_shape{inputs_[0]->shape()};
  out_shape.resize(GetAxis() + 1);
  out_shape[GetAxis()] = input1->shape()[0];
  output->set_shape(out_shape);
  output->set_data_type(input0->data_type());
  output->SetFormat(input0->GetFormat());

  return 0;
}
}  // namespace mindspore
