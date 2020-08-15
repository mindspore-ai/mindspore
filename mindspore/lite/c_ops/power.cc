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

#include "c_ops/power.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
float Power::GetPower() const { return this->primitive->value.AsPower()->power; }
float Power::GetScale() const { return this->primitive->value.AsPower()->scale; }
float Power::GetShift() const { return this->primitive->value.AsPower()->shift; }

void Power::SetPower(float power) { this->primitive->value.AsPower()->power = power; }
void Power::SetScale(float scale) { this->primitive->value.AsPower()->scale = scale; }
void Power::SetShift(float shift) { this->primitive->value.AsPower()->shift = shift; }

#else

float Power::GetPower() const { return this->primitive->value_as_Power()->power(); }
float Power::GetScale() const { return this->primitive->value_as_Power()->scale(); }
float Power::GetShift() const { return this->primitive->value_as_Power()->shift(); }

void Power::SetPower(float power) {}
void Power::SetScale(float scale) {}
void Power::SetShift(float shift) {}
#endif
int Power::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  auto x_tensor = inputs[0];
  MS_ASSERT(x_tensor != nullptr);
  lite::tensor::Tensor *exp_tensor = nullptr;
  if (inputs.size() == 2) {
    exp_tensor = inputs[1];
    MS_ASSERT(exp_tensor != nullptr);
  }
  auto output_tensor = outputs[0];
  MS_ASSERT(output_tensor != nullptr);
  if (exp_tensor) {
    if (exp_tensor->shape() != x_tensor->shape() || exp_tensor->data_type() != x_tensor->data_type()) {
      MS_LOG(ERROR) << "Power inputs shape or type is not equal!";
      return 1;
    }
  }

  output_tensor->SetFormat(x_tensor->GetFormat());
  output_tensor->set_shape(x_tensor->shape());
  output_tensor->set_data_type(x_tensor->data_type());
  return 0;
}
}  // namespace mindspore
