/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/fusion/activation.h"
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void Activation::set_alpha(const float alpha) { this->AddAttr(kAlpha, MakeValue(alpha)); }

void Activation::set_min_val(const float min_val) { this->AddAttr(kMinVal, MakeValue(min_val)); }

void Activation::set_max_val(const float max_val) { this->AddAttr(kMaxVal, MakeValue(max_val)); }

void Activation::set_activation_type(const ActivationType &activation_type) {
  int64_t swi;
  swi = activation_type;
  this->AddAttr(kActivationType, MakeValue(swi));
}

float Activation::get_alpha() const {
  auto value_ptr = this->GetAttr(kAlpha);
  return GetValue<float>(value_ptr);
}

float Activation::get_min_val() const {
  auto value_ptr = this->GetAttr(kMinVal);
  return GetValue<float>(value_ptr);
}

float Activation::get_max_val() const {
  auto value_ptr = this->GetAttr(kMaxVal);
  return GetValue<float>(value_ptr);
}

ActivationType Activation::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
void Activation::Init(const float alpha, const float min_val, const float max_val,
                      const ActivationType &activation_type) {
  this->set_alpha(alpha);
  this->set_min_val(min_val);
  this->set_max_val(max_val);
  this->set_activation_type(activation_type);
}
REGISTER_PRIMITIVE_C(kNameActivation, Activation);
}  // namespace ops
}  // namespace mindspore
