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

#include "ops/sgd.h"

namespace mindspore {
namespace ops {
void SGD::Init(const float dampening, const float weight_decay, const bool nesterov) {
  set_nesterov(nesterov);
  set_dampening(dampening);
  set_weight_decay(weight_decay);
}

void SGD::set_dampening(const float dampening) {
  if (get_nesterov()) CheckAndConvertUtils::CheckValue<float>(kDampening, dampening, kEqual, 0.0, name());
  AddAttr(kDampening, MakeValue(dampening));
}

void SGD::set_weight_decay(const float weight_decay) { AddAttr(kWeightDecay, MakeValue(weight_decay)); }

void SGD::set_nesterov(const bool nesterov) { AddAttr(kNesterov, MakeValue(nesterov)); }

float SGD::get_dampening() const {
  auto value_ptr = GetAttr(kDampening);
  return GetValue<float>(value_ptr);
}

float SGD::get_weight_decay() const {
  auto value_ptr = GetAttr(kWeightDecay);
  return GetValue<float>(value_ptr);
}

bool SGD::get_nesterov() const {
  auto value_ptr = GetAttr(kNesterov);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameSGD, SGD);
}  // namespace ops
}  // namespace mindspore
