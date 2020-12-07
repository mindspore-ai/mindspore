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

#include "c_ops/apply_momentum.h"
#include "c_ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
void ApplyMomentum::Init(bool use_nesterov, bool use_locking, float gradient_scale) {
  this->set_use_nesterov(use_nesterov);
  this->set_use_locking(use_locking);
  this->set_gradient_scale(gradient_scale);
}

void ApplyMomentum::set_use_nesterov(bool use_nesterov) { this->AddAttr(kUseNesterov, MakeValue(use_nesterov)); }

void ApplyMomentum::set_use_locking(bool use_locking) { this->AddAttr(kUseLocking, MakeValue(use_locking)); }

void ApplyMomentum::set_gradient_scale(float gradient_scale) {
  this->AddAttr(kGradientScale, MakeValue(gradient_scale));
}

bool ApplyMomentum::get_use_nesterov() const {
  auto value_ptr = GetAttr(kUseNesterov);
  return GetValue<bool>(value_ptr);
}

bool ApplyMomentum::get_use_locking() const {
  auto value_ptr = GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

float ApplyMomentum::get_gradient_scale() {
  auto value_ptr = GetAttr(kGradientScale);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameApplyMomentum, ApplyMomentum);
}  // namespace mindspore
