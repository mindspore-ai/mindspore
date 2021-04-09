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

#include "ops/grad/power_grad.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void PowerGrad::set_power(const float power) { this->AddAttr(kPower, MakeValue(power)); }
float PowerGrad::get_power() const {
  auto value_ptr = GetAttr(kPower);
  return GetValue<float>(value_ptr);
}

void PowerGrad::set_scale(const float scale) { this->AddAttr(kScale, MakeValue(scale)); }
float PowerGrad::get_scale() const {
  auto value_ptr = GetAttr(kScale);
  return GetValue<float>(value_ptr);
}

void PowerGrad::set_shift(const float shift) { this->AddAttr(kShift, MakeValue(shift)); }
float PowerGrad::get_shift() const {
  auto value_ptr = GetAttr(kShift);
  return GetValue<float>(value_ptr);
}

void PowerGrad::Init(const float power, const float scale, const float shift) {
  this->set_power(power);
  this->set_scale(scale);
  this->set_shift(shift);
}
REGISTER_PRIMITIVE_C(kNamePowerGrad, PowerGrad);
}  // namespace ops
}  // namespace mindspore
