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

#include "ops/fusion/exp_fusion.h"
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void ExpFusion::Init(const float base, const float scale, const float shift) {
  this->set_base(base);
  this->set_scale(scale);
  this->set_shift(shift);
}

void ExpFusion::set_base(const float base) { this->AddAttr(kBase, MakeValue(base)); }

void ExpFusion::set_scale(const float scale) { this->AddAttr(kScale, MakeValue(scale)); }

void ExpFusion::set_shift(const float shift) { this->AddAttr(kShift, MakeValue(shift)); }

float ExpFusion::get_base() const {
  auto value_ptr = GetAttr(kBase);
  return GetValue<float>(value_ptr);
}
float ExpFusion::get_scale() const {
  auto value_ptr = GetAttr(kScale);
  return GetValue<float>(value_ptr);
}
float ExpFusion::get_shift() const {
  auto value_ptr = GetAttr(kShift);
  return GetValue<float>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameExpFusion, ExpFusion);
}  // namespace ops
}  // namespace mindspore
