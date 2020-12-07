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

#include "c_ops/l2_normalize.h"

namespace mindspore {
void L2Normalize::Init(int64_t axis, float epsilon) {
  this->set_axis(axis);
  this->set_epsilon(epsilon);
}

void L2Normalize::set_axis(int64_t axis) { AddAttr(kAxis, MakeValue(axis)); }

void L2Normalize::set_epsilon(float epsilon) { AddAttr(kEpsilon, MakeValue(epsilon)); }

int64_t L2Normalize::get_axis() {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

float L2Normalize::get_epsilon() {
  auto value_ptr = GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameL2Normalize, L2Normalize);
}  // namespace mindspore
