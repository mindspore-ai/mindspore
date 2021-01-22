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

#include "ops/fusion/scale_fusion.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void ScaleFusion::Init(const int64_t axis, const ActivationType &activation_type) {
  this->set_axis(axis);
  this->set_activation_type(activation_type);
}

void ScaleFusion::set_activation_type(const ActivationType &activation_type) {
  int64_t swi;
  swi = activation_type;
  this->AddAttr(kActivationType, MakeValue(swi));
}

ActivationType ScaleFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
REGISTER_PRIMITIVE_C(kNameScaleFusion, ScaleFusion);
}  // namespace ops
}  // namespace mindspore
