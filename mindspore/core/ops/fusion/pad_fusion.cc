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

#include "ops/fusion/pad_fusion.h"
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void PadFusion::Init(const PaddingMode &padding_mode, const float constant_value) {
  this->set_padding_mode(padding_mode);
  this->set_constant_value(constant_value);
}

void PadFusion::set_padding_mode(const PaddingMode &padding_mode) {
  int64_t swi;
  swi = padding_mode;
  this->AddAttr(kPaddingMode, MakeValue(swi));
}

void PadFusion::set_constant_value(const float constant_value) {
  this->AddAttr(kConstantValue, MakeValue(constant_value));
}

PaddingMode PadFusion::get_padding_mode() const {
  auto value_ptr = GetAttr(kPaddingMode);
  return PaddingMode(GetValue<int64_t>(value_ptr));
}
float PadFusion::get_constant_value() const {
  auto value_ptr = GetAttr(kConstantValue);
  return GetValue<float>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNamePadFusion, PadFusion);
}  // namespace ops
}  // namespace mindspore
