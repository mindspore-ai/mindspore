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

#include <set>
#include <vector>
#include <memory>
#include "ops/crop_and_resize.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void CropAndResize::Init(const ResizeMethod method, const float extrapolation_value) {
  this->set_method(method);
  this->set_extrapolation_value(extrapolation_value);
}

void CropAndResize::set_method(const ResizeMethod method) {
  auto swi = (int64_t)method;
  this->AddAttr(kMethod, MakeValue(swi));
}

void CropAndResize::set_extrapolation_value(const float extrapolation_value) {
  this->AddAttr(kExtrapolationValue, MakeValue(extrapolation_value));
}

ResizeMethod CropAndResize::get_method() const {
  auto value_ptr = GetAttr(kMethod);
  return ResizeMethod(GetValue<int64_t>(value_ptr));
}

float CropAndResize::get_extrapolation_value() const {
  auto value_ptr = GetAttr(kExtrapolationValue);
  return GetValue<float>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameCropAndResize, CropAndResize);
}  // namespace ops
}  // namespace mindspore
