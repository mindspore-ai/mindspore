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

#include "ops/grad/minimum_grad.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void MinimumGrad::Init(const bool grad_x, const bool grad_y) {
  set_grad_x(grad_x);
  set_grad_y(grad_y);
}

void MinimumGrad::set_grad_x(const bool grad_x) { this->AddAttr(kGradX, MakeValue(grad_x)); }

void MinimumGrad::set_grad_y(const bool grad_y) { this->AddAttr(kGradY, MakeValue(grad_y)); }

bool MinimumGrad::get_grad_x() const {
  auto value_ptr = GetAttr(kGradX);
  return GetValue<bool>(value_ptr);
}

bool MinimumGrad::get_grad_y() const {
  auto value_ptr = GetAttr(kGradY);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameMinimumGrad, MinimumGrad);
}  // namespace ops
}  // namespace mindspore
