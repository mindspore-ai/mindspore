/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/activation_grad.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ActivationGrad, BaseOperator);
void ActivationGrad::Init(const ActivationType &type, const float alpha) {
  this->set_activation_type(type);
  this->set_alpha(alpha);
}

void ActivationGrad::set_activation_type(const ActivationType &type) {
  int64_t swi = type;
  (void)this->AddAttr(kActivationType, api::MakeValue(swi));
}

ActivationType ActivationGrad::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

void ActivationGrad::set_alpha(const float alpha) { (void)this->AddAttr(kAlpha, api::MakeValue(alpha)); }

float ActivationGrad::get_alpha() const {
  auto value_ptr = GetAttr(kAlpha);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameActivationGrad, ActivationGrad);
}  // namespace ops
}  // namespace mindspore
