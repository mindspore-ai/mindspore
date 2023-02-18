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

#include "ops/fusion/full_connection.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(FullConnection, BaseOperator);
void FullConnection::set_has_bias(const bool has_bias) { (void)this->AddAttr(kHasBias, api::MakeValue(has_bias)); }

bool FullConnection::get_has_bias() const {
  auto value_ptr = GetAttr(kHasBias);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

void FullConnection::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
int64_t FullConnection::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void FullConnection::set_use_axis(const bool use_axis) { (void)this->AddAttr(kUseAxis, api::MakeValue(use_axis)); }
bool FullConnection::get_use_axis() const {
  auto value_ptr = GetAttr(kUseAxis);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

void FullConnection::set_activation_type(const ActivationType &activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, api::MakeValue(swi));
}
ActivationType FullConnection::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
void FullConnection::Init(const bool has_bias, const int64_t axis, const bool use_axis,
                          const ActivationType &activation_type) {
  this->set_has_bias(has_bias);
  this->set_axis(axis);
  this->set_use_axis(use_axis);
  this->set_activation_type(activation_type);
}

REGISTER_PRIMITIVE_C(kNameFullConnection, FullConnection);
}  // namespace ops
}  // namespace mindspore
