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

#include "ops/lp_normalization.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(LpNormalization, BaseOperator);
void LpNormalization::Init(const int64_t axis, const int64_t p) {
  this->set_axis(axis);
  this->set_p(p);
}

void LpNormalization::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

int64_t LpNormalization::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

void LpNormalization::set_p(const int64_t p) { (void)this->AddAttr(kP, api::MakeValue(p)); }

int64_t LpNormalization::get_p() const {
  auto value_ptr = this->GetAttr(kP);
  return GetValue<int64_t>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameLpNormalization, LpNormalization);
}  // namespace ops
}  // namespace mindspore
