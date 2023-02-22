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

#include "ops/fusion/pow_fusion.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/base_operator.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void PowFusion::Init(const float &scale, const float &shift) {
  this->set_scale(scale);
  this->set_shift(shift);
}

void PowFusion::set_scale(const float &scale) { (void)this->AddAttr(kScale, api::MakeValue(scale)); }
void PowFusion::set_shift(const float &shift) { (void)this->AddAttr(kShift, api::MakeValue(shift)); }

float PowFusion::get_scale() const { return GetValue<float>(GetAttr(kScale)); }
float PowFusion::get_shift() const { return GetValue<float>(GetAttr(kShift)); }

MIND_API_OPERATOR_IMPL(PowFusion, Pow);
REGISTER_PRIMITIVE_C(kNamePowFusion, PowFusion);
}  // namespace ops
}  // namespace mindspore
