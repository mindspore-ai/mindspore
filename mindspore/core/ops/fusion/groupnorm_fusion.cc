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

#include "ops/fusion/groupnorm_fusion.h"

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(GroupNormFusion, PrimitiveC, BaseOperator);
void GroupNormFusion::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

void GroupNormFusion::set_num_groups(const int64_t num_groups) {
  (void)this->AddAttr(kNumGroups, api::MakeValue(num_groups));
}

void GroupNormFusion::set_affine(const bool affine) { (void)this->AddAttr(kAffine, api::MakeValue(affine)); }

float GroupNormFusion::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

int64_t GroupNormFusion::get_num_groups() const {
  auto value_ptr = this->GetAttr(kNumGroups);
  return GetValue<int64_t>(value_ptr);
}

bool GroupNormFusion::get_affine() const {
  auto value_ptr = this->GetAttr(kAffine);
  return GetValue<bool>(value_ptr);
}

void GroupNormFusion::Init(const int64_t num_groups, const float epsilon, bool affine) {
  this->set_epsilon(epsilon);
  this->set_num_groups(num_groups);
  this->set_affine(affine);
}
REGISTER_PRIMITIVE_C(kNameGroupNormFusion, GroupNormFusion);
}  // namespace ops
}  // namespace mindspore
