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

#include <set>

#include "ops/l2_normalize.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(L2Normalize, PrimitiveC, BaseOperator);
void L2Normalize::Init(const std::vector<int64_t> &axis, const float epsilon) {
  this->set_axis(axis);
  this->set_epsilon(epsilon);
}

void L2Normalize::set_axis(const std::vector<int64_t> &axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }

void L2Normalize::set_epsilon(const float epsilon) { (void)AddAttr(kEpsilon, api::MakeValue(epsilon)); }

std::vector<int64_t> L2Normalize::get_axis() const { return GetValue<std::vector<int64_t>>(GetAttr(kAxis)); }

float L2Normalize::get_epsilon() const {
  auto value_ptr = GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameL2Normalize, L2Normalize);
}  // namespace ops
}  // namespace mindspore
