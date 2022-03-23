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

#include "ops/dropout.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(Dropout, PrimitiveC, BaseOperator);
void Dropout::Init(const float keep_prob) { this->set_keep_prob(keep_prob); }

void Dropout::set_keep_prob(const float keep_prob) {
  CheckAndConvertUtils::CheckInRange<float>(kKeepProb, keep_prob, kIncludeRight, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kKeepProb, api::MakeValue(keep_prob));
}

float Dropout::get_keep_prob() const {
  auto value_ptr = this->GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameDropout, Dropout);
}  // namespace ops
}  // namespace mindspore
