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

#include "ops/fake_quant_with_min_max_vars.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void FakeQuantWithMinMaxVars::Init(const bool narrow_range, const int64_t num_bits) {
  this->set_narrow_range(narrow_range);
  this->set_num_bits(num_bits);
}

void FakeQuantWithMinMaxVars::set_narrow_range(const bool narrow_range) {
  (void)this->AddAttr(kNarrowRange, api::MakeValue(narrow_range));
}

bool FakeQuantWithMinMaxVars::get_narrow_range() const {
  auto value_ptr = this->GetAttr(kNarrowRange);
  return GetValue<bool>(value_ptr);
}

void FakeQuantWithMinMaxVars::set_num_bits(const int64_t num_bits) {
  (void)this->AddAttr(kNumBits, api::MakeValue(num_bits));
}

int64_t FakeQuantWithMinMaxVars::get_num_bits() const {
  auto value_ptr = this->GetAttr(kNumBits);
  return GetValue<int64_t>(value_ptr);
}

MIND_API_OPERATOR_IMPL(FakeQuantWithMinMaxVars, BaseOperator);
REGISTER_PRIMITIVE_C(kNameFakeQuantWithMinMaxVars, FakeQuantWithMinMaxVars);
}  // namespace ops
}  // namespace mindspore
