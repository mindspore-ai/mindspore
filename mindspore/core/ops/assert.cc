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

#include "ops/assert.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Assert, BaseOperator);
void Assert::Init(const int64_t summarize) { set_summarize(summarize); }

void Assert::set_summarize(const int64_t summarize) { (void)this->AddAttr(kSummarize, api::MakeValue(summarize)); }

int64_t Assert::get_summarize() const {
  auto value_ptr = GetAttr(kSummarize);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameAssert, Assert);
}  // namespace ops
}  // namespace mindspore
