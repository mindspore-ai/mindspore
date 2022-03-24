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
#include <utility>
#include "ops/topk.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(TopK, PrimitiveC, BaseOperator);
void TopK::Init(const bool sorted) { this->set_sorted(sorted); }
void TopK::set_sorted(const bool sorted) { (void)this->AddAttr(kSorted, api::MakeValue(sorted)); }

bool TopK::get_sorted() const {
  auto value_ptr = this->GetAttr(kSorted);
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameTopK, TopK);
}  // namespace ops
}  // namespace mindspore
