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
#include "ops/range.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(Range, PrimitiveC, BaseOperator);
void Range::set_d_type(const int64_t d_type) { (void)this->AddAttr(kDType, api::MakeValue(d_type)); }

int64_t Range::get_d_type() const {
  auto value_ptr = GetAttr(kDType);
  return GetValue<int64_t>(value_ptr);
}

void Range::set_start(const int64_t start) { (void)this->AddAttr(kStart, api::MakeValue(start)); }

int64_t Range::get_start() const { return GetValue<int64_t>(GetAttr(kStart)); }

void Range::set_limit(const int64_t limit) { (void)this->AddAttr(kLimit, api::MakeValue(limit)); }

int64_t Range::get_limit() const {
  auto value_ptr = GetAttr(kLimit);
  return GetValue<int64_t>(value_ptr);
}

void Range::set_delta(const int64_t delta) { (void)this->AddAttr(kDelta, api::MakeValue(delta)); }

int64_t Range::get_delta() const {
  auto value_ptr = GetAttr(kDelta);
  return GetValue<int64_t>(value_ptr);
}

void Range::Init(const int64_t d_type, const int64_t start, const int64_t limit, const int64_t delta) {
  this->set_d_type(d_type);
  this->set_start(start);
  this->set_limit(limit);
  this->set_delta(delta);
}

REGISTER_PRIMITIVE_C(kNameRange, Range);
}  // namespace ops
}  // namespace mindspore
