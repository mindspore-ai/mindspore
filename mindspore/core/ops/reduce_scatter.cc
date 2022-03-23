/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/reduce_scatter.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(ReduceScatter, PrimitiveC, BaseOperator);
void ReduceScatter::set_group(const string &group) {
  std::string g = group;
  (void)this->AddAttr(kGroup, api::MakeValue(g));
}
std::string ReduceScatter::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<std::string>(value_ptr);
}

void ReduceScatter::set_mode(const ReduceMode &mode) {
  int64_t m = mode;
  (void)this->AddAttr(kMode, api::MakeValue(m));
}

ReduceMode ReduceScatter::get_mode() const {
  auto value_ptr = this->GetAttr(kMode);
  return ReduceMode(GetValue<int64_t>(value_ptr));
}

void ReduceScatter::set_rank_size(int rank_size) {
  (void)this->AddAttr(kRankSize, api::MakeValue(static_cast<int64_t>(rank_size)));
}
int ReduceScatter::get_rank_size() const {
  auto value_ptr = GetAttr(kRankSize);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

REGISTER_PRIMITIVE_C(kNameReduceScatter, ReduceScatter);
}  // namespace ops
}  // namespace mindspore
