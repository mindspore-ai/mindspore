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

#include "ops/all_gather.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(AllGather, BaseOperator);
void AllGather::set_group(const string &group) {
  std::string g = group;
  (void)this->AddAttr(kGroup, api::MakeValue(g));
}
std::string AllGather::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<std::string>(value_ptr);
}

void AllGather::set_rank_size(int rank_size) {
  (void)this->AddAttr(kRankSize, api::MakeValue(static_cast<int64_t>(rank_size)));
}
int AllGather::get_rank_size() const {
  auto value_ptr = GetAttr(kRankSize);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

REGISTER_PRIMITIVE_C(kNameAllGather, AllGather);
}  // namespace ops
}  // namespace mindspore
