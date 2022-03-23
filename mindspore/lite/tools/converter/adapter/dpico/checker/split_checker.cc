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

#include "checker/split_checker.h"
#include <string>
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
bool SplitChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (output_num > kMaxTopNum) {
    MS_LOG(WARNING) << op->fullname_with_scope() << "'s output num " << output_num << " is greater than " << kMaxTopNum;
    return false;
  }
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    return false;
  }
  return primitive->GetAttr(ops::kAxis) != nullptr && api::GetValue<int64_t>(primitive->GetAttr(ops::kAxis)) != 0;
}

OpCheckerRegistrar g_SplitChecker("Split", new SplitChecker());
}  // namespace dpico
}  // namespace mindspore
