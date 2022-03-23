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

#include "checker/reverse_checker.h"
#include <vector>
#include <limits>
#include <string>
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
bool ReverseChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, kInputIndex1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }

  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  if (primitive->GetAttr(ops::kAxis) != nullptr) {
    auto axis = api::GetValue<std::vector<int64_t>>(primitive->GetAttr(ops::kAxis));
    if (axis.size() != 1) {
      MS_LOG(WARNING) << "reverse's axis size only supports 1 by dpico. " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_ReverseChecker("ReverseV2", new ReverseChecker());
}  // namespace dpico
}  // namespace mindspore
