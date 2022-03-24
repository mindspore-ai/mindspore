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

#include "checker/argmax_checker.h"
#include <vector>
#include <string>
#include "common/fetch_content.h"
#include "common/op_attr.h"
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
bool ArgMaxChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, 1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr." << op->fullname_with_scope();
    return false;
  }
  if (primitive->GetAttr(ops::kTopK) != nullptr) {
    auto top_k = static_cast<uint32_t>(api::GetValue<int64_t>(primitive->GetAttr(ops::kTopK)));
    if (top_k != 1) {
      MS_LOG(WARNING) << "top_k value only supports 1 for dpico. " << op->fullname_with_scope();
      return false;
    }
  }

  if (primitive->GetAttr(ops::kAxis) != nullptr) {
    auto axis = api::GetValue<int64_t>(primitive->GetAttr(ops::kAxis));
    if (axis <= kAxisLowerBound || axis > kAxisUpperBound || axis == 0) {
      MS_LOG(WARNING) << op->fullname_with_scope() << "'s axis should in range (-4, 0) and (0, 3], but in fact it's "
                      << axis;
      return false;
    }
  }
  if (primitive->GetAttr(dpico::kSelectLastIndex) != nullptr) {
    auto select_last_index = api::GetValue<bool>(primitive->GetAttr(dpico::kSelectLastIndex));
    if (!select_last_index) {
      MS_LOG(WARNING) << "select_last_index value only supports true for dpico. " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_ArgMaxChecker("ArgMaxFusion", new ArgMaxChecker());
}  // namespace dpico
}  // namespace mindspore
