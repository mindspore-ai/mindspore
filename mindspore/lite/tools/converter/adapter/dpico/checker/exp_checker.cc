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

#include "checker/exp_checker.h"
#include <vector>
#include <limits>
#include <cmath>
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
bool ExpFusionChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, kInputIndex1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }

  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  auto base_ptr = primitive->GetAttr(ops::kBase);
  if (base_ptr != nullptr) {
    auto base_data = api::GetValue<float>(base_ptr);
    if (base_data < 0 && std::fabs(base_data + 1.0) > std::numeric_limits<float>::epsilon()) {
      MS_LOG(WARNING) << "base val only supports -1 or positive num. " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_ExpFusionChecker("ExpFusion", new ExpFusionChecker());
}  // namespace dpico
}  // namespace mindspore
