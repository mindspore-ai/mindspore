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

#include "checker/scale_checker.h"
#include <vector>
#include <string>
#include <unordered_set>
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int kNegativeAxisCorrespondZero = -4;
constexpr int kNegativeAxisCorrespondOne = -3;
}  // namespace
bool ScaleChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, 1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }

  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  auto axis_ptr = primitive->GetAttr(ops::kAxis);
  if (axis_ptr != nullptr) {
    auto axis_data = api::GetValue<int64_t>(axis_ptr);
    std::unordered_set<int64_t> range = {1, kNegativeAxisCorrespondOne, 0, kNegativeAxisCorrespondZero};
    if (range.find(axis_data) == range.end()) {
      MS_LOG(WARNING) << "axis val only supports 1/-3/0/-4 by dpico. " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_ScaleChecker("ScaleFusion", new ScaleChecker());
}  // namespace dpico
}  // namespace mindspore
