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

#include "checker/slice_checker.h"
#include <vector>
#include <string>
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int kMaxSplitSize = 31;
}  // namespace

bool SliceChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (output_num > kMaxTopNum) {
    MS_LOG(WARNING) << "output num " << output_num << " is greater than " << kMaxNumOutput << " "
                    << op->fullname_with_scope();
    return false;
  }
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  auto axis_ptr = primitive->GetAttr(ops::kAxes);
  if (axis_ptr != nullptr) {
    auto axis_data = api::GetValue<std::vector<int64_t>>(axis_ptr);
    if (axis_data[0] < kAxisLowerBound || axis_data[0] > kAxisUpperBound) {
      MS_LOG(WARNING) << "axis val should in range [-4, 3]. " << op->fullname_with_scope();
      return false;
    }
  }

  if (primitive->GetAttr(ops::kSizeSplits) != nullptr) {
    auto splits = api::GetValue<std::vector<int64_t>>(primitive->GetAttr(ops::kSizeSplits));
    if (splits.size() > kMaxSplitSize) {
      MS_LOG(WARNING) << "split size should be less than " << kMaxSplitSize << " " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_SliceChecker("SliceFusion", new SliceChecker());
}  // namespace dpico
}  // namespace mindspore
