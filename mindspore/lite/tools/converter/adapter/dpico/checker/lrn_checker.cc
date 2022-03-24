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

#include "checker/lrn_checker.h"
#include <vector>
#include <string>
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int local_size_3 = 3;
constexpr int local_size_5 = 5;
constexpr int local_size_7 = 7;
}  // namespace
bool LRNChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, 1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }

  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  auto r_ptr = primitive->GetAttr(ops::kDepthRadius);
  if (r_ptr != nullptr) {
    auto r_data = api::GetValue<int64_t>(r_ptr) * 2 + 1;
    if (r_data != local_size_3 && r_data != local_size_5 && r_data != local_size_7) {
      MS_LOG(WARNING) << "local size only supports 3/5/7 " << op->fullname_with_scope();
      return false;
    }
  }
  auto norm_region_ptr = primitive->GetAttr(ops::kNormRegion);
  if (norm_region_ptr != nullptr) {
    auto norm_region_data = api::GetValue<std::string>(norm_region_ptr);
    if (norm_region_data != "ACROSS_CHANNEL") {
      MS_LOG(WARNING) << "norm region only supports ACROSS_CHANNEL " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_LRNChecker("LRN", new LRNChecker());
}  // namespace dpico
}  // namespace mindspore
