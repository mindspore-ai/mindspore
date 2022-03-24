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

#include "checker/eltwise_checker.h"
#include <vector>
#include <string>
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int kModeSize = 3;
}  // namespace
bool EltwiseChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  auto mode_ptr = primitive->GetAttr(ops::kMode);
  if (mode_ptr != nullptr) {
    auto mode_data = api::GetValue<int64_t>(mode_ptr);
    if (mode_data >= kModeSize) {  // only prod(0), sum(1), max(2) is supported
      MS_LOG(WARNING) << "mode only supports 0/1/2 " << op->fullname_with_scope();
      return false;
    }
  }

  if (op->inputs().size() - 1 > kMaxBottomNum) {
    MS_LOG(WARNING) << "op inputs size:" << op->inputs().size() << " is greater than max_bottom num:" << kMaxBottomNum
                    << " " << op->fullname_with_scope();
    return false;
  }

  std::vector<int64_t> input_shape;
  for (size_t i = 1; i < op->inputs().size(); i++) {
    if (!CheckInputW(op, i, format, kMaxInputWOf4Dims)) {
      MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
      return false;
    }
    input_shape.clear();
  }
  return true;
}

OpCheckerRegistrar g_EltwiseChecker("Eltwise", new EltwiseChecker());
}  // namespace dpico
}  // namespace mindspore
