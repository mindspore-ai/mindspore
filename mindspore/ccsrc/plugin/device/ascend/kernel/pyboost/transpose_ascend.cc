/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/pyboost/transpose_ascend.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
bool TransposeAscend::Launch(const tensor::TensorPtr &input, const vector<int64_t> &input_perm) { return true; }

tensor::TensorPtr TransposeAscend::Call(const tensor::TensorPtr &input, const ValueTuplePtr &input_perm) {
  std::vector<int64_t> axis;
  for (const auto &val : input_perm->value()) {
    (void)axis.emplace_back(GetValue<int64_t>(val));
  }
  PyboostProcessView(input, axis);
  return outputs_[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
