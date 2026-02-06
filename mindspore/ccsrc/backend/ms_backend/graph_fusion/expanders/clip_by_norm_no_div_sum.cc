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

#include <memory>
#include <vector>
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class ClipByNormNoDivSum : public OpDesc {
 public:
  ClipByNormNoDivSum() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~ClipByNormNoDivSum() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x0 = inputs[0];
    const auto &input_x1 = inputs[1];
    const auto &input_x2 = inputs[2];
    const auto &input_x3 = inputs[3];

    auto greater_res = gb.Greater(input_x0, input_x1);
    auto select_res0 = gb.Select(greater_res, input_x0, input_x2);
    auto sqrt_res = gb.Sqrt(select_res0);
    auto select_res1 = gb.Select(greater_res, sqrt_res, input_x0);
    auto result = gb.Emit("Maximum", {select_res1, input_x3});

    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("ClipByNormNoDivSum", ClipByNormNoDivSum);
}  // namespace mindspore::graphkernel::expanders
