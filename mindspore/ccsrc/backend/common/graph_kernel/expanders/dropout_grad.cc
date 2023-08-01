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
#include "backend/common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class DropoutGrad : public OpDesc {
 public:
  DropoutGrad() {
    (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>());
    std::initializer_list<std::string> attrs{"keep_prob"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~DropoutGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_dy = inputs[0];
    const auto &input_mask = inputs[1];
    auto keep_prob = GetValue<float>(attrs_["keep_prob"]);
    auto r_keep_prob = gb.Tensor(1.0f / keep_prob, input_dy->type);
    auto result = gb.Mul(input_dy, r_keep_prob);
    result = gb.Mul(result, input_mask);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("DropoutGrad", DropoutGrad);
}  // namespace mindspore::graphkernel::expanders
