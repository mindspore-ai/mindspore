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

#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class SigmoidGrad : public OpDesc {
 public:
  SigmoidGrad() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~SigmoidGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_y = inputs[0];
    const auto &dy = inputs[1];
    // Calculate sigmoid_grad(y, dy)
    // formula of sigmoid_grad is : (1 - y) * y * dy
    auto const_one = gb.Tensor(1.0, input_y->type);
    auto one_sub_y = gb.Sub(const_one, input_y);
    auto y_mul_dy = gb.Mul(input_y, dy);
    auto result = gb.Mul(one_sub_y, y_mul_dy);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("SigmoidGrad", SigmoidGrad);
}  // namespace mindspore::graphkernel::expanders
