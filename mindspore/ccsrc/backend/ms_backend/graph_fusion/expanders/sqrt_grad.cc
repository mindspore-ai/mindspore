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
class SqrtGrad : public OpDesc {
 public:
  SqrtGrad() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~SqrtGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &x = inputs[0];
    const auto &dout = inputs[1];
    // formula of sqrt_grad is dout / (2 * x)
    auto const_two = gb.Tensor(2.0, x->type);
    auto two_x = gb.Mul(x, const_two);
    auto result = gb.Div(dout, two_x);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("SqrtGrad", SqrtGrad);
}  // namespace mindspore::graphkernel::expanders
