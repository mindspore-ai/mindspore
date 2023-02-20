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

#include "common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class SiLUGrad : public OpDesc {
 public:
  SiLUGrad() = default;
  ~SiLUGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &dout = inputs[0];
    const auto &x = inputs[1];
    auto sigmoid_input = gb.Emit("Sigmoid", {x}, {});
    auto bc_dx = gb.Mul(x, dout);
    auto bc_dy = gb.Mul(sigmoid_input, dout);
    auto dx = gb.Emit("SigmoidGrad", {sigmoid_input, bc_dx}, {});
    return {gb.Add(dx, bc_dy)};
  }
};
EXPANDER_OP_DESC_REGISTER("SiLUGrad", SiLUGrad);
}  // namespace mindspore::graphkernel::expanders
