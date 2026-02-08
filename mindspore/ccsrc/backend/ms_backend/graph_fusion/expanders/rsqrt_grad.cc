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
class RsqrtGrad : public OpDesc {
 public:
  RsqrtGrad() = default;
  ~RsqrtGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_y = inputs[0];
    const auto &dy = inputs[1];
    auto y0 = gb.Mul(input_y, input_y);
    auto y1 = gb.Mul(y0, input_y);
    auto y2 = gb.Mul(y1, dy);
    auto result = gb.Mul(y2, gb.Tensor(-0.5, y2->type));
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("RsqrtGrad", RsqrtGrad);
}  // namespace mindspore::graphkernel::expanders
