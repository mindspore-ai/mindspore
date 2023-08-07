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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class ReluGrad : public OpDesc {
 public:
  ReluGrad() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~ReluGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    const auto &input_y = inputs[1];

    auto const_zero = gb.Tensor(0, input_y->type);
    auto ge_result = gb.Greater(input_y, const_zero);
    ge_result = gb.Cast(ge_result, input_x->type);
    auto result = gb.Mul(ge_result, input_x);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("ReluGrad", ReluGrad);
}  // namespace mindspore::graphkernel::expanders
