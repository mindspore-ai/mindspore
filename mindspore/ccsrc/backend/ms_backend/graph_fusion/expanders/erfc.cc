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
class Erfc : public OpDesc {
 public:
  Erfc() {}
  ~Erfc() = default;

  static NodePtr Exec(const inner::GraphBuilder &gb, const NodePtrList &inputs) {
    const auto &input_x = inputs[0];
    if (input_x->type == kNumberTypeFloat16) {
      auto const_one = gb.Tensor(1.0, TypeId::kNumberTypeFloat32);
      auto x_f32 = gb.Cast(input_x, TypeId::kNumberTypeFloat32);
      auto erf_result = gb.Emit("Erf", {x_f32});
      auto result = gb.Sub(const_one, erf_result);
      result = gb.Cast(result, TypeId::kNumberTypeFloat16);
      return result;
    }
    auto const_one = gb.Tensor(1.0, input_x->type);
    auto erf_result = gb.Emit("Erf", {input_x});
    auto result = gb.Sub(const_one, erf_result);
    return result;
  }

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override { return {Exec(gb, inputs)}; }
};
EXPANDER_OP_DESC_REGISTER("Erfc", Erfc);
}  // namespace mindspore::graphkernel::expanders
