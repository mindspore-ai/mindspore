/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class GeLU : public OpDesc {
 public:
  GeLU() = default;
  ~GeLU() = default;

  static NodePtr Exec(const inner::GraphBuilder &gb, const NodePtrList &inputs) {
    constexpr double csv_value = 0.044715;
    // np.sqrt(2/np.pi)
    constexpr double csv_value_sqrt_two_div_pi = 0.7978845608028564;
    const auto &input_x = inputs[0];
    auto dtype = input_x->type;

    // cal y
    auto mul_0 = gb.Mul(input_x, input_x);
    auto pow_0 = gb.Mul(mul_0, input_x);
    auto const_csvalue = gb.Const(csv_value, dtype);
    auto mul_1 = gb.Mul(pow_0, const_csvalue);
    auto tanh_res = gb.Add(input_x, mul_1);
    auto const_csvalue_sqrt_two_div_pi = gb.Const(csv_value_sqrt_two_div_pi, dtype);
    auto y = gb.Mul(tanh_res, const_csvalue_sqrt_two_div_pi);

    // cal gelu(x)
    auto tanh_y = gb.Emit("Tanh", {y});
    auto const_one = gb.Const(1, dtype);
    auto const_half = gb.Const(0.5, dtype);
    auto tanh_y_add_one = gb.Add(tanh_y, const_one);
    auto mul_x = gb.Mul(input_x, tanh_y_add_one);
    auto result = gb.Mul(mul_x, const_half);
    return result;
  }

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override { return {Exec(gb, inputs)}; }
};
EXPANDER_OP_DESC_REGISTER("GeLU", GeLU);

NodePtr GeluExpand(const inner::GraphBuilder &gb, const NodePtrList &inputs) { return GeLU::Exec(gb, inputs); }
}  // namespace mindspore::graphkernel::expanders
