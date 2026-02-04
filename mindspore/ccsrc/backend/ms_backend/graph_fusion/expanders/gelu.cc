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
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
namespace {
constexpr double csv_value = 0.044714998453855515;
// gelu(x) = 0.5 * x *(1 + Erf(x/sqrt(2)))
// gelu(x) = 0.5 * x *(1 + tanh(y)), y = sqrt(2/pi)*(x + 0.044715*x*x*x)
// Since in AKG or Ascend, there is no basic instruction for tanh, it is formed by combining basic instructions.
// Therefore, we expand tanh(x).
// tanh(x) = (e^x - e^{-x})/(e^x + e^{-x}) = 2/(1 + e^{-2x}) - 1
// gelu(x) = 0.5 *x * (1 + 2/(1 + e^{-2y}) - 1) = x/(1 + e^{-2y})
// After expanding, we find that the number of basic operators has reduced from 14 to 8, and memory can be reused
// (only input and output memory are needed to complete one GELU operation). Reflected in AKG, for 1024x1024 float16
// performing GELU, the original expansion on 910B required 64 cores, while the new code only needs 32 cores.
// Moreover, the basic instructions have been significantly reduced, leading to an 18% improvement in kernel
// performance.
NodePtr GeLUByTanh(const inner::GraphBuilder &gb, const NodePtr &input_x, const TypeId dtype) {
  // np.sqrt(2/np.pi)
  constexpr double csv_value_sqrt_two_div_pi = 0.7978845608028654;

  // cal y
  auto mul_0 = gb.Mul(input_x, input_x);
  auto pow_0 = gb.Mul(mul_0, input_x);
  auto const_csvalue = gb.Tensor(csv_value, dtype);
  auto mul_1 = gb.Mul(pow_0, const_csvalue);
  auto tanh_res = gb.Add(input_x, mul_1);
  auto const_csvalue_sqrt_two_div_pi = gb.Tensor(csv_value_sqrt_two_div_pi, dtype);
  auto y = gb.Mul(tanh_res, const_csvalue_sqrt_two_div_pi);

  // cal gelu(x)
  auto tanh_y = gb.Tanh(y);
  auto const_one = gb.Tensor(1, dtype);
  auto const_half = gb.Tensor(0.5, dtype);
  auto tanh_y_add_one = gb.Add(tanh_y, const_one);
  auto mul_x = gb.Mul(input_x, tanh_y_add_one);
  auto result = gb.Mul(mul_x, const_half);

  return result;
}

NodePtr GeLUAscend(const inner::GraphBuilder &gb, const NodePtr &input, const TypeId dtype) {
  // -np.sqrt(8/np.pi)
  constexpr double csv_value_sqrt_eight_div_pi = -1.5957691216057308;
  auto input_x = input;
  if (dtype != kNumberTypeFloat32) {
    input_x = gb.Cast(input_x, kNumberTypeFloat32);
  }
  auto mul_0 = gb.Mul(input_x, input_x);
  auto pow_0 = gb.Mul(mul_0, input_x);
  auto const_csvalue = gb.Tensor(csv_value, kNumberTypeFloat32);
  auto mul_1 = gb.Mul(pow_0, const_csvalue);
  auto tanh_res = gb.Add(input_x, mul_1);
  auto const_csvalue_sqrt_eight_div_pi = gb.Tensor(csv_value_sqrt_eight_div_pi, kNumberTypeFloat32);
  auto y = gb.Mul(tanh_res, const_csvalue_sqrt_eight_div_pi);

  auto exp_0 = gb.Exp(y);
  auto const_one = gb.Tensor(1, kNumberTypeFloat32);
  auto add_0 = gb.Add(exp_0, const_one);
  auto result = gb.Div(input_x, add_0);
  if (dtype != kNumberTypeFloat32) {
    result = gb.Cast(result, dtype);
  }
  return result;
}
}  // namespace
class GeLU : public OpDesc {
 public:
  GeLU() = default;
  ~GeLU() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    if (processor_ == "aicore") {
      return {GeLUAscend(gb, inputs[0], inputs[0]->type)};
    } else {
      return {GeLUByTanh(gb, inputs[0], inputs[0]->type)};
    }
  }
};
EXPANDER_OP_DESC_REGISTER("GeLU", GeLU);

NodePtr GeluExpand(const inner::GraphBuilder &gb, const NodePtrList &inputs) {
  return GeLUAscend(gb, inputs[0], inputs[0]->type);
}
}  // namespace mindspore::graphkernel::expanders
