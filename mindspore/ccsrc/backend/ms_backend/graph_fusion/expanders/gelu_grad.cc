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
class GeLUGrad : public OpDesc {
 public:
  GeLUGrad() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~GeLUGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    constexpr double cs_value = 0.044715;
    // np.sqrt(2/np.pi)
    constexpr double cs_sqrt_two_div_pi = 0.7978845608028564;
    // cs_value * 3
    constexpr double cs_value_tri = 0.134145;

    // cal formula are:
    // gelu_grad of dy and x is dy * y'
    // y' is 0.5 * (1.0 + tanh(tanh_para)) + 0.5 * x * (1.0 - tanh(tanh_para) * tanh(para)) * mul_right
    // tanh_para is 'sqrt(2.0 / pi) * (x + 0.044715 * x * x * x)'
    // mul_right is 'sqrt(2.0 / pi) * (1 + 3 * 0.044715 * x * x)'
    auto input_dy = inputs[0];
    auto input_x = inputs[1];
    auto dtype = input_x->type;
    if (dtype != kNumberTypeFloat32) {
      input_dy = gb.Cast(input_dy, kNumberTypeFloat32);
      input_x = gb.Cast(input_x, kNumberTypeFloat32);
    }
    // create const var
    auto const_csvalue = gb.Tensor(cs_value, input_dy->type);
    auto const_csvalue_sqrt_two_div_pi = gb.Tensor(cs_sqrt_two_div_pi, input_dy->type);
    auto const_csvalue_tri = gb.Tensor(cs_value_tri, input_dy->type);
    auto const_one = gb.Tensor(1.0, input_dy->type);
    auto const_half = gb.Tensor(0.5, input_dy->type);

    // cal mul_right
    auto mul_double = gb.Mul(input_x, input_x);
    auto mul_double_mul_tri = gb.Mul(const_csvalue_tri, mul_double);
    auto mul_add_one = gb.Add(const_one, mul_double_mul_tri);
    auto mul_right = gb.Mul(const_csvalue_sqrt_two_div_pi, mul_add_one);

    // cal tanh_para
    auto mul_triple = gb.Mul(input_x, mul_double);
    auto mul_triple_mul_csvalue = gb.Mul(const_csvalue, mul_triple);
    auto mul_add_x = gb.Add(input_x, mul_triple_mul_csvalue);
    auto tanh_para = gb.Mul(const_csvalue_sqrt_two_div_pi, mul_add_x);

    // cal 0.5 * (1.0 + tanh(tahn_para))
    auto tanh_res = gb.Tanh(tanh_para);
    auto tanh_res_add_one = gb.Add(const_one, tanh_res);
    auto half_mul_tanh_res_add_one = gb.Mul(const_half, tanh_res_add_one);

    // cal 0.5 * x * (1.0 - tanh(tanh_para) * tanh(para)) * mul_right
    auto tan_res_double = gb.Mul(tanh_res, tanh_res);
    auto one_sub_tan_res_double = gb.Sub(const_one, tan_res_double);
    auto half_mul_x = gb.Mul(const_half, input_x);
    auto mul_tmp = gb.Mul(half_mul_x, one_sub_tan_res_double);
    auto mul_final = gb.Mul(mul_tmp, mul_right);

    auto result_tmp = gb.Add(half_mul_tanh_res_add_one, mul_final);
    auto result = gb.Mul(input_dy, result_tmp);
    if (dtype != kNumberTypeFloat32) {
      result = gb.Cast(result, dtype);
    }
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("GeLUGrad", GeLUGrad);
}  // namespace mindspore::graphkernel::expanders
