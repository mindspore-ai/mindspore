/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
class Adam : public OpDesc {
 public:
  Adam() {
    std::initializer_list<std::string> attrs{"use_nesterov"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Adam() = default;

 protected:
  bool CheckInputs() override {
    const auto &var = inputs_info_[0];
    if (var.type != kNumberTypeFloat32 && var.type != kNumberTypeFloat16) {
      MS_LOG(INFO) << "In Adam, var's dtype must be float16 or float32";
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &var = inputs[0];
    const auto &m = inputs[1];
    const auto &v = inputs[2];
    const auto &beta1_power = inputs[3];
    const auto &beta2_power = inputs[4];
    const auto &lr = inputs[5];
    const auto &beta1 = inputs[6];
    const auto &beta2 = inputs[7];
    const auto &epsilon = inputs[8];
    const auto &grad = inputs[9];

    // calc m_new : m_new = beta1 * m + (1 - beta1) * grad
    auto m_b = gb.Mul(beta1, m);
    auto const_one = gb.Const(1.0, var->type);
    auto m1_beta1 = gb.Sub(const_one, beta1);
    auto m_g = gb.Mul(m1_beta1, grad);
    auto m_new = gb.Add(m_b, m_g);

    // calc v_new: v_new = beta2 * v + (1 - beta2) * grad * grad
    auto v_b = gb.Mul(beta2, v);
    auto m1_beta2 = gb.Sub(const_one, beta2);
    auto grad_mul = gb.Mul(grad, grad);
    auto v_g = gb.Mul(m1_beta2, grad_mul);
    auto v_new = gb.Add(v_b, v_g);

    // calc lr_t: lr_t = lr * sqrt(1 - beta2_power) / (1 - beta1_power);
    auto m1_beta2_power = gb.Sub(const_one, beta2_power);
    auto m1_beta2_power_sqrt = gb.Sqrt(m1_beta2_power);
    auto m1_beta1_power = gb.Sub(const_one, beta1_power);
    auto power_div = gb.Div(m1_beta2_power_sqrt, m1_beta1_power);
    auto lr_t = gb.Mul(lr, power_div);

    // if use_nesterov: var_new <- var - lr_t * (m_new * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_new))
    // if not use_nesterov: var_new <- var - lr_t * m_new / (epsilon + sqrt(v_new))
    auto v_new_sqrt = gb.Sqrt(v_new);
    auto v_new_sqrt_e = gb.Add(epsilon, v_new_sqrt);
    auto lr_t_div = gb.Div(lr_t, v_new_sqrt_e);
    mindspore::graphkernel::inner::NodePtr var_sub;
    if (GetValue<bool>(attrs_["use_nesterov"])) {
      auto m_new_mul = gb.Mul(m_new, beta1);
      auto m_new_mul_add = gb.Add(m_new_mul, m_g);
      var_sub = gb.Mul(lr_t_div, m_new_mul_add);
    } else {
      var_sub = gb.Mul(lr_t_div, m_new);
    }

    auto var_new = gb.Sub(var, var_sub);
    auto var_result = gb.Assign(var, var_new);
    auto m_result = gb.Assign(m, m_new);
    auto v_result = gb.Assign(v, v_new);
    auto result = {var_result, m_result, v_result};
    return result;
  }
};
EXPANDER_OP_DESC_REGISTER("Adam", Adam);
}  // namespace mindspore::graphkernel::expanders
