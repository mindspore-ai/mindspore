/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/graph_kernel/expanders/expander_factory.h"
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

  NodePtrList Expand() override {
    const auto &inputs = gb.Get()->inputs();
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

    // step1:  m_new <- beta1 * m + (1 - beta1) * grad
    auto m_b = gb.Emit("Mul", {beta1, m});
    tensor::TensorPtr data = std::make_shared<tensor::Tensor>(static_cast<double>(1.0), TypeIdToType(var->type));
    auto const_one = gb.Value(data);
    auto m1_beta1 = gb.Emit("Sub", {const_one, beta1});
    auto m_g = gb.Emit("Mul", {m1_beta1, grad});
    auto m_new = gb.Emit("Add", {m_b, m_g});

    // step2: v_new <- beta2 * v + (1 - beta2) * grad * grad
    auto v_b = gb.Emit("Mul", {beta2, v});
    auto m1_beta2 = gb.Emit("Sub", {const_one, beta2});
    auto grad_mul = gb.Emit("Mul", {grad, grad});
    auto v_g = gb.Emit("Mul", {m1_beta2, grad_mul});
    auto v_new = gb.Emit("Add", {v_b, v_g});
    // step3: lr_t <- lr * sqrt(1 - beta2_power) / (1 - beta1_power)
    auto m1_beta2_power = gb.Emit("Sub", {const_one, beta2_power});
    auto m1_beta2_power_sqrt = gb.Emit("Sqrt", {m1_beta2_power});
    auto m1_beta1_power = gb.Emit("Sub", {const_one, beta1_power});
    auto power_div = gb.Emit("RealDiv", {m1_beta2_power_sqrt, m1_beta1_power});
    auto lr_t = gb.Emit("Mul", {lr, power_div});

    // step4: if use_nesterov: var_new <- var - lr_t * (m_new * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_new))
    // if not use_nesterov: var_new <- var - lr_t * m_new / (epsilon + sqrt(v_new))
    auto v_new_sqrt = gb.Emit("Sqrt", {v_new});
    auto v_new_sqrt_e = gb.Emit("Add", {epsilon, v_new_sqrt});
    auto lr_t_div = gb.Emit("RealDiv", {lr_t, v_new_sqrt_e});
    mindspore::graphkernel::inner::NodePtr var_sub;
    if (GetValue<bool>(attrs_["use_nesterov"])) {
      auto m_new_mul = gb.Emit("Mul", {m_new, beta1});
      auto m_new_mul_add = gb.Emit("Add", {m_new_mul, m_g});
      var_sub = gb.Emit("Mul", {lr_t_div, m_new_mul_add});
    } else {
      var_sub = gb.Emit("Mul", {lr_t_div, m_new});
    }

    auto var_new = gb.Emit("Sub", {var, var_sub});
    auto var_result = gb.Emit("InplaceAssign", {var, var_new, var_new}, {{"fake_output", MakeValue(true)}});
    auto m_result = gb.Emit("InplaceAssign", {m, m_new, var_result}, {{"fake_output", MakeValue(true)}});
    auto v_result = gb.Emit("InplaceAssign", {v, v_new, m_result}, {{"fake_output", MakeValue(true)}});
    auto result = {var_result, m_result, v_result};
    return result;
  }
};
OP_EXPANDER_REGISTER("Adam", Adam);
}  // namespace mindspore::graphkernel::expanders
