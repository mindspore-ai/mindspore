/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

namespace mindspore::graphkernel::expanders {
constexpr size_t kGradIdx = 0;
constexpr size_t kVIdx = 1;
constexpr size_t kMIdx = 2;
constexpr size_t kVarIdx = 3;
constexpr size_t kLrIdx = 4;
constexpr size_t kBeta1Idx = 5;
constexpr size_t kBeta1ApplyOneIdx = 6;
constexpr size_t kBeta2Idx = 7;
constexpr size_t kBeta2ApplyOneIdx = 8;
constexpr size_t kDecayIdx = 9;
constexpr size_t kEpsilonIdx = 10;

class AdamApplyOneWithDecay : public OpDesc {
 public:
  AdamApplyOneWithDecay() {}
  ~AdamApplyOneWithDecay() = default;

 protected:
  NodePtrList Compute() {
    // calc m_new : m_new = beta1 * m + (1 - beta1) * grad
    auto m_b = gb.Mul(beta1_, m_);
    auto m_g = gb.Mul(beta1_apply_one_, grad_);
    auto m_new = gb.Add(m_b, m_g);

    // calc v_new: v_new = beta2 * v + (1 - beta2) * grad * grad
    auto v_b = gb.Mul(beta2_, v_);
    auto grad_mul = gb.Mul(grad_, grad_);
    auto v_g = gb.Mul(beta2_apply_one_, grad_mul);
    auto v_new = gb.Add(v_b, v_g);

    // calc var_new: var_new = var - (m_new / (sqrt(v_new) + epsilon) + decay * var) * lr
    auto v_sqrt = gb.Sqrt(v_new);
    auto sqrt_ep = gb.Add(v_sqrt, epsilon_);
    auto update = gb.Div(m_new, sqrt_ep);
    auto decay_var = gb.Mul(decay_, var_);
    auto new_update = gb.Add(update, decay_var);
    auto lr_update = gb.Mul(lr_, new_update);
    auto var_new = gb.Sub(var_, lr_update);

    auto result = {v_new, m_new, var_new};
    return result;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    var_ = inputs[kVarIdx];
    m_ = inputs[kMIdx];
    v_ = inputs[kVIdx];
    lr_ = inputs[kLrIdx];
    beta1_ = inputs[kBeta1Idx];
    beta2_ = inputs[kBeta2Idx];
    epsilon_ = inputs[kEpsilonIdx];
    decay_ = inputs[kDecayIdx];
    grad_ = inputs[kGradIdx];
    origin_var_ = inputs[kVarIdx];
    origin_m_ = inputs[kMIdx];
    origin_v_ = inputs[kVIdx];
    beta1_apply_one_ = inputs[kBeta1ApplyOneIdx];
    beta2_apply_one_ = inputs[kBeta2ApplyOneIdx];

    NodePtrList new_value;
    new_value = Compute();
    return new_value;
  }

  NodePtr var_;
  NodePtr m_;
  NodePtr v_;
  NodePtr lr_;
  NodePtr beta1_;
  NodePtr beta2_;
  NodePtr epsilon_;
  NodePtr decay_;
  NodePtr grad_;
  NodePtr origin_var_;
  NodePtr origin_m_;
  NodePtr origin_v_;
  NodePtr beta1_apply_one_;
  NodePtr beta2_apply_one_;
};
EXPANDER_OP_DESC_REGISTER("AdamApplyOneWithDecay", AdamApplyOneWithDecay);
}  // namespace mindspore::graphkernel::expanders
