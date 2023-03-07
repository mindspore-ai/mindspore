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
constexpr size_t kVarIdx = 0;
constexpr size_t kMIdx = 1;
constexpr size_t kVIdx = 2;
constexpr size_t kLrIdx = 3;
constexpr size_t kBeta1Idx = 4;
constexpr size_t kBeta2Idx = 5;
constexpr size_t kEpsilonIdx = 6;
constexpr size_t kDecayIdx = 7;
constexpr size_t kGradIdx = 8;
class AdamWeightDecay : public OpDesc {
 public:
  AdamWeightDecay() {}
  ~AdamWeightDecay() = default;

 protected:
  bool CheckInputs() override {
    const auto &var_info = inputs_info_[kVarIdx];
    const auto &m_info = inputs_info_[kMIdx];
    const auto &v_info = inputs_info_[kVIdx];
    const auto &lr_info = inputs_info_[kLrIdx];
    const auto &beta1_info = inputs_info_[kBeta1Idx];
    const auto &beta2_info = inputs_info_[kBeta2Idx];
    const auto &epsilon_info = inputs_info_[kEpsilonIdx];
    const auto &decay_info = inputs_info_[kDecayIdx];
    const auto &grad_info = inputs_info_[kGradIdx];
    if (var_info.type != grad_info.type) {
      MS_LOG(INFO) << "In AdamWeightDecay, var's dtype must be equal to grad's type";
      return false;
    }
    if (m_info.type != v_info.type) {
      MS_LOG(INFO) << "In AdamWeightDecay, m's dtype must be equal to v's type";
      return false;
    }
    if (lr_info.type != kNumberTypeFloat32 || beta1_info.type != kNumberTypeFloat32 ||
        beta2_info.type != kNumberTypeFloat32 || epsilon_info.type != kNumberTypeFloat32 ||
        decay_info.type != kNumberTypeFloat32) {
      MS_LOG(INFO) << "In AdamWeightDecay, lr, beta1 ,beta2, epsilon and decay's dtype must be float32";
      return false;
    }
    return true;
  }

  NodePtrList Compute() {
    // calc m_new : m_new = beta1 * m + (1 - beta1) * grad
    auto m_b = gb.Mul(beta1_, m_);
    auto const_one1 = gb.Const(1.0, beta1_->type);
    auto m1_beta1 = gb.Sub(const_one1, beta1_);
    auto m_g = gb.Mul(m1_beta1, grad_);
    auto m_new = gb.Add(m_b, m_g);

    // calc v_new: v_new = beta2 * v + (1 - beta2) * grad * grad
    auto v_b = gb.Mul(beta2_, v_);
    auto const_one2 = gb.Const(1.0, beta2_->type);
    auto m1_beta2 = gb.Sub(const_one2, beta2_);
    auto grad_mul = gb.Mul(grad_, grad_);
    auto v_g = gb.Mul(m1_beta2, grad_mul);
    auto v_new = gb.Add(v_b, v_g);

    // calc var_new: var_new = var - (m_new / (sqrt(v_new) + epsilon) + decay * var) * lr
    auto v_sqrt = gb.Sqrt(v_new);
    auto sqrt_ep = gb.Add(v_sqrt, epsilon_);
    auto update = gb.Div(m_new, sqrt_ep);
    auto decay_var = gb.Mul(decay_, var_);
    auto new_update = gb.Add(update, decay_var);
    auto lr_update = gb.Mul(lr_, new_update);
    auto var_new = gb.Sub(var_, lr_update);

    auto result = {var_new, m_new, v_new};
    return result;
  }

  NodePtrList DoAssign(const NodePtrList &new_value) {
    const auto &var_new = new_value[kVarIdx];
    const auto &m_new = new_value[kMIdx];
    const auto &v_new = new_value[kVIdx];
    auto var_result = gb.Assign(origin_var_, var_new);
    auto m_result = gb.Assign(origin_m_, m_new);
    auto v_result = gb.Assign(origin_v_, v_new);
    auto result = {var_result, m_result, v_result};
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

    NodePtrList new_value;
    if (origin_var_->type == kNumberTypeFloat16 && origin_m_->type == kNumberTypeFloat16) {
      lr_ = gb.Cast(lr_, kNumberTypeFloat16);
      beta1_ = gb.Cast(beta1_, kNumberTypeFloat16);
      beta2_ = gb.Cast(beta2_, kNumberTypeFloat16);
      epsilon_ = gb.Cast(epsilon_, kNumberTypeFloat16);
      decay_ = gb.Cast(decay_, kNumberTypeFloat16);
      new_value = Compute();
    } else if (origin_var_->type == kNumberTypeFloat16) {
      var_ = gb.Cast(var_, kNumberTypeFloat32);
      grad_ = gb.Cast(grad_, kNumberTypeFloat32);
      new_value = Compute();
      new_value[0] = gb.Cast(new_value[0], kNumberTypeFloat16);
    } else {
      new_value = Compute();
    }
    return DoAssign(new_value);
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
};
EXPANDER_OP_DESC_REGISTER("AdamWeightDecay", AdamWeightDecay);
}  // namespace mindspore::graphkernel::expanders
