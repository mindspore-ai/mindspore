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
class LambApplyOptimizerAssign : public OpDesc {
 public:
  LambApplyOptimizerAssign() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~LambApplyOptimizerAssign() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &grad = inputs[0];
    const auto &inputv = inputs[1];
    const auto &inputm = inputs[2];
    const auto &input_param = inputs[3];
    const auto &beta_1 = inputs[4];
    const auto &one_minus_beta_1 = inputs[5];
    const auto &beta_2 = inputs[6];
    const auto &one_minus_beta_2 = inputs[7];
    const auto &epsilon = inputs[8];
    const auto &steps = inputs[9];
    const auto &do_use_weight = inputs[10];
    const auto &weight_decay_rate = inputs[11];

    // next_v
    auto square_grad = gb.Mul(grad, grad);
    auto mul_3_result = gb.Mul(square_grad, one_minus_beta_2);
    auto mul_2_result = gb.Mul(inputv, beta_2);
    auto next_v = gb.Add(mul_2_result, mul_3_result);

    // next_m
    auto mul_0_result = gb.Mul(inputm, beta_1);
    auto mul_1_result = gb.Mul(grad, one_minus_beta_1);
    auto next_m = gb.Add(mul_0_result, mul_1_result);

    auto shape = next_m->shape;
    auto const_one = gb.Const(1.0, beta_2->type);

    auto beta_1_tensor = gb.BroadcastTo(beta_1, shape);
    auto beta_2_tensor = gb.BroadcastTo(beta_2, shape);

    // pow
    auto beta_1_log = gb.Log(beta_1_tensor);
    auto mul_res_1 = gb.Mul(beta_1_log, steps);
    auto beta_1_steps = gb.Exp(mul_res_1);

    auto neg_beta_1_step = gb.Neg(beta_1_steps);
    auto beta1_correction = gb.Add(neg_beta_1_step, const_one);

    auto next_m_unbiased = gb.Div(next_m, beta1_correction);

    // pow
    auto beta_2_log = gb.Log(beta_2_tensor);
    auto mul_res_2 = gb.Mul(beta_2_log, steps);
    auto beta_2_steps = gb.Exp(mul_res_2);

    auto neg_beta_2_step = gb.Neg(beta_2_steps);
    auto beta2_correction = gb.Add(neg_beta_2_step, const_one);

    auto next_v_unbiased = gb.Div(next_v, beta2_correction);
    // update
    auto sqrt_next_v = gb.Sqrt(next_v_unbiased);

    auto add_2_result = gb.Add(sqrt_next_v, epsilon);
    auto update = gb.Div(next_m_unbiased, add_2_result);
    // update do_use_weight_decay
    auto do_use_weight_mul = gb.Mul(input_param, weight_decay_rate);
    auto do_use_weight_decay = gb.Mul(do_use_weight_mul, do_use_weight);
    auto update_res = gb.Add(do_use_weight_decay, update);

    auto next_v_res = gb.Assign(inputv, next_v);
    auto next_m_res = gb.Assign(inputm, next_m);
    auto result = {update_res, next_v_res, next_m_res};

    return result;
  }
};
EXPANDER_OP_DESC_REGISTER("LambApplyOptimizerAssign", LambApplyOptimizerAssign);
}  // namespace mindspore::graphkernel::expanders
