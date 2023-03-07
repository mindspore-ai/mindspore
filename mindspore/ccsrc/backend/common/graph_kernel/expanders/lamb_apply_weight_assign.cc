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
#include <cmath>
#include <memory>
#include <vector>

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class LambApplyWeightAssign : public OpDesc {
 public:
  LambApplyWeightAssign() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~LambApplyWeightAssign() = default;

 protected:
  bool CheckInputs() override {
    const auto &g_norm = inputs_info_[1];
    if (g_norm.type != kNumberTypeFloat32 && g_norm.type != kNumberTypeFloat16) {
      MS_LOG(INFO) << "In LambApplyWeightAssign, g_norm's dtype must be float16 or float32";
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &w_norm = inputs[0];
    const auto &g_norm = inputs[1];
    const auto &input_lr = inputs[2];
    const auto &update = inputs[3];
    const auto &input_param = inputs[4];

    auto dtype = g_norm->type;

    double value_min;
    constexpr double num_two = 2;
    constexpr double pow_fp32 = -126;
    constexpr double pow_fp16 = -24;
    if (dtype == kNumberTypeFloat32) {
      value_min = std::pow(num_two, pow_fp32);
    } else {
      value_min = std::pow(num_two, pow_fp16);
    }
    auto data_min = gb.Const(value_min, dtype);
    auto const_zero = gb.Const(0.0, dtype);
    auto const_one = gb.Const(1.0, dtype);

    // w_norm >= 0, g_norm >= 0
    // ratio =  select(greater(w_norm, 0), select(greater(g_norm, 0), w_norm/g_norm, 1), 1)
    // cal ratio
    auto g_norm_greater_res = gb.Greater(g_norm, const_zero);
    auto g_norm_res = gb.Cast(g_norm_greater_res, dtype);

    auto g_norm_add = gb.Add(g_norm, data_min);
    auto w_norm_g_norm = gb.Div(w_norm, g_norm_add);

    auto g_norm_value_1 = gb.Mul(g_norm_res, w_norm_g_norm);

    // select
    auto g_norm_res_neg = gb.Neg(g_norm_res);
    auto g_norm_res_f = gb.Add(g_norm_res_neg, const_one);
    auto g_norm_value = gb.Add(g_norm_value_1, g_norm_res_f);

    auto w_norm_greater_res = gb.Greater(w_norm, const_zero);
    auto w_norm_res = gb.Cast(w_norm_greater_res, dtype);
    auto w_norm_value_1 = gb.Mul(w_norm_res, g_norm_value);

    // select
    auto w_norm_res_neg = gb.Neg(w_norm_res);
    auto w_norm_res_f = gb.Add(w_norm_res_neg, const_one);
    auto ratio = gb.Add(w_norm_value_1, w_norm_res_f);

    // ratio * input_lr * update
    auto update_with_ir = gb.Mul(update, input_lr);
    auto ratio_update_with_ir = gb.Mul(update_with_ir, ratio);

    // input_param - ratio_update_with_ir
    auto next_param = gb.Sub(input_param, ratio_update_with_ir);
    auto result = gb.Assign(input_param, next_param);

    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("LambApplyWeightAssign", LambApplyWeightAssign);
}  // namespace mindspore::graphkernel::expanders
