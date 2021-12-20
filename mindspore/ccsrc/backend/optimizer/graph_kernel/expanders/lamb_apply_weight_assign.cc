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
#include <cmath>
#include <memory>
#include <vector>

#include "backend/optimizer/graph_kernel/expanders/expander_factory.h"
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

  NodePtrList Expand() override {
    const auto &inputs = gb.Get()->inputs();
    const auto &w_norm = inputs[0];
    const auto &g_norm = inputs[1];
    const auto &input_lr = inputs[2];
    const auto &update = inputs[3];
    const auto &input_param = inputs[4];

    auto dtype = g_norm->type;

    tensor::TensorPtr value_min;
    constexpr double num_two = 2;
    constexpr double pow_fp32 = -126;
    constexpr double pow_fp16 = -24;
    if (dtype == kNumberTypeFloat32) {
      value_min = std::make_shared<tensor::Tensor>(std::pow(num_two, pow_fp32), TypeIdToType(dtype));
    } else {
      value_min = std::make_shared<tensor::Tensor>(std::pow(num_two, pow_fp16), TypeIdToType(dtype));
    }
    auto data_min = gb.Value(value_min);

    tensor::TensorPtr data_zero = std::make_shared<tensor::Tensor>(static_cast<double>(0.0), TypeIdToType(dtype));
    tensor::TensorPtr data_one = std::make_shared<tensor::Tensor>(static_cast<double>(1.0), TypeIdToType(dtype));
    auto const_zero = gb.Value(data_zero);
    auto const_one = gb.Value(data_one);

    // w_norm >= 0, g_norm >= 0
    // ratio =  select(greater(w_norm, 0), select(greater(g_norm, 0), w_norm/g_norm, 1), 1)
    // cal ratio
    auto g_norm_greater_res = gb.Emit("Greater", {g_norm, const_zero});
    auto g_norm_res = gb.Emit("Cast", {g_norm_greater_res}, {{"dst_type", TypeIdToType(dtype)}});

    auto g_norm_add = gb.Emit("Add", {g_norm, data_min});
    auto w_norm_g_norm = gb.Emit("RealDiv", {w_norm, g_norm_add});

    auto g_norm_value_1 = gb.Emit("Mul", {g_norm_res, w_norm_g_norm});
    // select
    auto g_norm_res_neg = gb.Emit("Neg", {g_norm_res});
    auto g_norm_res_f = gb.Emit("Add", {g_norm_res_neg, const_one});
    auto g_norm_value = gb.Emit("Add", {g_norm_value_1, g_norm_res_f});

    auto w_norm_greater_res = gb.Emit("Greater", {w_norm, const_zero});
    auto w_norm_res = gb.Emit("Cast", {w_norm_greater_res}, {{"dst_type", TypeIdToType(dtype)}});

    auto w_norm_value_1 = gb.Emit("Mul", {w_norm_res, g_norm_value});
    // select
    auto w_norm_res_neg = gb.Emit("Neg", {w_norm_res});
    auto w_norm_res_f = gb.Emit("Add", {w_norm_res_neg, const_one});
    auto ratio = gb.Emit("Add", {w_norm_value_1, w_norm_res_f});

    // ratio * input_lr * update
    auto update_with_ir = gb.Emit("Mul", {update, input_lr});
    auto ratio_update_with_ir = gb.Emit("Mul", {update_with_ir, ratio});

    // input_param - ratio_update_with_ir
    auto next_param = gb.Emit("Sub", {input_param, ratio_update_with_ir});
    auto result = gb.Emit("Assign", {input_param, next_param});

    return {result};
  }
};
OP_EXPANDER_REGISTER("LambApplyWeightAssign", LambApplyWeightAssign);
}  // namespace mindspore::graphkernel::expanders
