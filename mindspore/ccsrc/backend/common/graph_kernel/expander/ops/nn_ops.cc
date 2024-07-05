/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include "backend/common/graph_kernel/expander/base/ir_builder.h"
#include "backend/common/graph_kernel/expander/base/utils.h"
#include "kernel/common_utils.h"

namespace mindspore::graphkernel::expander {
REG_EXPANDER_FUNC("Sigmoid").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  auto const_one = ib->Tensor(1.0, input_x->GetDtype());
  auto neg_x = ib->Neg(input_x);
  auto exp_neg_x = ib->Exp(neg_x);
  auto add_exp = ib->Add(exp_neg_x, const_one);
  auto result = ib->Div(const_one, add_exp);
  return {result};
});

REG_EXPANDER_FUNC("SoftmaxBackward").SetBody(BODYFUNC(ib) {
  auto dout = ib->input(kIndex0);
  auto out = ib->input(kIndex1);
  auto dim = ib->input(kIndex2);
  auto dim_value_ptr = dim->GetValue();
  if (dim_value_ptr == nullptr || dim_value_ptr->isa<ValueAny>() || dim_value_ptr->isa<None>()) {
    MS_LOG(INFO) << "dim is not const value";
    return {};
  }
  auto dim_value = GetValue<int64_t>(dim_value_ptr);
  auto shp = out->GetShape();
  bool is_last_axis = true;
  if (IsDynamicRank(shp)) {
    is_last_axis = (dim_value == -1);
  } else {
    auto nd = SizeToLong(shp.size());
    is_last_axis = dim_value < 0 ? (dim_value == -1) : (dim_value == nd - 1);
  }
  if (!is_last_axis) {
    MS_LOG(INFO) << "dim is not last axis";
    return {};
  }
  ShapeVector axis{-1};
  auto result = ib->Mul(out, ib->Sub(dout, ib->ReduceSum(ib->Mul(out, dout), ib->Value(axis), ib->Value(true))));
  return {result};
});

REG_EXPANDER_FUNC("ApplyMomentum").SetBody(BODYFUNC(ib) {
  auto weight = ib->input(kIndex0);
  auto accumulate = ib->input(kIndex1);
  auto lr = ib->input(kIndex2);
  auto gradient = ib->input(kIndex3);
  auto moment = ib->input(kIndex4);
  auto mul1 = ib->Mul(accumulate, moment);
  auto acc_new = ib->Add(mul1, gradient);
  auto mul2 = ib->Mul(acc_new, lr);
  auto weight_new = ib->Sub(weight, mul2);

  auto assign1 = ib->Assign(accumulate, acc_new);
  auto assign2 = ib->Assign(weight, weight_new);

  auto result = {assign1, assign2};
  return result;
});

REG_EXPANDER_FUNC("Adam").SetBody(BODYFUNC(ib) {
  // Check Inputs and Attrs
  if (!CheckAttrs(ib, {"use_nesterov"})) {
    return {};
  }
  const auto &var = ib->input(0);
  if (var->GetDtype() != TypeIdToType(kNumberTypeFloat32) && var->GetDtype() != TypeIdToType(kNumberTypeFloat16)) {
    MS_LOG(INFO) << "In Adam, var's dtype must be float16 or float32, but got " << var->GetDtype()->ToString();
    return {};
  }
  // Expand
  const auto &m = ib->input(1);
  const auto &v = ib->input(2);
  const auto &beta1_power = ib->input(3);
  const auto &beta2_power = ib->input(4);
  const auto &lr = ib->input(5);
  const auto &beta1 = ib->input(6);
  const auto &beta2 = ib->input(7);
  const auto &epsilon = ib->input(8);
  const auto &grad = ib->input(9);

  // calc m_new : m_new = beta1 * m + (1 - beta1) * grad
  auto m_b = ib->Mul(beta1, m);
  auto const_one = ib->Tensor(1.0, var->GetDtype());
  auto m1_beta1 = ib->Sub(const_one, beta1);
  auto m_g = ib->Mul(m1_beta1, grad);
  auto m_new = ib->Add(m_b, m_g);

  // calc v_new: v_new = beta2 * v + (1 - beta2) * grad * grad
  auto v_b = ib->Mul(beta2, v);
  auto m1_beta2 = ib->Sub(const_one, beta2);
  auto grad_mul = ib->Mul(grad, grad);
  auto v_g = ib->Mul(m1_beta2, grad_mul);
  auto v_new = ib->Add(v_b, v_g);

  // calc lr_t: lr_t = lr * sqrt(1 - beta2_power) / (1 - beta1_power);
  auto m1_beta2_power = ib->Sub(const_one, beta2_power);
  auto m1_beta2_power_sqrt = ib->Sqrt(m1_beta2_power);
  auto m1_beta1_power = ib->Sub(const_one, beta1_power);
  auto power_div = ib->Div(m1_beta2_power_sqrt, m1_beta1_power);
  auto lr_t = ib->Mul(lr, power_div);

  // if use_nesterov: var_new <- var - lr_t * (m_new * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_new))
  // if not use_nesterov: var_new <- var - lr_t * m_new / (epsilon + sqrt(v_new))
  auto v_new_sqrt = ib->Sqrt(v_new);
  auto v_new_sqrt_e = ib->Add(epsilon, v_new_sqrt);
  auto lr_t_div = ib->Div(lr_t, v_new_sqrt_e);
  NodePtr var_sub;
  if (GetValue<bool>(ib->attr("use_nesterov"))) {
    auto m_new_mul = ib->Mul(m_new, beta1);
    auto m_new_mul_add = ib->Add(m_new_mul, m_g);
    var_sub = ib->Mul(lr_t_div, m_new_mul_add);
  } else {
    var_sub = ib->Mul(lr_t_div, m_new);
  }

  auto var_new = ib->Sub(var, var_sub);
  auto var_result = ib->Assign(var, var_new);
  auto m_result = ib->Assign(m, m_new);
  auto v_result = ib->Assign(v, v_new);
  NodePtrList result = {var_result, m_result, v_result};
  return result;
});

REG_EXPANDER_FUNC("DropoutGrad").SetBody(BODYFUNC(ib) {
  // Check Inputs and Attrs
  if (!CheckAllFormatsSame(ib) || !CheckAttrs(ib, {"keep_prob"})) {
    return {};
  }
  // Expand
  const auto &input_dy = ib->input(0);
  const auto &input_mask = ib->input(1);
  auto keep_prob = GetValue<float>(ib->attr("keep_prob"));
  auto r_keep_prob = ib->Tensor(1.0f / keep_prob, input_dy->GetDtype());
  auto result = ib->Mul(input_dy, r_keep_prob);
  result = ib->Mul(result, input_mask);
  return {result};
});

REG_EXPANDER_FUNC("BiasAdd").SetBody(BODYFUNC(ib) {
  // Expand
  auto input_x = ib->input(0);
  auto input_y = ib->input(1);
  auto y_shape = input_y->GetShape();
  if (IsDynamicRank(y_shape) || std::count_if(y_shape.begin(), y_shape.end(), [](int64_t sh) { return sh < 0; }) > 1) {
    MS_LOG(DEBUG) << "Bias is dynamic shape";
    return {};
  }
  if (input_x->GetFormat() == kOpFormat_NCHW) {
    auto target_shape = ExpandDimsInferShape(y_shape, {1, 2});
    input_y = ib->Reshape(input_y, target_shape);
  } else if (input_x->GetFormat() == kOpFormat_DEFAULT) {
    auto x_shape = input_x->GetShape();
    if (IsDynamicRank(x_shape)) {
      MS_LOG(DEBUG) << "Input is dynamic rank";
      return {};
    }
    auto data_format = GetValue<std::string>(ib->attr("data_format"));
    size_t channel_idx = (data_format == kOpFormat_NHWC) ? x_shape.size() - 1 : 1;
    std::vector<int64_t> axis((x_shape.size() - channel_idx) - 1, -1);
    if (!axis.empty()) {
      auto target_shape = ExpandDimsInferShape(y_shape, axis);
      input_y = ib->Reshape(input_y, target_shape);
    }
  }
  auto result = ib->Add(input_x, input_y);
  return {result};
});
}  // namespace mindspore::graphkernel::expander
