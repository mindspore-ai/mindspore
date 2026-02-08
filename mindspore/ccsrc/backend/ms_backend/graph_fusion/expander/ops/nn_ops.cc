/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include <string>
#include <vector>
#include "backend/ms_backend/graph_fusion/expander/base/ir_builder.h"
#include "backend/ms_backend/graph_fusion/expander/base/utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore::graphkernel::expander {
namespace {
NodePtrList ComputeAdamApplyOneWithDecay(const DefaultIrBuilder *ib) {
  auto grad = ib->input(kIndex0);
  auto v = ib->input(kIndex1);
  auto m = ib->input(kIndex2);
  auto var = ib->input(kIndex3);
  auto lr = ib->input(kIndex4);
  auto beta1 = ib->input(kIndex5);
  auto beta1_apply_one = ib->input(kIndex6);
  auto beta2 = ib->input(kIndex7);
  auto beta2_apply_one = ib->input(kIndex8);
  auto decay = ib->input(kIndex9);
  auto epsilon = ib->input(kIndex10);
  // calc m_new : m_new = beta1 * m + (1 - beta1) * grad
  auto m_b = ib->Mul(beta1, m);
  auto m_g = ib->Mul(beta1_apply_one, grad);
  auto m_new = ib->Add(m_b, m_g);

  // calc v_new: v_new = beta2 * v + (1 - beta2) * grad * grad
  auto v_b = ib->Mul(beta2, v);
  auto grad_mul = ib->Mul(grad, grad);
  auto v_g = ib->Mul(beta2_apply_one, grad_mul);
  auto v_new = ib->Add(v_b, v_g);

  // calc var_new: var_new = var - (m_new / (sqrt(v_new) + epsilon) + decay * var) * lr
  auto v_sqrt = ib->Sqrt(v_new);
  auto sqrt_ep = ib->Add(v_sqrt, epsilon);
  auto update = ib->Div(m_new, sqrt_ep);
  auto decay_var = ib->Mul(decay, var);
  auto new_update = ib->Add(update, decay_var);
  auto lr_update = ib->Mul(lr, new_update);
  auto var_new = ib->Sub(var, lr_update);
  return {v_new, m_new, var_new};
}

ShapeVector AllAxis(size_t rank) {
  ShapeVector axis(rank);
  for (int64_t i = 0; i < static_cast<int64_t>(rank); ++i) {
    axis[i] = i;
  }
  return axis;
}

NodePtr CastOp(const DefaultIrBuilder *ib, const NodePtr &node, TypeId dst_type) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_type = node->GetDtype();
  MS_EXCEPTION_IF_NULL(node_type);
  return node_type->type_id() == dst_type ? node : ib->Cast(node, dst_type);
}
}  // namespace

REG_EXPANDER_FUNC("AdamApplyOneWithDecay").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib) || !CheckAllDataTypeSame(ib)) {
    return {};
  }
  return ComputeAdamApplyOneWithDecay(ib);
});

REG_EXPANDER_FUNC("AdamApplyOneWithDecayAssign").SetBody(BODYFUNC(ib) {
  if (!CheckAllFormatsSame(ib) || !CheckAllDataTypeSame(ib)) {
    return {};
  }
  auto compute_res = ComputeAdamApplyOneWithDecay(ib);
  auto origin_v = ib->input(kIndex1);
  auto origin_m = ib->input(kIndex2);
  auto origin_var = ib->input(kIndex3);
  auto v_res = ib->Assign(origin_v, compute_res[0]);
  auto m_res = ib->Assign(origin_m, compute_res[1]);
  auto var_res = ib->Assign(origin_var, compute_res[2]);
  return {v_res, m_res, var_res};
});

REG_EXPANDER_FUNC("Sigmoid").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  auto need_cast = input_x->GetDtype()->type_id() != kNumberTypeFloat32;
  auto cast_x = need_cast ? ib->Cast(input_x, kFloat32) : input_x;
  auto const_one = ib->Tensor(1.0, kFloat32);
  auto neg_x = ib->Neg(cast_x);
  auto exp_neg_x = ib->Exp(neg_x);
  auto add_exp = ib->Add(exp_neg_x, const_one);
  auto result = ib->Div(const_one, add_exp);
  if (need_cast && !ib->is_int_or_bool(input_x)) {
    result = ib->Cast(result, input_x->GetDtype());
  }
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

REG_EXPANDER_FUNC("ApplyMomentum").SetRealOutputIndices({1}).SetBody(BODYFUNC(ib) {
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
  if (!CheckAttrs(ib, {"use_nesterov"}) || !CheckAllFormatsSame(ib)) {
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
  auto beta1_power = ib->input(3);
  auto beta2_power = ib->input(4);
  auto lr = ib->input(5);
  auto beta1 = ib->input(6);
  auto beta2 = ib->input(7);
  auto epsilon = ib->input(8);
  const auto &grad = ib->input(9);
  if (var->GetDtype() != TypeIdToType(kNumberTypeFloat32)) {
    auto dtype = var->GetDtype();
    beta1_power = ib->Cast(beta1_power, dtype);
    beta2_power = ib->Cast(beta2_power, dtype);
    lr = ib->Cast(lr, dtype);
    beta1 = ib->Cast(beta1, dtype);
    beta2 = ib->Cast(beta2, dtype);
    epsilon = ib->Cast(epsilon, dtype);
  }

  // calc m_new : m_new = m + (1 - beta1) * (grad - m)
  auto const_one = ib->Tensor(1.0, var->GetDtype());
  auto one_minus_beta1 = ib->Sub(const_one, beta1);
  auto m_g = ib->Mul(one_minus_beta1, grad);
  auto grad_minus_m = ib->Sub(grad, m);
  auto m_new = ib->Add(m, ib->Mul(one_minus_beta1, grad_minus_m));

  // calc v_new: v_new = v + (1 - beta2) * (grad * grad - v)
  auto one_minus_beta2 = ib->Sub(const_one, beta2);
  auto grad_square = ib->Mul(grad, grad);
  auto grad_square_minus_v = ib->Sub(grad_square, v);
  auto v_new = ib->Add(v, ib->Mul(one_minus_beta2, grad_square_minus_v));

  // calc lr_t: lr_t = lr * sqrt(1 - beta2_power) / (1 - beta1_power)
  auto one_minus_beta2_power = ib->Sub(const_one, beta2_power);
  auto sqrt_res = ib->Sqrt(one_minus_beta2_power);
  auto lr_mul = ib->Mul(lr, sqrt_res);
  auto m1_beta1_power = ib->Sub(const_one, beta1_power);
  auto lr_t = ib->Div(lr_mul, m1_beta1_power);

  // if use_nesterov: var_new = var - lr_t * (m_new * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_new))
  // if not use_nesterov: var_new = var - lr_t * m_new / (epsilon + sqrt(v_new))
  auto v_new_sqrt = ib->Sqrt(v_new);
  auto v_new_sqrt_e = ib->Add(epsilon, v_new_sqrt);
  NodePtr div_res;
  if (GetValue<bool>(ib->attr("use_nesterov"))) {
    auto m_new_mul = ib->Mul(m_new, beta1);
    auto m_new_mul_add = ib->Add(m_new_mul, m_g);
    auto lr_mul = ib->Mul(lr_t, m_new_mul_add);
    div_res = ib->Div(lr_mul, v_new_sqrt_e);
  } else {
    auto lr_m_mul = ib->Mul(lr_t, m_new);
    div_res = ib->Div(lr_m_mul, v_new_sqrt_e);
  }

  auto var_new = ib->Sub(var, div_res);
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
  if (!CheckAttrs(ib, {"data_format"})) {
    return {};
  }
  auto input_x = ib->input(0);
  auto input_y = ib->input(1);
  auto y_shape = input_y->GetShape();
  if (IsDynamicRank(input_x->GetShape()) || IsDynamicShape(y_shape)) {
    return {};
  }
  // default is NCHW
  auto data_format = GetValue<std::string>(ib->attr("data_format"));
  if (data_format != kOpFormat_DEFAULT && data_format != kOpFormat_NCHW && data_format != kOpFormat_NHWC) {
    return {};
  }
  data_format = data_format != kOpFormat_NHWC ? kOpFormat_NCHW : kOpFormat_NHWC;
  auto x_shape = input_x->GetShape();
  size_t channel_idx = (data_format == kOpFormat_NHWC) ? x_shape.size() - 1 : 1;
  std::vector<int64_t> axis((x_shape.size() - channel_idx) - 1, -1);
  if (!axis.empty()) {
    auto target_shape = ExpandDimsInferShape(y_shape, axis);
    input_y = ib->Reshape(input_y, target_shape);
  }
  auto result = ib->Add(input_x, input_y);
  return {result};
});

REG_EXPANDER_FUNC("RmsNorm").SetBody(BODYFUNC(ib) {
  auto x = ib->input(kIndex0);
  auto x_shape = x->GetShape();
  if (IsDynamicRank(x_shape) || x_shape.empty() || x_shape.back() <= 0) {
    MS_LOG(DEBUG) << "Skip shape: " << x_shape;
    return {};
  }
  auto gamma = ib->input(kIndex1);
  auto eps = ib->input(kIndex2);

  auto compute_type = kNumberTypeFloat32;
  auto x_type = x->GetDtype()->type_id();
  auto need_cast = x_type != compute_type;
  if (need_cast) {
    x = ib->Cast(x, compute_type);
    gamma = ib->Cast(gamma, compute_type);
  }
  auto x2 = ib->Mul(x, x);
  auto x2_mean = ib->ReduceSum(ib->Mul(x2, ib->Tensor(1.0 / x_shape.back(), x->GetDtype())), ib->Value(ShapeVector{-1}),
                               ib->Value(true));  // mean square of x
  auto rstd = ib->Rsqrt(ib->Add(x2_mean, eps));
  auto x_scale = ib->Mul(x, rstd);
  auto y = ib->Mul(x_scale, gamma);
  if (need_cast) {
    y = ib->Cast(y, x_type);
  }
  return {y, rstd};
});

REG_EXPANDER_FUNC("RmsNormGrad").SetBody(BODYFUNC(ib) {
  auto x = ib->input(kIndex1);
  auto x_shape = x->GetShape();
  if (IsDynamicRank(x_shape) || x_shape.empty() || x_shape.back() <= 0) {
    MS_LOG(DEBUG) << "Skip shape: " << x_shape;
    return {};
  }
  auto dy = ib->input(kIndex0);
  auto rstd = ib->input(kIndex2);
  auto gamma = ib->input(kIndex3);

  auto compute_type = kNumberTypeFloat32;
  auto x_type = x->GetDtype()->type_id();
  auto need_cast = x_type != compute_type;
  if (need_cast) {
    dy = ib->Cast(dy, compute_type);
    x = ib->Cast(x, compute_type);
    gamma = ib->Cast(gamma, compute_type);
  }
  ShapeVector reduce_axis;
  for (int64_t i = 0; i < SizeToLong(x_shape.size()) - 1; ++i) {
    reduce_axis.push_back(i);
  }
  // dgamma
  auto x_rstd = ib->Mul(x, rstd);
  auto dgamma = ib->ReduceSum(ib->Mul(dy, x_rstd), ib->Value(reduce_axis), ib->Value(false));
  // dx
  auto dy_gamma = ib->Mul(dy, gamma);
  auto dy_gamma_sum = ib->ReduceSum(ib->Mul(x, dy_gamma), ib->Value(ShapeVector{-1}), ib->Value(true));
  auto t0 = ib->Mul(ib->Mul(ib->Mul(rstd, rstd), rstd), dy_gamma_sum);
  auto t1 = ib->Mul(t0, ib->Tensor(-1.0 / x_shape.back(), x->GetDtype()));
  auto dx = ib->Add(ib->Mul(t1, x), ib->Mul(rstd, dy_gamma));
  if (need_cast) {
    dx = ib->Cast(dx, x_type);
  }
  return {dx, dgamma};
});

REG_EXPANDER_FUNC("LeakyReLUExt").SetBody(BODYFUNC(ib) {
  auto input = ib->input(kIndex0);
  auto alpha = ib->input(kIndex1);
  if (alpha->GetDtype() != TypeIdToType(kNumberTypeFloat32)) {
    MS_LOG(INFO) << "In LeakyReLU, negative_slope's dtype must be float32, but got " << alpha->GetDtype()->ToString();
    return {};
  }
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto need_cast = input->GetDtype() != f32;
  auto cast_1 = need_cast ? ib->Cast(input, f32) : input;
  auto mul = ib->Mul(cast_1, alpha);
  auto gre = ib->Less(cast_1, ib->Tensor(0, f32));
  auto result = ib->Select(gre, mul, cast_1);
  result = need_cast ? ib->Cast(result, input->GetDtype()) : result;
  return {result};
});

REG_EXPANDER_FUNC("EluExt").SetBody(BODYFUNC(ib) {
  auto input = ib->input(kIndex0);
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto need_cast = input->GetDtype() != f32;
  auto cast_1 = need_cast ? ib->Cast(input, f32) : input;
  auto alpha = ib->input(kIndex1);
  auto min = ib->Minimum(cast_1, 0);
  auto exp = ib->Exp(min);
  auto sub = ib->Sub(exp, 1);
  auto mul = ib->Mul(sub, alpha);
  NodePtr result;
  auto m_x = ib->Maximum(cast_1, 0);
  result = ib->Add(m_x, mul);
  result = need_cast ? ib->Cast(result, input->GetDtype()) : result;
  return {result};
});

REG_EXPANDER_FUNC("SoftplusExt").SetBody(BODYFUNC(ib) {
  auto input_x = ib->input(kIndex0);
  auto beta = ib->input(kIndex1);
  auto threshold = ib->input(kIndex2);
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto need_cast = input_x->GetDtype() != f32;
  auto cast_1 = need_cast ? ib->Cast(input_x, f32) : input_x;
  threshold = ib->ScalarToTensor(threshold, f32);
  beta = ib->ScalarToTensor(beta, f32);
  auto mul = ib->Mul(cast_1, beta);
  auto exp_x = ib->Exp(mul);
  auto exp_x_add_one = ib->Add(exp_x, 1);
  auto result = ib->Div(ib->Log(exp_x_add_one), beta);
  auto greater_t = ib->Greater(mul, threshold);
  result = ib->Select(greater_t, cast_1, result);
  result = need_cast ? ib->Cast(result, input_x->GetDtype()) : result;
  return {result};
});

REG_EXPANDER_FUNC("HShrink").SetBody(BODYFUNC(ib) {
  auto input = ib->input(kIndex0);
  if (IsDynamic(input->GetShape())) {
    MS_LOG(DEBUG) << "FOr HShrink, input cannot be dynamic";
    return {};
  }
  auto lambd = ib->input(kIndex1);
  MS_EXCEPTION_IF_NULL(input->GetDtype());
  MS_EXCEPTION_IF_NULL(lambd->GetDtype());
  auto input_type = input->GetDtype()->type_id();
  auto lambd_type = lambd->GetDtype()->type_id();
  if (lambd_type != input_type) {
    TypeId compute_type = input_type;
    if (input_type == kNumberTypeFloat32 || input_type == kNumberTypeBFloat16 || lambd_type == kNumberTypeFloat32 ||
        lambd_type == kNumberTypeBFloat16) {
      compute_type = kNumberTypeFloat32;
    }
    if (input_type != compute_type) {
      input = ib->Cast(input, compute_type);
    }
    if (lambd_type != compute_type) {
      lambd = ib->Cast(lambd, compute_type);
    }
  }
  auto abs = ib->Abs(input);
  auto const_zero = ib->Tensor(0, input->GetDtype());
  auto le_cmp = ib->LessEqual(abs, lambd);
  auto result = ib->Select(le_cmp, const_zero, input);
  if (result->GetDtype()->type_id() != input_type) {
    result = ib->Cast(result, input_type);
  }
  return {result};
});

REG_EXPANDER_FUNC("HSigmoid").SetBody(BODYFUNC(ib) {
  auto input = ib->input(kIndex0);
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto need_cast = input->GetDtype() != f32;
  auto cast_1 = need_cast ? ib->Cast(input, f32) : input;
  auto div_add = ib->Add(ib->Div(cast_1, 6), 0.5);
  auto result = ib->Minimum(div_add, 1);
  result = ib->Maximum(result, 0);
  result = need_cast ? ib->Cast(result, input->GetDtype()) : result;
  return {result};
});

REG_EXPANDER_FUNC("HSwish").SetBody(BODYFUNC(ib) {
  auto input = ib->input(kIndex0);
  if (input->GetDtype()->type_id() == kNumberTypeBool) {
    return {};
  }
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto need_cast = input->GetDtype() != f32;
  auto cast_1 = need_cast ? ib->Cast(input, f32) : input;
  auto in_add = ib->Add(cast_1, 3);
  auto in_max = ib->Maximum(in_add, 0);
  auto in_min = ib->Minimum(in_max, 6);
  auto in_div = ib->RealDiv(in_min, 6);
  auto result = ib->Mul(in_div, cast_1);
  result = need_cast ? ib->Cast(result, input->GetDtype()) : result;
  return {result};
});

REG_EXPANDER_FUNC("BinaryCrossEntropy").SetBody(BODYFUNC(ib) {
  auto logits = ib->input(kIndex0);
  auto labels = ib->input(kIndex1);
  auto weight = ib->input(kIndex2);
  if (weight->GetDtype() == TypeIdToType(kMetaTypeNone)) {
    weight = ib->Tensor(1, logits->GetDtype());
  }
  if (logits->GetDtype() != labels->GetDtype() || logits->GetDtype() != weight->GetDtype()) {
    MS_LOG(DEBUG) << "for BinaryCrossEntropy, all inputs should have same type";
    return {};
  }
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto dtype = logits->GetDtype();
  auto need_cast = dtype != f32;
  if (need_cast) {
    logits = ib->Cast(logits, f32);
    labels = ib->Cast(labels, f32);
    weight = ib->Cast(weight, f32);
  }
  auto mode = ib->input(kIndex3);
  auto log1 = ib->Log(logits);
  auto threshold = ib->Tensor(-100, logits->GetDtype());
  auto max1 = ib->Maximum(log1, threshold);
  auto mul1 = ib->Mul(labels, max1);
  auto log2 = ib->Log(ib->Add(ib->Neg(logits), 1));
  auto max2 = ib->Maximum(log2, threshold);
  auto mul2 = ib->Mul(ib->Add(ib->Neg(labels), 1), max2);
  auto ln = ib->Mul(ib->Neg(ib->Add(mul1, mul2)), weight);
  auto mode_num = GetValue<int64_t>(mode->GetValue());
  if (mode_num == 2) {
    ln = need_cast ? ib->Cast(ln, dtype) : ln;
    return {ln};
  }
  if (IsDynamic(ln->GetShape())) {
    MS_LOG(DEBUG) << "for BinaryCrossEntropy reduce mode, input cannot be dynamic";
    return {};
  }
  ShapeVector reduce_axis;
  for (int64_t i = 0; i < SizeToLong(ln->GetShape().size()); ++i) {
    reduce_axis.push_back(i);
  }
  auto result = ib->ReduceSum(ln, ib->Value(reduce_axis), ib->Value(false));
  if (mode_num == 1) {
    int64_t sz = 1;
    for (int64_t i = 0; i < SizeToLong(reduce_axis.size()); ++i) {
      sz *= ln->GetShape()[i];
    }
    result = ib->Div(result, sz);
  }
  result = need_cast ? ib->Cast(result, dtype) : result;
  return {result};
});

REG_EXPANDER_FUNC("BCEWithLogitsLoss").SetBody(BODYFUNC(ib) {
  auto logits = ib->input(kIndex0);
  auto labels = ib->input(kIndex1);
  auto weight = ib->input(kIndex2);
  auto post = ib->input(kIndex3);
  if (weight->GetDtype() == TypeIdToType(kMetaTypeNone)) {
    weight = ib->Tensor(1, logits->GetDtype());
  }
  if (post->GetDtype() == TypeIdToType(kMetaTypeNone)) {
    post = ib->Tensor(1, logits->GetDtype());
  }
  if (logits->GetDtype() != labels->GetDtype() || logits->GetDtype() != weight->GetDtype() ||
      logits->GetDtype() != post->GetDtype()) {
    MS_LOG(DEBUG) << "for BCEWithLogitsLoss, all inputs should have same type";
    return {};
  }
  auto f32 = TypeIdToType(kNumberTypeFloat32);
  auto dtype = logits->GetDtype();
  auto need_cast = dtype != f32;
  if (need_cast) {
    logits = ib->Cast(logits, f32);
    labels = ib->Cast(labels, f32);
    weight = ib->Cast(weight, f32);
    post = ib->Cast(post, f32);
  }
  auto const_one = ib->Tensor(1, f32);
  logits = ib->Div(const_one, ib->Add(const_one, ib->Exp(ib->Neg(logits))));
  auto mode = ib->input(kIndex4);
  auto log1 = ib->Log(logits);
  auto mul1 = ib->Mul(labels, log1);
  mul1 = ib->Mul(mul1, post);
  auto log2 = ib->Log(ib->Add(ib->Neg(logits), 1));
  auto mul2 = ib->Mul(ib->Add(ib->Neg(labels), 1), log2);
  auto ln = ib->Mul(ib->Neg(ib->Add(mul1, mul2)), weight);
  auto mode_num = GetValue<int64_t>(mode->GetValue());
  if (mode_num == 2) {
    ln = need_cast ? ib->Cast(ln, dtype) : ln;
    return {ln};
  }
  if (IsDynamic(ln->GetShape())) {
    MS_LOG(DEBUG) << "for BCEWithLogitsLoss reduce mode, input cannot be dynamic";
    return {};
  }
  ShapeVector reduce_axis;
  for (int64_t i = 0; i < SizeToLong(ln->GetShape().size()); ++i) {
    reduce_axis.push_back(i);
  }
  auto result = ib->ReduceSum(ln, ib->Value(reduce_axis), ib->Value(false));
  if (mode_num == 1) {
    int64_t sz = 1;
    for (int64_t i = 0; i < SizeToLong(reduce_axis.size()); ++i) {
      sz *= ln->GetShape()[i];
    }
    result = ib->Div(result, sz);
  }
  result = need_cast ? ib->Cast(result, dtype) : result;
  return {result};
});

REG_EXPANDER_FUNC("BatchNormStats").SetBody(BODYFUNC(ib) {
  auto x = ib->input(kIndex0);
  auto eps = ib->input(kIndex1);
  auto x_type = x->GetDtype()->type_id();
  if (x_type != kNumberTypeFloat32) {
    MS_LOG(DEBUG) << "Skip data type: " << TypeIdToString(x_type);
    return {};
  }
  const auto &input_shape = x->GetShape();
  ShapeVector axis;
  axis.reserve(input_shape.size());
  int64_t count = 1;
  for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); ++i) {
    if (i != 1) {  // reduce all axis except C channel(axis 1)
      axis.push_back(i);
      count *= input_shape[i];
    }
  }
  if (count <= 0) {
    return {};
  }
  auto coef = ib->Tensor(1.0 / static_cast<double>(count), x->GetDtype());

  // calc mean
  auto input_mean = ib->ReduceSum(ib->Mul(x, coef), ib->Value(axis), ib->Value(true));

  // calc invstd
  auto input_sub_mean = ib->Sub(x, input_mean);
  auto input_var = ib->Mul(input_sub_mean, input_sub_mean);
  input_var = ib->ReduceSum(ib->Mul(input_var, coef), ib->Value(axis), ib->Value(false));
  auto invstd = ib->Reciprocal(ib->Sqrt(ib->Add(input_var, eps)));

  return {ib->Reshape(input_mean, invstd->GetShape()), invstd};
});

REG_EXPANDER_FUNC("BatchNormGatherStatsWithCounts").SetRealOutputIndices({0, 1}).SetBody(BODYFUNC(ib) {
  auto x = ib->input(kIndex0);
  auto x_type = x->GetDtype()->type_id();
  if (x_type != kNumberTypeFloat32) {
    MS_LOG(DEBUG) << "Skip data type: " << TypeIdToString(x_type);
    return {};
  }
  auto mean_all = ib->input(kIndex1);
  auto invstd_all = ib->input(kIndex2);
  auto running_mean = ib->input(kIndex3);  // optional input
  auto running_var = ib->input(kIndex4);   // optional input
  auto momentum = ib->input(kIndex5);
  auto eps = ib->input(kIndex6);
  auto counts_all = ib->input(kIndex7);  // optional input
  if (counts_all->GetDtype()->type_id() == kMetaTypeNone) {
    MS_LOG(DEBUG) << "Skip counts_all is None";
    return {};
  }
  NodePtr counts_all_new = counts_all;
  if (counts_all->GetShape().size() != mean_all->GetShape().size()) {
    ShapeVector count_shape(mean_all->GetShape());
    count_shape[count_shape.size() - 1] = 1;
    counts_all_new = ib->Reshape(counts_all, count_shape);
  }

  auto momentum_value = GetValue<float>(momentum->GetValue());
  auto m1 = ib->Tensor(momentum_value, x->GetDtype());
  auto m2 = ib->Tensor(1.0f - momentum_value, x->GetDtype());

  auto global_counts = ib->ReduceSum(counts_all, ib->Value(AllAxis(counts_all->GetShape().size())), ib->Value(false));
  ShapeVector reduce_axis;
  reduce_axis.reserve(mean_all->GetShape().size());
  for (int64_t i = 0; i < static_cast<int64_t>(mean_all->GetShape().size()) - 1; ++i) {  // last axis is C channel
    reduce_axis.push_back(i);
  }

  // calc global mean
  auto mean_all_obj = CastOp(ib, mean_all, x_type);
  auto global_mean = ib->ReduceSum(ib->Div(ib->Mul(mean_all_obj, counts_all_new), global_counts),
                                   ib->Value(reduce_axis), ib->Value(false));

  // calc global invstd
  auto mean_sub_all = ib->Sub(mean_all_obj, global_mean);
  auto std_all = ib->Reciprocal(CastOp(ib, invstd_all, x_type));
  auto eps_value = GetValue<float>(eps->GetValue());
  auto var_all = ib->Add(ib->Mul(std_all, std_all), ib->Tensor(eps_value, x->GetDtype()));
  var_all = ib->Add(var_all, ib->Mul(mean_sub_all, mean_sub_all));
  var_all = ib->Mul(var_all, counts_all_new);
  auto global_var_sum = ib->ReduceSum(var_all, ib->Value(reduce_axis), ib->Value(false));
  auto global_var = ib->Div(global_var_sum, global_counts);
  auto global_invstd = ib->Reciprocal(ib->Sqrt(ib->Add(global_var, eps)));
  NodePtrList results{global_mean, global_invstd};
  // update running_mean
  if (running_mean->GetDtype()->type_id() != kMetaTypeNone) {
    auto running_mean_new = ib->Add(ib->Mul(global_mean, m1), ib->Mul(CastOp(ib, running_mean, x_type), m2));
    running_mean_new = CastOp(ib, running_mean_new, running_mean->GetDtype()->type_id());
    auto running_mean_res = ib->Assign(running_mean, running_mean_new);
    results.push_back(running_mean_res);
  }
  // update running_var
  if (running_var->GetDtype()->type_id() != kMetaTypeNone) {
    auto const_one = ib->Tensor(1.0, x->GetDtype());
    auto global_var1 = ib->Mul(global_var, ib->Div(global_counts, ib->Sub(global_counts, const_one)));
    auto running_var_new = ib->Add(ib->Mul(global_var1, m1), ib->Mul(CastOp(ib, running_var, x_type), m2));
    running_var_new = CastOp(ib, running_var_new, running_var->GetDtype()->type_id());
    auto running_var_res = ib->Assign(running_var, running_var_new);
    results.push_back(running_var_res);
  }
  return results;
});

REG_EXPANDER_FUNC("BatchNormElemt").SetBody(BODYFUNC(ib) {
  auto x = ib->input(kIndex0);
  auto x_type = x->GetDtype()->type_id();
  auto x_shape = x->GetShape();
  if (x_type != kNumberTypeFloat32 || x_shape.size() < kDim2) {
    MS_LOG(DEBUG) << "Skip data type: " << TypeIdToString(x_type) << " shape: " << x_shape;
    return {};
  }
  auto weight = ib->input(kIndex1);  // optional input
  auto bias = ib->input(kIndex2);    // optional input
  auto mean = ib->input(kIndex3);    // optional input
  auto invstd = ib->input(kIndex4);  // optional input
  ShapeVector new_shape(x_shape.size(), 1);
  new_shape[1] = x_shape[1];
  if (mean->GetDtype()->type_id() != kMetaTypeNone) {
    x = ib->Sub(x, ib->Reshape(CastOp(ib, mean, x_type), new_shape));
  }
  if (invstd->GetDtype()->type_id() != kMetaTypeNone) {
    x = ib->Mul(x, ib->Reshape(CastOp(ib, invstd, x_type), new_shape));
  }
  if (weight->GetDtype()->type_id() != kMetaTypeNone) {
    x = ib->Mul(x, ib->Reshape(CastOp(ib, weight, x_type), new_shape));
  }
  if (bias->GetDtype()->type_id() != kMetaTypeNone) {
    x = ib->Add(x, ib->Reshape(CastOp(ib, bias, x_type), new_shape));
  }
  return {x};
});

REG_EXPANDER_FUNC("BatchNormElemtGrad").SetBody(BODYFUNC(ib) {
  auto dout = ib->input(kIndex0);
  auto x = ib->input(kIndex1);
  auto x_type = x->GetDtype()->type_id();
  auto x_shape = x->GetShape();
  if (x_type != kNumberTypeFloat32 || x_shape.size() < kDim2) {
    MS_LOG(DEBUG) << "Skip data type: " << TypeIdToString(x_type) << " shape: " << x_shape;
    return {};
  }
  auto mean = ib->input(kIndex2);
  auto invstd = ib->input(kIndex3);
  auto weight = ib->input(kIndex4);
  auto sumd_dy = ib->input(kIndex5);
  auto sum_dy_xmu = ib->input(kIndex6);
  auto count = ib->input(kIndex7);
  ShapeVector new_shape(x_shape.size(), 1);
  new_shape[1] = x_shape[1];
  auto global_counts = ib->ReduceSum(count, ib->Value(AllAxis(count->GetShape().size())), ib->Value(false));
  invstd = CastOp(ib, ib->Reshape(invstd, new_shape), x_type);
  auto invstd_dy_xmu =
    ib->Mul(ib->Mul(invstd, invstd), ib->Div(CastOp(ib, ib->Reshape(sum_dy_xmu, new_shape), x_type), global_counts));
  auto x_sub_mean = ib->Sub(x, CastOp(ib, ib->Reshape(mean, new_shape), x_type));
  auto x_invstd = ib->Mul(x_sub_mean, invstd_dy_xmu);
  auto result = ib->Mul(ib->Sub(ib->Sub(CastOp(ib, dout, x_type),
                                        ib->Div(CastOp(ib, ib->Reshape(sumd_dy, new_shape), x_type), global_counts)),
                                x_invstd),
                        ib->Mul(invstd, CastOp(ib, ib->Reshape(weight, new_shape), x_type)));
  return {result};
});
}  // namespace mindspore::graphkernel::expander
