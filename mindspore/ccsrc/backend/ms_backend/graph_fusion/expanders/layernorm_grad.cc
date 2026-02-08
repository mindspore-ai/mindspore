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

#include <algorithm>
#include <memory>
#include <vector>
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class LayerNormGrad : public OpDesc {
 public:
  LayerNormGrad() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat(
      {kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    (void)validators_.emplace_back(std::move(support_format));
    std::initializer_list<std::string> attrs{"begin_norm_axis", "begin_params_axis"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~LayerNormGrad() = default;

 protected:
  bool CheckInputs() override {
    if (processor_ != "aicore") {
      auto dtype = inputs_info_[0].type;
      if (std::any_of(inputs_info_.begin(), inputs_info_.end(), [&dtype](auto &info) { return info.type != dtype; })) {
        MS_LOG(INFO) << "Inputs of LayerNormGrad are not all same, LayerNormGrad would not expand";
        return false;
      }
    }
    return true;
  }
  NodePtrList Expand(const NodePtrList &inputs) override {
    auto x = inputs[0];
    auto dy = inputs[1];
    auto variance = inputs[2];
    auto mean = inputs[3];
    auto gamma = inputs[4];

    float epsilon = 1e-12;
    if (attrs_.find("epsilon") != attrs_.end()) {
      epsilon = GetValue<float>(attrs_["epsilon"]);
    }

    auto ori_type = x->type;
    if (processor_ == "aicore" && ori_type == TypeId::kNumberTypeFloat16) {
      x = gb.Cast(x, TypeId::kNumberTypeFloat32);
      dy = gb.Cast(dy, TypeId::kNumberTypeFloat32);
      variance = gb.Cast(variance, TypeId::kNumberTypeFloat32);
      mean = gb.Cast(mean, TypeId::kNumberTypeFloat32);
      gamma = gb.Cast(gamma, TypeId::kNumberTypeFloat32);
    }

    auto ori_shape_x = x->shape;

    auto begin_norm_axis = GetValue<int64_t>(attrs_["begin_norm_axis"]);
    if (begin_norm_axis < 0) {
      begin_norm_axis += static_cast<int64_t>(ori_shape_x.size());
    }
    auto begin_params_axis = GetValue<int64_t>(attrs_["begin_params_axis"]);
    if (begin_params_axis < 0) {
      begin_params_axis += static_cast<int64_t>(ori_shape_x.size());
    }

    auto norm_axis = ShapeVector();
    if (static_cast<int64_t>(ori_shape_x.size()) - begin_norm_axis < 0) {
      MS_LOG(INFO) << "begin_norm_axis should be less than or equal to the dimension of x, but got begin_norm_axis: "
                   << begin_norm_axis << ", the dimension of x: " << ori_shape_x.size();
      return {};
    }
    norm_axis.reserve(ori_shape_x.size() - LongToSize(begin_norm_axis));
    for (int64_t i = begin_norm_axis; i < SizeToLong(ori_shape_x.size()); ++i) {
      norm_axis.push_back(i);
    }
    auto param_axis = ShapeVector();
    if (begin_params_axis < 0) {
      MS_LOG(INFO) << "begin_param_axis should be greater than or euqal to 0, but is: " << begin_params_axis;
      return {};
    }
    param_axis.reserve(LongToSize(begin_params_axis));
    for (int64_t i = 0; i < static_cast<int64_t>(begin_params_axis); ++i) {
      param_axis.push_back(i);
    }

    auto reduce_size = std::accumulate(norm_axis.begin(), norm_axis.end(), 1.0f,
                                       [&ori_shape_x](auto a, auto i) { return a * ori_shape_x[i]; });

    // Set some constant value
    auto eps = gb.Tensor(epsilon, x->type);
    auto const_neg_half = gb.Tensor(-0.5, x->type);
    auto const_neg_two = gb.Tensor(-2.0, x->type);
    auto const_two = gb.Tensor(2.0, x->type);
    auto const_neg_one = gb.Tensor(-1.0, x->type);
    auto mean_cof = gb.Tensor(1.0 / reduce_size, x->type);

    // Calculate dg db
    auto var_add_eps = gb.Add(variance, eps);
    auto log_var_add_eps = gb.Log(var_add_eps);
    auto var_eps_mul = gb.Mul(log_var_add_eps, const_neg_half);
    auto rsqrt_var_eps = gb.Exp(var_eps_mul);

    auto x_sub_mean = gb.Sub(x, mean);

    auto x_sub_mean_mul_exp_var_eps = gb.Mul(x_sub_mean, rsqrt_var_eps);
    auto dg_mul = gb.Mul(dy, x_sub_mean_mul_exp_var_eps);
    auto dg = gb.ReduceSum(dg_mul, param_axis, false);
    auto db = gb.ReduceSum(dy, param_axis, false);

    // pd_var
    auto tmp_var_eps = gb.Mul(rsqrt_var_eps, rsqrt_var_eps);
    auto r_tmp_var_eps = gb.Mul(rsqrt_var_eps, tmp_var_eps);

    auto dy_mul_gamma = gb.Mul(dy, gamma);
    auto tmp_mul = gb.Mul(dy_mul_gamma, x_sub_mean);
    auto padvar_mul1 = gb.ReduceSum(tmp_mul, norm_axis, true);
    auto padvar_mul3 = gb.Mul(padvar_mul1, r_tmp_var_eps);
    auto pd_var = gb.Mul(padvar_mul3, const_neg_half);

    // pd_mean
    auto pdmean1_sum = gb.ReduceSum(dy_mul_gamma, norm_axis, true);
    auto neg_rsqrt_var_eps = gb.Mul(rsqrt_var_eps, const_neg_one);
    auto pd_mean_1 = gb.Mul(neg_rsqrt_var_eps, pdmean1_sum);
    auto pdmean2_mul1 = gb.Mul(const_neg_two, x_sub_mean);
    auto pdmean2_sum = gb.ReduceSum(pdmean2_mul1, norm_axis, true);
    auto pdmean2_mul3 = gb.Mul(pdmean2_sum, mean_cof);
    auto pd_mean_2 = gb.Mul(pdmean2_mul3, pd_var);

    auto pd_mean = gb.Add(pd_mean_1, pd_mean_2);

    // Calculate dx
    auto pd_x_1 = gb.Mul(dy_mul_gamma, rsqrt_var_eps);
    auto pdx2_mul = gb.Mul(pd_var, x_sub_mean);
    auto pdx2_mul_two = gb.Mul(pdx2_mul, const_two);
    auto pd_x_2 = gb.Mul(pdx2_mul_two, mean_cof);

    auto pd_x_3 = gb.Mul(pd_mean, mean_cof);

    auto dx_tmp = gb.Add(pd_x_1, pd_x_2);
    auto dx = gb.Add(dx_tmp, pd_x_3);

    if (processor_ == "aicore" && ori_type == TypeId::kNumberTypeFloat16) {
      dx = gb.Cast(dx, TypeId::kNumberTypeFloat16);
      dg = gb.Cast(dg, TypeId::kNumberTypeFloat16);
      db = gb.Cast(db, TypeId::kNumberTypeFloat16);
    }

    return {dx, dg, db};
  }
};
EXPANDER_OP_DESC_REGISTER("LayerNormGrad", LayerNormGrad);
}  // namespace mindspore::graphkernel::expanders
