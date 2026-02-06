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

#include <memory>
#include <vector>
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class LayerNorm : public OpDesc {
 public:
  LayerNorm() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_FRAC_NZ, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    (void)validators_.emplace_back(std::move(support_format));
    std::initializer_list<std::string> attrs{"begin_norm_axis", "begin_params_axis", "epsilon"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~LayerNorm() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    const auto &input_gamma = inputs[1];
    const auto &input_beta = inputs[2];
    const auto epsilon = GetValue<float>(attrs_["epsilon"]);

    auto x = input_x;
    auto gamma = input_gamma;
    auto beta = input_beta;
    auto ori_type = input_x->type;
    if (processor_ == "aicore" && ori_type == TypeId::kNumberTypeFloat16) {
      x = gb.Cast(x, TypeId::kNumberTypeFloat32);
      gamma = gb.Cast(gamma, TypeId::kNumberTypeFloat32);
      beta = gb.Cast(beta, TypeId::kNumberTypeFloat32);
    }

    auto ori_shape_x = input_x->shape;
    if (input_x->format == kOpFormat_FRAC_NZ) {
      ori_shape_x = InferShapeFromFractalnz(ori_shape_x);
    }

    auto begin_norm_axis = GetValue<int64_t>(attrs_["begin_norm_axis"]);
    if (begin_norm_axis < 0) {
      begin_norm_axis += static_cast<int64_t>(ori_shape_x.size());
    }

    ShapeVector reduce_axis;
    int64_t reduce_elts = 1;
    for (size_t i = 0; i < ori_shape_x.size(); i++) {
      auto axis_int64 = SizeToLong(i);
      if (axis_int64 >= begin_norm_axis) {
        reduce_axis.push_back(axis_int64);
        reduce_elts *= ori_shape_x[i];
      }
    }

    auto ori_reduced_shape_x = GetReducedOriShape(ori_shape_x, reduce_axis);
    auto axis = reduce_axis;
    if (x->format == kOpFormat_FRAC_NZ) {
      axis = ToFracZAxis(ori_shape_x, reduce_axis);
    }

    auto mean_cof_tensor = gb.Tensor(1.0 / reduce_elts, x->type);

    // Calculate mean
    auto mean_red = gb.ReduceSum(x, axis, true);
    auto mean = gb.Mul(mean_red, mean_cof_tensor);
    if (x->format == kOpFormat_FRAC_NZ) {
      mean = gb.Reshape(mean, ori_reduced_shape_x);
    }

    // Calculate variance
    auto variance_sub = gb.Sub(x, mean);
    auto variance_mul = gb.Mul(variance_sub, variance_sub);
    auto variance_red = gb.ReduceSum(variance_mul, axis, true);
    auto variance = gb.Mul(variance_red, mean_cof_tensor);
    if (x->format == kOpFormat_FRAC_NZ) {
      variance = gb.Reshape(variance, ori_reduced_shape_x);
    }

    // Calculate normalize
    auto epsilon_tensor = gb.Tensor(epsilon, x->type);
    auto normalize_add = gb.Add(variance, epsilon_tensor);
    auto normalize_rsqrt = gb.Emit("Rsqrt", {normalize_add});
    auto normalize_mul = gb.Mul(variance_sub, normalize_rsqrt);

    // Calculate scale and translate
    auto scale_mul = gb.Mul(normalize_mul, gamma);
    auto res = gb.Add(scale_mul, beta);

    if (processor_ == "aicore" && ori_type == TypeId::kNumberTypeFloat16) {
      res = gb.Cast(res, TypeId::kNumberTypeFloat16);
      mean = gb.Cast(mean, TypeId::kNumberTypeFloat16);
      variance = gb.Cast(variance, TypeId::kNumberTypeFloat16);
    }
    return {res, mean, variance};
  }
};
EXPANDER_OP_DESC_REGISTER("LayerNorm", LayerNorm);
}  // namespace mindspore::graphkernel::expanders
