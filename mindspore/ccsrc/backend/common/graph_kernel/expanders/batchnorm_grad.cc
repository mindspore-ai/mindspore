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
#include "backend/common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class BatchNormGrad : public OpDesc {
 public:
  BatchNormGrad() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT,
                               kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    support_format->AddFormat({
      kOpFormat_NCHW,
      kOpFormat_NCHW,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
    });
    support_format->AddFormat({
      kOpFormat_NHWC,
      kOpFormat_NHWC,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
    });
    (void)validators_.emplace_back(std::move(support_format));
    std::initializer_list<std::string> attrs{"is_training", "epsilon"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~BatchNormGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_dy = inputs[0];
    const auto &input_x = inputs[1];
    const auto &input_scale = inputs[2];
    const auto &input_save_mean = inputs[3];
    const auto &input_save_inv_variance = inputs[4];

    ShapeVector reduce_axis;
    int64_t num;
    auto shape_x = input_x->shape;
    if (input_x->format == kOpFormat_NHWC) {
      // reduce_axis: idx_n, idx_h, idx_w
      reduce_axis = {kDim0, kDim1, kDim2};
      num = shape_x[kDim0] * shape_x[kDim1] * shape_x[kDim2];
    } else {
      reduce_axis = {kDim0, kDim2, kDim3};
      num = shape_x[kDim0] * shape_x[kDim2] * shape_x[kDim3];
    }
    auto ori_type = input_x->type;
    auto x = input_x;
    if (ori_type == TypeId::kNumberTypeFloat16) {
      x = gb.Cast(input_x, TypeId::kNumberTypeFloat32);
    }
    auto dy = input_dy;
    if (input_dy->type == TypeId::kNumberTypeFloat32) {
      dy = gb.Cast(input_dy, TypeId::kNumberTypeFloat32);
    }
    auto num_rec = -1.0f / num;
    auto num_rec_tensor = gb.Tensor(num_rec, input_scale->type);
    auto dbeta = gb.ReduceSum(dy, reduce_axis, false);

    // in training input_save_inv_variance means 1 / sqrt(variance + epsilon), which is calculated in forward pass
    auto inv_variance = input_save_inv_variance;
    if (!GetValue<bool>(attrs_["is_training"])) {
      auto epsilon_tensor = gb.Tensor(GetValue<float>(attrs_["epsilon"]), input_scale->type);
      auto var_add = gb.Add(input_save_inv_variance, epsilon_tensor);
      auto sqrt_var_eps = gb.Sqrt(var_add);
      auto scalar_one_tensor = gb.Tensor(1.0, input_scale->type);
      inv_variance = gb.Div(scalar_one_tensor, sqrt_var_eps);
    }

    // compute dgamma
    NodePtr dx;
    auto scale = input_scale;
    auto save_mean = input_save_mean;
    if (input_x->format != kOpFormat_NHWC) {
      save_mean = gb.Reshape(save_mean, ExpandDimsInferShape(save_mean->shape, {-1, -1}));
      scale = gb.Reshape(scale, ExpandDimsInferShape(scale->shape, {-1, -1}));
      inv_variance = gb.Reshape(inv_variance, ExpandDimsInferShape(inv_variance->shape, {-1, -1}));
    }
    auto x_sub_mean = gb.Sub(x, save_mean);
    auto x_div = gb.Mul(x_sub_mean, inv_variance);
    auto dgamma_param = gb.Mul(dy, x_div);
    auto dgamma = gb.ReduceSum(dgamma_param, reduce_axis, false);

    // compute dx
    if (GetValue<bool>(attrs_["is_training"])) {
      auto tmp_b = gb.Mul(num_rec_tensor, dbeta);
      auto dgamma_expand = dgamma;
      if (input_x->format != kOpFormat_NHWC) {
        dgamma_expand = gb.Reshape(dgamma, ExpandDimsInferShape(dgamma->shape, {-1, -1}));
        tmp_b = gb.Reshape(tmp_b, ExpandDimsInferShape(tmp_b->shape, {-1, -1}));
      }
      auto x_sub_mean_dgamma_mul = gb.Mul(x_div, dgamma_expand);
      auto tmp_c = gb.Mul(num_rec_tensor, x_sub_mean_dgamma_mul);
      auto tmp_ab_add = gb.Add(dy, tmp_b);
      auto tmp_abc_add = gb.Add(tmp_ab_add, tmp_c);
      auto gamma_mul = gb.Mul(scale, tmp_abc_add);
      dx = gb.Mul(inv_variance, gamma_mul);
    } else {
      auto y_scale = gb.Mul(scale, dy);
      dx = gb.Mul(inv_variance, y_scale);
    }

    if (ori_type == TypeId::kNumberTypeFloat16) {
      dx = gb.Cast(dx, TypeId::kNumberTypeFloat16);
    }

    constexpr size_t idx_dx = 0;
    constexpr size_t idx_dgamma = 1;
    constexpr size_t idx_dbeta = 2;
    dx->format = outputs_info_[idx_dx].format;
    dgamma->format = outputs_info_[idx_dgamma].format;
    dbeta->format = outputs_info_[idx_dbeta].format;

    return {dx, dgamma, dbeta};
  }
};
EXPANDER_OP_DESC_REGISTER("BatchNormGrad", BatchNormGrad);
}  // namespace mindspore::graphkernel::expanders
