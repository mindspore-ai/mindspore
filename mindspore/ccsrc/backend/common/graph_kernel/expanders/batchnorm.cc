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
class BatchNorm : public OpDesc {
 public:
  BatchNorm() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
    });
    support_format->AddFormat({
      kOpFormat_NCHW,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
    });
    support_format->AddFormat({
      kOpFormat_NHWC,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
      kOpFormat_DEFAULT,
    });
    (void)validators_.emplace_back(std::move(support_format));
    std::initializer_list<std::string> attrs{"is_training", "momentum", "epsilon"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~BatchNorm() = default;

 private:
  // expand BatchNorm for training mode
  std::tuple<NodePtr, NodePtr, NodePtr, NodePtr, NodePtr> Train(const inner::GraphBuilder &gb,
                                                                const NodePtrList &inputs) {
    const auto &input_x = inputs[0];
    const auto &input_scale = inputs[1];
    const auto &input_offset = inputs[2];
    const auto &input_mean = inputs[3];
    const auto &input_variance = inputs[4];
    auto eps = gb.Tensor(GetValue<float>(attrs_["epsilon"]), input_x->type);
    auto shape_x = input_x->shape;
    ShapeVector reduce_axis;
    int64_t num;
    if (input_x->format == kOpFormat_NHWC) {
      constexpr size_t idx_n = 0;
      constexpr size_t idx_h = 1;
      constexpr size_t idx_w = 2;
      reduce_axis = {idx_n, idx_h, idx_w};
      num = shape_x[idx_n] * shape_x[idx_h] * shape_x[idx_w];
    } else {
      constexpr size_t idx_n = 0;
      constexpr size_t idx_h = 2;
      constexpr size_t idx_w = 3;
      reduce_axis = {idx_n, idx_h, idx_w};
      num = shape_x[idx_n] * shape_x[idx_h] * shape_x[idx_w];
    }
    float num_rec = 1.0f / static_cast<float>(num);
    auto num_rec_tensor = gb.Tensor(num_rec, input_scale->type);

    // compute mean value of input_x
    auto mean_sum = gb.ReduceSum(input_x, reduce_axis, false);
    auto mean_muls = gb.Mul(mean_sum, num_rec_tensor);

    // compute variance of input_x
    NodePtr mean_muls_expand;
    if (input_x->format == kOpFormat_NHWC) {
      mean_muls_expand = mean_muls;
    } else {
      mean_muls_expand = gb.Reshape(mean_muls, ExpandDimsInferShape(mean_muls->shape, {-1, -1}));
    }
    auto var_sub = gb.Sub(input_x, mean_muls_expand);
    auto var_mul = gb.Mul(var_sub, var_sub);
    auto var_sum = gb.ReduceSum(var_mul, reduce_axis, false);
    var_mul = gb.Mul(var_sum, num_rec_tensor);

    // y_sqrt_rec means 1 / sqrt(variance + epsilon), which is calculated in backward pass
    auto scalar_one_tensor = gb.Tensor(1.0, input_scale->type);
    auto y_add = gb.Add(var_mul, eps);
    auto y_sqrt = gb.Sqrt(y_add);
    auto y_sqrt_rec = gb.Div(scalar_one_tensor, y_sqrt);

    // compute res_y
    auto tmp_sub = gb.Sub(input_x, mean_muls_expand);
    NodePtr y_sqrt_rec_expand;
    NodePtr input_scale_expand;
    NodePtr input_offset_expand;
    if (input_x->format == kOpFormat_CHWN) {
      y_sqrt_rec_expand = y_sqrt_rec;
      input_scale_expand = input_scale;
      input_offset_expand = input_offset;
    } else {
      y_sqrt_rec_expand = gb.Reshape(y_sqrt_rec, ExpandDimsInferShape(y_sqrt_rec->shape, {-1, -1}));
      input_scale_expand = gb.Reshape(input_scale, ExpandDimsInferShape(input_scale->shape, {-1, -1}));
      input_offset_expand = gb.Reshape(input_offset, ExpandDimsInferShape(input_offset->shape, {-1, -1}));
    }
    auto y_norm = gb.Mul(tmp_sub, y_sqrt_rec_expand);
    auto res_y_mul = gb.Mul(input_scale_expand, y_norm);
    auto res_y = gb.Add(res_y_mul, input_offset_expand);

    // compute mean_res
    auto momentum = GetValue<float>(attrs_["momentum"]);
    auto momentum_sub = 1.0f - momentum;
    auto momentum_sub_tensor = gb.Tensor(momentum_sub, input_scale->type);
    auto new_running_mean_tmp = gb.Mul(momentum_sub_tensor, input_mean);
    auto momentum_tensor = gb.Tensor(momentum, input_scale->type);
    auto current_mean_tmp = gb.Mul(momentum_tensor, mean_muls);
    auto updated_moving_mean = gb.Add(new_running_mean_tmp, current_mean_tmp);
    auto mean_res = gb.Assign(input_mean, updated_moving_mean);

    // variance_res is calculated by sample variance, and need to multyply by num / (num - 1)
    auto var_num = static_cast<float>(num) / (num - 1);
    auto var_num_tensor = gb.Tensor(var_num, input_scale->type);
    auto var_mul_update = gb.Mul(var_num_tensor, var_mul);
    auto new_running_var_tmp = gb.Mul(momentum_sub_tensor, input_variance);
    auto current_var_tmp = gb.Mul(momentum_tensor, var_mul_update);
    auto updated_moving_variance = gb.Add(new_running_var_tmp, current_var_tmp);
    auto variance_res = gb.Assign(input_variance, updated_moving_variance);
    return {res_y, mean_res, variance_res, mean_muls, y_sqrt_rec};
  }

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    const auto &input_scale = inputs[1];
    const auto &input_offset = inputs[2];
    const auto &input_mean = inputs[3];
    const auto &input_variance = inputs[4];
    auto eps = gb.Tensor(GetValue<float>(attrs_["epsilon"]), input_x->type);

    auto x_ori_type = input_x->type;
    auto x_new_type = input_x->type;
    auto x = input_x;
    if (x_ori_type == TypeId::kNumberTypeFloat16 && input_scale->type == TypeId::kNumberTypeFloat32 &&
        input_offset->type == TypeId::kNumberTypeFloat32 && input_mean->type == TypeId::kNumberTypeFloat32 &&
        input_variance->type == TypeId::kNumberTypeFloat32) {
      x_new_type = TypeId::kNumberTypeFloat32;
    }
    if (x_new_type != x_ori_type) {
      x = gb.Cast(input_x, x_new_type);
    }

    if (GetValue<bool>(attrs_["is_training"])) {
      auto new_inputs = inputs;
      new_inputs[0] = x;
      auto [res_y, mean_res, variance_res, mean_muls, y_sqrt_rec] = Train(gb, new_inputs);
      if (x_new_type != x_ori_type) {
        res_y = gb.Cast(res_y, x_ori_type);
      }
      return {res_y, mean_res, variance_res, mean_muls, y_sqrt_rec};
    }

    // infer mode
    NodePtr input_mean_expand;
    NodePtr input_scale_expand;
    NodePtr input_offset_expand;
    if (input_x->format == kOpFormat_CHWN) {
      input_mean_expand = input_mean;
      input_scale_expand = input_scale;
      input_offset_expand = input_offset;
    } else {
      input_mean_expand = gb.Reshape(input_mean, ExpandDimsInferShape(input_mean->shape, {-1, -1}));
      input_scale_expand = gb.Reshape(input_scale, ExpandDimsInferShape(input_scale->shape, {-1, -1}));
      input_offset_expand = gb.Reshape(input_offset, ExpandDimsInferShape(input_offset->shape, {-1, -1}));
    }

    auto x_sub = gb.Sub(input_x, input_mean_expand);
    auto x_sub_mul = gb.Mul(input_scale_expand, x_sub);
    auto var_add = gb.Add(eps, input_variance);
    auto var_add_sqrt = gb.Sqrt(var_add);
    if (input_x->format != kOpFormat_CHWN) {
      var_add_sqrt = gb.Reshape(var_add_sqrt, ExpandDimsInferShape(var_add_sqrt->shape, {-1, -1}));
    }
    auto x_div = gb.Div(x_sub_mul, var_add_sqrt);
    auto res_y = gb.Add(input_offset_expand, x_div);
    if (x_new_type != x_ori_type) {
      res_y = gb.Cast(res_y, x_ori_type);
    }
    return {res_y, var_add, var_add, var_add, var_add};
  }
};
EXPANDER_OP_DESC_REGISTER("BatchNorm", BatchNorm);
}  // namespace mindspore::graphkernel::expanders
