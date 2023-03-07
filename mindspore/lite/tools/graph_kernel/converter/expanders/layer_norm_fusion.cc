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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class LayerNormFusion : public OpDesc {
 public:
  LayerNormFusion() {
    std::initializer_list<std::string> attrs{"begin_norm_axis", "begin_params_axis", "epsilon"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~LayerNormFusion() = default;

 protected:
  void Init() override {
    begin_norm_axis_ = GetAxisList(attrs_["begin_norm_axis"]);
    begin_params_axis_ = GetAxisList(attrs_["begin_params_axis"]);
  }

  bool CheckInputs() override {
    if (begin_norm_axis_.size() != 1) {
      MS_LOG(INFO) << "begin_norm_axis should only contain 1 axis, but got " << begin_norm_axis_.size();
      return false;
    }

    if (begin_params_axis_.size() != 1) {
      MS_LOG(INFO) << "begin_params_axis should only contain 1 axis, but got " << begin_params_axis_.size();
      return false;
    }

    if (begin_params_axis_[0] != begin_norm_axis_[0]) {
      MS_LOG(INFO) << "Expander doesn't support begin_norm_axis and begin_params_axis with different value";
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    // only work for infer
    const size_t input_index = 0;
    const size_t gamma_index = 1;
    const size_t beta_index = 2;
    const auto &input = inputs[input_index];
    const auto &gamma = inputs[gamma_index];
    const auto &beta = inputs[beta_index];

    int64_t real_axis =
      begin_norm_axis_[0] < 0 ? SizeToLong(input->shape.size()) + begin_norm_axis_[0] : begin_norm_axis_[0];
    std::vector<int64_t> reduce_axis;
    int64_t mean_cof_v = 1;
    for (size_t i = 0; i < input->shape.size(); i++) {
      if (auto axis_int64 = SizeToLong(i); axis_int64 >= real_axis) {
        reduce_axis.push_back(axis_int64);
        mean_cof_v *= input->shape[i];
      }
    }

    // const
    auto num = gb.Const(mean_cof_v, input->type);
    auto eps = gb.Const(GetValue<float>(attrs_["epsilon"]), input->type);

    // Calculate mean
    auto sum_res = gb.ReduceSum(input, reduce_axis, true);
    auto mean = gb.Div(sum_res, num);

    // Calculate variance
    auto variance_sub = gb.Sub(input, mean);
    auto variance_mul = gb.Mul(variance_sub, variance_sub);
    auto variance_red = gb.ReduceSum(variance_mul, reduce_axis, true);
    auto variance = gb.Div(variance_red, num);

    // Calculate normalize
    auto normalize_sqrt = gb.Sqrt(gb.Add(variance, eps));
    auto normalize = gb.Div(variance_sub, normalize_sqrt);

    // Calculate scale and translate
    auto scale_mul = gb.Mul(normalize, gamma);
    auto result = gb.Add(scale_mul, beta);
    return {result};
  }

  std::vector<int64_t> begin_norm_axis_;
  std::vector<int64_t> begin_params_axis_;
};
EXPANDER_OP_DESC_REGISTER("LayerNormFusion", LayerNormFusion);
}  // namespace mindspore::graphkernel::expanders
