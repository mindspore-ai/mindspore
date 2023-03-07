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
#include "kernel/common_utils.h"

namespace mindspore::graphkernel::expanders {
constexpr size_t kInputIdx = 0;
constexpr size_t kScaleIdx = 1;
constexpr size_t kBiasIdx = 2;
constexpr size_t kInputMinRank = 2;
constexpr int64_t kDimHeight = -2;
constexpr int64_t kDimWidth = -1;
class InstanceNorm : public OpDesc {
 public:
  InstanceNorm() {
    std::initializer_list<std::string> attrs{"epsilon"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~InstanceNorm() = default;

 protected:
  bool CheckInputs() override {
    const auto &var = inputs_info_[0];
    if (var.shape.size() < kInputMinRank) {
      MS_LOG(INFO) << "In InstanceNorm, input[0]'s rank must be at least 2, but got " << var.shape.size();
      return false;
    }
    if (var.format != kOpFormat_NCHW) {
      MS_LOG(INFO) << "In InstanceNorm, input[0]'s format must be NCHW, but got " << var.format;
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input = inputs[kInputIdx];
    const auto &scale = inputs[kScaleIdx];
    const auto &bias = inputs[kBiasIdx];

    int64_t rank = SizeToLong(input->shape.size());
    std::vector<int64_t> reduce_axis{kDimHeight + rank, kDimWidth + rank};
    int64_t mean_cof_v = input->shape[LongToSize(kDimHeight + rank)] * input->shape[LongToSize(kDimWidth + rank)];

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
    auto scale_mul = gb.Mul(normalize, scale);
    auto result = gb.Add(scale_mul, bias);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("InstanceNorm", InstanceNorm);
}  // namespace mindspore::graphkernel::expanders
