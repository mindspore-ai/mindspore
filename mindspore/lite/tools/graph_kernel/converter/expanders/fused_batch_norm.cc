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
class FusedBatchNorm : public OpDesc {
 public:
  FusedBatchNorm() {
    std::initializer_list<std::string> attrs{"epsilon"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~FusedBatchNorm() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    // only work for infer
    const size_t input_index = 0;
    const size_t scale_index = 1;
    const size_t bias_index = 2;
    const size_t mean_index = 3;
    const size_t var_index = 4;
    const auto &input = inputs[input_index];
    const auto &scale = inputs[scale_index];
    const auto &bias = inputs[bias_index];
    const auto &mean = inputs[mean_index];
    const auto &var = inputs[var_index];
    auto eps = gb.Const(GetValue<float>(attrs_["epsilon"]), input->type);
    auto fuse_scale = gb.Div(scale, gb.Sqrt(gb.Add(var, eps)));
    auto fuse_offset = gb.Sub(bias, gb.Mul(fuse_scale, mean));
    auto result = gb.Add(gb.Mul(input, fuse_scale), fuse_offset);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("FusedBatchNorm", FusedBatchNorm);
}  // namespace mindspore::graphkernel::expanders
