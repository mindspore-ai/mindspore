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

#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class GkDropout : public OpDesc {
 public:
  GkDropout() {
    (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>());
    std::initializer_list<std::string> attrs{"keep_prob"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~GkDropout() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    const auto &input_mask = inputs[1];
    auto keep_prob = GetValue<float>(attrs_["keep_prob"]);
    auto keep_prob_tensor = gb.Tensor(keep_prob, input_x->type);
    auto r_keep_prob_tensor = gb.Tensor(1.0f / keep_prob, input_x->type);

    auto mask = input_mask;
    if (mask->type != input_x->type) {
      mask = gb.Cast(mask, input_x->type);
    }

    auto new_mask = gb.LessEqual(mask, keep_prob_tensor);
    new_mask = gb.Cast(new_mask, input_x->type);
    auto result = gb.Mul(r_keep_prob_tensor, input_x);
    result = gb.Mul(result, new_mask);
    return {result, new_mask};
  }
};
EXPANDER_OP_DESC_REGISTER("GkDropout", GkDropout);
}  // namespace mindspore::graphkernel::expanders
