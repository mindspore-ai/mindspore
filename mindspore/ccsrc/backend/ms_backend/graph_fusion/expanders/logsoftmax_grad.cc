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
class LogSoftmaxGrad : public OpDesc {
 public:
  LogSoftmaxGrad() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    (void)validators_.emplace_back(std::move(support_format));
    std::initializer_list<std::string> attrs{"axis"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~LogSoftmaxGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_logits = inputs[0];
    const auto &input_dy = inputs[1];
    auto axis = GetAxisList(attrs_["axis"]);

    auto softmax = gb.Exp(input_logits);
    auto dy_sum = gb.ReduceSum(input_dy, axis, true);
    auto mul_result = gb.Mul(softmax, dy_sum);
    auto result = gb.Sub(input_dy, mul_result);

    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("LogSoftmaxGrad", LogSoftmaxGrad);
}  // namespace mindspore::graphkernel::expanders
