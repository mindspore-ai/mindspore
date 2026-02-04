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

#include <vector>
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class SoftmaxCrossEntropyWithLogits : public OpDesc {
 public:
  SoftmaxCrossEntropyWithLogits() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    (void)validators_.emplace_back(std::move(support_format));
  }
  ~SoftmaxCrossEntropyWithLogits() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &logits = inputs[0];
    const auto &label = inputs[1];
    // Calculate softmax_cross_entropy_with_logits(logits, label)
    // formula of softmax_cross_entropy_with_logits is :
    // -reduce_sum(label * log(softmax(logits)))

    auto axis = ShapeVector{-1};
    auto max_x = gb.ReduceMax(logits, axis, true);
    auto data_sub = gb.Sub(logits, max_x);
    auto data_exp = gb.Exp(data_sub);
    auto data_expsum = gb.ReduceSum(data_exp, axis, true);
    auto data_softmax = gb.Div(data_exp, data_expsum);
    auto const_eps = gb.Tensor(0.000001, logits->type);
    auto data_softmax_safety = gb.Emit("Maximum", {data_softmax, const_eps});
    auto softmax_log = gb.Log(data_softmax_safety);
    auto label_mul_log = gb.Mul(label, softmax_log);
    auto tmp_res = gb.ReduceSum(label_mul_log, axis, false);
    auto loss = gb.Neg(tmp_res);
    auto dlogits = gb.Sub(data_softmax, label);
    return {loss, dlogits};
  }
};
EXPANDER_OP_DESC_REGISTER("SoftmaxCrossEntropyWithLogits", SoftmaxCrossEntropyWithLogits);
}  // namespace mindspore::graphkernel::expanders
