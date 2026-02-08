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
class SigmoidCrossEntropyWithLogits : public OpDesc {
 public:
  SigmoidCrossEntropyWithLogits() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~SigmoidCrossEntropyWithLogits() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &logits = inputs[0];
    const auto &labels = inputs[1];
    // Calculate sigmoid_cross_entropy_with_logits(logits, labels)
    // formula of sigmoid_cross_entropy_with_logits is:
    //     -(labels * log(sigmoid(logits)) + (1 - labels) * log(1 - sigmoid(logits)))
    // To ensure stability and avoid overflow, the formula equal to :
    //      max(logits, 0) - logits * labels + log(1 + exp(-abs(logits)))
    auto const_one = gb.Tensor(1.0, logits->type);
    auto const_zero = gb.Tensor(0.0, logits->type);
    auto max_logits = gb.Emit("Maximum", {logits, const_zero});
    auto logits_mul_labels = gb.Mul(logits, labels);
    auto abs_logits = gb.Abs(logits);
    auto neg_abs_logits = gb.Neg(abs_logits);
    auto exp_neg_abs_logits = gb.Exp(neg_abs_logits);
    auto one_add_exp_neg_abs_logits = gb.Add(const_one, exp_neg_abs_logits);
    auto log_one_add_exp_neg_abs_logits = gb.Log(one_add_exp_neg_abs_logits);
    auto res_tmp = gb.Sub(max_logits, logits_mul_labels);
    auto result = gb.Add(res_tmp, log_one_add_exp_neg_abs_logits);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("SigmoidCrossEntropyWithLogits", SigmoidCrossEntropyWithLogits);
}  // namespace mindspore::graphkernel::expanders
