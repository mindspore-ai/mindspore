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
class SigmoidCrossEntropyWithLogitsGrad : public OpDesc {
 public:
  SigmoidCrossEntropyWithLogitsGrad() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~SigmoidCrossEntropyWithLogitsGrad() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &logits = inputs[0];
    const auto &label = inputs[1];
    const auto &dout = inputs[2];
    // Calculate sigmoid_cross_entropy_with_logits_grad(logits, label, dout)
    // formula of sigmoid_cross_entropy_with_logits_grad is :
    //      (sigmoid(logits) - label) * dout
    auto const_one = gb.Tensor(1.0, logits->type);
    auto neg_x = gb.Neg(logits);
    auto exp_neg_x = gb.Exp(neg_x);
    auto add_exp = gb.Add(const_one, exp_neg_x);
    auto sigmoid_res = gb.Div(const_one, add_exp);
    auto sigmoid_res_sub_label = gb.Sub(sigmoid_res, label);
    auto result = gb.Mul(sigmoid_res_sub_label, dout);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("SigmoidCrossEntropyWithLogitsGrad", SigmoidCrossEntropyWithLogitsGrad);
}  // namespace mindspore::graphkernel::expanders
