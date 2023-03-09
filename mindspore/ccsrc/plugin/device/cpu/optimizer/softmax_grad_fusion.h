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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_OPTIMIZER_SOFTMAX_GRAD_FUSION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_OPTIMIZER_SOFTMAX_GRAD_FUSION_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class SoftmaxGradFusionCpu : public PatternProcessPass {
 public:
  explicit SoftmaxGradFusionCpu(bool multigraph = true) : PatternProcessPass("softmax_grad_fusion_cpu", multigraph) {
    reduce_sum_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimReduceSum->name()));
    input0_ = std::make_shared<Var>();
    input1_ = std::make_shared<Var>();
    axis_ = std::make_shared<Var>();
  }
  ~SoftmaxGradFusionCpu() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 private:
  VarPtr reduce_sum_;
  VarPtr input0_;
  VarPtr input1_;
  VarPtr axis_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_OPTIMIZER_SOFTMAX_GRAD_FUSION_H_
