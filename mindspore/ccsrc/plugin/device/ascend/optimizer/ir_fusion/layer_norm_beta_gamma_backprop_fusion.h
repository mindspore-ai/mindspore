/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAYER_NORM_BETA_GAMMA_BACKPROP_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAYER_NORM_BETA_GAMMA_BACKPROP_FUSION_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class LayerNormBetaGammaBackpropFusion : public PatternProcessPass {
 public:
  explicit LayerNormBetaGammaBackpropFusion(bool multigraph = true)
      : PatternProcessPass("layer_norm_beta_gamma_backprop_fusion", multigraph),
        kernel_query_(std::make_shared<KernelQuery>()) {}

  ~LayerNormBetaGammaBackpropFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  KernelQueryPtr kernel_query_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAYER_NORM_BETA_GAMMA_BACKPROP_FUSION_H_
