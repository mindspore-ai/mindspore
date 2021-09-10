/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_LAYER_NORM_GRAD_SPLIT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_LAYER_NORM_GRAD_SPLIT_H_

#include <vector>
#include <memory>
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/common/helper.h"
#include "backend/optimizer/ascend/ascend_helper.h"

namespace mindspore {
namespace opt {
class LayerNormGradSplit : public PatternProcessPass {
 public:
  explicit LayerNormGradSplit(bool multigraph = true) : PatternProcessPass("layer_norm_grad_split", multigraph) {}
  ~LayerNormGradSplit() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  void CreateOutputsOfLayerNormXBackpropV2(const FuncGraphPtr &graph, const CNodePtr &layer_norm_grad,
                                           std::vector<AnfNodePtr> *layer_norm_grad_outputs, bool is_dynamic) const;
  void CreateOutputsOfLayerNormBetaGammaBackpropV2(const FuncGraphPtr &graph, const CNodePtr &layer_norm_grad,
                                                   const AnfNodePtr &res_for_gamma,
                                                   std::vector<AnfNodePtr> *layer_norm_beta_gamma_outputs,
                                                   bool is_dynamic) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_LAYER_NORM_GRAD_SPLIT_H_
