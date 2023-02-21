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
#include "plugin/device/gpu/optimizer/apply_momentum_weight_fusion.h"

#include <vector>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
const BaseRef ApplyMomentumWeightDecayFusion::DefinePattern() const {
  VectorRef load_para = VectorRef({prim::kPrimLoad, variable_, monad_});
  VectorRef weight_decay =
    VectorRef({prim::kPrimAddN, VectorRef({prim::kPrimMul, load_para, weight_decay_}), gradient_});
  VectorRef apply_momentum = VectorRef(
    {prim::kPrimApplyMomentum, variable_, accumulation_, learning_rate_, weight_decay, momentum_, monad_state_});
  return apply_momentum;
}

const AnfNodePtr ApplyMomentumWeightDecayFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                         const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto weight_decay = utils::cast<AnfNodePtr>((*equiv)[weight_decay_]);
  auto variable = utils::cast<AnfNodePtr>((*equiv)[variable_]);
  auto accumulation = utils::cast<AnfNodePtr>((*equiv)[accumulation_]);
  auto learning_rate = utils::cast<AnfNodePtr>((*equiv)[learning_rate_]);
  auto gradient = utils::cast<AnfNodePtr>((*equiv)[gradient_]);
  auto momentum = utils::cast<AnfNodePtr>((*equiv)[momentum_]);
  auto monad_state = utils::cast<AnfNodePtr>((*equiv)[monad_state_]);
  MS_EXCEPTION_IF_NULL(weight_decay);
  MS_EXCEPTION_IF_NULL(variable);
  MS_EXCEPTION_IF_NULL(accumulation);
  MS_EXCEPTION_IF_NULL(learning_rate);
  MS_EXCEPTION_IF_NULL(gradient);
  MS_EXCEPTION_IF_NULL(momentum);
  MS_EXCEPTION_IF_NULL(monad_state);

  auto prim = std::make_shared<Primitive>(kFusedWeightApplyMomentum);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), weight_decay, variable, accumulation,
                                    learning_rate,      gradient,     momentum, monad_state};
  auto replace_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(replace_node);
  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(node, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, replace_node.get());
  replace_node->set_scope(node->scope());
  return replace_node;
}
}  // namespace opt
}  // namespace mindspore
