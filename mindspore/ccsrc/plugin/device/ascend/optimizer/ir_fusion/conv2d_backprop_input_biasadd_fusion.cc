/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/conv2d_backprop_input_biasadd_fusion.h"
#include <memory>
#include <vector>
#include "mindspore/core/ops/conv_pool_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef Conv2dBackpropInputBiasaddFusion::DefinePattern() const {
  VectorRef conv2d_bp_input({conv2d_bp_input_var_, x0_, x1_});
  VectorRef pattern({prim::kPrimBiasAdd, conv2d_bp_input, x2_});
  return pattern;
}

const AnfNodePtr Conv2dBackpropInputBiasaddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                           const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto conv2d_bp_input = GetAnfNodeByVar(equiv, conv2d_bp_input_var_);
  if (conv2d_bp_input == nullptr || !conv2d_bp_input->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Get cnode Conv2DBackpropInput failed!" << trace::DumpSourceLines(conv2d_bp_input);
  }

  // If there is a side-effect operator in the fusion, do not merge
  if (!IsStateEquivalent(node, conv2d_bp_input)) {
    return node;
  }

  std::vector<AnfNodePtr> inputs;
  (void)inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimConv2DTranspose->name())));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x0_));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x1_));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x2_));
  auto new_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(node->scope());
  new_node->set_abstract(node->abstract());

  common::AnfAlgo::CopyNodeAttrs(conv2d_bp_input, new_node);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
