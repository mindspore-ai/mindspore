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
#include "pre_activate/ascend/ir_fusion/adam_apply_one_with_decay_rule.h"

#include <memory>
#include <vector>

#include "session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
std::vector<AnfNodePtr> AdamApplyOneWithDecayRule::GetFusionNodeInputs(const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(equiv);
  auto input0 = utils::cast<AnfNodePtr>((*equiv)[input0_]);
  auto input1 = utils::cast<AnfNodePtr>((*equiv)[input1_]);
  auto input2 = utils::cast<AnfNodePtr>((*equiv)[input2_]);
  auto input3 = utils::cast<AnfNodePtr>((*equiv)[input3_]);
  auto input4 = utils::cast<AnfNodePtr>((*equiv)[input4_]);
  auto mul0_x = utils::cast<AnfNodePtr>((*equiv)[mul0_x_]);
  auto mul1_x = utils::cast<AnfNodePtr>((*equiv)[mul1_x_]);
  auto mul2_x = utils::cast<AnfNodePtr>((*equiv)[mul2_x_]);
  auto mul3_x = utils::cast<AnfNodePtr>((*equiv)[mul3_x_]);
  auto mul4_x = utils::cast<AnfNodePtr>((*equiv)[mul4_x_]);
  auto add2_y = utils::cast<AnfNodePtr>((*equiv)[add2_y_]);
  auto prim = std::make_shared<Primitive>(kAdamApplyOneWithDecayOpName);
  return {NewValueNode(prim), input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y};
}

const BaseRef AdamApplyOneWithDecayRule::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0_pattern({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1_pattern({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0_pattern({prim::kPrimSquare, input0_});
  VectorRef add0_pattern({add0_var_, mul0_pattern, mul1_pattern});
  VectorRef mul2_pattern({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3_pattern({prim::kPrimMul, mul3_x_, square0_pattern});
  VectorRef add1_pattern({add1_var_, mul2_pattern, mul3_pattern});
  VectorRef sqrt0_pattern({sqrt, add1_pattern});
  VectorRef add2_pattern({prim::kPrimTensorAdd, sqrt0_pattern, add2_y_});
  VectorRef mul4_pattern({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div_pattern({real_div, add0_pattern, add2_pattern});
  VectorRef add3_pattern({prim::kPrimTensorAdd, real_div_pattern, mul4_pattern});
  VectorRef mul5_pattern({prim::kPrimMul, input4_, add3_pattern});
  VectorRef sub0_pattern({prim::kPrimSub, input3_, mul5_pattern});
  return sub0_pattern;
}

const AnfNodePtr AdamApplyOneWithDecayRule::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  if (graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }

  std::vector<AnfNodePtr> inputs = GetFusionNodeInputs(equiv);
  auto fusion_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_scope(node->scope());

  auto iter_add0 = (*equiv).find(add0_var_);
  if (iter_add0 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the add0 var after matched.";
  }
  auto iter_add1 = (*equiv).find(add1_var_);
  if (iter_add1 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the add1 var after matched.";
  }
  auto add0 = utils::cast<AnfNodePtr>(iter_add0->second);
  MS_EXCEPTION_IF_NULL(add0);
  auto add1 = utils::cast<AnfNodePtr>(iter_add1->second);
  MS_EXCEPTION_IF_NULL(add1);
  auto types = {AnfAlgo::GetOutputInferDataType(add1, 0), AnfAlgo::GetOutputInferDataType(add0, 0),
                AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(add1, 0), AnfAlgo::GetOutputInferShape(add0, 0),
                 AnfAlgo::GetOutputInferShape(node, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, fusion_node.get());

  std::vector<AnfNodePtr> fusion_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, fusion_node, kAdamApplyOneWithDecayOutputNum, &fusion_node_outputs);
  if (fusion_node_outputs.size() != kAdamApplyOneWithDecayOutputNum) {
    MS_LOG(ERROR) << "create multiple outputs for fusion node fail!";
    return nullptr;
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(add1, fusion_node_outputs[0]);
  (void)manager->Replace(add0, fusion_node_outputs[1]);
  return fusion_node_outputs[2];
}
}  // namespace opt
}  // namespace mindspore
