/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/lamb_next_right_rule.h"
#include <vector>
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace opt {
AnfNodePtr LambNextRightRule::CreateLambNextRightNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  std::vector<AnfNodePtr> new_node_inputs;
  auto prim = std::make_shared<Primitive>(kLambNextRightOpName);
  MS_EXCEPTION_IF_NULL(prim);
  new_node_inputs.push_back(NewValueNode(prim));
  auto input0 = utils::cast<AnfNodePtr>((*equiv)[input0_]);
  MS_EXCEPTION_IF_NULL(input0);
  new_node_inputs.push_back(input0);
  auto input1 = utils::cast<AnfNodePtr>((*equiv)[input1_]);
  MS_EXCEPTION_IF_NULL(input1);
  new_node_inputs.push_back(input1);
  auto mul2_x = utils::cast<AnfNodePtr>((*equiv)[mul2_x_]);
  MS_EXCEPTION_IF_NULL(mul2_x);
  new_node_inputs.push_back(mul2_x);
  auto mul3_x = utils::cast<AnfNodePtr>((*equiv)[mul3_x_]);
  MS_EXCEPTION_IF_NULL(mul3_x);
  new_node_inputs.push_back(mul3_x);
  auto true_div1_recip = utils::cast<AnfNodePtr>((*equiv)[true_div1_recip_]);
  MS_EXCEPTION_IF_NULL(true_div1_recip);
  new_node_inputs.push_back(true_div1_recip);
  auto add2_y = utils::cast<AnfNodePtr>((*equiv)[add2_y_]);
  MS_EXCEPTION_IF_NULL(add2_y);
  new_node_inputs.push_back(add2_y);
  auto new_node = NewCNode(new_node_inputs, func_graph);
  return new_node;
}

const BaseRef LambNextRightRule::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_sqrt);
  VectorRef mul3 = VectorRef({prim::kPrimMul, mul3_x_, VectorRef({prim::kPrimSquare, input0_})});
  VectorRef add1 = VectorRef({add1_var_, VectorRef({prim::kPrimMul, mul2_x_, input1_}), mul3});
  return VectorRef(
    {prim::kPrimAdd, VectorRef({prim_sqrt, VectorRef({prim::kPrimMul, add1, true_div1_recip_})}), add2_y_});
}

const AnfNodePtr LambNextRightRule::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!CheckSupportDataType(node, kFloatDataTypeSet)) {
    return nullptr;
  }
  auto new_node = CreateLambNextRightNode(func_graph, equiv);
  MS_EXCEPTION_IF_NULL(new_node);
  // Set abstract of new node
  auto iter_add1 = (*equiv).find(add1_var_);
  if (iter_add1 == (*equiv).cend()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the add1 var after matched."
                      << trace::DumpSourceLines(node);
  }
  auto add1 = utils::cast<AnfNodePtr>(iter_add1->second);
  MS_EXCEPTION_IF_NULL(add1);
  AbstractBasePtrList new_node_abstract_list;
  new_node_abstract_list.push_back(add1->abstract());
  new_node_abstract_list.push_back(node->abstract());
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(new_node_abstract_list);
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  new_node->set_abstract(abstract_tuple);
  // Create tuple_getitem node for outputs
  std::vector<AnfNodePtr> new_node_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, new_node, kLambNextRightOutputNum, &new_node_outputs);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(add1, new_node_outputs[0]);
  return new_node_outputs[1];
}
}  // namespace opt
}  // namespace mindspore
