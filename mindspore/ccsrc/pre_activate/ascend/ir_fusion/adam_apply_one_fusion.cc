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
#include "pre_activate/ascend/ir_fusion/adam_apply_one_fusion.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
AnfNodePtr AdamApplyOneFusion::CreateAdamApplyOneNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  auto prim = std::make_shared<Primitive>(kAdamApplyOneOpName);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(prim)};
  for (const auto &input_var : input_vars_) {
    auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_var]);
    MS_EXCEPTION_IF_NULL(input_node);
    new_node_inputs.push_back(input_node);
  }
  for (const auto &mul_x_input_var : mul_x_input_vars_) {
    auto mul_x_input_node = utils::cast<AnfNodePtr>((*equiv)[mul_x_input_var]);
    MS_EXCEPTION_IF_NULL(mul_x_input_node);
    new_node_inputs.push_back(mul_x_input_node);
  }
  auto add2_y_node = utils::cast<AnfNodePtr>((*equiv)[add2_y_]);
  MS_EXCEPTION_IF_NULL(add2_y_node);
  new_node_inputs.push_back(add2_y_node);
  auto new_node = func_graph->NewCNode(new_node_inputs);
  return new_node;
}

const BaseRef AdamApplyOneFusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_deal_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[2], input_vars_[1]});
  VectorRef mul3 = VectorRef({prim::kPrimMul, mul_x_input_vars_[3], VectorRef({prim::kPrimSquare, input_vars_[0]})});
  VectorRef sqrt0 = VectorRef({prim_sqrt, VectorRef({add1_var_, mul2, mul3})});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[1], input_vars_[0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[0], input_vars_[2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_deal_div, add0, VectorRef({prim::kPrimTensorAdd, sqrt0, add2_y_})});
  return VectorRef({prim::kPrimSub, input_vars_[3], VectorRef({prim::kPrimMul, input_vars_[4], true_div0})});
}

const AnfNodePtr AdamApplyOneFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto new_node = CreateAdamApplyOneNode(func_graph, equiv);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(node->scope());
  // Set abstract of new node
  AbstractBasePtrList new_node_abstract_list;
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
  new_node_abstract_list.push_back(add1->abstract());
  new_node_abstract_list.push_back(add0->abstract());
  new_node_abstract_list.push_back(node->abstract());
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(new_node_abstract_list);
  new_node->set_abstract(abstract_tuple);
  // Create tuple_getitem node for outputs
  std::vector<AnfNodePtr> new_node_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, new_node, kAdamApplyOneOutputNum, &new_node_outputs);
  if (new_node_outputs.size() != kAdamApplyOneOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node " << new_node->DebugString() << " should be "
                      << kAdamApplyOneOutputNum;
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(add1, new_node_outputs[0]);
  (void)manager->Replace(add0, new_node_outputs[1]);
  return new_node_outputs[2];
}
}  // namespace opt
}  // namespace mindspore
