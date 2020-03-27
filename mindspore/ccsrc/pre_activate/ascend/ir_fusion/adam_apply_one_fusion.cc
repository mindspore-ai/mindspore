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
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
void GetAdd0AndAdd1(const AnfNodePtr &sub0, AnfNodePtr *add0, AnfNodePtr *add1) {
  MS_EXCEPTION_IF_NULL(sub0);
  MS_EXCEPTION_IF_NULL(add0);
  MS_EXCEPTION_IF_NULL(add1);
  auto sub0_cnode = sub0->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sub0_cnode);
  CheckCNodeInputSize(sub0_cnode, kSubInputNum);
  AnfNodePtr mul4 = sub0_cnode->input(2);
  MS_EXCEPTION_IF_NULL(mul4);
  auto mul4_cnode = mul4->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul4_cnode);
  CheckCNodeInputSize(mul4_cnode, kMulInputNum);
  AnfNodePtr true_div0 = mul4_cnode->input(2);
  MS_EXCEPTION_IF_NULL(true_div0);
  auto true_div0_cnode = true_div0->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(true_div0_cnode);
  CheckCNodeInputSize(true_div0_cnode, kRealDivInputNum);
  *add0 = true_div0_cnode->input(1);
  AnfNodePtr add2 = true_div0_cnode->input(2);
  MS_EXCEPTION_IF_NULL(add2);
  auto add2_cnode = add2->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add2_cnode);
  CheckCNodeInputSize(add2_cnode, kAddInputNum);
  AnfNodePtr sqrt0 = add2_cnode->input(1);
  MS_EXCEPTION_IF_NULL(sqrt0);
  auto sqrt0_cnode = sqrt0->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sqrt0_cnode);
  CheckCNodeInputSize(sqrt0_cnode, kSqrtInputNum);
  *add1 = sqrt0_cnode->input(1);
}
}  // namespace

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
  VectorRef sqrt0 = VectorRef({prim_sqrt, VectorRef({prim::kPrimTensorAdd, mul2, mul3})});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[1], input_vars_[0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[0], input_vars_[2]});
  VectorRef add0 = VectorRef({prim::kPrimTensorAdd, mul0, mul1});
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
  AnfNodePtr add0 = nullptr;
  AnfNodePtr add1 = nullptr;
  GetAdd0AndAdd1(node, &add0, &add1);
  MS_EXCEPTION_IF_NULL(add0);
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
