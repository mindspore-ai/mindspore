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
#include "plugin/device/ascend/optimizer/ir_fusion/adam_apply_one_fusion.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
namespace mindspore {
namespace opt {
const BaseRef AdamApplyOneFusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex3], VectorRef({prim::kPrimSquare, input_vars_[kIndex0]})});
  VectorRef sqrt0 = VectorRef({prim_sqrt, VectorRef({add1_var_, mul2, mul3})});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, sqrt0, add2_y_})});
  return VectorRef(
    {prim::kPrimSub, input_vars_[kIndex3], VectorRef({prim::kPrimMul, input_vars_[kIndex4], true_div0})});
}

const BaseRef AdamApplyOneCond1Fusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex3], VectorRef({prim::kPrimSquare, input_vars_[kIndex0]})});
  VectorRef sqrt0 = VectorRef({prim_sqrt, VectorRef({add1_var_, mul2, mul3})});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, add2_y_, sqrt0})});
  return VectorRef(
    {prim::kPrimSub, input_vars_[kIndex3], VectorRef({prim::kPrimMul, input_vars_[kIndex4], true_div0})});
}

const BaseRef AdamApplyOneCond2Fusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, VectorRef({prim::kPrimSquare, input_vars_[kIndex0]}), mul_x_input_vars_[kIndex3]});
  VectorRef sqrt0 = VectorRef({prim_sqrt, VectorRef({add1_var_, mul2, mul3})});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, sqrt0, add2_y_})});
  return VectorRef(
    {prim::kPrimSub, input_vars_[kIndex3], VectorRef({prim::kPrimMul, true_div0, input_vars_[kIndex4]})});
}

const BaseRef AdamApplyOneCond3Fusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex3], VectorRef({prim::kPrimSquare, input_vars_[kIndex0]})});
  VectorRef sqrt0 = VectorRef({prim_sqrt, VectorRef({add1_var_, mul2, mul3})});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, sqrt0, add2_y_})});
  return VectorRef(
    {prim::kPrimSub, input_vars_[kIndex3], VectorRef({prim::kPrimMul, true_div0, input_vars_[kIndex4]})});
}

const BaseRef AdamApplyOneCond4Fusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex3], VectorRef({prim::kPrimSquare, input_vars_[kIndex0]})});
  VectorRef sqrt0 = VectorRef({prim_sqrt, VectorRef({add1_var_, mul2, mul3})});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, add2_y_, sqrt0})});
  return VectorRef(
    {prim::kPrimSub, input_vars_[kIndex3], VectorRef({prim::kPrimMul, true_div0, input_vars_[kIndex4]})});
}

const BaseRef AdamApplyOneAssignFusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex3], VectorRef({prim::kPrimSquare, input_vars_[kIndex0]})});
  VectorRef add1 = VectorRef({add1_var_, mul2, mul3});
  VectorRef sqrt0 = VectorRef({prim_sqrt, add1});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, sqrt0, add2_y_})});
  VectorRef sub0 =
    VectorRef({sub0_var_, input_vars_[kIndex3], VectorRef({prim::kPrimMul, input_vars_[kIndex4], true_div0})});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input_vars_[kIndex3], sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input_vars_[kIndex2], add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input_vars_[kIndex1], add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

const BaseRef AdamApplyOneAssignCond1Fusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex3], VectorRef({prim::kPrimSquare, input_vars_[kIndex0]})});
  VectorRef add1 = VectorRef({add1_var_, mul2, mul3});
  VectorRef sqrt0 = VectorRef({prim_sqrt, add1});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, add2_y_, sqrt0})});
  VectorRef sub0 =
    VectorRef({sub0_var_, input_vars_[kIndex3], VectorRef({prim::kPrimMul, input_vars_[kIndex4], true_div0})});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input_vars_[kIndex3], sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input_vars_[kIndex2], add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input_vars_[kIndex1], add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

const BaseRef AdamApplyOneAssignCond2Fusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, VectorRef({prim::kPrimSquare, input_vars_[kIndex0]}), mul_x_input_vars_[kIndex3]});
  VectorRef add1 = VectorRef({add1_var_, mul2, mul3});
  VectorRef sqrt0 = VectorRef({prim_sqrt, add1});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, sqrt0, add2_y_})});
  VectorRef sub0 =
    VectorRef({sub0_var_, input_vars_[kIndex3], VectorRef({prim::kPrimMul, true_div0, input_vars_[kIndex4]})});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input_vars_[kIndex3], sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input_vars_[kIndex2], add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input_vars_[kIndex1], add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

const BaseRef AdamApplyOneAssignCond3Fusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex3], VectorRef({prim::kPrimSquare, input_vars_[kIndex0]})});
  VectorRef add1 = VectorRef({add1_var_, mul2, mul3});
  VectorRef sqrt0 = VectorRef({prim_sqrt, add1});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, sqrt0, add2_y_})});
  VectorRef sub0 =
    VectorRef({sub0_var_, input_vars_[kIndex3], VectorRef({prim::kPrimMul, true_div0, input_vars_[kIndex4]})});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input_vars_[kIndex3], sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input_vars_[kIndex2], add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input_vars_[kIndex1], add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

const BaseRef AdamApplyOneAssignCond4Fusion::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul2 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex2], input_vars_[kIndex1]});
  VectorRef mul3 =
    VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex3], VectorRef({prim::kPrimSquare, input_vars_[kIndex0]})});
  VectorRef add1 = VectorRef({add1_var_, mul2, mul3});
  VectorRef sqrt0 = VectorRef({prim_sqrt, add1});
  VectorRef mul1 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex1], input_vars_[kIndex0]});
  VectorRef mul0 = VectorRef({prim::kPrimMul, mul_x_input_vars_[kIndex0], input_vars_[kIndex2]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef true_div0 = VectorRef({prim_real_div, add0, VectorRef({prim::kPrimAdd, add2_y_, sqrt0})});
  VectorRef sub0 =
    VectorRef({sub0_var_, input_vars_[kIndex3], VectorRef({prim::kPrimMul, true_div0, input_vars_[kIndex4]})});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input_vars_[kIndex3], sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input_vars_[kIndex2], add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input_vars_[kIndex1], add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

AnfNodePtr AdamApplyOneFusion::CreateAdamApplyOneNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                      const AnfNodePtr &final_node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  PrimitivePtr prim = nullptr;
  if (common::AnfAlgo::CheckPrimitiveType(final_node, prim::kPrimDepend)) {
    prim = std::make_shared<Primitive>(kAdamApplyOneAssignOpName);
  } else {
    prim = std::make_shared<Primitive>(kAdamApplyOneOpName);
  }
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
  auto new_node = NewCNode(new_node_inputs, func_graph);
  return new_node;
}

const AnfNodePtr AdamApplyOneFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto sub0 = node;
  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend)) {
    auto iter_sub0 = (*equiv).find(sub0_var_);
    if (iter_sub0 == (*equiv).end()) {
      MS_LOG(EXCEPTION) << "The equiv map is expected to contains the sub0 var after matched."
                        << trace::DumpSourceLines(node);
    }
    sub0 = utils::cast<AnfNodePtr>(iter_sub0->second);
  }
  MS_EXCEPTION_IF_NULL(sub0);
  if (!CheckSupportDataType(sub0, kFloatDataTypeSet)) {
    return nullptr;
  }
  auto new_node = CreateAdamApplyOneNode(func_graph, equiv, node);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(sub0->scope());
  // Set abstract of new node
  AbstractBasePtrList new_node_abstract_list;
  auto iter_add0 = (*equiv).find(add0_var_);
  if (iter_add0 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the add0 var after matched."
                      << trace::DumpSourceLines(node);
  }
  auto iter_add1 = (*equiv).find(add1_var_);
  if (iter_add1 == (*equiv).cend()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the add1 var after matched."
                      << trace::DumpSourceLines(node);
  }
  auto add0 = utils::cast<AnfNodePtr>(iter_add0->second);
  MS_EXCEPTION_IF_NULL(add0);
  auto add1 = utils::cast<AnfNodePtr>(iter_add1->second);
  MS_EXCEPTION_IF_NULL(add1);
  new_node_abstract_list.push_back(add1->abstract());
  new_node_abstract_list.push_back(add0->abstract());
  new_node_abstract_list.push_back(sub0->abstract());
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(new_node_abstract_list);
  new_node->set_abstract(abstract_tuple);
  // Create tuple_getitem node for outputs
  std::vector<AnfNodePtr> new_node_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, new_node, kAdamApplyOneOutputNum, &new_node_outputs);
  if (new_node_outputs.size() != kAdamApplyOneOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node " << new_node->DebugString() << " should be "
                      << kAdamApplyOneOutputNum << trace::DumpSourceLines(node);
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(add1, new_node_outputs[kIndex0]);
  (void)manager->Replace(add0, new_node_outputs[kIndex1]);
  return new_node_outputs[kIndex2];
}
}  // namespace opt
}  // namespace mindspore
