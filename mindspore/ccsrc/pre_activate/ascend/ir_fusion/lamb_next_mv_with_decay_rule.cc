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
#include "pre_activate/ascend/ir_fusion/lamb_next_mv_with_decay_rule.h"
#include <utility>
#include "session/anf_runtime_algorithm.h"
#include "optimizer/opt.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr GetLambNextMVWithDecayOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &new_node,
                                        const AnfNodePtr &add3, const AnfNodePtr &add5, const AnfNodePtr &real_div0,
                                        const AnfNodePtr &real_div1) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  MS_EXCEPTION_IF_NULL(add3);
  MS_EXCEPTION_IF_NULL(real_div0);
  MS_EXCEPTION_IF_NULL(real_div1);
  MS_EXCEPTION_IF_NULL(add5);
  // Set abstract of new node
  AbstractBasePtrList new_node_list;
  new_node_list.push_back(add3->abstract());
  auto real_div0_cnode = real_div0->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_div0_cnode);
  AnfNodePtr add0 = real_div0_cnode->input(1);
  MS_EXCEPTION_IF_NULL(add0);
  new_node_list.push_back(add0->abstract());
  auto real_div1_cnode = real_div1->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_div1_cnode);
  AnfNodePtr add1 = real_div1_cnode->input(1);
  MS_EXCEPTION_IF_NULL(add1);
  new_node_list.push_back(add1->abstract());
  new_node_list.push_back(add5->abstract());
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(new_node_list);
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  new_node->set_abstract(abstract_tuple);
  // Create tuple_getitem node for outputs
  std::vector<AnfNodePtr> new_node_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, new_node, kLambNextMVWithDecayOutputNum, &new_node_outputs);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(add3, new_node_outputs[0]);
  (void)manager->Replace(add0, new_node_outputs[1]);
  (void)manager->Replace(add1, new_node_outputs[2]);
  return new_node_outputs[3];
}

void GetSharedInputNodesByAdd5(const AnfNodePtr &node, AnfNodePtr *mul4, AnfNodePtr *real_div0, AnfNodePtr *real_div1,
                               AnfNodePtr *constant_add2_y_input) {
  MS_EXCEPTION_IF_NULL(node);
  auto add5_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add5_cnode);
  if (add5_cnode->inputs().size() < kAddInputNum) {
    MS_LOG(EXCEPTION) << "The input size of Add5 is less than " << kAddInputNum;
  }
  *mul4 = add5_cnode->input(2);

  AnfNodePtr real_div4 = add5_cnode->input(1);
  MS_EXCEPTION_IF_NULL(real_div4);
  auto real_div4_cnode = real_div4->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_div4_cnode);
  if (real_div4_cnode->inputs().size() < kRealDivInputNum) {
    MS_LOG(EXCEPTION) << "The input size of RealDiv4 is less than " << kRealDivInputNum;
  }
  *real_div0 = real_div4_cnode->input(1);

  AnfNodePtr add4 = real_div4_cnode->input(2);
  MS_EXCEPTION_IF_NULL(add4);
  auto add4_cnode = add4->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add4_cnode);
  if (add4_cnode->inputs().size() < kAddInputNum) {
    MS_LOG(EXCEPTION) << "The input size of Add4 is less than " << kAddInputNum;
  }
  AnfNodePtr sqrt1 = add4_cnode->input(1);
  MS_EXCEPTION_IF_NULL(sqrt1);
  auto sqrt1_cnode = sqrt1->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sqrt1_cnode);
  if (sqrt1_cnode->inputs().size() < kSqrtInputNum) {
    MS_LOG(EXCEPTION) << "The input size of Sqrt1 is less than " << kSqrtInputNum;
  }
  *real_div1 = sqrt1_cnode->input(1);
  *constant_add2_y_input = add4_cnode->input(2);
}

bool MatchAdd3(const AnfNodePtr &add3, const AnfNodePtr &mul4, const AnfNodePtr &real_div0, const AnfNodePtr &real_div1,
               const AnfNodePtr &constant_add2_y) {
  if (add3 == nullptr || !add3->isa<CNode>()) {
    return false;
  }
  auto add3_cnode = add3->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add3_cnode);
  if (AnfAlgo::GetCNodeName(add3_cnode) != prim::kPrimTensorAdd->name() ||
      add3_cnode->inputs().size() != kAddInputNum) {
    return false;
  }
  // Check the shared input nodes.
  if (add3_cnode->input(2) != mul4) {
    return false;
  }
  AnfNodePtr real_div2 = add3_cnode->input(1);
  MS_EXCEPTION_IF_NULL(real_div2);
  auto real_div2_cnode = real_div2->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_div2_cnode);
  if (AnfAlgo::GetCNodeName(real_div2_cnode) != prim::kPrimMul->name() ||
      real_div2_cnode->inputs().size() != kMulInputNum) {
    return false;
  }
  if (real_div2_cnode->input(1) != real_div0) {
    return false;
  }
  AnfNodePtr sqrt0 = real_div2_cnode->input(2);
  MS_EXCEPTION_IF_NULL(sqrt0);
  auto sqrt0_cnode = sqrt0->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sqrt0_cnode);
  if (AnfAlgo::GetCNodeName(sqrt0_cnode) != kRsqrtOpName || sqrt0_cnode->inputs().size() != kRsqrtInputNum) {
    return false;
  }
  AnfNodePtr add2 = sqrt0_cnode->input(1);
  MS_EXCEPTION_IF_NULL(add2);
  auto add2_cnode = add2->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add2_cnode);
  if (AnfAlgo::GetCNodeName(add2_cnode) != prim::kPrimTensorAdd->name() ||
      add2_cnode->inputs().size() != kAddInputNum) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(add2_cnode->input(2));
  MS_EXCEPTION_IF_NULL(constant_add2_y);
  return add2_cnode->input(1) == real_div1 && *(add2_cnode->input(2)) == *constant_add2_y;
}
}  // namespace

AnfNodePtr LambNextMVWithDecayRule::CreateLambNextMVWithDecayNode(const FuncGraphPtr &func_graph,
                                                                  const AnfNodePtr &add3, const AnfNodePtr &add5,
                                                                  const AnfNodePtr &real_div0,
                                                                  const AnfNodePtr &real_div1,
                                                                  const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(add3);
  MS_EXCEPTION_IF_NULL(equiv);
  // Create new node with all the inputs
  auto prim = std::make_shared<Primitive>(kLambNextMVWithDecayOpName);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(prim)};
  for (size_t i = 0; i < kLambNextMVWithDecayInputNum; ++i) {
    auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_vars_[i]]);
    MS_EXCEPTION_IF_NULL(input_node);
    new_node_inputs.push_back(input_node);
  }
  for (size_t i = 0; i < kLambNextMVWithDecayConstantMulInputNum; ++i) {
    auto constant_mul_input_node = utils::cast<AnfNodePtr>((*equiv)[constant_mul_input_vars_[i]]);
    MS_EXCEPTION_IF_NULL(constant_mul_input_node);
    new_node_inputs.push_back(constant_mul_input_node);
  }
  auto constant_add2_y_node = utils::cast<AnfNodePtr>((*equiv)[constant_add2_y_]);
  MS_EXCEPTION_IF_NULL(constant_add2_y_node);
  new_node_inputs.push_back(constant_add2_y_node);
  auto new_node = func_graph->NewCNode(new_node_inputs);
  return GetLambNextMVWithDecayOutput(func_graph, new_node, add3, add5, real_div0, real_div1);
}

const BaseRef LambNextMVWithDecayRule::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_sqrt);
  const auto prim_deal_div = std::make_shared<Primitive>(kRealDivOpName);
  MS_EXCEPTION_IF_NULL(prim_deal_div);
  VectorRef mul4 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[4], input_vars_[6]});
  VectorRef add0 =
    VectorRef({prim::kPrimTensorAdd, VectorRef({prim::kPrimMul, constant_mul_input_vars_[0], input_vars_[4]}),
               VectorRef({prim::kPrimMul, constant_mul_input_vars_[1], input_vars_[3]})});
  VectorRef real_div0 = VectorRef({prim_deal_div, add0, input_vars_[5]});
  VectorRef add1 =
    VectorRef({prim::kPrimTensorAdd, VectorRef({prim::kPrimMul, constant_mul_input_vars_[2], input_vars_[1]}),
               VectorRef({prim::kPrimMul, constant_mul_input_vars_[3], input_vars_[0]})});
  VectorRef real_div1 = VectorRef({prim_deal_div, add1, input_vars_[2]});
  VectorRef real_div4 = VectorRef(
    {prim_deal_div, real_div0, VectorRef({prim::kPrimTensorAdd, VectorRef({prim_sqrt, real_div1}), constant_add2_y_})});
  return VectorRef({prim::kPrimTensorAdd, real_div4, mul4});
}

const AnfNodePtr LambNextMVWithDecayRule::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  // Get the shared input nodes in patterns of add5 and add3
  AnfNodePtr mul4 = nullptr;
  AnfNodePtr real_div0 = nullptr;
  AnfNodePtr real_div1 = nullptr;
  AnfNodePtr constant_add2_y_input = nullptr;
  GetSharedInputNodesByAdd5(node, &mul4, &real_div0, &real_div1, &constant_add2_y_input);
  // Get add3 and try to match the add3 pattern
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(mul4) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "The Mul4 should be used by at least another node input";
  }
  AnfNodeIndexSet mul4_output_node_index_set = manager->node_users()[mul4];
  auto iter = std::find_if(
    mul4_output_node_index_set.begin(), mul4_output_node_index_set.end(),
    [&node, &mul4, &real_div0, &real_div1, &constant_add2_y_input](const std::pair<AnfNodePtr, int> &node_index) {
      return node_index.first != node && MatchAdd3(node_index.first, mul4, real_div0, real_div1, constant_add2_y_input);
    });
  if (iter != mul4_output_node_index_set.end()) {
    return CreateLambNextMVWithDecayNode(func_graph, iter->first, node, real_div0, real_div1, equiv);
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
