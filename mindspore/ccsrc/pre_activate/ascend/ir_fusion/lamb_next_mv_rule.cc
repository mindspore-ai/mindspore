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

#include "pre_activate/ascend/ir_fusion/lamb_next_mv_rule.h"
#include <memory>
#include <utility>
#include <tuple>
#include <algorithm>
#include <unordered_map>
#include "session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "pre_activate/common/helper.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace {
std::tuple<CNodePtr, CNodePtr, AnfNodePtr> GetSharedNodesByPattern(const AnfNodePtr &node) {
  auto add3_cnode = CheckAnfNodeIfCNodeAndInputSize(node, kAddInputNum);
  MS_EXCEPTION_IF_NULL(add3_cnode);
  auto real_div2_cnode = CheckAnfNodeIfCNodeAndInputSize(add3_cnode->input(1), kMulInputNum);
  MS_EXCEPTION_IF_NULL(real_div2_cnode);
  auto real_div0_cnode = CheckAnfNodeIfCNodeAndInputSize(real_div2_cnode->input(1), kRealDivInputNum);
  MS_EXCEPTION_IF_NULL(real_div0_cnode);
  auto sqrt0_cnode = CheckAnfNodeIfCNodeAndInputSize(real_div2_cnode->input(2), kSqrtInputNum);
  MS_EXCEPTION_IF_NULL(sqrt0_cnode);
  auto add2_cnode = CheckAnfNodeIfCNodeAndInputSize(sqrt0_cnode->input(1), kAddInputNum);
  MS_EXCEPTION_IF_NULL(add2_cnode);
  auto real_div1_cnode = CheckAnfNodeIfCNodeAndInputSize(add2_cnode->input(1), kRealDivInputNum);
  auto constant_add2_y = add2_cnode->input(2);

  return std::make_tuple(real_div0_cnode, real_div1_cnode, constant_add2_y);
}

bool MatchRealDiv4(const AnfNodePtr &real_div4, const AnfNodePtr &real_div1, const AnfNodePtr &constant_add2_y) {
  if (real_div4 == nullptr || !real_div4->isa<CNode>()) {
    return false;
  }
  auto real_div4_cnode = real_div4->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_div4_cnode);
  if (AnfAlgo::GetCNodeName(real_div4_cnode) != kRealDivOpName || real_div4_cnode->inputs().size() < kRealDivInputNum) {
    return false;
  }

  CNodePtr add4_cnode = nullptr;
  if (!CheckIfCNodeAndInputSize(real_div4_cnode->input(2), kAddInputNum, &add4_cnode) ||
      AnfAlgo::GetCNodeName(add4_cnode) != prim::kPrimTensorAdd->name()) {
    return false;
  }
  CNodePtr sqrt1_cnode = nullptr;
  if (!CheckIfCNodeAndInputSize(add4_cnode->input(1), kSqrtInputNum, &sqrt1_cnode) ||
      AnfAlgo::GetCNodeName(sqrt1_cnode) != kSqrtOpName) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(add4_cnode->input(2));
  MS_EXCEPTION_IF_NULL(constant_add2_y);
  return sqrt1_cnode->input(1) == real_div1 && *(add4_cnode->input(2)) == *constant_add2_y;
}
}  // namespace

const BaseRef LambNextMVRule::DefinePattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_rsqrt);
  const auto prim_deal_div = std::make_shared<Primitive>(kRealDivOpName);
  MS_EXCEPTION_IF_NULL(prim_deal_div);

  auto mul0 = VectorRef({prim::kPrimMul, input_varptr_[7], input_varptr_[4]});
  auto mul1 = VectorRef({prim::kPrimMul, input_varptr_[8], input_varptr_[3]});
  auto mul2 = VectorRef({prim::kPrimMul, input_varptr_[9], input_varptr_[1]});
  auto mul3 = VectorRef({prim::kPrimMul, input_varptr_[10], input_varptr_[0]});
  auto mul4 = VectorRef({prim::kPrimMul, input_varptr_[11], input_varptr_[6]});
  auto add0 = VectorRef({prim::kPrimTensorAdd, mul0, mul1});
  auto add1 = VectorRef({prim::kPrimTensorAdd, mul2, mul3});

  auto real_div0 = VectorRef({prim_deal_div, add0, input_varptr_[5]});
  auto real_div1 = VectorRef({prim_deal_div, add1, input_varptr_[2]});

  auto add2 = VectorRef({prim::kPrimTensorAdd, real_div1, input_varptr_[12]});
  auto sqrt0 = VectorRef({prim_rsqrt, add2});
  auto real_div2 = VectorRef({prim::kPrimMul, real_div0, sqrt0});

  return VectorRef({prim::kPrimTensorAdd, real_div2, mul4});
}

bool LambNextMVRule::IsRuleMatched(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                   std::vector<AnfNodePtr> *old_pattern_outputs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr real_div0 = nullptr;
  CNodePtr real_div1 = nullptr;
  AnfNodePtr constant_add2_y = nullptr;
  std::tie(real_div0, real_div1, constant_add2_y) = GetSharedNodesByPattern(node);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &users = manager->node_users();
  if (users.find(real_div0) == users.end() || users[real_div0].size() < 2) {
    return false;
  }
  AnfNodeIndexSet real_div0_outputs = users[real_div0];
  auto iter = std::find_if(real_div0_outputs.begin(), real_div0_outputs.end(),
                           [&node, &real_div1, &constant_add2_y](const std::pair<AnfNodePtr, int> &node_index) {
                             return node_index.first != node && node_index.second == 1 &&
                                    MatchRealDiv4(node_index.first, real_div1, constant_add2_y);
                           });
  if (iter == real_div0_outputs.end()) {
    return false;
  }

  auto add0_cnode = CheckAnfNodeIfCNodeAndInputSize(real_div0->input(1), kAddInputNum);
  auto add1_cnode = CheckAnfNodeIfCNodeAndInputSize(real_div1->input(1), kAddInputNum);
  (*old_pattern_outputs).push_back(node);
  (*old_pattern_outputs).push_back(add0_cnode);
  (*old_pattern_outputs).push_back(add1_cnode);
  (*old_pattern_outputs).push_back(iter->first);

  return true;
}

AnfNodePtr LambNextMVRule::CreateLambNextMVNode(const FuncGraphPtr &func_graph,
                                                const std::vector<AnfNodePtr> &old_pattern_outputs,
                                                const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto prim = std::make_shared<Primitive>(kLambNextMVOpName);
  std::vector<AnfNodePtr> lamb_next_mv_rule_inputs = {NewValueNode(prim)};
  (void)std::transform(input_varptr_.begin(), input_varptr_.end(), std::back_inserter(lamb_next_mv_rule_inputs),
                       [&equiv](const VarPtr &in) { return utils::cast<AnfNodePtr>((*equiv)[in]); });
  auto lamb_next_mv_rule = func_graph->NewCNode(lamb_next_mv_rule_inputs);
  MS_EXCEPTION_IF_NULL(lamb_next_mv_rule);

  // Set abstract of new node
  AbstractBasePtrList new_abstracts;
  (void)std::transform(old_pattern_outputs.begin(), old_pattern_outputs.end(), std::back_inserter(new_abstracts),
                       [](const AnfNodePtr &out) { return out->abstract(); });
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(new_abstracts);
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  lamb_next_mv_rule->set_abstract(abstract_tuple);

  // Create tuple_getitem node for outputs
  std::vector<AnfNodePtr> lamb_next_mv_rule_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, lamb_next_mv_rule, kLambNextMVRuleOutputNum, &lamb_next_mv_rule_outputs);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(old_pattern_outputs[1], lamb_next_mv_rule_outputs[1]);
  (void)manager->Replace(old_pattern_outputs[2], lamb_next_mv_rule_outputs[2]);
  (void)manager->Replace(old_pattern_outputs[3], lamb_next_mv_rule_outputs[3]);

  return lamb_next_mv_rule_outputs[0];
}

const AnfNodePtr LambNextMVRule::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                         const EquivPtr &equiv) const {
  std::vector<AnfNodePtr> old_pattern_outputs;
  if (!IsRuleMatched(func_graph, node, &old_pattern_outputs)) {
    return nullptr;
  }

  return CreateLambNextMVNode(func_graph, old_pattern_outputs, equiv);
}
}  // namespace opt
}  // namespace mindspore
