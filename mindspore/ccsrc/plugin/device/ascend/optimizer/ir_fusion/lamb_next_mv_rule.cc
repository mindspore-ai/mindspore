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

#include "plugin/device/ascend/optimizer/ir_fusion/lamb_next_mv_rule.h"
#include <memory>
#include <utility>
#include <algorithm>
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/core/ops/core_ops.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
bool LambNextMVRule::IsRuleMatched(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv,
                                   std::vector<AnfNodePtr> *const old_pattern_outputs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  auto real_div0 = GetAnfNodeByVar(equiv, real_div0_var_);
  auto real_div2 = GetAnfNodeByVar(equiv, real_div2_var_);
  constexpr size_t kRealDiv0Size = 2;

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &users = manager->node_users();
  if (users.find(real_div0) == users.end() || users[real_div0].size() < kRealDiv0Size) {
    return false;
  }
  AnfNodeIndexSet real_div0_outputs = users[real_div0];
  auto iter = std::find_if(real_div0_outputs.begin(), real_div0_outputs.end(),
                           [&real_div2, &equiv, this](const std::pair<AnfNodePtr, int> &node_index) {
                             return node_index.first != real_div2 && node_index.second == 1 &&
                                    MatchAnotherPattern(node_index.first, equiv);
                           });
  if (iter == real_div0_outputs.end()) {
    return false;
  }

  (*old_pattern_outputs).push_back(node);
  (*old_pattern_outputs).push_back(GetAnfNodeByVar(equiv, add0_var_));
  (*old_pattern_outputs).push_back(GetAnfNodeByVar(equiv, add1_var_));
  (*old_pattern_outputs).push_back(iter->first);

  return true;
}

AnfNodePtr LambNextMVRule::CreateLambNextMVNode(const FuncGraphPtr &func_graph,
                                                const std::vector<AnfNodePtr> &old_pattern_outputs,
                                                const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto prim = std::make_shared<Primitive>(kLambNextMVOpName);
  std::vector<AnfNodePtr> lamb_next_mv_rule_inputs = {NewValueNode(prim)};
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[input0_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[input1_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[input2_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[input3_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[input4_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[input5_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[input6_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[mul0_x_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[mul1_sub_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[mul2_x_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[mul3_sub1_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[mul4_x_]));
  lamb_next_mv_rule_inputs.push_back(utils::cast<AnfNodePtr>((*equiv)[add2_y_]));
  auto lamb_next_mv_rule = NewCNode(lamb_next_mv_rule_inputs, func_graph);
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
  (void)manager->Replace(old_pattern_outputs[kIndex1], lamb_next_mv_rule_outputs[kIndex1]);
  (void)manager->Replace(old_pattern_outputs[kIndex2], lamb_next_mv_rule_outputs[kIndex2]);
  (void)manager->Replace(old_pattern_outputs[kIndex3], lamb_next_mv_rule_outputs[kIndex3]);

  return lamb_next_mv_rule_outputs[0];
}

bool LambNextMVRule::IsShareNodes(const EquivPtr &equiv1, const EquivPtr &equiv2) const {
  return IsSameNode(equiv1, equiv2, real_div0_var_) && IsSameNode(equiv1, equiv2, real_div1_var_) &&
         IsSameNode(equiv1, equiv2, add2_y_);
}

const AnfNodePtr LambNextMVRule::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                         const EquivPtr &equiv) const {
  if (!CheckSupportDataType(node, kFloatDataTypeSet)) {
    return nullptr;
  }
  std::vector<AnfNodePtr> old_pattern_outputs;
  if (!IsRuleMatched(func_graph, node, equiv, &old_pattern_outputs)) {
    return nullptr;
  }

  return CreateLambNextMVNode(func_graph, old_pattern_outputs, equiv);
}

const BaseRef LambNextMVRuleCond1::DefinePattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);

  auto mul0 = VectorRef({prim::kPrimMul, mul0_x_, input4_});
  auto mul1 = VectorRef({prim::kPrimMul, mul1_sub_, input3_});
  auto mul2 = VectorRef({prim::kPrimMul, mul2_x_, input1_});
  auto mul3 = VectorRef({prim::kPrimMul, mul3_sub1_, input0_});
  auto mul4 = VectorRef({prim::kPrimMul, mul4_x_, input6_});
  auto add0 = VectorRef({add0_var_, mul0, mul1});
  auto add1 = VectorRef({add1_var_, mul2, mul3});

  auto real_div0 = VectorRef({real_div0_var_, add0, input5_});
  auto real_div1 = VectorRef({real_div1_var_, add1, input2_});

  auto add2 = VectorRef({prim::kPrimAdd, add2_y_, real_div1});
  auto sqrt0 = VectorRef({prim_rsqrt, add2});
  auto real_div2 = VectorRef({real_div2_var_, sqrt0, real_div0});

  return VectorRef({prim::kPrimAdd, mul4, real_div2});
}

BaseRef LambNextMVRuleCond1::DefineAnotherPattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Ys = std::make_shared<SeqVar>();
  // Two patterns share: real_div0, real_div1, add2_y_
  VectorRef real_div0 = VectorRef({real_div0_var_, Xs});
  VectorRef real_div1 = VectorRef({real_div1_var_, Ys});

  VectorRef sqrt1 = VectorRef({prim_sqrt, real_div1});
  VectorRef add4 = VectorRef({prim::kPrimAdd, add2_y_, sqrt1});
  VectorRef real_div4 = VectorRef({prim_real_div, real_div0, add4});
  return real_div4;
}

const BaseRef LambNextMVRuleCond2::DefinePattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);

  auto mul0 = VectorRef({prim::kPrimMul, input4_, mul0_x_});
  auto mul1 = VectorRef({prim::kPrimMul, input3_, mul1_sub_});
  auto mul2 = VectorRef({prim::kPrimMul, input1_, mul2_x_});
  auto mul3 = VectorRef({prim::kPrimMul, mul3_sub1_, input0_});
  auto mul4 = VectorRef({prim::kPrimMul, input6_, mul4_x_});
  auto add0 = VectorRef({add0_var_, mul0, mul1});
  auto add1 = VectorRef({add1_var_, mul2, mul3});

  auto real_div0 = VectorRef({real_div0_var_, add0, input5_});
  auto real_div1 = VectorRef({real_div1_var_, add1, input2_});

  auto add2 = VectorRef({prim::kPrimAdd, add2_y_, real_div1});
  auto sqrt0 = VectorRef({prim_rsqrt, add2});
  auto real_div2 = VectorRef({real_div2_var_, sqrt0, real_div0});

  return VectorRef({prim::kPrimAdd, mul4, real_div2});
}

BaseRef LambNextMVRuleCond2::DefineAnotherPattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Ys = std::make_shared<SeqVar>();
  // Two patterns share: real_div0, real_div1, add2_y_
  VectorRef real_div0 = VectorRef({real_div0_var_, Xs});
  VectorRef real_div1 = VectorRef({real_div1_var_, Ys});

  VectorRef sqrt1 = VectorRef({prim_sqrt, real_div1});
  VectorRef add4 = VectorRef({prim::kPrimAdd, sqrt1, add2_y_});
  VectorRef real_div4 = VectorRef({prim_real_div, real_div0, add4});
  return real_div4;
}

const BaseRef LambNextMVRuleCond3::DefinePattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);

  auto mul0 = VectorRef({prim::kPrimMul, input4_, mul0_x_});
  auto mul1 = VectorRef({prim::kPrimMul, input3_, mul1_sub_});
  auto mul2 = VectorRef({prim::kPrimMul, input1_, mul2_x_});
  auto mul3 = VectorRef({prim::kPrimMul, input0_, mul3_sub1_});
  auto mul4 = VectorRef({prim::kPrimMul, input6_, mul4_x_});
  auto add0 = VectorRef({add0_var_, mul0, mul1});
  auto add1 = VectorRef({add1_var_, mul2, mul3});

  auto real_div0 = VectorRef({real_div0_var_, add0, input5_});
  auto real_div1 = VectorRef({real_div1_var_, add1, input2_});

  auto add2 = VectorRef({prim::kPrimAdd, real_div1, add2_y_});
  auto sqrt0 = VectorRef({prim_rsqrt, add2});
  auto real_div2 = VectorRef({real_div2_var_, sqrt0, real_div0});

  return VectorRef({prim::kPrimAdd, mul4, real_div2});
}

BaseRef LambNextMVRuleCond3::DefineAnotherPattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Ys = std::make_shared<SeqVar>();
  // Two patterns share: real_div0, real_div1, add2_y_
  VectorRef real_div0 = VectorRef({real_div0_var_, Xs});
  VectorRef real_div1 = VectorRef({real_div1_var_, Ys});

  VectorRef sqrt1 = VectorRef({prim_sqrt, real_div1});
  VectorRef add4 = VectorRef({prim::kPrimAdd, sqrt1, add2_y_});
  VectorRef real_div4 = VectorRef({prim_real_div, real_div0, add4});
  return real_div4;
}

const BaseRef LambNextMVRuleCond4::DefinePattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);

  auto mul0 = VectorRef({prim::kPrimMul, mul0_x_, input4_});
  auto mul1 = VectorRef({prim::kPrimMul, mul1_sub_, input3_});
  auto mul2 = VectorRef({prim::kPrimMul, mul2_x_, input1_});
  auto mul3 = VectorRef({prim::kPrimMul, mul3_sub1_, input0_});
  auto mul4 = VectorRef({prim::kPrimMul, mul4_x_, input6_});
  auto add0 = VectorRef({add0_var_, mul0, mul1});
  auto add1 = VectorRef({add1_var_, mul2, mul3});

  auto real_div0 = VectorRef({real_div0_var_, add0, input5_});
  auto real_div1 = VectorRef({real_div1_var_, add1, input2_});

  auto add2 = VectorRef({prim::kPrimAdd, real_div1, add2_y_});
  auto sqrt0 = VectorRef({prim_rsqrt, add2});
  auto real_div2 = VectorRef({real_div2_var_, real_div0, sqrt0});

  return VectorRef({prim::kPrimAdd, real_div2, mul4});
}

BaseRef LambNextMVRuleCond4::DefineAnotherPattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Ys = std::make_shared<SeqVar>();
  // Two patterns share: real_div0, real_div1, add2_y_
  VectorRef real_div0 = VectorRef({real_div0_var_, Xs});
  VectorRef real_div1 = VectorRef({real_div1_var_, Ys});

  VectorRef sqrt1 = VectorRef({prim_sqrt, real_div1});
  VectorRef add4 = VectorRef({prim::kPrimAdd, sqrt1, add2_y_});
  VectorRef real_div4 = VectorRef({prim_real_div, real_div0, add4});
  return real_div4;
}
}  // namespace opt
}  // namespace mindspore
