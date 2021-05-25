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
#include "backend/optimizer/ascend/ir_fusion/lamb_next_mv_with_decay_rule.h"
#include <utility>
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/optimizer/opt.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace {
constexpr size_t kZeroIndex = 0;
constexpr size_t kFirstIndex = 1;
constexpr size_t kSecondIndex = 2;
constexpr size_t kThirdIndex = 3;
constexpr size_t kFourthIndex = 4;
constexpr size_t kFifthIndex = 5;
constexpr size_t kSixthIndex = 6;
}  // namespace

namespace opt {
AnfNodePtr LambNextMVWithDecayRule::GetLambNextMVWithDecayOutput(const FuncGraphPtr &func_graph,
                                                                 const AnfNodePtr &new_node, const AnfNodePtr &add3,
                                                                 const AnfNodePtr &add5, const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  MS_EXCEPTION_IF_NULL(add3);
  MS_EXCEPTION_IF_NULL(add5);
  MS_EXCEPTION_IF_NULL(equiv);
  constexpr size_t kOutputIndex0 = 0;
  constexpr size_t kOutputIndex1 = 1;
  constexpr size_t kOutputIndex2 = 2;
  constexpr size_t kOutputIndex3 = 3;
  auto add0 = GetAnfNodeByVar(equiv, add0_var_);
  MS_EXCEPTION_IF_NULL(add0);
  auto add1 = GetAnfNodeByVar(equiv, add1_var_);
  MS_EXCEPTION_IF_NULL(add1);

  // Set abstract of new node
  AbstractBasePtrList new_node_list;
  new_node_list.push_back(add3->abstract());
  new_node_list.push_back(add0->abstract());
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
  (void)manager->Replace(add3, new_node_outputs[kOutputIndex0]);
  (void)manager->Replace(add0, new_node_outputs[kOutputIndex1]);
  (void)manager->Replace(add1, new_node_outputs[kOutputIndex2]);
  return new_node_outputs[kOutputIndex3];
}

AnfNodePtr LambNextMVWithDecayRule::CreateLambNextMVWithDecayNode(const FuncGraphPtr &func_graph,
                                                                  const AnfNodePtr &add3, const AnfNodePtr &add5,
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
  return GetLambNextMVWithDecayOutput(func_graph, new_node, add3, add5, equiv);
}

bool LambNextMVWithDecayRule::IsShareNodes(const EquivPtr &equiv1, const EquivPtr &equiv2) const {
  return IsSameNode(equiv1, equiv2, mul4_var_) && IsSameNode(equiv1, equiv2, real_div0_var_) &&
         IsSameNode(equiv1, equiv2, real_div1_var_) && IsSameNode(equiv1, equiv2, constant_add2_y_);
}

const AnfNodePtr LambNextMVWithDecayRule::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!CheckSupportDataType(node, kFloatDataTypeSet)) {
    return nullptr;
  }
  AnfNodePtr mul4 = GetAnfNodeByVar(equiv, mul4_var_);
  MS_EXCEPTION_IF_NULL(mul4);
  // Get add3 and match the add3 pattern
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(mul4) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "The Mul4 should be used by at least another node input"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  AnfNodeIndexSet mul4_outputs = manager->node_users()[mul4];
  auto iter = std::find_if(mul4_outputs.begin(), mul4_outputs.end(),
                           [&node, &equiv, this](const std::pair<AnfNodePtr, int> &node_index) {
                             return node_index.first != node && MatchAnotherPattern(node_index.first, equiv);
                           });
  if (iter != mul4_outputs.end()) {
    return CreateLambNextMVWithDecayNode(func_graph, iter->first, node, equiv);
  }
  return nullptr;
}

BaseRef LambNextMVWithDecayRuleCond1::DefineAnotherPattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_rsqrt);
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Ys = std::make_shared<SeqVar>();
  VarPtr Zs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  MS_EXCEPTION_IF_NULL(Ys);
  MS_EXCEPTION_IF_NULL(Zs);
  VectorRef real_div0 = VectorRef({real_div0_var_, Xs});
  VectorRef real_div1 = VectorRef({real_div1_var_, Ys});
  VectorRef mul4 = VectorRef({mul4_var_, Zs});

  VectorRef add2 = VectorRef({prim::kPrimAdd, constant_add2_y_, real_div1});
  VectorRef sqrt0 = VectorRef({prim_rsqrt, add2});
  VectorRef real_div2 = VectorRef({prim::kPrimMul, sqrt0, real_div0});
  VectorRef add3 = VectorRef({prim::kPrimAdd, mul4, real_div2});
  return add3;
}

const BaseRef LambNextMVWithDecayRuleCond1::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_sqrt);
  const auto prim_deal_div = std::make_shared<Primitive>(kRealDivOpName);
  MS_EXCEPTION_IF_NULL(prim_deal_div);
  VectorRef mul2 = VectorRef({prim::kPrimMul, input_vars_[kFirstIndex], constant_mul_input_vars_[kSecondIndex]});
  VectorRef mul3 = VectorRef({prim::kPrimMul, input_vars_[kZeroIndex], constant_mul_input_vars_[kThirdIndex]});
  VectorRef add1 = VectorRef({add1_var_, mul2, mul3});
  VectorRef real_div1 = VectorRef({real_div1_var_, add1, input_vars_[kSecondIndex]});
  VectorRef sqrt1 = VectorRef({prim_sqrt, real_div1});
  VectorRef add4 = VectorRef({prim::kPrimAdd, sqrt1, constant_add2_y_});
  VectorRef mul0 = VectorRef({prim::kPrimMul, input_vars_[kFourthIndex], constant_mul_input_vars_[kZeroIndex]});
  VectorRef mul1 = VectorRef({prim::kPrimMul, input_vars_[kThirdIndex], constant_mul_input_vars_[kFirstIndex]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef real_div0 = VectorRef({real_div0_var_, add0, input_vars_[kFifthIndex]});
  VectorRef real_div4 = VectorRef({prim_deal_div, real_div0, add4});
  VectorRef mul4 = VectorRef({mul4_var_, constant_mul_input_vars_[kFourthIndex], input_vars_[kSixthIndex]});
  VectorRef add5 = VectorRef({prim::kPrimAdd, mul4, real_div4});
  return add5;
}

BaseRef LambNextMVWithDecayRuleCond2::DefineAnotherPattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_rsqrt);
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Ys = std::make_shared<SeqVar>();
  VarPtr Zs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  MS_EXCEPTION_IF_NULL(Ys);
  MS_EXCEPTION_IF_NULL(Zs);
  VectorRef real_div0 = VectorRef({real_div0_var_, Xs});
  VectorRef real_div1 = VectorRef({real_div1_var_, Ys});
  VectorRef mul4 = VectorRef({mul4_var_, Zs});

  VectorRef add2 = VectorRef({prim::kPrimAdd, constant_add2_y_, real_div1});
  VectorRef sqrt0 = VectorRef({prim_rsqrt, add2});
  VectorRef real_div2 = VectorRef({prim::kPrimMul, sqrt0, real_div0});
  VectorRef add3 = VectorRef({prim::kPrimAdd, mul4, real_div2});
  return add3;
}

const BaseRef LambNextMVWithDecayRuleCond2::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_sqrt);
  const auto prim_deal_div = std::make_shared<Primitive>(kRealDivOpName);
  MS_EXCEPTION_IF_NULL(prim_deal_div);
  VectorRef mul2 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[kSecondIndex], input_vars_[kFirstIndex]});
  VectorRef mul3 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[kThirdIndex], input_vars_[kZeroIndex]});
  VectorRef add1 = VectorRef({add1_var_, mul2, mul3});
  VectorRef real_div1 = VectorRef({real_div1_var_, add1, input_vars_[kSecondIndex]});
  VectorRef sqrt1 = VectorRef({prim_sqrt, real_div1});
  VectorRef add4 = VectorRef({prim::kPrimAdd, constant_add2_y_, sqrt1});
  VectorRef mul0 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[kZeroIndex], input_vars_[kFourthIndex]});
  VectorRef mul1 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[kFirstIndex], input_vars_[kThirdIndex]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef real_div0 = VectorRef({real_div0_var_, add0, input_vars_[kFifthIndex]});
  VectorRef real_div4 = VectorRef({prim_deal_div, real_div0, add4});
  VectorRef mul4 = VectorRef({mul4_var_, constant_mul_input_vars_[kFourthIndex], input_vars_[kSixthIndex]});
  VectorRef add5 = VectorRef({prim::kPrimAdd, mul4, real_div4});
  return add5;
}

BaseRef LambNextMVWithDecayRuleCond3::DefineAnotherPattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_rsqrt);
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Ys = std::make_shared<SeqVar>();
  VarPtr Zs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  MS_EXCEPTION_IF_NULL(Ys);
  MS_EXCEPTION_IF_NULL(Zs);
  VectorRef real_div0 = VectorRef({real_div0_var_, Xs});
  VectorRef real_div1 = VectorRef({real_div1_var_, Ys});
  VectorRef mul4 = VectorRef({mul4_var_, Zs});

  VectorRef add2 = VectorRef({prim::kPrimAdd, real_div1, constant_add2_y_});
  VectorRef sqrt0 = VectorRef({prim_rsqrt, add2});
  VectorRef real_div2 = VectorRef({prim::kPrimMul, sqrt0, real_div0});
  VectorRef add3 = VectorRef({prim::kPrimAdd, mul4, real_div2});
  return add3;
}

const BaseRef LambNextMVWithDecayRuleCond3::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_sqrt);
  const auto prim_deal_div = std::make_shared<Primitive>(kRealDivOpName);
  MS_EXCEPTION_IF_NULL(prim_deal_div);
  VectorRef mul2 = VectorRef({prim::kPrimMul, input_vars_[kFirstIndex], constant_mul_input_vars_[kSecondIndex]});
  VectorRef mul3 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[kThirdIndex], input_vars_[kZeroIndex]});
  VectorRef add1 = VectorRef({add1_var_, mul2, mul3});
  VectorRef real_div1 = VectorRef({real_div1_var_, add1, input_vars_[kSecondIndex]});
  VectorRef sqrt1 = VectorRef({prim_sqrt, real_div1});
  VectorRef add4 = VectorRef({prim::kPrimAdd, sqrt1, constant_add2_y_});
  VectorRef mul0 = VectorRef({prim::kPrimMul, input_vars_[kFourthIndex], constant_mul_input_vars_[kZeroIndex]});
  VectorRef mul1 = VectorRef({prim::kPrimMul, input_vars_[kThirdIndex], constant_mul_input_vars_[kFirstIndex]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef real_div0 = VectorRef({real_div0_var_, add0, input_vars_[kFifthIndex]});
  VectorRef real_div4 = VectorRef({prim_deal_div, real_div0, add4});
  VectorRef mul4 = VectorRef({mul4_var_, input_vars_[kSixthIndex], constant_mul_input_vars_[kFourthIndex]});
  VectorRef add5 = VectorRef({prim::kPrimAdd, mul4, real_div4});
  return add5;
}

BaseRef LambNextMVWithDecayRuleCond4::DefineAnotherPattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_rsqrt);
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Ys = std::make_shared<SeqVar>();
  VarPtr Zs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  MS_EXCEPTION_IF_NULL(Ys);
  MS_EXCEPTION_IF_NULL(Zs);
  // Two patterns share: real_div0, real_div1, mul4, constant_add2_y_
  VectorRef real_div0 = VectorRef({real_div0_var_, Xs});
  VectorRef real_div1 = VectorRef({real_div1_var_, Ys});
  VectorRef mul4 = VectorRef({mul4_var_, Zs});

  VectorRef add2 = VectorRef({prim::kPrimAdd, real_div1, constant_add2_y_});
  VectorRef sqrt0 = VectorRef({prim_rsqrt, add2});
  VectorRef real_div2 = VectorRef({prim::kPrimMul, real_div0, sqrt0});
  VectorRef add3 = VectorRef({prim::kPrimAdd, real_div2, mul4});
  return add3;
}

const BaseRef LambNextMVWithDecayRuleCond4::DefinePattern() const {
  const auto prim_sqrt = std::make_shared<Primitive>(kSqrtOpName);
  MS_EXCEPTION_IF_NULL(prim_sqrt);
  const auto prim_deal_div = std::make_shared<Primitive>(kRealDivOpName);
  MS_EXCEPTION_IF_NULL(prim_deal_div);
  VectorRef mul2 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[kSecondIndex], input_vars_[kFirstIndex]});
  VectorRef mul3 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[kThirdIndex], input_vars_[kZeroIndex]});
  VectorRef add1 = VectorRef({add1_var_, mul2, mul3});
  VectorRef real_div1 = VectorRef({real_div1_var_, add1, input_vars_[kSecondIndex]});
  VectorRef sqrt1 = VectorRef({prim_sqrt, real_div1});
  VectorRef add4 = VectorRef({prim::kPrimAdd, sqrt1, constant_add2_y_});
  VectorRef mul0 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[kZeroIndex], input_vars_[kFourthIndex]});
  VectorRef mul1 = VectorRef({prim::kPrimMul, constant_mul_input_vars_[kFirstIndex], input_vars_[kThirdIndex]});
  VectorRef add0 = VectorRef({add0_var_, mul0, mul1});
  VectorRef real_div0 = VectorRef({real_div0_var_, add0, input_vars_[kFifthIndex]});
  VectorRef real_div4 = VectorRef({prim_deal_div, real_div0, add4});
  VectorRef mul4 = VectorRef({mul4_var_, constant_mul_input_vars_[kFourthIndex], input_vars_[kSixthIndex]});
  VectorRef add5 = VectorRef({prim::kPrimAdd, real_div4, mul4});
  return add5;
}
}  // namespace opt
}  // namespace mindspore
