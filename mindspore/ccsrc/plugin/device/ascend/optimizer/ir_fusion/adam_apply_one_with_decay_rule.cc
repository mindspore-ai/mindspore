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
#include "plugin/device/ascend/optimizer/ir_fusion/adam_apply_one_with_decay_rule.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "backend/common/optimizer/helper.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
std::vector<AnfNodePtr> AdamApplyOneWithDecayRule::GetFusionNodeInputs(const EquivPtr &equiv,
                                                                       const AnfNodePtr &final_node) const {
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
  PrimitivePtr prim = nullptr;
  if (common::AnfAlgo::CheckPrimitiveType(final_node, prim::kPrimDepend)) {
    prim = std::make_shared<Primitive>(kAdamApplyOneWithDecayAssignOpName);
  } else {
    prim = std::make_shared<Primitive>(kAdamApplyOneWithDecayOpName);
  }
  return {NewValueNode(prim), input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y};
}

const BaseRef AdamApplyOneWithDecayRuleCond1::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3({prim::kPrimMul, mul3_x_, square0});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, add2_y_, sqrt0});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, input4_, add3});
  VectorRef sub0({prim::kPrimSub, input3_, mul5});
  return sub0;
}

const BaseRef AdamApplyOneWithDecayRuleCond2::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, input2_, mul0_x_});
  VectorRef mul1({prim::kPrimMul, input0_, mul1_x_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, input1_, mul2_x_});
  VectorRef mul3({prim::kPrimMul, mul3_x_, square0});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, sqrt0, add2_y_});
  VectorRef mul4({prim::kPrimMul, input3_, mul4_x_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, add3, input4_});
  VectorRef sub0({prim::kPrimSub, input3_, mul5});
  return sub0;
}

const BaseRef AdamApplyOneWithDecayRuleCond3::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3({prim::kPrimMul, square0, mul3_x_});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, sqrt0, add2_y_});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, add3, input4_});
  VectorRef sub0({prim::kPrimSub, input3_, mul5});
  return sub0;
}

const BaseRef AdamApplyOneWithDecayRuleCond4::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3({prim::kPrimMul, mul3_x_, square0});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, add2_y_, sqrt0});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, add3, input4_});
  VectorRef sub0({prim::kPrimSub, input3_, mul5});
  return sub0;
}

const BaseRef AdamApplyOneWithDecayRuleCond5::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3({prim::kPrimMul, mul3_x_, square0});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, sqrt0, add2_y_});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, add3, input4_});
  VectorRef sub0({prim::kPrimSub, input3_, mul5});
  return sub0;
}

const BaseRef AdamApplyOneWithDecayRuleCond6::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3({prim::kPrimMul, mul3_x_, square0});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, add2_y_, sqrt0});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, input4_, add3});
  VectorRef sub0({prim::kPrimSub, input3_, mul5});
  return sub0;
}

const BaseRef AdamApplyOneWithDecayAssignRuleCond1::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3({prim::kPrimMul, mul3_x_, square0});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, add2_y_, sqrt0});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, input4_, add3});
  VectorRef sub0({sub0_var_, input3_, mul5});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input3_, sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input2_, add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input1_, add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

const BaseRef AdamApplyOneWithDecayAssignRuleCond2::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, input2_, mul0_x_});
  VectorRef mul1({prim::kPrimMul, input0_, mul1_x_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, input1_, mul2_x_});
  VectorRef mul3({prim::kPrimMul, mul3_x_, square0});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, sqrt0, add2_y_});
  VectorRef mul4({prim::kPrimMul, input3_, mul4_x_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, add3, input4_});
  VectorRef sub0({sub0_var_, input3_, mul5});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input3_, sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input2_, add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input1_, add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

const BaseRef AdamApplyOneWithDecayAssignRuleCond3::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3({prim::kPrimMul, square0, mul3_x_});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, sqrt0, add2_y_});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, add3, input4_});
  VectorRef sub0({sub0_var_, input3_, mul5});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input3_, sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input2_, add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input1_, add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

const BaseRef AdamApplyOneWithDecayAssignRuleCond4::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3({prim::kPrimMul, mul3_x_, square0});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, add2_y_, sqrt0});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, add3, input4_});
  VectorRef sub0({sub0_var_, input3_, mul5});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input3_, sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input2_, add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input1_, add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

const BaseRef AdamApplyOneWithDecayAssignRuleCond5::DefinePattern() const {
  auto sqrt = std::make_shared<Primitive>(kSqrtOpName);
  auto real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul0({prim::kPrimMul, mul0_x_, input2_});
  VectorRef mul1({prim::kPrimMul, mul1_x_, input0_});
  VectorRef square0({prim::kPrimSquare, input0_});
  VectorRef add0({add0_var_, mul0, mul1});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3({prim::kPrimMul, mul3_x_, square0});
  VectorRef add1({add1_var_, mul2, mul3});
  VectorRef sqrt0({sqrt, add1});
  VectorRef add2({prim::kPrimAdd, sqrt0, add2_y_});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input3_});
  VectorRef real_div0({real_div, add0, add2});
  VectorRef add3({prim::kPrimAdd, mul4, real_div0});
  VectorRef mul5({prim::kPrimMul, add3, input4_});
  VectorRef sub0({sub0_var_, input3_, mul5});
  VectorRef assign0 = VectorRef({prim::kPrimAssign, input3_, sub0});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, sub0, assign0});
  VectorRef assign1 = VectorRef({prim::kPrimAssign, input2_, add0});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, depend0, assign1});
  VectorRef assign2 = VectorRef({prim::kPrimAssign, input1_, add1});
  return VectorRef({prim::kPrimDepend, depend1, assign2});
}

const AnfNodePtr AdamApplyOneWithDecayRule::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
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
  std::vector<AnfNodePtr> inputs = GetFusionNodeInputs(equiv, node);
  auto fusion_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_scope(sub0->scope());

  auto iter_add0 = (*equiv).find(add0_var_);
  if (iter_add0 == (*equiv).cend()) {
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
  auto types = {common::AnfAlgo::GetOutputInferDataType(add1, 0), common::AnfAlgo::GetOutputInferDataType(add0, 0),
                common::AnfAlgo::GetOutputInferDataType(sub0, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(add1, 0), AnfAlgo::GetOutputDetailShape(add0, 0),
                 AnfAlgo::GetOutputDetailShape(sub0, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, fusion_node.get());

  std::vector<AnfNodePtr> fusion_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, fusion_node, kAdamApplyOneWithDecayOutputNum, &fusion_node_outputs);
  if (fusion_node_outputs.size() != kAdamApplyOneWithDecayOutputNum) {
    MS_LOG(ERROR) << "Create multiple outputs for fusion node failed, should have " << kAdamApplyOneWithDecayOutputNum
                  << " outputs, but got " << fusion_node_outputs.size() << " outputs.";
    return nullptr;
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(add1, fusion_node_outputs[kIndex0]);
  (void)manager->Replace(add0, fusion_node_outputs[kIndex1]);
  return fusion_node_outputs[kIndex2];
}
}  // namespace opt
}  // namespace mindspore
