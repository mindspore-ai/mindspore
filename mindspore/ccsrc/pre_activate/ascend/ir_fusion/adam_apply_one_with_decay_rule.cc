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
#include <tuple>

#include "session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
std::tuple<AnfNodePtr, AnfNodePtr> GetAdd0Add1Node(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto sub0 = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sub0);
  auto mul5_anf = sub0->input(2);
  MS_EXCEPTION_IF_NULL(mul5_anf);
  auto mul5 = mul5_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul5);
  auto add3_anf = mul5->input(2);
  MS_EXCEPTION_IF_NULL(add3_anf);
  auto add3 = add3_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add3);
  auto real_div0_anf = add3->input(1);
  MS_EXCEPTION_IF_NULL(real_div0_anf);
  auto real_div0 = real_div0_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_div0);
  auto add0_anf = real_div0->input(1);
  MS_EXCEPTION_IF_NULL(add0_anf);
  auto add2_anf = real_div0->input(2);
  MS_EXCEPTION_IF_NULL(add2_anf);
  auto add2 = add2_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add2);
  auto sqrt0_anf = add2->input(1);
  MS_EXCEPTION_IF_NULL(sqrt0_anf);
  auto sqrt0 = sqrt0_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sqrt0);
  auto add1_anf = sqrt0->input(1);
  MS_EXCEPTION_IF_NULL(add1_anf);
  return std::make_tuple(add0_anf, add1_anf);
}
}  // namespace

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
  VectorRef add0_pattern({prim::kPrimTensorAdd, mul0_pattern, mul1_pattern});
  VectorRef mul2_pattern({prim::kPrimMul, mul2_x_, input1_});
  VectorRef mul3_pattern({prim::kPrimMul, mul3_x_, square0_pattern});
  VectorRef add1_pattern({prim::kPrimTensorAdd, mul2_pattern, mul3_pattern});
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

  AnfNodePtr add0 = nullptr;
  AnfNodePtr add1 = nullptr;
  std::tie(add0, add1) = GetAdd0Add1Node(node);
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
