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
#include "plugin/device/ascend/optimizer/ir_fusion/lamb_next_mv_with_decay_v1_rule.h"

#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/optimizer/opt.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace opt {
namespace {
std::tuple<AnfNodePtr, AnfNodePtr, AnfNodePtr, AnfNodePtr> GetSharedNodes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto add3 = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add3);
  CheckCNodeInputSize(add3, kAddInputTensorNum);
  auto real_div2_anf = add3->input(kIndex1);
  MS_EXCEPTION_IF_NULL(real_div2_anf);
  auto real_div2 = real_div2_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_div2);
  CheckCNodeInputSize(real_div2, kRealDivInputTensorNum);
  auto sqrt0_anf = real_div2->input(kIndex2);
  MS_EXCEPTION_IF_NULL(sqrt0_anf);
  auto sqrt0 = sqrt0_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sqrt0);
  CheckCNodeInputSize(sqrt0, kSqrtInputTensorNum);
  auto add2_anf = sqrt0->input(kIndex1);
  MS_EXCEPTION_IF_NULL(add2_anf);
  auto add2 = add2_anf->cast<CNodePtr>();
  CheckCNodeInputSize(add2, kAddInputTensorNum);
  return std::make_tuple(add3->input(kIndex2), real_div2->input(kIndex1), add2->input(kIndex1), add2->input(kIndex2));
}

bool MatchAdd5Pattern(const AnfNodePtr &node, const AnfNodePtr &mul4, const AnfNodePtr &real_div0,
                      const AnfNodePtr &real_div1, const AnfNodePtr &add2_y) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  auto add5 = node->cast<CNodePtr>();
  if (common::AnfAlgo::GetCNodeName(add5) != prim::kPrimAdd->name() ||
      common::AnfAlgo::GetInputTensorNum(add5) != kAddInputTensorNum) {
    return false;
  }
  auto real_div4_anf = add5->input(kIndex1);
  if (real_div4_anf == nullptr || !real_div4_anf->isa<CNode>()) {
    return false;
  }
  auto real_div4 = real_div4_anf->cast<CNodePtr>();
  if (common::AnfAlgo::GetCNodeName(real_div4) != kRealDivOpName ||
      common::AnfAlgo::GetInputTensorNum(real_div4) != kRealDivInputTensorNum) {
    return false;
  }
  auto add4_anf = real_div4->input(kIndex2);
  if (add4_anf == nullptr || !add4_anf->isa<CNode>()) {
    return false;
  }
  auto add4 = add4_anf->cast<CNodePtr>();
  if (common::AnfAlgo::GetCNodeName(add4) != prim::kPrimAdd->name() ||
      common::AnfAlgo::GetInputTensorNum(add4) != kAddInputTensorNum) {
    return false;
  }
  auto sqrt1_anf = add4->input(kIndex1);
  if (sqrt1_anf == nullptr || !sqrt1_anf->isa<CNode>()) {
    return false;
  }
  auto sqrt1 = sqrt1_anf->cast<CNodePtr>();
  if (common::AnfAlgo::GetCNodeName(sqrt1) != kSqrtOpName ||
      common::AnfAlgo::GetInputTensorNum(sqrt1) != kSqrtInputTensorNum) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(add2_y);
  return add5->input(kIndex2) == mul4 && real_div4->input(kIndex1) == real_div0 && sqrt1->input(kIndex1) == real_div1 &&
         *add4->input(kIndex2) == *add2_y;
}

std::tuple<AnfNodePtr, AnfNodePtr> GetAdd0Add1Nodes(const AnfNodePtr &real_div0_anf, const AnfNodePtr &real_div1_anf) {
  MS_EXCEPTION_IF_NULL(real_div0_anf);
  MS_EXCEPTION_IF_NULL(real_div1_anf);
  auto real_div0 = real_div0_anf->cast<CNodePtr>();
  auto real_div1 = real_div1_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_div0);
  MS_EXCEPTION_IF_NULL(real_div1);
  CheckCNodeInputSize(real_div0, kRealDivInputTensorNum);
  CheckCNodeInputSize(real_div1, kRealDivInputTensorNum);
  return std::make_tuple(real_div0->input(1), real_div1->input(1));
}
}  // namespace

std::vector<AnfNodePtr> LambNextMVWithDecayV1Rule::GetFusionNodeInputs(const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(equiv);
  auto i0 = utils::cast<AnfNodePtr>((*equiv)[input0_]);
  auto i1 = utils::cast<AnfNodePtr>((*equiv)[input1_]);
  auto i2 = utils::cast<AnfNodePtr>((*equiv)[input2_]);
  auto i3 = utils::cast<AnfNodePtr>((*equiv)[input3_]);
  auto i4 = utils::cast<AnfNodePtr>((*equiv)[input4_]);
  auto i5 = utils::cast<AnfNodePtr>((*equiv)[input5_]);
  auto i6 = utils::cast<AnfNodePtr>((*equiv)[input6_]);
  auto i7 = utils::cast<AnfNodePtr>((*equiv)[mul0_x_]);
  auto i8 = utils::cast<AnfNodePtr>((*equiv)[mul1_sub_]);
  auto i9 = utils::cast<AnfNodePtr>((*equiv)[mul2_x_]);
  auto i10 = utils::cast<AnfNodePtr>((*equiv)[mul3_sub1_]);
  auto i11 = utils::cast<AnfNodePtr>((*equiv)[mul4_x_]);
  auto i12 = utils::cast<AnfNodePtr>((*equiv)[add2_y_]);
  auto prim = std::make_shared<Primitive>(kLambNextMVWithDecayV1OpName);
  return {NewValueNode(prim), i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12};
}

const BaseRef LambNextMVWithDecayV1Rule::DefinePattern() const {
  const auto prim_rsqrt = std::make_shared<Primitive>(kRsqrtOpName);
  const auto prim_real_div = std::make_shared<Primitive>(kRealDivOpName);
  VectorRef mul3({prim::kPrimMul, mul3_sub1_, input0_});
  VectorRef mul2({prim::kPrimMul, mul2_x_, input1_});
  VectorRef add1({prim::kPrimAdd, mul2, mul3});
  VectorRef real_div1({prim_real_div, add1, input2_});
  VectorRef add2({prim::kPrimAdd, real_div1, add2_y_});
  VectorRef mul0({prim::kPrimMul, mul0_x_, input4_});
  VectorRef mul1({prim::kPrimMul, mul1_sub_, input3_});
  VectorRef sqrt0({prim_rsqrt, add2});
  VectorRef add0({prim::kPrimAdd, mul0, mul1});
  VectorRef real_div0({prim_real_div, add0, input5_});
  VectorRef real_div2({prim::kPrimMul, real_div0, sqrt0});
  VectorRef mul4({prim::kPrimMul, mul4_x_, input6_});
  VectorRef add3({prim::kPrimAdd, real_div2, mul4});
  return add3;
}

const AnfNodePtr LambNextMVWithDecayV1Rule::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if (!CheckSupportDataType(node, kFloatDataTypeSet)) {
    return nullptr;
  }
  AnfNodePtr mul4 = nullptr;
  AnfNodePtr real_div0 = nullptr;
  AnfNodePtr real_div1 = nullptr;
  AnfNodePtr add2_y = nullptr;
  std::tie(mul4, real_div0, real_div1, add2_y) = GetSharedNodes(node);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(mul4) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "The Mul4 should be used by at least another node input."
                      << " trace: " << trace::DumpSourceLines(node);
  }
  AnfNodeIndexSet mul4_output_node_index_set = manager->node_users()[mul4];
  auto iter = std::find_if(
    mul4_output_node_index_set.begin(), mul4_output_node_index_set.end(),
    [&node, &mul4, &real_div0, &real_div1, &add2_y](const std::pair<AnfNodePtr, int> &node_index) {
      return node_index.first != node && MatchAdd5Pattern(node_index.first, mul4, real_div0, real_div1, add2_y);
    });
  if (iter == mul4_output_node_index_set.end()) {
    return nullptr;
  }

  std::vector<AnfNodePtr> inputs = GetFusionNodeInputs(equiv);
  auto fusion_node = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_scope(node->scope());

  AnfNodePtr add0 = nullptr;
  AnfNodePtr add1 = nullptr;
  AnfNodePtr add5 = iter->first;
  std::tie(add0, add1) = GetAdd0Add1Nodes(real_div0, real_div1);
  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0), common::AnfAlgo::GetOutputInferDataType(add0, 0),
                common::AnfAlgo::GetOutputInferDataType(add1, 0), common::AnfAlgo::GetOutputInferDataType(add5, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(node, 0), AnfAlgo::GetOutputDetailShape(add0, 0),
                 AnfAlgo::GetOutputDetailShape(add1, 0), AnfAlgo::GetOutputDetailShape(add5, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, fusion_node.get());

  std::vector<AnfNodePtr> fusion_node_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, fusion_node, kLambNextMVWithDecayV1OutputNum, &fusion_node_outputs);
  if (fusion_node_outputs.size() != kLambNextMVWithDecayV1OutputNum) {
    MS_LOG(EXCEPTION) << "create multiple outputs for fusion node fail!"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  (void)manager->Replace(add0, fusion_node_outputs[kIndex1]);
  (void)manager->Replace(add1, fusion_node_outputs[kIndex2]);
  (void)manager->Replace(add5, fusion_node_outputs[kIndex3]);
  return fusion_node_outputs[kIndex0];
}
}  // namespace opt
}  // namespace mindspore
