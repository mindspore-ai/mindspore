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
#include "plugin/device/ascend/optimizer/ir_fusion/confusion_mul_grad_fusion.h"
#include <utility>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "backend/common/optimizer/helper.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace opt {
namespace {
const size_t kConfusionMulGradOutputNum = 2;

AnfNodePtr GetMul0(const FuncGraphPtr &graph, const AnfNodePtr &input2, const AnfNodePtr &mul1) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input2);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(input2) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager" << trace::DumpSourceLines(input2);
  }

  AnfNodePtr mul0 = nullptr;
  const AnfNodeIndexSet &outputs_set = manager->node_users()[input2];
  // input2 must be the 2rd input of mul0
  auto it = std::find_if(outputs_set.begin(), outputs_set.end(), [&mul1](const std::pair<AnfNodePtr, int> &node_index) {
    constexpr int kMul0InputIndex2 = 2;
    return node_index.first != mul1 && node_index.second == kMul0InputIndex2;
  });
  if (it != outputs_set.end() && common::AnfAlgo::GetCNodeName(it->first) == prim::kPrimMul->name()) {
    mul0 = it->first;
  }
  return mul0;
}

bool QuitFusion(const FuncGraphPtr &graph, const AnfNodePtr &mul0_anf, const AnfNodePtr &mul1_anf,
                const AnfNodePtr &reduce_sum, const AnfNodePtr &input2) {
  MS_EXCEPTION_IF_NULL(mul0_anf);
  MS_EXCEPTION_IF_NULL(mul1_anf);
  MS_EXCEPTION_IF_NULL(reduce_sum);
  MS_EXCEPTION_IF_NULL(input2);
  auto addn = input2->cast<CNodePtr>();
  constexpr size_t kInferShapeIndex = 2;
  constexpr ShapeValueDType kShape2Dim1 = 1024;
  constexpr ShapeValueDType kShape2Dim2 = 768;
  if (addn == nullptr || common::AnfAlgo::GetCNodeName(addn) != prim::kPrimAddN->name()) {
    MS_LOG(INFO) << "Mul's second input is not Addn, quit fusion";
    return true;
  }
  if (common::AnfAlgo::IsDynamicShape(addn)) {
    return true;
  }
  auto shape = common::AnfAlgo::GetOutputInferShape(addn, 0);
  if (shape.size() != kInferShapeIndex || !(shape[1] == kShape2Dim1 || shape[1] == kShape2Dim2)) {
    MS_LOG(INFO) << "Addn's infer shape is not equal to [x,1024] or [x,768], quit fusion";
    return true;
  }
  if (!mul0_anf->isa<CNode>() || !mul1_anf->isa<CNode>()) {
    return true;
  }
  auto mul1 = mul1_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul1);
  auto mul0 = mul0_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul0);

  if (IsDepend(*graph, mul0->input(1), {reduce_sum})) {
    MS_LOG(INFO) << "mul0->input(1) depends on reduce_sum, quit fusion";
    return true;
  }
  if (IsDepend(*graph, mul1->input(1), {mul0})) {
    MS_LOG(INFO) << "mul1->input(1) depends on mul0, quit fusion";
    return true;
  }
  return false;
}
}  // namespace

CNodePtr ConfusionMulGradFusion::CreateFusionNode(const FuncGraphPtr &graph, const CNodePtr &reduce_sum,
                                                  const AnfNodePtr &mul0_anf, const AnfNodePtr &input3) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(reduce_sum);
  MS_EXCEPTION_IF_NULL(mul0_anf);
  MS_EXCEPTION_IF_NULL(input3);
  auto mul0 = mul0_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul0);

  auto prim = std::make_shared<Primitive>(kConfusionMulGradOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), mul0->input(kIndex1), mul0->input(kIndex2), input3};
  auto fusion_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_scope(reduce_sum->scope());
  common::AnfAlgo::CopyNodeAttr(kAttrAxis, reduce_sum, fusion_node);
  common::AnfAlgo::CopyNodeAttr(kAttrKeepDims, reduce_sum, fusion_node);
  auto types = {common::AnfAlgo::GetOutputInferDataType(mul0, 0),
                common::AnfAlgo::GetOutputInferDataType(reduce_sum, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(mul0, 0), AnfAlgo::GetOutputDetailShape(reduce_sum, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, fusion_node.get());
  return fusion_node;
}

const BaseRef ConfusionMulGradFusion::DefinePattern() const {
  VectorRef mul1({prim::kPrimMul, input3_, input2_});
  VectorRef reduce_sum({prim::kPrimReduceSumD, mul1});
  return reduce_sum;
}

const AnfNodePtr ConfusionMulGradFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto input2 = utils::cast<AnfNodePtr>((*equiv)[input2_]);
  auto input3 = utils::cast<AnfNodePtr>((*equiv)[input3_]);
  auto reduce_sum = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(reduce_sum);
  auto mul1 = reduce_sum->input(1);
  if (IsUsedByOthers(graph, mul1)) {
    MS_LOG(INFO) << "Mul1 is used by others, quit fusion!";
    return nullptr;
  }
  auto mul0 = GetMul0(graph, input2, mul1);
  if (mul0 == nullptr) {
    MS_LOG(INFO) << "Mul0 do not exist, quit fusion";
    return nullptr;
  }
  if (QuitFusion(graph, mul0, mul1, node, input2)) {
    return nullptr;
  }

  auto fusion_node = CreateFusionNode(graph, reduce_sum, mul0, input3);
  std::vector<AnfNodePtr> fusion_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, fusion_node, kConfusionMulGradOutputNum, &fusion_node_outputs);

  auto manage = graph->manager();
  MS_EXCEPTION_IF_NULL(manage);
  (void)manage->Replace(mul0, fusion_node_outputs[0]);
  return fusion_node_outputs[1];
}
}  // namespace opt
}  // namespace mindspore
