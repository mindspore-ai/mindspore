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
#include "pre_activate/ascend/ir_fusion/confusion_softmax_grad_rule.h"

#include <memory>
#include <vector>

#include "session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
void SetAttrsForFusionNode(const AnfNodePtr &sub_anf, const AnfNodePtr &fusion_node) {
  MS_EXCEPTION_IF_NULL(sub_anf);
  MS_EXCEPTION_IF_NULL(fusion_node);
  auto sub = sub_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sub);
  if (sub->size() != kSubInputNum) {
    MS_LOG(EXCEPTION) << "Sub's size is not equal with 3";
  }
  auto reduce_sum_anf = sub->input(2);
  MS_EXCEPTION_IF_NULL(reduce_sum_anf);
  auto reduce_sum = reduce_sum_anf->cast<CNodePtr>();
  if (reduce_sum == nullptr) {
    MS_LOG(EXCEPTION) << "Sub's second input is not a cnode";
  }
  AnfAlgo::CopyNodeAttr(kAttrAxis, reduce_sum, fusion_node);
  AnfAlgo::CopyNodeAttr(kAttrKeepDims, reduce_sum, fusion_node);
}
}  // namespace

const BaseRef ConfusionSoftmaxGradRule::DefinePattern() const {
  return VectorRef(
    {prim::kPrimSub, input0_, VectorRef({prim::kPrimReduceSum, VectorRef({prim::kPrimMul, input1_, input0_})})});
}

const AnfNodePtr ConfusionSoftmaxGradRule::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto input0 = utils::cast<AnfNodePtr>((*equiv)[input0_]);
  auto input1 = utils::cast<AnfNodePtr>((*equiv)[input1_]);
  MS_EXCEPTION_IF_NULL(input0);
  MS_EXCEPTION_IF_NULL(input1);

  auto prim = std::make_shared<Primitive>(kConfusionSoftmaxGradOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), input0, input1};
  auto confusion_softmax_grad = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(confusion_softmax_grad);
  auto types = {AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(node, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, confusion_softmax_grad.get());
  confusion_softmax_grad->set_scope(node->scope());
  SetAttrsForFusionNode(node, confusion_softmax_grad);
  return confusion_softmax_grad;
}
}  // namespace opt
}  // namespace mindspore
