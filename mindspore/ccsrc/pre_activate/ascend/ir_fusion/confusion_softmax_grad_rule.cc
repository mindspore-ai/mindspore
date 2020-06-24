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
const BaseRef ConfusionSoftmaxGradRule::DefinePattern() const {
  return VectorRef({prim::kPrimSub, input0_, VectorRef({reduce_sum_, VectorRef({prim::kPrimMul, input1_, input0_})})});
}

const AnfNodePtr ConfusionSoftmaxGradRule::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  AnfNodePtr input0 = GetAnfNodeByVar(equiv, input0_);
  AnfNodePtr input1 = GetAnfNodeByVar(equiv, input1_);
  AnfNodePtr sum_anf = GetAnfNodeByVar(equiv, reduce_sum_);
  if (sum_anf == nullptr || !sum_anf->isa<CNode>()) {
    MS_LOG(WARNING) << "Matched ReduceSum is not a CNode!";
    return nullptr;
  }
  if (!GetBoolAttr(sum_anf, kAttrKeepDims)) {
    MS_LOG(INFO) << "ReduceSum's attr keep_dims should be true if do fusion. Otherwise the calculation will be wrong";
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kConfusionSoftmaxGradOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), input0, input1};
  auto fusion_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_abstract(node->abstract());
  fusion_node->set_scope(node->scope());
  AnfAlgo::CopyNodeAttr(kAttrAxis, sum_anf, fusion_node);
  AnfAlgo::CopyNodeAttr(kAttrKeepDims, sum_anf, fusion_node);
  return fusion_node;
}
}  // namespace opt
}  // namespace mindspore
