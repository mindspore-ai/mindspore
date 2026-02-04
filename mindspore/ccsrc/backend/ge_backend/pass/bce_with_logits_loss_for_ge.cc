/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/ge_backend/pass/bce_with_logits_loss_for_ge.h"
#include <vector>
#include <memory>
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "ops_utils/op_utils.h"
#include "primitive/nn_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace opt {

const BaseRef BCEWithLogitsLossForGe::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  return VectorRef({prim::kPrimBCEWithLogitsLoss, Xs});
}

const AnfNodePtr BCEWithLogitsLossForGe::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto input_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kIndex0);
  auto target_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kIndex1);
  if (input_type_id == target_type_id) {
    return node;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  AnfNodePtrList sigmoid_node_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimSigmoidCrossEntropyWithLogitsV2->name()))};
  auto cnode_inputs = cnode->inputs();
  (void)sigmoid_node_inputs.insert(sigmoid_node_inputs.cend(), cnode_inputs.cbegin() + 1, cnode_inputs.cend());
  CNodePtr sigmoid_node = NewCNode(sigmoid_node_inputs, graph);
  MS_EXCEPTION_IF_NULL(sigmoid_node);
  auto new_node_shape = {common::AnfAlgo::GetOutputInferShape(node, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape({input_type_id}, new_node_shape, sigmoid_node.get());

  auto cast_node = AddCastNode(graph, target_type_id, sigmoid_node, false);
  return cast_node;
}
}  // namespace opt
}  // namespace mindspore
