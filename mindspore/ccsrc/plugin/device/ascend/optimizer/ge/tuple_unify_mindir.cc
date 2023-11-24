/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ge/tuple_unify_mindir.h"
#include <memory>
#include "mindspore/core/ops/arithmetic_ops.h"
#include "include/common/utils/anfalgo.h"

/* This pass changes the following pattern.

  TupleToTensor and TensorToTuple ops are temporary placeholders. The last step is to remove them.
    ###############
    Pattern:
    TensorToTuple -> TupleToTensor

    Replace:
    remove both, the Tensor are connected.
    ###############
*/

namespace mindspore {
namespace opt {

const BaseRef TupleUnifyMindIR::DefinePattern() const {
  VarPtr x = std::make_shared<Var>();
  VectorRef tensor_to_tuple({std::make_shared<Primitive>(kTensorToTupleOpName), x});
  return VectorRef({std::make_shared<Primitive>(kTupleToTensorOpName), tensor_to_tuple});
}

const AnfNodePtr TupleUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto tuple_to_tensor_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_to_tensor_cnode);
  auto tensor_to_tuple_node = tuple_to_tensor_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(tensor_to_tuple_node);
  auto tensor_to_tuple_cnode = tensor_to_tuple_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tensor_to_tuple_cnode);
  auto x = tensor_to_tuple_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(x);
  auto &child_node_users = manager->node_users()[tuple_to_tensor_cnode];
  for (auto &child_node_user : child_node_users) {
    auto child_node = child_node_user.first->cast<CNodePtr>();
    manager->SetEdge(child_node, GetInputNodeIndex(tuple_to_tensor_cnode, child_node) + kSizeOne, x);
  }
  return x;
}
}  // namespace opt
}  // namespace mindspore
