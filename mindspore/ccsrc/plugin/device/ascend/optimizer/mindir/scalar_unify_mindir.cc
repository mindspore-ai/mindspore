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

#include "plugin/device/ascend/optimizer/mindir/scalar_unify_mindir.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "include/common/utils/anfalgo.h"

/* This pass changes the following pattern.

  ScalarToTensor and TensorToScalar ops are temporary placeholders. The last step is to remove them.
    ###############
    Pattern:
    TensorToScalar -> ScalarToTensor

    Replace:
    remove both, the Tensor are connected.
    ###############
*/

namespace mindspore {
namespace opt {

const BaseRef ScalarUnifyMindIR::DefinePattern() const {
  VarPtr x = std::make_shared<Var>();
  VectorRef tensor_to_scalar({std::make_shared<Primitive>(kTensorToScalarOpName), x});
  return VectorRef({std::make_shared<Primitive>(kScalarToTensorOpName), tensor_to_scalar});
}

const AnfNodePtr ScalarUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto scalar_to_tensor_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(scalar_to_tensor_cnode);
  auto tensor_to_scalar_node = scalar_to_tensor_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(tensor_to_scalar_node);
  auto tensor_to_scalar_cnode = tensor_to_scalar_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tensor_to_scalar_cnode);
  auto x = tensor_to_scalar_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(x);
  auto child_node_users = manager->node_users()[scalar_to_tensor_cnode];
  for (auto &child_node_user : child_node_users) {
    auto child_node = child_node_user.first->cast<CNodePtr>();
    manager->SetEdge(child_node, GetInputNodeIndex(scalar_to_tensor_cnode, child_node) + kSizeOne, x);
  }
  return x;
}
}  // namespace opt
}  // namespace mindspore
