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

#include "plugin/device/ascend/optimizer/ge/convert_data_depend_to_control_depend.h"

#include <memory>
#include <vector>
#include "ops/framework_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef ConvertDataDependToControlDepend::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VectorRef send_node({prim::kPrimSend, x1});
  return VectorRef({prim::kPrimDepend, send_node, x2});
}

const AnfNodePtr ConvertDataDependToControlDepend::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Process node: " << node->fullname_with_scope()
               << ", input node: " << cnode->input(1)->fullname_with_scope();
  auto manager = func_graph->manager();
  if (func_graph->manager() == nullptr) {
    std::vector<FuncGraphPtr> graphs{func_graph};
    FuncGraphManagerPtr new_manager = std::make_shared<FuncGraphManager>(graphs);
    new_manager->AddFuncGraph(func_graph);
  }
  auto node_users = manager->node_users()[node];
  for (auto node_user : node_users) {
    if (AnfUtils::IsRealCNodeKernel(node_user.first)) {
      MS_LOG(DEBUG) << "Node: " << node->fullname_with_scope() << " is used by real kernel "
                    << node_user.first->fullname_with_scope();
      return nullptr;
    }
  }

  auto tensor = std::make_shared<tensor::Tensor>(0.0);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  ValueNodePtr value_node = kernel_graph->NewValueNode(tensor->ToAbstract(), tensor);
  kernel_graph->AddValueNodeToGraph(value_node);

  std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(kDependOpName)), value_node,
                                          cnode->input(1)};
  auto depend_node = NewCNode(depend_input, func_graph);
  MS_EXCEPTION_IF_NULL(depend_node);
  MS_LOG(INFO) << "Replace depend: " << node->fullname_with_scope()
               << " by new node: " << depend_node->fullname_with_scope();
  depend_node->set_abstract(value_node->abstract());
  return depend_node;
}
}  // namespace opt
}  // namespace mindspore
