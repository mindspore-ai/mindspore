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

#include "frontend/optimizer/irpass/pack_expand.h"

#include "ir/func_graph_cloner.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/expander/pack/packfunc.h"

namespace mindspore::opt::irpass {
bool PackExpand::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const {
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &all_nodes = manager->all_nodes();
  AnfNodePtrList pack_nodes;
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimPackFunc)) {
      (void)pack_nodes.emplace_back(node);
    }
  }
  for (auto &node : pack_nodes) {
    auto prim = GetCNodePrimitive(node);
    abstract::AbstractBasePtrList args_abs;
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->size(); i++) {
      (void)args_abs.emplace_back(cnode->input(i)->abstract());
    }
    auto fg = expander::ExpandPackFuncGraph(prim, args_abs);
    auto node_input = cnode->inputs();
    node_input[0] = NewValueNode(fg);
    (void)manager->Replace(node, node->func_graph()->NewCNodeInOrder(node_input));
  }
  manager->KeepRoots({func_graph});
  return !pack_nodes.empty();
}
}  // namespace mindspore::opt::irpass
