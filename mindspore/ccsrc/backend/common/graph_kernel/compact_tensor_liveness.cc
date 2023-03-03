/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/compact_tensor_liveness.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::graphkernel {
bool CompactTensorLiveness::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  mindspore::HashSet<AnfNodePtr> target_nodes;
  for (auto &node : todos) {
    if (auto cnode = node->cast<CNodePtr>(); cnode != nullptr) {
      bool any_cnode_input = std::any_of(cnode->inputs().cbegin(), cnode->inputs().cend(),
                                         [](const AnfNodePtr &n) { return n->isa<CNode>() || n->isa<Parameter>(); });
      if (any_cnode_input || common::AnfAlgo::IsGraphKernel(cnode) || mng->node_users()[cnode].size() != 1) {
        continue;
      }
      (void)target_nodes.insert(node);
    }
  }
  for (auto &node : target_nodes) {
    auto cnode = node->cast<CNodePtr>();
    auto user = mng->node_users()[cnode].front().first;
    if (auto user_cnode = user->cast<CNodePtr>(); user_cnode != nullptr) {
      const auto iter = std::find_if(
        user_cnode->inputs().cbegin() + 1, user_cnode->inputs().cend(),
        [&node, &target_nodes](const AnfNodePtr &n) { return n->isa<CNode>() && target_nodes.count(n) == 0; });
      if (iter != user_cnode->inputs().end()) {
        // Insert update_state_node, need mount a monad node.
        auto u = NewValueNode(kUMonad);
        u->set_abstract(kUMonad->ToAbstract());
        AnfNodePtrList update_state_inputs = {NewValueNode(prim::kPrimUpdateState), u};
        update_state_inputs.push_back(*iter);
        auto update_state_cnode = func_graph->NewCNode(update_state_inputs);
        update_state_cnode->set_abstract(u->abstract());
        func_graph->AddNode(update_state_cnode);

        // reset inputs
        auto origin_inputs = cnode->inputs();
        origin_inputs.push_back(update_state_cnode);
        cnode->set_inputs(origin_inputs);
        changed = true;
      }
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace mindspore::graphkernel
