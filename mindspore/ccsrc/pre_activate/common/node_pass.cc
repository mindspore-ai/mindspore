/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "pre_activate/common/node_pass.h"

#include <unordered_set>
#include <deque>
#include <algorithm>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
bool NodePass::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);

  std::unordered_set<AnfNodePtr> seen_node;
  std::deque<AnfNodePtr> todo{func_graph->output()};
  bool changes = false;
  while (!todo.empty()) {
    AnfNodePtr node = todo.front();
    todo.pop_front();
    if (seen_node.count(node) > 0 || !manager->all_nodes().contains(node)) {
      continue;
    }
    (void)seen_node.insert(node);
    AnfNodePtr new_node = Run(func_graph, node);
    bool change = (new_node != nullptr);
    if (new_node != nullptr && new_node != node) {
      (void)manager->Replace(node, new_node);
      (void)seen_node.erase(node);
    } else if (new_node == nullptr) {
      new_node = node;
    }
    if (new_node && IsValueNode<FuncGraph>(new_node)) {
      auto const_func_graph = GetValueNode<FuncGraphPtr>(new_node);
      MS_EXCEPTION_IF_NULL(const_func_graph);
      if (!const_func_graph->has_attr(FUNC_GRAPH_FLAG_COMPOSITE)) {
        todo.push_back(const_func_graph->output());
      }
    } else if (new_node && new_node->isa<CNode>()) {
      if (AnfAlgo::IsCompositeKernel(new_node)) {
        todo.push_back(new_node);
      }
      auto cnode = new_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto inputs = cnode->inputs();
      (void)todo.insert(todo.end(), inputs.begin(), inputs.end());
    }
    changes = changes || change;
  }
  return changes;
}
}  // namespace opt
}  // namespace mindspore
