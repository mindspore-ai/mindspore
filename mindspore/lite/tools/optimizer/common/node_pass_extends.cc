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
#include "backend/optimizer/common/node_pass.h"
#include <unordered_set>
#include <deque>
#include <algorithm>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
bool NodePass::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  manager->AddFuncGraph(func_graph);

  std::unordered_set<AnfNodePtr> seen_node;
  std::deque<AnfNodePtr> to_process{func_graph->output()};
  bool changes = false;
  while (!to_process.empty()) {
    AnfNodePtr node = to_process.front();
    to_process.pop_front();
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
    if (IsValueNode<FuncGraph>(new_node)) {
      auto const_func_graph = GetValueNode<FuncGraphPtr>(new_node);
      if (const_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      to_process.push_back(const_func_graph->output());
    } else if (new_node->isa<CNode>()) {
      if (IsGraphKernel(new_node)) {
        to_process.push_back(new_node);
      }
      auto cnode = new_node->cast<CNodePtr>();
      if (cnode == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      auto inputs = cnode->inputs();
      (void)to_process.insert(to_process.end(), inputs.begin(), inputs.end());
    }
    changes = changes || change;
    if (changes) {
      MS_LOG(DEBUG) << "pass " << this->name() << "changed node:" << new_node->fullname_with_scope();
    }
  }
  return changes;
}
}  // namespace opt
}  // namespace mindspore
