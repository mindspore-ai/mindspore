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

#define USE_DEPRECATED_API
#include "tools/converter/parser/unused_node_remove_pass.h"
#include <deque>
#include <unordered_set>
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {

STATUS UnusedNodeRemovePass::ProcessGraph(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *has_visited) {
  MS_ASSERT(func_graph != nullptr && has_visited != nullptr);
  if (has_visited->find(func_graph) != has_visited->end()) {
    return lite::RET_OK;
  }
  has_visited->insert(func_graph);
  auto return_node = func_graph->get_return();
  if (return_node == nullptr) {
    return RET_OK;
  }
  std::unordered_set<AnfNodePtr> vis;
  std::deque<AnfNodePtr> q;
  q.push_back(return_node);
  while (!q.empty()) {
    auto node = q.front();
    vis.insert(node);
    q.pop_front();
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = utils::cast<CNodePtr>(node);
      MS_ASSERT(cnode != nullptr);
      for (auto &input : cnode->inputs()) {
        if (vis.find(input) == vis.end() && std::find(q.begin(), q.end(), input) == q.end()) {
          q.push_back(input);
        }
      }
    }
    if (utils::isa<ValueNode>(node) && GetValueNode<FuncGraphPtr>(node) != nullptr) {
      auto sub_graph = GetValueNode<FuncGraphPtr>(node);
      MS_ASSERT(sub_graph != nullptr);
      auto status = ProcessGraph(sub_graph, has_visited);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "process sub graph failed";
        return RET_ERROR;
      }
    }
  }
  auto nodes = func_graph->nodes();
  auto graph_inputs = func_graph->get_inputs();
  for (auto &node : nodes) {
    if (vis.find(node) == vis.end() &&
        std::find(graph_inputs.begin(), graph_inputs.end(), node) == graph_inputs.end()) {
      func_graph->DropNode(node);
    }
  }
  return RET_OK;
}

bool UnusedNodeRemovePass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  std::set<FuncGraphPtr> has_visited;
  auto status = ProcessGraph(func_graph, &has_visited);
  return status == RET_OK;
}
}  // namespace mindspore::opt
