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

#include <memory>
#include <set>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "tools/converter/legacy_optimizer/graph/const_node_reorder_pass.h"
#include "tools/common/node_util.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "tools/common/meta_graph_utils.h"

namespace mindspore {
namespace lite {
STATUS ConstNodeReorderPass::Run(schema::MetaGraphT *graph) {
  for (size_t i = 0; i < graph->subGraph.size(); i++) {
    std::set<unsigned int> const_node_indices;
    std::vector<unsigned int> final_order;
    std::vector<unsigned int> non_const_node_order;
    auto subgraph_node_indices = graph->subGraph[i]->nodeIndices;
    std::unordered_map<unsigned int, std::vector<unsigned int>> dependent_nodes_map;
    for (size_t j = 0; j < subgraph_node_indices.size(); j++) {
      auto node_idx = subgraph_node_indices[j];
      auto &node = graph->nodes[node_idx];
      auto post_node_idxes = GetOutputNodeIdx(*graph, *(node.get()));
      if (IsConstNode(node, *graph) && !post_node_idxes.empty()) {
        const_node_indices.insert(node_idx);
        for (auto idx : post_node_idxes) {
          dependent_nodes_map[idx].emplace_back(node_idx);
        }
      } else {
        non_const_node_order.emplace_back(node_idx);
      }
    }

    for (auto idx : non_const_node_order) {
      for (auto depend_node : dependent_nodes_map[idx]) {
        if (const_node_indices.find(depend_node) != const_node_indices.end()) {
          final_order.emplace_back(depend_node);
          const_node_indices.erase(depend_node);
        }
      }
      final_order.emplace_back(idx);
    }
    graph->subGraph[i]->nodeIndices = final_order;
    if (subgraph_node_indices.size() != final_order.size()) {
      MS_LOG(ERROR) << "Unknown error in ConstNodeReorderSort, old nodes size: " << subgraph_node_indices.size()
                    << ", current nodes size: " << final_order.size();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool ConstNodeReorderPass::IsConstNode(const std::unique_ptr<schema::CNodeT> &node, const schema::MetaGraphT &graph) {
  // Const node is node whose inputs are all const value.
  return std::all_of(node->inputIndex.begin(), node->inputIndex.end(),
                     [&](size_t idx) { return graph.allTensors[idx]->nodeType == NodeType_ValueNode; });
}
}  // namespace lite
}  // namespace mindspore
