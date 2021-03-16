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

#include <queue>
#include <utility>
#include <memory>
#include <vector>
#include "tools/converter/legacy_optimizer/graph/topological_sort_pass.h"
#include "tools/common/node_util.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
STATUS TopologicalSortPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  std::vector<std::unique_ptr<schema::CNodeT>> new_nodes;
  std::vector<size_t> sinked_tensor_idxes;
  // put all const tensor index into sinked_tensor_idxes
  for (size_t i = 0; i < graph->allTensors.size(); i++) {
    if (graph->allTensors.at(i)->nodeType == NodeType_ValueNode) {
      sinked_tensor_idxes.insert(sinked_tensor_idxes.end(), i);
    }
  }
  auto &old_nodes = graph->nodes;
  std::queue<std::unique_ptr<schema::CNodeT>> op_queue;
  // put all none depend node into queue
  for (size_t i = 0; i < graph->subGraph.size(); i++) {
    std::vector<unsigned int> new_subgraph_node_indices = {};
    auto subgraph_node_indices = graph->subGraph[i]->nodeIndices;

    for (size_t j = 0; j < subgraph_node_indices.size(); j++) {
      auto &node = old_nodes[subgraph_node_indices[j]];
      if (IsNodeNonDepend(node, sinked_tensor_idxes)) {
        sinked_tensor_idxes.insert(sinked_tensor_idxes.end(), node->outputIndex.begin(), node->outputIndex.end());
        op_queue.push(std::move(node));
      }
    }
    while (!op_queue.empty()) {
      auto &node = op_queue.front();
      auto post_node_idxes = GetOutputNodeIdx(*graph, *(node.get()));
      sinked_tensor_idxes.insert(sinked_tensor_idxes.end(), node->outputIndex.begin(), node->outputIndex.end());
      for (auto post_node_idx : post_node_idxes) {
        if (IsContain(subgraph_node_indices, (unsigned int)(post_node_idx))) {
          auto &post_node = old_nodes.at(post_node_idx);
          // check if post_node is non-depended
          if (IsNodeNonDepend(post_node, sinked_tensor_idxes)) {
            op_queue.push(std::move(post_node));
          }
        }
      }
      new_nodes.emplace_back(std::move(node));
      new_subgraph_node_indices.push_back(new_nodes.size() - 1);
      op_queue.pop();
    }
    graph->subGraph[i]->nodeIndices.swap(new_subgraph_node_indices);
  }
  if (new_nodes.size() != old_nodes.size()) {
    MS_LOG(ERROR) << "Unknown error in TopologicalSort, old_nodes size: " << old_nodes.size()
                  << ", new_nodes size: " << new_nodes.size();
    return RET_ERROR;
  }
  graph->nodes.swap(new_nodes);
  return RET_OK;
}

bool TopologicalSortPass::IsNodeNonDepend(const std::unique_ptr<schema::CNodeT> &node,
                                          const std::vector<size_t> &sinked_tensor_idxes) {
  MS_ASSERT(node != nullptr);
  if (node->primitive->value.type == schema::PrimitiveType_Merge) {
    auto node_input_index = node->inputIndex;
    MS_ASSERT(node_input_index.size() % 2 == 0);
    return std::all_of(node_input_index.begin(), node_input_index.begin() + node_input_index.size() / 2,
                       [&](size_t input_idx) { return IsContain(sinked_tensor_idxes, input_idx); }) ||
           std::all_of(node_input_index.begin() + node_input_index.size() / 2, node_input_index.end(),
                       [&](size_t input_idx) { return IsContain(sinked_tensor_idxes, input_idx); });
  } else {
    return std::all_of(node->inputIndex.begin(), node->inputIndex.end(),
                       [&](size_t input_idx) { return IsContain(sinked_tensor_idxes, size_t(input_idx)); });
  }
}
}  // namespace lite
}  // namespace mindspore
