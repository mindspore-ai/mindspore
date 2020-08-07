/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/common/converter_op_utils.h"
#include "utils/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
STATUS TopologicalSortPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  std::vector<std::unique_ptr<schema::CNodeT>> newNodes;
  std::vector<size_t> sinkedTensorIdxes;
  // put all const tensor index into sinkedTensorIdxes
  for (size_t i = 0; i < graph->allTensors.size(); i++) {
    if (graph->allTensors.at(i)->nodeType == schema::NodeType_ValueNode) {
      sinkedTensorIdxes.insert(sinkedTensorIdxes.end(), i);
    }
  }
  auto &oldNodes = graph->nodes;
  std::queue<std::unique_ptr<schema::CNodeT>> opQueue;
  // put all non depend node into queue
  for (auto &node : graph->nodes) {
    if (IsNodeNonDepend(node, sinkedTensorIdxes)) {
      sinkedTensorIdxes.insert(sinkedTensorIdxes.end(), node->outputIndex.begin(), node->outputIndex.end());
      opQueue.push(std::move(node));
    }
  }
  // bfs
  while (!opQueue.empty()) {
    auto &node = opQueue.front();
    auto postNodeIdxes = GetOutputNodeIdx(*graph, *(node.get()));
    for (auto postNodeIdx : postNodeIdxes) {
      auto &postNode = oldNodes.at(postNodeIdx);
      // check if postNode is non-depended
      if (IsNodeNonDepend(postNode, sinkedTensorIdxes)) {
        sinkedTensorIdxes.insert(sinkedTensorIdxes.end(), postNode->outputIndex.begin(), postNode->outputIndex.end());
        opQueue.push(std::move(postNode));
      }
    }
    newNodes.emplace_back(std::move(node));
    opQueue.pop();
  }
  if (newNodes.size() != oldNodes.size()) {
    MS_LOG(ERROR) << "Unknow error in TopologicalSort, oldNodesSize: " << oldNodes.size()
                  << ", newNodesSize: " << newNodes.size();
    return RET_ERROR;
  }
  graph->nodes.swap(newNodes);
  return RET_OK;
}

bool TopologicalSortPass::IsNodeNonDepend(const std::unique_ptr<schema::CNodeT> &node,
                                          const std::vector<size_t> &sinkedTensorIdxes) {
  for (auto inputIdx : node->inputIndex) {
    if (!IsContain(sinkedTensorIdxes, size_t(inputIdx))) {
      return false;
    }
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore

