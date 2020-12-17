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

#include <vector>
#include <algorithm>
#include <memory>
#include "tools/converter/legacy_optimizer/graph/subgraph_node_pass.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {

void SubgraphNodePass::UpdateSubgraphNodeIndices(const size_t &node_idx, schema::MetaGraphT *graph) {
  for (auto &subgraph : graph->subGraph) {
    for (auto &idx : subgraph->nodeIndices) {
      if (idx > node_idx) {
        idx--;
      }
    }
  }
}

STATUS SubgraphNodePass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  std::vector<schema::CNodeT *> new_nodes{};
  std::transform(graph->nodes.begin(), graph->nodes.end(), std::back_inserter(new_nodes),
                 [](std::unique_ptr<CNodeT> &node) { return node.get(); });

  for (auto it = old_nodes_.begin(); it != old_nodes_.end();) {
    if (!IsContain(new_nodes, *it)) {
      size_t node_idx = it - old_nodes_.begin();
      for (auto &subgraph : graph->subGraph) {
        auto node_idx_pos = std::find(subgraph->nodeIndices.begin(), subgraph->nodeIndices.end(), node_idx);
        if (node_idx_pos != subgraph->nodeIndices.end()) {
          subgraph->nodeIndices.erase(node_idx_pos);
          UpdateSubgraphNodeIndices(node_idx, graph);
          break;
        }
      }
      it = old_nodes_.erase(it);
    } else {
      it++;
    }
  }

  for (uint32_t i = 0; i < new_nodes.size(); i++) {
    if (!IsContain(old_nodes_, new_nodes[i])) {
      for (auto &subgraph : graph->subGraph) {
        if (IsContain(subgraph->nodeIndices, i - 1) || IsContain(subgraph->nodeIndices, i + 1)) {
          subgraph->nodeIndices.push_back(old_nodes_.size());
          old_nodes_.push_back(new_nodes[i]);
        }
      }
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
