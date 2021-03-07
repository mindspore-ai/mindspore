/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <set>
#include <algorithm>
#include <memory>
#include "tools/converter/legacy_optimizer/graph/nested_loop_expand_pass.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
bool NestedLoopExpandPass::IsNestedPartial(const std::unique_ptr<CNodeT> &node) {
  if (node->primitive->value.type != PrimitiveType_PartialFusion) {
    return false;
  }
  auto subgraph_idx = ((schema::PartialFusionT *)(node->primitive->value.value))->sub_graph_index;
  auto &this_subgraph = graph_->subGraph.at(subgraph_idx);

  for (auto &node_idx : this_subgraph->nodeIndices) {
    auto &cnode = graph_->nodes.at(node_idx);
    if (cnode->primitive->value.type == PrimitiveType_PartialFusion) {
      return true;
    }
  }
  return false;
}

void NestedLoopExpandPass::ReplacePartialNodeWithSubgraph(const std::unique_ptr<SubGraphT> &main_graph) {
  bool is_changed = false;
  for (auto iter = main_graph->nodeIndices.begin(); iter != main_graph->nodeIndices.end();) {
    auto &node = graph_->nodes.at(*iter);
    if (!IsNestedPartial(node)) {
      iter++;
      continue;
    }
    is_changed = true;
    auto subgraph_idx = ((schema::PartialFusionT *)(node->primitive->value.value))->sub_graph_index;
    auto &this_subgraph = graph_->subGraph.at(subgraph_idx);
    subgraph_to_drop_.push_back(subgraph_idx);
    iter = main_graph->nodeIndices.erase(iter);
    main_graph->nodeIndices.insert(iter, this_subgraph->nodeIndices.begin(), this_subgraph->nodeIndices.end());
    break;
  }

  if (is_changed) {
    ReplacePartialNodeWithSubgraph(main_graph);
  }
}

STATUS NestedLoopExpandPass::Run(schema::MetaGraphT *graph) {
  graph_ = graph;
  auto &main_graph = graph_->subGraph[0];

  ReplacePartialNodeWithSubgraph(main_graph);

  for (auto idx : subgraph_to_drop_) {
    graph_->subGraph.at(idx) = nullptr;
  }

  for (auto &node_idx : main_graph->nodeIndices) {
    auto &node = graph_->nodes.at(node_idx);
    if (node->primitive->value.type == PrimitiveType_PartialFusion) {
      auto &subgraph_idx = ((schema::PartialFusionT *)(node->primitive->value.value))->sub_graph_index;
      if (graph_->subGraph.at(subgraph_idx) == nullptr) {
        node = nullptr;
        continue;
      }
      auto tmp_subgraph_idx = subgraph_idx;
      for (auto i = 0; i < tmp_subgraph_idx; ++i) {
        if (graph_->subGraph.at(i) == nullptr) {
          subgraph_idx--;
        }
      }
    }
  }

  for (auto it = graph_->subGraph.begin(); it != graph_->subGraph.end();) {
    if ((*it) == nullptr) {
      it = graph_->subGraph.erase(it);
    } else {
      it++;
    }
  }

  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
