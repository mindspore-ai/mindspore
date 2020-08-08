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

#include <fstream>
#include <sstream>
#include <utility>
#include "src/common/graph_util.h"
#include "src/common/utils.h"
#include "utils/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
std::vector<size_t> GetGraphInputNodes(const schema::MetaGraph *meta_graph) {
  MS_ASSERT(nullptr != meta_graph);
  std::vector<size_t> ret;
  for (size_t i = 0; i < meta_graph->inputIndex()->size(); i++) {
    auto input_index = meta_graph->inputIndex()->GetAs<uint32_t>(i);
    for (size_t j = 0; j < meta_graph->nodes()->size(); j++) {
      auto *cNode = meta_graph->nodes()->GetAs<schema::CNode>(j);
      MS_ASSERT(nullptr != cNode);
      for (size_t k = 0; k < cNode->inputIndex()->size(); k++) {
        if (cNode->inputIndex()->GetAs<uint32_t>(k) == input_index) {
          if (!IsContain<size_t>(ret, j)) {
            ret.emplace_back(j);
          }
          break;
        }
      }
    }
  }
  return std::move(ret);
}

std::vector<size_t> GetGraphOutputNodes(const schema::MetaGraph *meta_graph) {
  MS_ASSERT(nullptr != meta_graph);
  std::vector<size_t> ret;
  for (size_t i = 0; i < meta_graph->outputIndex()->size(); i++) {
    auto output_index = meta_graph->outputIndex()->GetAs<uint32_t>(i);
    for (size_t j = 0; j < meta_graph->nodes()->size(); j++) {
      auto *cNode = meta_graph->nodes()->GetAs<schema::CNode>(j);
      MS_ASSERT(nullptr != cNode);
      for (size_t k = 0; k < cNode->outputIndex()->size(); k++) {
        if (cNode->outputIndex()->GetAs<uint32_t>(k) == output_index) {
          if (!IsContain<size_t>(ret, j)) {
            ret.emplace_back(j);
          }
          break;
        }
      }
    }
  }
  return std::move(ret);
}

// NODE_ID OpNode::ID() { return id; }
//
// void OpNode::AddInEdge(NODE_ID nodeId) { inEdges.insert(nodeId); }
//
// void OpNode::AddOutEdge(NODE_ID nodeId) { outEdges.insert(nodeId); }
//
// std::unordered_set<NODE_ID> OpNode::GetAllInEdges() { return inEdges; }
//
// std::unordered_set<NODE_ID> OpNode::GetAllOutEdges() { return outEdges; }

}  // namespace lite
}  // namespace mindspore

