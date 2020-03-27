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

#ifndef PREDICT_COMMON_GRAPH_UTIL_H_
#define PREDICT_COMMON_GRAPH_UTIL_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <memory>
#include "common/utils.h"
#include "schema/inner/ms_generated.h"

namespace mindspore {
namespace predict {
using NODE_ID = std::string;

class OpNode {
 public:
  explicit OpNode(NODE_ID nodeId) : id(std::move(nodeId)) {}
  NODE_ID ID();
  void AddInEdge(const NODE_ID &nodeId);
  void AddOutEdge(const NODE_ID &nodeId);
  std::unordered_set<NODE_ID> GetAllInEdge();
  std::unordered_set<NODE_ID> GetAllOutEdge();

 protected:
  NODE_ID id;
  std::unordered_set<NODE_ID> inEdges;
  std::unordered_set<NODE_ID> outEdges;
};

class OpGraph {
 public:
  OpGraph() = default;

  ~OpGraph();

  static OpGraph *Build(const SubGraphDef &subGraphDef);

  OpNode *GetNode(const NODE_ID &nodeId);
  OpNode *AddNode(const NODE_ID &nodeId);
  std::unordered_set<NODE_ID> GetInputNode();
  std::unordered_set<NODE_ID> GetOutputNode();

 private:
  int AddEdge(const NODE_ID &srcId, const NODE_ID &dstId);
  int AddEdge(const NodeDef &srcNodeDef, const flatbuffers::Vector<flatbuffers::Offset<NodeDef>> &nodeDefs);

 protected:
  std::unordered_map<NODE_ID, OpNode *> nodes;
};
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_COMMON_GRAPH_UTIL_H_
