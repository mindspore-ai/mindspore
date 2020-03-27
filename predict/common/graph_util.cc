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

#include "common/graph_util.h"
#include <fstream>
#include <sstream>
#include "common/mslog.h"
#include "include/errorcode.h"

namespace mindspore {
namespace predict {
OpGraph *OpGraph::Build(const SubGraphDef &subGraphDef) {
  auto graph = std::unique_ptr<OpGraph>(new OpGraph());
  if (graph == nullptr) {
    MS_LOGE("malloc opgraph failed");
    return nullptr;
  }

  auto nodeDefs = subGraphDef.nodes();
  if (nodeDefs == nullptr) {
    MS_LOGE("nodeDefs from subGraphDef is nullptr");
    return nullptr;
  }

  uint32_t opCount = nodeDefs->size();
  for (uint32_t i = 0; i < opCount; i++) {
    auto nodeDef = nodeDefs->GetAs<NodeDef>(i);
    MS_ASSERT(nodeDef != nullptr);
    auto ret = graph->AddEdge(*nodeDef, *nodeDefs);
    if (ret != RET_OK) {
      MS_LOGE("%s add edge failed. ret:%d", nodeDef->opDef()->name()->c_str(), ret);
      return nullptr;
    }
  }

  return graph.release();
}

int OpGraph::AddEdge(const NodeDef &srcNodeDef, const flatbuffers::Vector<flatbuffers::Offset<NodeDef>> &nodeDefs) {
  MS_ASSERT(srcNodeDef.opDef() != nullptr);
  MS_ASSERT(srcNodeDef.opDef()->name() != nullptr);
  NODE_ID srcId = std::string(srcNodeDef.opDef()->name()->c_str());
  uint32_t opCount = nodeDefs.size();

  MS_ASSERT(srcNodeDef.opDef()->outputIndex() != nullptr);
  for (auto index : *(srcNodeDef.opDef()->outputIndex())) {
    for (uint32_t i = 0; i < opCount; i++) {
      auto dstNodeDef = nodeDefs.GetAs<NodeDef>(i);
      bool find = false;
      MS_ASSERT(dstNodeDef != nullptr);
      MS_ASSERT(dstNodeDef->opDef() != nullptr);
      auto inputIndex = dstNodeDef->opDef()->inputIndex();
      MS_ASSERT(inputIndex != nullptr);
      if (std::any_of(inputIndex->begin(), inputIndex->end(), [&index](int i) { return i == index; })) {
        find = true;
      }

      if (!find) {
        continue;
      }
      MS_ASSERT(dstNodeDef->opDef()->name() != nullptr);
      NODE_ID dstId = std::string(dstNodeDef->opDef()->name()->c_str());
      auto ret = AddEdge(srcId, dstId);
      if (ret != RET_OK) {
        return ret;
      }
    }
  }

  return RET_OK;
}

int OpGraph::AddEdge(const NODE_ID &srcId, const NODE_ID &dstId) {
  auto srcNode = AddNode(srcId);
  if (srcNode == nullptr) {
    MS_LOGE("add srcNode failed");
    return RET_ERROR;
  }
  srcNode->AddOutEdge(dstId);
  auto dstNode = AddNode(dstId);
  if (dstNode == nullptr) {
    MS_LOGE("add dstNode failed");
    return RET_ERROR;
  }
  dstNode->AddInEdge(srcId);
  return RET_OK;
}

OpNode *OpGraph::GetNode(const NODE_ID &nodeId) {
  auto node = nodes.find(nodeId);
  if (node == nodes.end()) {
    return nullptr;
  }
  return node->second;
}

OpNode *OpGraph::AddNode(const NODE_ID &nodeId) {
  auto node = GetNode(nodeId);
  if (node != nullptr) {
    return node;
  }
  node = new (std::nothrow) OpNode(nodeId);
  if (node == nullptr) {
    MS_LOGE("new node failed");
    return nullptr;
  }
  nodes[nodeId] = node;
  return node;
}

std::unordered_set<NODE_ID> OpGraph::GetInputNode() {
  std::unordered_set<NODE_ID> inputNodes;
  for (const auto &iter : nodes) {
    auto node = iter.second;
    MS_ASSERT(node != nullptr);
    if (node->GetAllInEdge().empty()) {
      inputNodes.insert(node->ID());
    }
  }
  return inputNodes;
}

std::unordered_set<NODE_ID> OpGraph::GetOutputNode() {
  std::unordered_set<NODE_ID> outputNodes;
  for (const auto &iter : nodes) {
    auto node = iter.second;
    MS_ASSERT(node != nullptr);
    if (node->GetAllOutEdge().empty()) {
      outputNodes.insert(node->ID());
    }
  }
  return outputNodes;
}

OpGraph::~OpGraph() {
  for (auto iter : nodes) {
    if (iter.second != nullptr) {
      delete iter.second;
    }
  }
  nodes.clear();
}

NODE_ID OpNode::ID() { return id; }

void OpNode::AddInEdge(const NODE_ID &nodeId) { inEdges.insert(nodeId); }

void OpNode::AddOutEdge(const NODE_ID &nodeId) { outEdges.insert(nodeId); }

std::unordered_set<NODE_ID> OpNode::GetAllInEdge() { return inEdges; }

std::unordered_set<NODE_ID> OpNode::GetAllOutEdge() { return outEdges; }
}  // namespace predict
}  // namespace mindspore
