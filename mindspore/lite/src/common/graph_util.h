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

#ifndef MINDSPORE_LITE_COMMON_GRAPH_UTIL_H_
#define MINDSPORE_LITE_COMMON_GRAPH_UTIL_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include "schema/model_generated.h"
#include "utils//log_adapter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
using NODE_ID = std::string;

std::vector<size_t> GetGraphInputNodes(const schema::MetaGraph *meta_graph);

std::vector<size_t> GetGraphOutputNodes(const schema::MetaGraph *meta_graph);

class OpNode {
 public:
    explicit OpNode(const NODE_ID &nodeId) : id(nodeId) {}
    NODE_ID ID() { return id; };
    void AddInEdge(NODE_ID nodeId) { inEdges.insert(nodeId); }
    void AddOutEdge(NODE_ID nodeId) { outEdges.insert(nodeId); }
    std::unordered_set<NODE_ID> GetAllInEdges() { return inEdges; }
    std::unordered_set<NODE_ID> GetAllOutEdges() { return outEdges; }

 protected:
    NODE_ID id;
    std::unordered_set<NODE_ID> inEdges;
    std::unordered_set<NODE_ID> outEdges;
};


template <typename NODE_T>
class OpGraph {
 public:
  OpGraph() {}

  ~OpGraph();

  int Build(const schema::MetaGraph *subGraphDef);
  NODE_T *GetNode(NODE_ID nodeId);
  NODE_T *AddNode(NODE_ID nodeId);
  std::unordered_set<NODE_T *> GetInputNode();
  std::unordered_set<NODE_T *> GetOutputNode();

  void AddNodes(std::vector<NODE_T *> addNodes);
  void DeleteNodes(std::vector<NODE_T *> deleteNodes);

  void AddEdge(NODE_ID nodeId);
  int AddEdge(NODE_ID srcId, NODE_ID dstId);
  int AddEdge(const schema::CNode *srcNodeDef, const flatbuffers::Vector<flatbuffers::Offset<schema::CNode>> *opDefs);
  std::unordered_map<NODE_T *, std::unordered_set<NODE_T *>> GetDepends();

 protected:
  std::unordered_map<NODE_ID, NODE_T *> nodes;
};

template <typename NODE_T>
int OpGraph<NODE_T>::Build(const schema::MetaGraph *subGraphDef) {
  if (subGraphDef == nullptr) {
    // MS_LOGE("subGraphDef is nullptr");
    return RET_ERROR;
  }


  auto opDefs = subGraphDef->nodes();

  uint32_t opCount = opDefs->size();
  for (uint32_t i = 0; i < opCount; i++) {
    auto opDef = opDefs->GetAs<schema::CNode>(i);
    auto node = AddNode(std::string(opDef->name()->c_str()));
    if (node == nullptr) {
      // MS_LOGE("add srcNode failed,name %s", opDef->name()->c_str());
      return RET_ERROR;
    }
    auto ret = AddEdge(opDef, opDefs);
    if (ret != RET_OK) {
      // MS_LOGE("%s add edge failed. ret:%d", opDef->name()->c_str(), ret);
      return RET_ERROR;
    }
  }

  return RET_OK;
}
template <typename NODE_T>
int OpGraph<NODE_T>::AddEdge(const schema::CNode *srcNodeDef,
        const flatbuffers::Vector<flatbuffers::Offset<schema::CNode>> *nodeDefs) {
  MS_ASSERT(srcNodeDef != nullptr);
  MS_ASSERT(nodeDefs != nullptr);
  NODE_ID srcId = std::string(srcNodeDef->name()->c_str());
  uint32_t opCount = nodeDefs->size();
  // for single op condition
  AddNode(srcId);
  for (auto index : *(srcNodeDef->outputIndex())) {
    for (uint32_t i = 0; i < opCount; i++) {
      auto dstNodeDef = nodeDefs->GetAs<schema::CNode>(i);
      bool find = false;
      auto inputIndex = dstNodeDef->inputIndex();
      if (std::any_of(inputIndex->begin(), inputIndex->end(), [&index](int i) { return i == index; })) {
        find = true;
      }

      if (!find) {
        continue;
      }
      NODE_ID dstId = std::string(dstNodeDef->name()->c_str());
      auto ret = AddEdge(srcId, dstId);
      if (ret != RET_OK) {
        return ret;
      }
    }
  }

  return RET_OK;
}

template <typename NODE_T>
int OpGraph<NODE_T>::AddEdge(NODE_ID srcId, NODE_ID dstId) {
  auto srcNode = AddNode(srcId);
  if (srcNode == nullptr) {
    // MS_LOGE("add srcNode failed");
    return RET_ERROR;
  }
  auto dstNode = AddNode(dstId);
  if (dstNode == nullptr) {
    // MS_LOGE("add dstNode failed");
    return RET_ERROR;
  }

  srcNode->AddOutEdge(dstNode);

  dstNode->AddInEdge(srcNode);
  return RET_OK;
}

template <typename NODE_T>
NODE_T *OpGraph<NODE_T>::GetNode(NODE_ID nodeId) {
  auto node = nodes.find(nodeId);
  if (node == nodes.end()) {
    return nullptr;
  }
  return node->second;
}

template <typename NODE_T>
NODE_T *OpGraph<NODE_T>::AddNode(NODE_ID nodeId) {
  auto node = GetNode(nodeId);
  if (node != nullptr) {
    return node;
  }
  node = new (std::nothrow) NODE_T(nodeId);
  if (node == nullptr) {
    // MS_LOGE("new node failed");
    return nullptr;
  }
  nodes[nodeId] = node;
  return node;
}

template <typename NODE_T>
void OpGraph<NODE_T>::AddNodes(std::vector<NODE_T *> addNodes) {
  for (auto node : addNodes) {
    if (node == nullptr) {
      return;
    }

    nodes[node->ID()] = node;
  }
}

template <typename NODE_T>
void OpGraph<NODE_T>::DeleteNodes(std::vector<NODE_T *> deleteNodes) {
  for (auto deletenode : deleteNodes) {
    if (deletenode == nullptr) {
      continue;
    }
    auto node = GetNode(deletenode->ID());
    if (node == nullptr) {
      continue;
    }
    nodes.erase(deletenode->ID());
  }
}

template <typename NODE_T>
std::unordered_set<NODE_T *> OpGraph<NODE_T>::GetInputNode() {
  std::unordered_set<NODE_T *> inputNodes;
  for (const auto &iter : nodes) {
    auto node = iter.second;
    if (node->GetAllInEdges().empty()) {
      inputNodes.insert(node);
    }
  }
  return inputNodes;
}

template <typename NODE_T>
std::unordered_set<NODE_T *> OpGraph<NODE_T>::GetOutputNode() {
  std::unordered_set<NODE_T *> outputNodes;
  for (const auto &iter : nodes) {
    auto node = iter.second;
    if (node->GetAllOutEdges().empty()) {
      outputNodes.insert(node);
    }
  }
  return outputNodes;
}

template <typename NODE_T>
std::unordered_map<NODE_T *, std::unordered_set<NODE_T *>> OpGraph<NODE_T>::GetDepends() {
  std::unordered_map<NODE_T *, std::unordered_set<NODE_T *>> depends;
  for (auto nodeIter : nodes) {
    depends[nodeIter.second] = nodeIter.second->GetAllInEdges();
  }
  return depends;
}

template <typename NODE_T>
OpGraph<NODE_T>::~OpGraph() {
  for (auto iter : nodes) {
    delete iter.second;
  }
  nodes.clear();
}

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_COMMON_GRAPH_UTIL_H_

