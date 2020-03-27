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

#include "src/graph.h"
#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include "schema/ms_generated.h"
#include "common/graph_util.h"
#include "common/mslog.h"
#include "include/errorcode.h"
#include "src/graph_execution.h"

namespace mindspore {
namespace predict {
static const uint32_t G_MAX_OP_COUNT = 10000;

Graph *Graph::CreateFromBuf(const char *buf, size_t size, const Context &ctx) {
  if (buf == nullptr) {
    MS_LOGE("the input buffer is nullptr");
    return nullptr;
  }

  flatbuffers::Verifier verify((const uint8_t *)buf, size);
  if (!VerifyGraphDefBuffer(verify)) {
    MS_LOGE("the buffer is invalid and fail to create graph");
    return nullptr;
  }

  auto graphDef = GetGraphDef(buf);
  std::unique_ptr<Graph> graph(new (std::nothrow) Graph());
  if (graph == nullptr) {
    MS_LOGE("graph malloc fail");
    return nullptr;
  }
  auto ret = graph->Build(*graphDef, ctx);
  if (ret != RET_OK) {
    MS_LOGE("build graph fail");
    return nullptr;
  }
  return graph.release();
}

Graph::Graph() = default;

Graph::~Graph() {
  for (auto &subgraph : subgraphs) {
    delete subgraph;
  }
  subgraphs.clear();
}

int Graph::Build(const GraphDef &graphDef, const Context &ctx) {
  MS_ASSERT(graphDef.subgraphs() != nullptr);
  for (size_t i = 0; i < graphDef.subgraphs()->size(); i++) {
    MS_ASSERT(graphDef.subgraphs()->GetAs<SubGraphDef>(i) != nullptr);
    SubGraph *subGraph = SubGraph::CreateSubGraph(*(graphDef.subgraphs()->GetAs<SubGraphDef>(i)), ctx);
    if (subGraph == nullptr) {
      MS_LOGE("converter subgraph failed");
      return RET_ERROR;
    }
    subgraphs.push_back(subGraph);
    auto subDepends = subGraph->GetDepends();
    depends.insert(subDepends.begin(), subDepends.end());
  }

  auto iter = depends.begin();
  while (iter != depends.end()) {
    if (iter->second.empty()) {
      readyQue.push_back(iter->first);
      iter = depends.erase(iter);
    } else {
      iter++;
    }
  }

  return RET_OK;
}

std::vector<Tensor *> Graph::GetInputs() {
  MS_ASSERT(subgraphs.front() != nullptr);
  return subgraphs.front()->GetInputs();
}

std::vector<Tensor *> Graph::GetOutputs() {
  MS_ASSERT(subgraphs.back() != nullptr);
  return subgraphs.back()->GetOutputs();
}

std::map<NODE_ID, std::vector<Tensor *>> &Graph::GetOutputsMap() {
  MS_ASSERT(subgraphs.back() != nullptr);
  return subgraphs.back()->GetOutputsMap();
}

void Graph::FreeAllTensors() {
  for (auto iter : subgraphs) {
    iter->FreeAllTensors();
  }
}

std::vector<SubGraph *> *Graph::Subgraphs() { return &subgraphs; }

SubGraph::SubGraph() = default;

SubGraph::~SubGraph() {
  for (auto iter = nodes.begin(); iter != nodes.end();) {
    if (iter->second != nullptr) {
      delete iter->second;
    }
    iter = nodes.erase(iter);
  }
  nodes.clear();

  for (auto &allTensor : allTensors) {
    if (allTensor != nullptr) {
      delete allTensor;
    }
  }
  allTensors.clear();
}

SubGraph *SubGraph::CreateSubGraph(const SubGraphDef &subGraphDef, const Context &ctx) {
  std::unique_ptr<SubGraph> subGraph(new (std::nothrow) SubGraph());
  if (subGraph == nullptr) {
    MS_LOGE("subGraph malloc fail");
    return nullptr;
  }

  auto ret = subGraph->Build(subGraphDef, ctx);
  if (ret != RET_OK) {
    MS_LOGE("subGraph Build fail");
    return nullptr;
  }

  return subGraph.release();
}

int SubGraph::Build(const SubGraphDef &subGraphDef, const Context &ctx) {
  int ret;
  MS_ASSERT(subGraphDef.inputIndex() != nullptr);
  ret = ConverterIndex(*(subGraphDef.inputIndex()), &inputIndices);
  if (ret != RET_OK) {
    MS_LOGE("ConverterIndex failed: %d", ret);
    return ret;
  }
  MS_LOGD("converter inputIndex succ");

  MS_ASSERT(subGraphDef.outputIndex() != nullptr);
  ret = ConverterIndex(*(subGraphDef.outputIndex()), &outputIndices);
  if (ret != RET_OK) {
    MS_LOGE("ConverterIndex failed: %d", ret);
    return ret;
  }
  MS_LOGD("converter outputIndex succ");
  MS_ASSERT(subGraphDef.allTensors() != nullptr);
  ret = ConverterAllTensor(*(subGraphDef.allTensors()));
  if (ret != RET_OK) {
    MS_LOGE("ConverterAllTensor failed: %d", ret);
    return ret;
  }
  MS_LOGD("converter AllTensor succ");
  MS_ASSERT(subGraphDef.nodes() != nullptr);
  ret = ConverterNodes(*(subGraphDef.nodes()), ctx);
  if (ret != RET_OK) {
    MS_LOGE("ConverterNodes failed: %d", ret);
    return ret;
  }
  MS_LOGD("converter nodes succ");

  ret = ConverterEdges(subGraphDef);
  if (ret != RET_OK) {
    MS_LOGE("ConverterEdges failed: %d", ret);
    return ret;
  }
  MS_LOGD("converter edges succ");

  ret = InitOutputsMap();
  if (ret != RET_OK) {
    MS_LOGE("InitOutputsMap failed: %d", ret);
    return ret;
  }
  MS_LOGD("init outputs map succ");

  MS_LOGD("build graph succ");
  return RET_OK;
}

int SubGraph::ConverterIndex(const flatbuffers::Vector<uint32_t> &srcIndex, std::vector<uint32_t> *dstIndex) {
  if (dstIndex == nullptr) {
    MS_LOGE("input dstIndex is nullptr");
    return RET_PARAM_INVALID;
  }
  dstIndex->resize(srcIndex.size());
  std::copy(srcIndex.begin(), srcIndex.end(), dstIndex->begin());
  return RET_OK;
}

int SubGraph::ConverterAllTensor(const flatbuffers::Vector<flatbuffers::Offset<TensorDef>> &srcTensors) {
  uint32_t tensorsSize = srcTensors.size();

  allTensors.clear();
  allTensors.reserve(tensorsSize);
  for (uint32_t i = 0; i < tensorsSize; i++) {
    auto tensorDef = srcTensors.GetAs<TensorDef>(i);
    if (tensorDef == nullptr) {
      MS_LOGE("%ud th tensordef is null", i);
      return RET_ERROR;
    }
    auto tensor = Tensor::CopyFromTensorDef(*tensorDef);
    if (tensor == nullptr) {
      return RET_ERROR;
    }
    allTensors.push_back(tensor);
  }

  return RET_OK;
}

int SubGraph::ConverterNodes(const flatbuffers::Vector<flatbuffers::Offset<NodeDef>> &nodeDefs, const Context &ctx) {
  uint32_t opCount = nodeDefs.size();
  // for dfx
  if (opCount > G_MAX_OP_COUNT) {
    MS_LOGE("opCount(%u) bigger than maxOpCount(%u)", opCount, G_MAX_OP_COUNT);
    return RET_ERROR;
  }

  nodes.clear();

  for (uint32_t i = 0; i < opCount; i++) {
    auto nodeDef = nodeDefs.GetAs<NodeDef>(i);
    MS_ASSERT(nodeDef != nullptr);
    auto node = std::unique_ptr<Node>(new (std::nothrow) Node(nodeDef));
    if (node == nullptr) {
      MS_LOGE("new node failed");
      return RET_NULL_PTR;
    }

    node->SetTensors(*nodeDef, allTensors);

    auto ret = node->InitOp(*(nodeDef->opDef()), ctx);
    if (ret != RET_OK) {
      MS_LOGE("node (%s) InitOP failed. ret:%d", node->ID().c_str(), ret);
      return ret;
    }

    auto nodeId = node->ID();
    nodes[nodeId] = node.release();
    MS_LOGD("add node succ, id:%s", nodeId.c_str());
  }

  return RET_OK;
}

int SubGraph::ConverterEdges(const SubGraphDef &subGraphDef) {
  auto opGraph = OpGraph::Build(subGraphDef);
  if (opGraph == nullptr) {
    MS_LOGE("opGraph Build fail");
    return RET_ERROR;
  }

  for (auto nodeIter : nodes) {
    auto node = opGraph->GetNode(nodeIter.first);
    if (node == nullptr) {
      MS_LOGI("node %s not found", nodeIter.first.c_str());
      continue;
    }
    for (const auto &edge : node->GetAllInEdge()) {
      MS_ASSERT(nodeIter.second != nullptr);
      nodeIter.second->AddInEdge(GetNode(edge));
    }
    for (const auto &edge : node->GetAllOutEdge()) {
      MS_ASSERT(nodeIter.second != nullptr);
      nodeIter.second->AddOutEdge(GetNode(edge));
    }
  }
  delete opGraph;
  return RET_OK;
}

int SubGraph::InitOutputsMap() {
  if (nodes.empty()) {
    MS_LOGE("nodes are empty");
    return RET_ERROR;
  }
  for (auto node : nodes) {
    NODE_ID realNodeName = node.second->ID();
    MS_ASSERT(node.second != nullptr);
    if (node.second->GetAllOutEdges().empty()) {
      auto nodeType = node.second->Type();
      if (nodeType == "Nhwc2Nchw" || nodeType == "Nchw2Nhwc") {
        auto dependNode = *(this->GetDepends().at(this->GetNode(realNodeName)).begin());
        realNodeName = dependNode->ID();
      }
      this->outputsMap.emplace(
        std::pair<NODE_ID, std::vector<Tensor *>>(realNodeName, node.second->GetOutputTensors()));
    }
  }
  return RET_OK;
}

std::unordered_map<Node *, std::unordered_set<Node *>> SubGraph::GetDepends() {
  std::unordered_map<Node *, std::unordered_set<Node *>> depends;
  for (auto nodeIter : nodes) {
    MS_ASSERT(nodeIter.second != nullptr);
    depends[nodeIter.second] = nodeIter.second->GetAllInEdges();
  }
  return depends;
}

Node *SubGraph::GetNode(const NODE_ID &id) {
  auto node = nodes.find(id);
  if (node == nodes.end()) {
    return nullptr;
  }
  return node->second;
}

std::vector<Tensor *> SubGraph::GetInputs() {
  std::vector<Tensor *> inputTensor;
  inputTensor.resize(inputIndices.size());
  std::transform(inputIndices.begin(), inputIndices.end(), inputTensor.begin(),
                 [this](int i) { return this->allTensors[i]; });

  return inputTensor;
}

std::vector<Tensor *> SubGraph::GetOutputs() {
  std::vector<Tensor *> outputTensor;
  outputTensor.resize(outputIndices.size());
  std::transform(outputIndices.begin(), outputIndices.end(), outputTensor.begin(),
                 [this](int i) { return this->allTensors[i]; });

  return outputTensor;
}

std::map<NODE_ID, std::vector<Tensor *>> &SubGraph::GetOutputsMap() { return outputsMap; }

void SubGraph::FreeAllTensors() {
  for (auto &allTensor : allTensors) {
    if (allTensor != nullptr) {
      auto refcount = allTensor->RefCount();
      if (refcount != MSConst_WEIGHT_REFCOUNT) {
        allTensor->DefRef(refcount);
        allTensor->FreeData();
      }
    }
  }
}

const std::vector<uint32_t> *SubGraph::GetInputIndices() const { return &inputIndices; }

const std::vector<uint32_t> *SubGraph::GetOutputIndices() const { return &outputIndices; }

bool SubGraph::IsInputIndex(uint32_t i) {
  auto iter = std::find(inputIndices.begin(), inputIndices.end(), i);
  return !(iter == inputIndices.end());
}

bool SubGraph::IsOutputIndex(uint32_t i) {
  auto iter = std::find(outputIndices.begin(), outputIndices.end(), i);
  return !(iter == outputIndices.end());
}
}  // namespace predict
}  // namespace mindspore
