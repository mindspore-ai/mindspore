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

#include "tools/common/meta_graph_utils.h"
#include <vector>
#include <set>
#include "inner/model_generated.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"
namespace mindspore::lite {
namespace {
size_t GetRefCount(schema::MetaGraphT *graphT, uint32_t tensorIdx) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(graphT->allTensors.size() > tensorIdx);
  size_t refCount = 0;
  for (auto &node : graphT->nodes) {
    MS_ASSERT(node != nullptr);
    if (IsContain(node->inputIndex, tensorIdx)) {
      refCount++;
    }
  }
  return refCount;
}
}  // namespace

std::vector<size_t> GetLinkedPostIdx(const schema::MetaGraphT &graphT, const size_t &tensorIdx) {
  std::vector<size_t> postNodeIdx;
  for (size_t i = 0; i < graphT.nodes.size(); i++) {
    auto &oldNode = graphT.nodes.at(i);
    if (oldNode == nullptr) {
      continue;
    }
    auto inputIndexes = oldNode->inputIndex;
    if (IsContain<uint32_t>(inputIndexes, tensorIdx)) {
      postNodeIdx.emplace_back(i);
    }
  }
  return postNodeIdx;
}

std::vector<size_t> GetLinkedPreIdx(const schema::MetaGraphT &graphT, const size_t &tensorIdx) {
  std::vector<size_t> preNodeIdx;
  for (size_t i = 0; i < graphT.nodes.size(); i++) {
    auto &oldNode = graphT.nodes.at(i);
    if (oldNode == nullptr) {
      continue;
    }
    auto outputIndexes = oldNode->outputIndex;
    if (IsContain<uint32_t>(outputIndexes, tensorIdx)) {
      preNodeIdx.emplace_back(i);
    }
  }
  return preNodeIdx;
}

std::vector<size_t> GetInputNodeIdx(const schema::MetaGraphT &graphT, const schema::CNodeT &node,
                                    const int inputIndexIdx) {
  std::vector<uint32_t> inputIndexes;
  if (inputIndexIdx == -1) {
    inputIndexes = node.inputIndex;
  } else {
    MS_ASSERT(node.inputIndex.size() > static_cast<uint32_t>(inputIndexIdx));
    inputIndexes.emplace_back(node.inputIndex.at(inputIndexIdx));
  }
  std::set<size_t> inputNodeIdx;
  for (uint32_t inputIdx : inputIndexes) {
    auto linkedPreIdx = GetLinkedPreIdx(graphT, inputIdx);
    inputNodeIdx.insert(linkedPreIdx.begin(), linkedPreIdx.end());
  }
  std::vector<size_t> ret;
  ret.insert(ret.end(), inputNodeIdx.begin(), inputNodeIdx.end());
  return ret;
}

std::vector<size_t> GetInputNodeIdx(const schema::MetaGraphT &graphT, const size_t &nodeIdx, const int inputIndexIdx) {
  return GetInputNodeIdx(graphT, *(graphT.nodes.at(nodeIdx).get()), inputIndexIdx);
}

std::vector<size_t> GetOutputNodeIdx(const schema::MetaGraphT &graphT, const schema::CNodeT &node,
                                     const int outputIndexIdx) {
  std::vector<uint32_t> outputIndexes;
  if (outputIndexIdx == -1) {
    outputIndexes = node.outputIndex;
  } else {
    MS_ASSERT(node.outputIndex.size() > static_cast<uint32_t>(outputIndexIdx));
    outputIndexes.emplace_back(node.outputIndex.at(outputIndexIdx));
  }
  std::set<size_t> outputNodeIdx;
  for (uint32_t outputIdx : outputIndexes) {
    auto linkedPostIdx = GetLinkedPostIdx(graphT, outputIdx);
    outputNodeIdx.insert(linkedPostIdx.begin(), linkedPostIdx.end());
  }
  std::vector<size_t> ret;
  ret.insert(ret.end(), outputNodeIdx.begin(), outputNodeIdx.end());
  return ret;
}

std::vector<size_t> GetOutputNodeIdx(const schema::MetaGraphT &graphT, const size_t &nodeIdx,
                                     const int outputIndexIdx) {
  return GetOutputNodeIdx(graphT, *(graphT.nodes.at(nodeIdx).get()), outputIndexIdx);
}

void ReplaceOutput(const uint32_t &old_index, const uint32_t &new_index, schema::MetaGraphT *graphT) {
  std::replace_if(
    std::begin(graphT->outputIndex), std::end(graphT->outputIndex),
    [&old_index](uint32_t outputIndex) { return outputIndex == old_index; }, new_index);

  for (auto &subGraph : graphT->subGraph) {
    std::replace_if(
      std::begin(subGraph->outputIndices), std::end(subGraph->outputIndices),
      [&old_index](uint32_t outputIndex) { return outputIndex == old_index; }, new_index);
  }
}

STATUS UpdateNodeIndex(schema::CNodeT *node, uint32_t deleteIdx) {
  MS_ASSERT(node != nullptr);
  for (auto inIdxIt = node->inputIndex.begin(); inIdxIt != node->inputIndex.end();) {
    if (*inIdxIt == deleteIdx) {
      inIdxIt = node->inputIndex.erase(inIdxIt);
    } else {
      if (*inIdxIt > deleteIdx) {
        (*inIdxIt)--;
      }
      inIdxIt++;
    }
  }
  // update nodes output indexes
  for (auto outIdxIt = node->outputIndex.begin(); outIdxIt != node->outputIndex.end();) {
    if (*outIdxIt == deleteIdx) {
      outIdxIt = node->outputIndex.erase(outIdxIt);
    } else {
      if (*outIdxIt > deleteIdx) {
        (*outIdxIt)--;
      }
      outIdxIt++;
    }
  }
  return RET_OK;
}

STATUS RemoveTensor(schema::MetaGraphT *graphT, std::vector<uint32_t> toDeleteTensorIdxes, bool forceDelete) {
  MS_CHECK_TRUE_MSG(graphT != nullptr, RET_NULL_PTR, "graphT is nullptr");
  for (auto iter = toDeleteTensorIdxes.begin(); iter != toDeleteTensorIdxes.end();) {
    uint32_t deleteIdx = *iter;
    if (!forceDelete) {
      if (GetRefCount(graphT, deleteIdx) > 1) {
        iter++;
        continue;
      }
    }
    // update graph input indices
    for (auto gInIdx = graphT->inputIndex.begin(); gInIdx != graphT->inputIndex.end(); gInIdx++) {
      if (*gInIdx > deleteIdx) {
        (*gInIdx)--;
      }
    }
    // update graph output indices
    for (auto gOutIdx = graphT->outputIndex.begin(); gOutIdx != graphT->outputIndex.end(); gOutIdx++) {
      if (*gOutIdx > deleteIdx) {
        (*gOutIdx)--;
      }
    }

    for (auto &subgraph : graphT->subGraph) {
      // update subgraph input indices
      for (auto gInIdx = subgraph->inputIndices.begin(); gInIdx != subgraph->inputIndices.end(); gInIdx++) {
        if (*gInIdx > deleteIdx) {
          (*gInIdx)--;
        }
      }
      // update subgraph output indices
      for (auto gOutIdx = subgraph->outputIndices.begin(); gOutIdx != subgraph->outputIndices.end(); gOutIdx++) {
        if (*gOutIdx > deleteIdx) {
          (*gOutIdx)--;
        }
      }
      // update subgraph output indices
      for (auto idx = subgraph->tensorIndices.begin(); idx != subgraph->tensorIndices.end(); idx++) {
        if (*idx > deleteIdx) {
          (*idx)--;
        }
      }
    }

    // update nodes indexes
    for (auto node_iter = graphT->nodes.begin(); node_iter != graphT->nodes.end(); node_iter++) {
      // update nodes input indexes
      UpdateNodeIndex((*node_iter).get(), deleteIdx);
    }
    // update deleteTensorIdx
    for (auto selfIt = toDeleteTensorIdxes.begin(); selfIt != toDeleteTensorIdxes.end(); selfIt++) {
      if (*selfIt > deleteIdx) {
        (*selfIt)--;
      }
    }
    graphT->allTensors.erase(graphT->allTensors.begin() + deleteIdx);
    iter = toDeleteTensorIdxes.erase(iter);
  }
  return RET_OK;
}

STATUS IsolateNode(schema::MetaGraphT *graphT, schema::CNodeT *node) {
  MS_CHECK_TRUE_MSG(graphT != nullptr, RET_NULL_PTR, "graphT is nullptr");
  MS_CHECK_TRUE_MSG(node != nullptr, RET_NULL_PTR, "node is nullptr");
  size_t nodeIdx = 0;
  for (size_t i = 0; i < graphT->nodes.size(); i++) {
    auto &inNode = graphT->nodes.at(i);
    MS_CHECK_TRUE_MSG(inNode != nullptr, RET_NULL_PTR, "inNode is nullptr");
    if (inNode->name == node->name) {
      nodeIdx = i;
      break;
    }
  }
  auto inputTensorIdxes = node->inputIndex;
  auto outputTensorIdxes = node->outputIndex;
  if (inputTensorIdxes.empty()) {
    MS_LOG(ERROR) << "Node " << node->name.c_str() << "should has no inputs";
    return RET_ERROR;
  }
  if (outputTensorIdxes.size() != 1) {
    MS_LOG(ERROR) << "FakeQuantNode " << node->name.c_str()
                  << "should has 1 output, in fact: " << outputTensorIdxes.size();
    return RET_ERROR;
  }
  auto inDataTensorIdx = inputTensorIdxes.front();
  auto outDataTensorIdx = outputTensorIdxes.front();

  MS_ASSERT(graphT->allTensors.size() > inDataTensorIdx);
  ReplaceOutput(outDataTensorIdx, inDataTensorIdx, graphT);

  // find poseNode
  auto postNodeIdxes = GetOutputNodeIdx(*graphT, nodeIdx, 0);
  for (auto postNodeIdx : postNodeIdxes) {
    MS_ASSERT(graphT->nodes.size() > postNodeIdx);
    auto &postNode = graphT->nodes.at(postNodeIdx);
    MS_CHECK_TRUE_MSG(postNode != nullptr, RET_NULL_PTR, "postNode is nullptr");
    for (auto iter = postNode->inputIndex.begin(); iter != postNode->inputIndex.end(); iter++) {
      if (*iter == outDataTensorIdx) {
        *iter = inDataTensorIdx;
        break;
      }
    }
  }
  RemoveTensor(graphT, outputTensorIdxes);
  node->inputIndex.clear();
  node->outputIndex.clear();
  return RET_OK;
}

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, size_t nodeIdx, bool removeTensor) {
  MS_CHECK_TRUE_MSG(graphT != nullptr, RET_NULL_PTR, "graphT is nullptr");
  if (graphT->nodes.size() <= nodeIdx) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  schema::CNodeT *node = graphT->nodes.at(nodeIdx).get();
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is null";
    return RET_NULL_PTR;
  }
  auto inputTensorIdxes = node->inputIndex;
  auto outputTensorIdxes = node->outputIndex;
  auto preNodeIdxes = GetInputNodeIdx(*graphT, nodeIdx);
  if (preNodeIdxes.size() > 1 || outputTensorIdxes.size() > 1) {
    MS_LOG(ERROR) << "Only support node who has no more than one input and one output";
    return RET_ERROR;
  }
  if (inputTensorIdxes.empty()) {
    MS_LOG(ERROR) << "Error, " << nodeIdx << "th node has no input tensor";
    return RET_ERROR;
  }
  auto inDataTensorIdx = inputTensorIdxes.front();
  if (!outputTensorIdxes.empty()) {
    auto outDataTensorIdx = outputTensorIdxes.front();
    MS_ASSERT(graphT->allTensors.size() > inDataTensorIdx);
    MS_ASSERT(graphT->allTensors.at(inDataTensorIdx) != nullptr);
    ReplaceOutput(outDataTensorIdx, inDataTensorIdx, graphT);

    // find poseNode
    auto postNodeIdxes = GetOutputNodeIdx(*graphT, nodeIdx, 0);
    for (auto postNodeIdx : postNodeIdxes) {
      MS_ASSERT(graphT->nodes.size() > postNodeIdx);
      auto &postNode = graphT->nodes.at(postNodeIdx);
      MS_CHECK_TRUE_MSG(postNode != nullptr, RET_NULL_PTR, "postNode is nullptr");
      for (auto iter = postNode->inputIndex.begin(); iter != postNode->inputIndex.end(); iter++) {
        if (*iter == outDataTensorIdx) {
          *iter = inDataTensorIdx;
          break;
        }
      }
    }
  }
  if (removeTensor) {
    // now all node's outputTensors are useless
    // remove all node's outputTensors
    auto status = RemoveTensor(graphT, outputTensorIdxes);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "RemoveOutputTensors of node " << node->name.c_str() << "failed";
      return RET_ERROR;
    }
  }
  node->inputIndex.clear();
  node->outputIndex.clear();
  return RET_OK;
}

STATUS IsolateOneWayNode(schema::MetaGraphT *graph, size_t subGraphIdx, size_t nodeIdx, bool removeTensor) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "graph is nullptr");
  return IsolateOneWayNode(graph, nodeIdx, removeTensor);
}

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, schema::CNodeT *node, bool removeTensor) {
  MS_CHECK_TRUE_MSG(graphT != nullptr, RET_NULL_PTR, "graphT is nullptr");
  MS_CHECK_TRUE_MSG(node != nullptr, RET_NULL_PTR, "node is nullptr");
  bool isSubNode = false;
  size_t nodeIdx = 0;
  for (size_t i = 0; i < graphT->nodes.size(); i++) {
    auto &inNode = graphT->nodes.at(i);
    MS_CHECK_TRUE_MSG(inNode != nullptr, RET_NULL_PTR, "inNode is nullptr");
    if (inNode->name == node->name) {
      isSubNode = true;
      nodeIdx = i;
      break;
    }
  }
  if (!isSubNode) {
    MS_LOG(ERROR) << "Node " << node->name.c_str() << "is not in graphT " << graphT->name.c_str();
    return RET_PARAM_INVALID;
  } else {
    return IsolateOneWayNode(graphT, nodeIdx, removeTensor);
  }
}
}  // namespace mindspore::lite
