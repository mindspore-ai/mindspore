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

#include "tools/common/graph_util.h"
#include <algorithm>
#include <ctime>
#include <utility>
#include <set>
#include "schema/inner/model_generated.h"
#include "tools/common/tensor_util.h"
#include "tools/common/node_util.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
OpDefCopyer GetSimpleOpCopyer() {
  return [](CNodeT *inCNode) -> std::unique_ptr<CNodeT> {
    std::unique_ptr<CNodeT> newCNode = std::make_unique<CNodeT>();
    if (newCNode == nullptr) {
      return nullptr;
    }

    newCNode->name = inCNode->name;
    newCNode->quantType = inCNode->quantType;
    newCNode->primitive = std::make_unique<schema::PrimitiveT>();
    newCNode->primitive->value.type = inCNode->primitive->value.type;
    return newCNode;
  };
}

std::vector<size_t> GetInputNodeIdx(const schema::MetaGraphT &graphT, const size_t &nodeIdx, const int inputIndexIdx) {
  return GetInputNodeIdx(graphT, *(graphT.nodes.at(nodeIdx).get()), inputIndexIdx);
}

std::vector<size_t> GetInputNodeIdx(const schema::MetaGraphT &graphT, const CNodeT &node, const int inputIndexIdx) {
  std::vector<uint32_t> inputIndexes;
  if (inputIndexIdx == -1) {
    inputIndexes = node.inputIndex;
  } else {
    MS_ASSERT(node.inputIndex.size() > inputIndexIdx);
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

std::vector<size_t> GetOutputNodeIdx(const schema::MetaGraphT &graphT, const size_t &nodeIdx,
                                     const int outputIndexIdx) {
  return GetOutputNodeIdx(graphT, *(graphT.nodes.at(nodeIdx).get()), outputIndexIdx);
}

std::vector<size_t> GetOutputNodeIdx(const schema::MetaGraphT &graphT, const CNodeT &node, const int outputIndexIdx) {
  std::vector<uint32_t> outputIndexes;
  if (outputIndexIdx == -1) {
    outputIndexes = node.outputIndex;
  } else {
    MS_ASSERT(node.outputIndex.size() > outputIndexIdx);
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

STATUS IsolateNode(schema::MetaGraphT *graphT, CNodeT *node) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(node != nullptr);
  size_t nodeIdx = 0;
  for (size_t i = 0; i < graphT->nodes.size(); i++) {
    auto &inNode = graphT->nodes.at(i);
    MS_ASSERT(inNode != nullptr);
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
  auto &gOutTensorIdx = graphT->outputIndex;
  for (auto iter = gOutTensorIdx.begin(); iter != gOutTensorIdx.end(); iter++) {
    if (*iter == outDataTensorIdx) {
      *iter = inDataTensorIdx;
      break;
    }
  }

  // find poseNode
  auto postNodeIdxes = GetOutputNodeIdx(*graphT, nodeIdx, 0);
  for (auto postNodeIdx : postNodeIdxes) {
    MS_ASSERT(graphT->nodes.size() > postNodeIdx);
    auto &postNode = graphT->nodes.at(postNodeIdx);
    MS_ASSERT(postNode != nullptr);
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

STATUS IsolateOneWayNode(schema::MetaGraphT *graph, size_t subGraphIdx, size_t nodeIdx, bool removeTensor) {
  MS_ASSERT(graph != nullptr);
  return IsolateOneWayNode(graph, nodeIdx, removeTensor);
}

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, size_t nodeIdx, bool removeTensor) {
  MS_ASSERT(graphT != nullptr);
  if (graphT->nodes.size() <= nodeIdx) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  CNodeT *node = graphT->nodes.at(nodeIdx).get();
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
    auto &gOutTensorIdx = graphT->outputIndex;
    for (auto iter = gOutTensorIdx.begin(); iter != gOutTensorIdx.end(); iter++) {
      if (*iter == outDataTensorIdx) {
        *iter = inDataTensorIdx;
        break;
      }
    }
    // find poseNode
    auto postNodeIdxes = GetOutputNodeIdx(*graphT, nodeIdx, 0);
    for (auto postNodeIdx : postNodeIdxes) {
      MS_ASSERT(graphT->nodes.size() > postNodeIdx);
      auto &postNode = graphT->nodes.at(postNodeIdx);
      MS_ASSERT(postNode != nullptr);
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

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, CNodeT *node, bool removeTensor) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(node != nullptr);
  bool isSubNode = false;
  size_t nodeIdx = 0;
  for (size_t i = 0; i < graphT->nodes.size(); i++) {
    auto &inNode = graphT->nodes.at(i);
    MS_ASSERT(inNode != nullptr);
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

STATUS RemoveTensor(schema::MetaGraphT *graphT, std::vector<uint32_t> toDeleteTensorIdxes, bool forceDelete) {
  MS_ASSERT(graphT != nullptr);
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

STATUS UpdateNodeIndex(CNodeT *node, uint32_t deleteIdx) {
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

STATUS AddTensor2Node(schema::MetaGraphT *graphT, uint32_t nodeIdx, std::unique_ptr<TensorT> tensor,
                      InsertPlace place) {
  if (nodeIdx >= graphT->nodes.size()) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  graphT->allTensors.emplace_back(std::move(tensor));
  uint32_t newTensorIdx = graphT->allTensors.size() - 1;
  auto node = graphT->nodes.at(nodeIdx).get();
  MS_ASSERT(node != nullptr);
  if (place == kBefore) {
    node->inputIndex.emplace_back(newTensorIdx);
  } else {
    node->outputIndex.emplace_back(newTensorIdx);
  }
  return RET_OK;
}

STATUS ReplaceTensorOfNode(schema::MetaGraphT *graphT, uint32_t nodeIdx, uint32_t inTensorIdx,
                           std::unique_ptr<TensorT> tensor) {
  MS_ASSERT(graphT != nullptr);
  if (nodeIdx >= graphT->nodes.size()) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  auto node = graphT->nodes.at(nodeIdx).get();
  MS_ASSERT(node != nullptr);
  if (inTensorIdx >= graphT->allTensors.size()) {
    MS_LOG(ERROR) << "inTensorIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  if (!IsContain(node->inputIndex, inTensorIdx)) {
    MS_LOG(ERROR) << "inTensorIdx(" << inTensorIdx << ") is not a inputIdx of node(" << nodeIdx << ")";
    return RET_PARAM_INVALID;
  }
  graphT->allTensors.at(inTensorIdx).swap(tensor);
  return RET_OK;
}

NodeIter InsertNode(schema::MetaGraphT *graphT, uint32_t existNodeIdx, InsertPlace place, size_t inoutIndex,
                    std::unique_ptr<CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(errorCode != nullptr);
  if (existNodeIdx >= graphT->nodes.size()) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << existNodeIdx;
    return graphT->nodes.end();
  }
  auto node_iter = graphT->nodes.begin() + existNodeIdx;
  MS_ASSERT(node_iter != graphT->nodes.begin());
  MS_ASSERT((*node_iter) != nullptr);
  return InsertNode(graphT, node_iter, place, inoutIndex, std::move(toAddNode), errorCode, insert_num);
}

NodeIter InsertNode(schema::MetaGraphT *graphT, NodeIter existNodeIter, InsertPlace place, size_t inoutIndexIdx,
                    std::unique_ptr<CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(errorCode != nullptr);
  if (place == kBefore) {
    return InsertNodeBefore(graphT, existNodeIter, inoutIndexIdx, std::move(toAddNode), errorCode, insert_num,
                            opDefCopyer);
  } else if (place == kAfter) {
    return InsertNodeAfter(graphT, existNodeIter, inoutIndexIdx, std::move(toAddNode), errorCode, insert_num,
                           opDefCopyer);
  } else {
    MS_LOG(ERROR) << "Invalid InsertPlace : " << place;
    return graphT->nodes.end();
  }
}

NodeIter InsertNodeBefore(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t inputIndexIdx,
                          std::unique_ptr<CNodeT> toAddNodeIn, STATUS *errorCode, int *insert_num,
                          const OpDefCopyer &opDefCopyer) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(errorCode != nullptr);
  auto &existNode = *existNodeIter;
  MS_ASSERT(existNode != nullptr);
  MS_ASSERT(existNode->inputIndex.size() > inputIndexIdx);
  MS_ASSERT(toAddNodeIn != nullptr);
  auto preTensorIdx = existNode->inputIndex.at(inputIndexIdx);
  MS_ASSERT(graphT->allTensors.size() > preTensorIdx);

  auto preNodeIdxes = GetInputNodeIdx(*graphT, *(existNode), inputIndexIdx);
  size_t insert_node_num = preNodeIdxes.empty() ? 1 : preNodeIdxes.size();
  std::vector<std::unique_ptr<CNodeT>> toAddNodes;
  for (size_t i = 0; i < insert_node_num; ++i) {
    auto &preTensor = graphT->allTensors.at(preTensorIdx);
    MS_ASSERT(preTensor != nullptr);
    auto toAddTensor = CopyTensorDefT(preTensor);
    if (toAddTensor == nullptr) {
      *errorCode = RET_NULL_PTR;
      MS_LOG(ERROR) << "Copy Tensor failed";
      return graphT->nodes.end();
    }
    toAddTensor->nodeType = NodeType_CNode;
    toAddTensor->refCount = 0;
    toAddTensor->data.clear();
    MS_ASSERT(toAddNodeIn->primitive != nullptr);
    if (toAddNodeIn->primitive->value.type == schema::PrimitiveType_QuantDTypeCast) {
      auto prim = toAddNodeIn->primitive->value.AsQuantDTypeCast();
      MS_ASSERT(prim != nullptr);
      preTensor->dataType = prim->src_t;
      toAddTensor->dataType = prim->dst_t;
      if (prim->src_t == TypeId::kNumberTypeUInt8 && prim->dst_t == TypeId::kNumberTypeInt8) {
        preTensor->quantParams.front()->zeroPoint += 128;
      } else if (prim->src_t == TypeId::kNumberTypeInt8 && prim->dst_t == TypeId::kNumberTypeUInt8) {
        toAddTensor->quantParams.front()->zeroPoint += 128;
      }
    }
    graphT->allTensors.emplace_back(std::move(toAddTensor));
    size_t toAddTensorIdx = graphT->allTensors.size() - 1;
    auto toAddNode = opDefCopyer(toAddNodeIn.get());
    if (toAddNode == nullptr) {
      MS_LOG(ERROR) << "copy toAddNodeIn failed";
      *errorCode = RET_NULL_PTR;
      return graphT->nodes.end();
    }
    if (!preNodeIdxes.empty()) {
      toAddNode->name = toAddNodeIn->name + "_" + std::to_string(i);
    }
    toAddNode->inputIndex.clear();
    toAddNode->inputIndex.push_back(preTensorIdx);
    toAddNode->outputIndex.clear();
    toAddNode->outputIndex.push_back(toAddTensorIdx);
    for (auto iter = existNode->inputIndex.begin(); iter != existNode->inputIndex.end(); iter++) {
      if (*iter == preTensorIdx) {
        *iter = toAddTensorIdx;
        break;
      }
    }
    toAddNodes.emplace_back(std::move(toAddNode));
  }
  for (auto &toAddNode : toAddNodes) {
    existNodeIter = graphT->nodes.insert(existNodeIter, std::move(toAddNode));
    existNodeIter++;
    *insert_num += 1;
  }
  *errorCode = RET_OK;
  return existNodeIter;
}

NodeIter InsertNodeAfter(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t outputIndexIdx,
                         std::unique_ptr<schema::CNodeT> toAddNodeIn, STATUS *errorCode, int *insert_num,
                         const OpDefCopyer &opDefCopyer) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(errorCode != nullptr);
  auto &existNode = *existNodeIter;
  MS_ASSERT(existNode != nullptr);
  MS_ASSERT(existNode->outputIndex.size() > outputIndexIdx);
  MS_ASSERT(toAddNodeIn != nullptr);
  auto postTensorIdx = existNode->outputIndex.at(outputIndexIdx);
  MS_ASSERT(graphT->allTensors.size() > postTensorIdx);
  auto postNodeIdxes = GetOutputNodeIdx(*graphT, *(existNode), outputIndexIdx);
  bool is_output_index = IsContain(graphT->outputIndex, postTensorIdx);
  size_t insert_node_num = (postNodeIdxes.empty() || is_output_index) ? postNodeIdxes.size() + 1 : postNodeIdxes.size();
  bool has_insert_for_graph_out = postNodeIdxes.empty() || is_output_index;
  std::vector<std::unique_ptr<schema::CNodeT>> toAddNodes;
  for (size_t i = 0; i < insert_node_num; ++i) {
    auto &postTensor = graphT->allTensors.at(postTensorIdx);
    MS_ASSERT(postTensor != nullptr);
    auto toAddTensor = CopyTensorDefT(postTensor);
    if (toAddTensor == nullptr) {
      MS_LOG(ERROR) << "Copy TensorT failed";
      *errorCode = RET_NULL_PTR;
      return graphT->nodes.end();
    }
    toAddTensor->nodeType = NodeType_CNode;
    MS_ASSERT(toAddNodeIn->primitive != nullptr);
    if (toAddNodeIn->primitive->value.type == schema::PrimitiveType_QuantDTypeCast) {
      auto prim = toAddNodeIn->primitive->value.AsQuantDTypeCast();
      MS_ASSERT(prim != nullptr);
      postTensor->dataType = prim->src_t;
      toAddTensor->dataType = prim->dst_t;
      if (prim->src_t == TypeId::kNumberTypeInt8 && prim->dst_t == TypeId::kNumberTypeUInt8) {
        toAddTensor->quantParams.front()->zeroPoint += 128;
      } else if (prim->src_t == TypeId::kNumberTypeUInt8 && prim->dst_t == TypeId::kNumberTypeInt8) {
        postTensor->quantParams.front()->zeroPoint += 128;
      }
    }
    graphT->allTensors.emplace_back(std::move(toAddTensor));
    size_t toAddTensorIdx = graphT->allTensors.size() - 1;
    auto toAddNode = opDefCopyer(toAddNodeIn.get());
    if (toAddNode == nullptr) {
      MS_LOG(ERROR) << "copy toAddNodeIn failed";
      *errorCode = RET_NULL_PTR;
      return graphT->nodes.end();
    }
    toAddNode->inputIndex.clear();
    toAddNode->inputIndex.push_back(postTensorIdx);
    toAddNode->outputIndex.clear();
    toAddNode->outputIndex.push_back(toAddTensorIdx);
    if (!postNodeIdxes.empty()) {
      toAddNode->name = toAddNodeIn->name + "_" + std::to_string(i);
    }
    if (has_insert_for_graph_out) {
      for (auto iter = graphT->outputIndex.begin(); iter != graphT->outputIndex.end(); iter++) {
        if (*iter == postTensorIdx) {
          *iter = toAddTensorIdx;
        }
      }
      has_insert_for_graph_out = false;
    } else {
      auto &postNode = graphT->nodes.at(postNodeIdxes[is_output_index ? i - 1 : i]);
      for (auto iter = postNode->inputIndex.begin(); iter != postNode->inputIndex.end(); iter++) {
        if (*iter == postTensorIdx) {
          *iter = toAddTensorIdx;
        }
      }
    }
    toAddNodes.emplace_back(std::move(toAddNode));
  }
  for (auto &toAddNode : toAddNodes) {
    existNodeIter = graphT->nodes.insert(existNodeIter, std::move(toAddNode));
    existNodeIter++;
    *insert_num += 1;
  }
  *errorCode = RET_OK;
  return existNodeIter;
}

STATUS ValidateFileStr(const std::string &modelFile, const std::string &fileType) {
  if (modelFile.size() > fileType.size() && modelFile.substr(modelFile.size() - fileType.size()) == fileType) {
    return RET_OK;
  } else {
    return RET_ERROR;
  }
}

std::string GetModelName(const std::string &modelFile) {
  std::string modelName = modelFile;
  modelName = modelName.substr(modelName.find_last_of('/') + 1);
  modelName = modelName.substr(0, modelName.find_last_of('.'));
  return modelName;
}

int SetSubgraphTensorIndices(schema::MetaGraphT *meta_graphT) {
  for (auto &subgraph : meta_graphT->subGraph) {
    std::vector<uint32_t> subgraph_indices{};
    for (auto &node_idx : subgraph->nodeIndices) {
      auto &node = meta_graphT->nodes.at(node_idx);
      for (auto &input_idx : node->inputIndex) {
        if (IsContain(subgraph_indices, input_idx)) {
          continue;
        } else {
          subgraph_indices.push_back(input_idx);
        }
      }
      for (auto &output_idx : node->outputIndex) {
        if (IsContain(subgraph_indices, output_idx)) {
          continue;
        } else {
          subgraph_indices.push_back(output_idx);
        }
      }
    }
    subgraph->tensorIndices.assign(subgraph_indices.begin(), subgraph_indices.end());
  }
  return RET_OK;
}

std::vector<int> GetTransposePerm(MetaGraphT *graph, const std::unique_ptr<CNodeT> &cnode) {
  MS_ASSERT(graph != nullptr && cnode != nullptr);
  std::vector<int> perm;
  if (cnode->primitive->value.type != schema::PrimitiveType_Transpose) {
    return perm;
  }
  if (cnode->inputIndex.size() < 2) {
    MS_LOG(ERROR) << "transpose node input size is less than 2.";
    return perm;
  }
  MS_ASSERT(cnode->outputIndex.at(1) < graph->allTensors.size());
  auto &perm_tensor = graph->allTensors.at(cnode->inputIndex.at(1));
  if (perm_tensor->data.empty()) {
    return perm;
  }
  MS_ASSERT(perm_tensor->dims.size() != 0);
  perm.resize(perm_tensor->dims[0]);
  if (memcpy_s(perm.data(), perm_tensor->dims[0] * sizeof(int), perm_tensor->data.data(),
               perm_tensor->dims[0] * sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return {};
  }
  return perm;
}
}  // namespace lite
}  // namespace mindspore
