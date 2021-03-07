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

#include <cfloat>
#include <queue>
#include <algorithm>
#include <cassert>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "tools/converter/legacy_optimizer/fusion/fusion_pass.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
STATUS FusionPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  auto ret = DefinePattern();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DefinePattern Error " << ret;
    return ret;
  }
  for (auto pattern : patterns) {
    if (pattern == nullptr) {
      MS_LOG(ERROR) << "FusionPattern has not been set";
      return RET_PARAM_INVALID;
    }

    if (!pattern->Check()) {
      MS_LOG(ERROR) << "FusionPattern is invalid";
      return RET_PARAM_INVALID;
    }
  }

  ret = MatchPatterns(graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatchPattern Error " << ret;
    return ret;
  }

  if (this->matchedPaths.empty()) {
    return RET_NO_CHANGE;
  } else {
    ret = Fuse(graph);
    if (ret != RET_OK && ret != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Fuse Error " << ret;
    }
    return ret;
  }
}

STATUS FusionPass::MatchPatterns(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  this->matchedPaths.clear();
  STATUS status;
  for (auto pattern : patterns) {
    status = MatchOnePattern(graph, pattern);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "MatchOnePatternInSubGraph failed: " << status;
      return status;
    }
  }
  this->mapedMatchedPaths.clear();
  for (auto iter = matchedPaths.begin(); iter != matchedPaths.end(); iter++) {
    auto patternName = iter->first;
    auto patternOps = iter->second;
    std::vector<std::unordered_map<std::string, std::shared_ptr<Path>>> mapedPaths;
    for (const auto &patternOp : patternOps) {
      std::queue<std::shared_ptr<PatternOp>> opQueue;
      std::unordered_map<std::string, std::shared_ptr<Path>> mapedPath;
      opQueue.push(patternOp);
      while (!opQueue.empty()) {
        auto curPatternOp = opQueue.front();
        opQueue.pop();
        MS_ASSERT(curPatternOp != nullptr);
        mapedPath.insert(std::make_pair(curPatternOp->id, curPatternOp->path));
        if (curPatternOp->left != nullptr) {
          opQueue.push(curPatternOp->left);
        }
        if (curPatternOp->right != nullptr) {
          opQueue.push(curPatternOp->right);
        }
      }
      mapedPaths.emplace_back(mapedPath);
    }
    this->mapedMatchedPaths.insert(std::make_pair(patternName, mapedPaths));
  }
  return RET_OK;
}

// assume that all nodes have only one output. if node has multi-outputs,
// some errors may happen
STATUS FusionPass::MatchOnePattern(schema::MetaGraphT *graph, FusionPattern *pattern) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(pattern != nullptr);
  //  std::vector<std::unordered_map<std::string, Path *>> patternMatchPaths;
  auto outputOp = pattern->GetPatternOp(pattern->GetOutput());
  if (outputOp == nullptr) {
    MS_LOG(ERROR) << "Can not find the output of the pattern";
    return RET_NULL_PTR;
  }
  MS_ASSERT(outputOp->isTail);
  if (graph->nodes.empty()) {
    return RET_OK;
  }
  // find all matched entries
  std::vector<size_t> entries;
  std::queue<size_t> nodeQueue;
  std::vector<size_t> sinkIdes;
  for (auto index : graph->outputIndex) {
    auto subGraphOutputNodeIdxes = GetLinkedPreIdx(*graph, index);
    for (auto subGraphOutputNodeIdx : subGraphOutputNodeIdxes) {
      MS_ASSERT((subGraph->nodes.size() > subGraphOutputNodeIdx));
      nodeQueue.push(subGraphOutputNodeIdx);
    }
  }
  while (!nodeQueue.empty()) {
    auto nodeIdx = nodeQueue.front();
    nodeQueue.pop();
    if (IsContain(sinkIdes, nodeIdx)) {
      continue;
    }
    MS_ASSERT(subGraph->nodes.size() > nodeIdx);
    auto &node = graph->nodes.at(nodeIdx);
    sinkIdes.emplace_back(nodeIdx);

    MS_ASSERT(nullptr != node->primitive);
    if (IsContain(outputOp->types, node->primitive->value.type)) {
      entries.emplace_back(nodeIdx);
    }
    auto preNodeIdxes = GetInputNodeIdx(*graph, nodeIdx);
    for (auto preNodeIdx : preNodeIdxes) {
      MS_ASSERT((subGraph->nodes.size() > preNodeIdx));
      nodeQueue.push(preNodeIdx);
    }
  }

  // check each entry
  std::vector<std::shared_ptr<PatternOp>> paths;
  sinkIdes.clear();
  std::vector<size_t> pathSinkIdes;
  for (auto nodeIdx : entries) {
    if (IsContain(sinkIdes, nodeIdx)) {
      continue;
    }
    pathSinkIdes.clear();
    auto path = PatternOp::Copy(outputOp);
    auto ret = MatchTree(graph, nodeIdx, path, sinkIdes, pathSinkIdes);
    if (ret && CheckMatch(graph, path)) {
      paths.emplace_back(path);
    }
  }
  auto patternName = pattern->GetName();
  this->matchedPaths.insert(std::make_pair(patternName, paths));
  return RET_OK;
}

bool FusionPass::CheckMatch(schema::MetaGraphT *graph, const std::shared_ptr<PatternOp> &patternOp) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(patternOp != nullptr);
  // find included nodes
  std::queue<std::shared_ptr<PatternOp>> opQueue;
  std::vector<size_t> matchedNodeIdxes;
  std::vector<std::shared_ptr<PatternOp>> inputNodes;
  std::shared_ptr<PatternOp> outputNode = nullptr;
  opQueue.push(patternOp);
  while (!opQueue.empty()) {
    auto curPatternOp = opQueue.front();
    opQueue.pop();
    matchedNodeIdxes.push_back(curPatternOp->path->nodeIdx);
    if (curPatternOp->isHead) {
      inputNodes.emplace_back(curPatternOp);
    }
    if (curPatternOp->isTail) {
      if (outputNode != nullptr && outputNode != curPatternOp) {
        return false;
      }
      outputNode = curPatternOp;
    }
    if (curPatternOp->left != nullptr) {
      opQueue.push(curPatternOp->left);
    }
    if (curPatternOp->right != nullptr) {
      opQueue.push(curPatternOp->right);
    }
  }
  // all post node of input node should be in path except input node is placeHold
  for (const auto &inputNode : inputNodes) {
    if (inputNode->isPlaceHold) {
      continue;
    }
    auto inputNodePostNodeIdxes = GetOutputNodeIdx(*graph, inputNode->path->nodeIdx);
    for (auto inputNodePostNodeIdx : inputNodePostNodeIdxes) {
      if (!IsContain(matchedNodeIdxes, inputNodePostNodeIdx)) {
        return false;
      }
    }
  }
  // all pre node of output node should be in path
  auto outputNodePreNodeIdxes = GetInputNodeIdx(*graph, outputNode->path->nodeIdx);
  for (auto outputNodePreNodeIdx : outputNodePreNodeIdxes) {
    if (!IsContain(matchedNodeIdxes, outputNodePreNodeIdx)) {
      return false;
    }
  }
  return true;
}

bool FusionPass::MatchTree(schema::MetaGraphT *graph, size_t nodeIdx, const std::shared_ptr<PatternOp> &target,
                           std::vector<size_t> &sinkIdes, std::vector<size_t> &pathSinkIdes) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(nodeIdx < graph->nodes.size());
  // check the func params
  if (!CheckMatchParams(graph, nodeIdx, target, sinkIdes, pathSinkIdes)) {
    return false;
  }
  // path is set and not pointer to this node
  if (target->pathSetted) {
    MS_ASSERT(target->path != nullptr);
    if (target->path->nodeIdx != static_cast<int>(nodeIdx)) {
      return false;
    }
  }
  target->SetPath(-1, nodeIdx);
  sinkIdes.push_back(nodeIdx);
  pathSinkIdes.push_back(nodeIdx);
  // target is marked head, no need to check left and right. head-target's left
  // and right is always nullptr
  if (target->isHead) {
    return true;
  }
  auto preNodeIdxes = GetInputNodeIdx(*graph, nodeIdx);
  if (preNodeIdxes.empty() && target->left == nullptr && target->right == nullptr) {
    return true;
  }
  for (auto preNodeIdx : preNodeIdxes) {
    MS_ASSERT(graph->nodes.size() > preNodeIdx);
    // Case of multiple outputs is not supported.
    if (GetInputNodeIdx(*graph, preNodeIdx).size() > 2 || GetOutputNodeIdx(*graph, preNodeIdx).size() > 1) {
      sinkIdes.erase((sinkIdes.end() - 1));
      pathSinkIdes.erase((pathSinkIdes.end() - 1));
      target->UnSetPath();
      return false;
    }
    if (!MatchTree(graph, preNodeIdx, target->left, sinkIdes, pathSinkIdes)) {
      continue;
    }
    // match left then match right
    if (preNodeIdxes.size() == 1 && target->right == nullptr) {
      return true;
    }
    for (auto preNodeIdxInner : preNodeIdxes) {
      if (preNodeIdxInner == preNodeIdx) {
        continue;
      }
      MS_ASSERT(subGraph->nodes.size() > preNodeIdxInner);
      if (MatchTree(graph, preNodeIdxInner, target->right, sinkIdes, pathSinkIdes)) {
        return true;  // ignore follow match, pick the first match
      }
    }
  }
  sinkIdes.erase((sinkIdes.end() - 1));
  pathSinkIdes.erase((pathSinkIdes.end() - 1));
  target->UnSetPath();
  return false;
}

bool FusionPass::CheckMatchParams(schema::MetaGraphT *graph, size_t nodeIdx, const std::shared_ptr<PatternOp> &target,
                                  std::vector<size_t> &sinkIdes, std::vector<size_t> &pathSinkIdes) {
  auto &scope = graph->nodes.at(nodeIdx);
  MS_ASSERT(scope != nullptr);
  // if target(except target is marked head) is nullptr, it means the preNode
  // has no left or right, but scope is not nullptr
  if (target == nullptr) {
    return false;
  }
  // if node is sinked and not in the pathSinkId, then return false
  if (IsContain(sinkIdes, nodeIdx) && !IsContain(pathSinkIdes, nodeIdx)) {
    return false;
  }
  // type not match
  if (!target->isPlaceHold && !IsContain(target->types, scope->primitive->value.type)) {
    return false;
  }
  return true;
}

STATUS FusionPass::Fuse(schema::MetaGraphT *graph) {
  STATUS ret;
  bool isChange = false;
  for (auto iter = mapedMatchedPaths.begin(); iter != mapedMatchedPaths.end(); iter++) {
    for (auto &matchedPath : iter->second) {
      ret = DoFusion(graph, iter->first, matchedPath);
      if (ret != RET_OK && ret != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "DoFusion Error " << ret;
        return ret;
      } else {
        if (ret == RET_OK) {
          isChange = true;
        }
      }
    }
  }
  return isChange ? RET_OK : RET_NO_CHANGE;
}

FusionPass::~FusionPass() {
  for (auto pattern : patterns) {
    if (pattern != nullptr) {
      delete (pattern);
    }
  }
}

void FusionPass::MergeNodeAttrFromPost(std::unique_ptr<schema::CNodeT> &dstOp, std::unique_ptr<schema::CNodeT> &postOp,
                                       size_t dstOpOutIdx) {}
}  // namespace lite
}  // namespace mindspore
