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

#include <set>
#include <utility>
#include "tools/converter/legacy_optimizer/fusion/fusion_pattern.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
FusionPattern::FusionPattern(std::string name) { this->name = std::move(name); }

FusionPattern::~FusionPattern() = default;

FusionPattern &FusionPattern::AddPatternOp(const std::string &id,
                                           const std::initializer_list<schema::PrimitiveType> &types) {
  return AddPatternOp(id, std::vector<schema::PrimitiveType>(types));
}

FusionPattern &FusionPattern::AddPatternOp(const std::string &id, const std::vector<schema::PrimitiveType> &types) {
  if (id.empty()) {
    MS_LOG(ERROR) << "Id cannot be empty";
    hasError = true;
    return *this;
  }

  if (GetPatternOp(id) != nullptr) {
    MS_LOG(ERROR) << "Id repeated. id: " << id;
    hasError = true;
    return *this;
  }

  auto op = std::make_shared<PatternOp>();
  if (op == nullptr) {
    MS_LOG(ERROR) << "new an object failed";
    hasError = true;
    return *this;
  } else {
    op->id = id;
    op->types = types;
    ops.push_back(op);
    opMap[id] = op;
  }

  return *this;
}

bool FusionPattern::Check() {
  if (hasError) {
    MS_LOG(ERROR) << "Has Error in previous Func";
    return false;
  }

  if (GetPatternOp(this->outputOpId) == nullptr) {
    MS_LOG(ERROR) << "Can not find the output of the pattern";
    return false;
  }

  return true;
}

std::shared_ptr<PatternOp> FusionPattern::GetPatternOp(const std::string &id) const {
  auto it = opMap.find(id);
  if (it != opMap.end()) {
    return it->second;
  }
  return nullptr;
}

std::string FusionPattern::GetOutput() const { return this->outputOpId; }

FusionPattern &FusionPattern::AddPatternOp(const std::shared_ptr<PatternOp> &patternOp) {
  if (patternOp == nullptr) {
    MS_LOG(ERROR) << "Input pattern op is nullptr";
    hasError = true;
    return *this;
  }
  if (GetPatternOp(patternOp->id) != nullptr) {
    MS_LOG(ERROR) << "Id repeated. id: " << patternOp->id;
    hasError = true;
    return *this;
  }

  ops.push_back(patternOp);
  opMap[patternOp->id] = patternOp;
  return *this;
}

FusionPattern &FusionPattern::Finish() {
  std::vector<std::string> ids;
  std::set<std::string> nodeInputIds;
  std::vector<std::string> inputNodeIds;
  for (const auto &patternOp : ops) {
    MS_ASSERT(patternOp != nullptr);
    if (IsContain(ids, patternOp->id)) {
      MS_LOG(ERROR) << "Duplicate id find: " << patternOp->id;
      hasError = true;
      return *this;
    }
    ids.emplace_back(patternOp->id);
    if (patternOp->left != nullptr) {
      nodeInputIds.insert(patternOp->left->id);
    }
    if (patternOp->right != nullptr) {
      nodeInputIds.insert(patternOp->right->id);
    }
    if (patternOp->left == nullptr && patternOp->right == nullptr) {
      inputNodeIds.emplace_back(patternOp->id);
    }
  }
  for (auto iter = ids.begin(); iter != ids.end();) {
    if (nodeInputIds.find(*iter) != nodeInputIds.end()) {
      iter = ids.erase(iter);
    } else {
      iter++;
    }
  }
  if (ids.size() > 1) {
    MS_LOG(ERROR) << "Multi-output node find, only support pattern with one output";
    hasError = true;
    return *this;
  }
  if (ids.empty()) {
    MS_LOG(ERROR) << "No output node find, only support pattern with one output";
    hasError = true;
    return *this;
  }
  this->outputOpId = std::string(ids.front());
  auto outputNode = GetPatternOp(this->outputOpId);
  if (outputNode != nullptr) {
    outputNode->isTail = true;
  }

  for (const auto &inputNodeId : inputNodeIds) {
    auto inputNode = GetPatternOp(inputNodeId);
    if (inputNode != nullptr) {
      inputNode->isHead = true;
    }
  }
  return *this;
}

std::string FusionPattern::GetName() { return this->name; }
}  // namespace lite
}  // namespace mindspore
