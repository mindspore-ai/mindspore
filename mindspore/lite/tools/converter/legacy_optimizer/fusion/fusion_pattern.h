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

#ifndef MINDSPORE_PREDICT_FUSION_PATTERN_H
#define MINDSPORE_PREDICT_FUSION_PATTERN_H

#include <string>
#include <utility>
#include <vector>
#include <map>
#include <memory>
#include "src/common/log_adapter.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
struct Path {
 public:
  Path(int32_t subGraphIdx, int32_t nodeIdx) : subGraphIdx(subGraphIdx), nodeIdx(nodeIdx) {}
  int32_t subGraphIdx = -1;
  int32_t nodeIdx = -1;
};

// Op description in pattern
struct PatternOp {
  std::string id;                            // id of op in pattern
  std::vector<schema::PrimitiveType> types;  // type of matchable op
  // only support node with no more than two preNode now
  // avoid loop reference
  std::shared_ptr<PatternOp> left;   // left input patternOp of this patternOp
  std::shared_ptr<PatternOp> right;  // right input patternOp of this patternOp
  std::shared_ptr<Path> path = std::make_shared<Path>(-1, -1);
  bool pathSetted = false;
  bool isHead = false;
  bool isTail = false;
  bool isPlaceHold = false;

  PatternOp() = default;
  explicit PatternOp(const std::string &inId) : id(inId) {}
  ~PatternOp() = default;
  void SetPath(size_t subGraphIdx, size_t nodeIdx) {
    MS_ASSERT(this->path != nullptr);
    this->path->subGraphIdx = subGraphIdx;
    this->path->nodeIdx = nodeIdx;
    this->pathSetted = true;
  }
  void UnSetPath() {
    MS_ASSERT(this->path != nullptr);
    this->path->subGraphIdx = -1;
    this->path->nodeIdx = -1;
    this->pathSetted = false;
  }
  static std::shared_ptr<PatternOp> Copy(const std::shared_ptr<PatternOp> &src) {
    if (src == nullptr) {
      return nullptr;
    }
    auto dst = std::make_shared<PatternOp>();
    dst->id = src->id;
    dst->types = src->types;
    if (src->path != nullptr) {
      dst->path = std::make_shared<Path>(src->path->subGraphIdx, src->path->nodeIdx);
    }
    dst->pathSetted = src->pathSetted;
    dst->isTail = src->isTail;
    dst->isHead = src->isHead;
    dst->isPlaceHold = src->isPlaceHold;
    dst->left = PatternOp::Copy(src->left);
    dst->right = PatternOp::Copy(src->right);
    return dst;
  }
};

class FusionPattern {
 public:
  explicit FusionPattern(std::string name = "");

  ~FusionPattern();

  std::string GetName();

  FusionPattern &SetName(const std::string &name);

  FusionPattern &AddPatternOp(const std::string &id, const std::initializer_list<schema::PrimitiveType> &types = {});

  FusionPattern &AddPatternOp(const std::string &id, const std::vector<schema::PrimitiveType> &types);

  FusionPattern &AddPatternOp(const std::shared_ptr<PatternOp> &patternOp);

  FusionPattern &RemovePatternOp(const std::string &id);

  // set id of patternOp
  // set isTail and isHead for patternOps
  FusionPattern &Finish();

  bool Check();
  // get the id of the output Op of th pattern
  std::string GetOutput() const;

  void Dump() const;

  // return nullptr if not find
  std::shared_ptr<PatternOp> GetPatternOp(const std::string &id) const;

 private:
  FusionPattern(const FusionPattern &) = default;

  FusionPattern &operator=(const FusionPattern &) = default;

 private:
  std::string name;

  std::vector<std::shared_ptr<PatternOp>> ops;

  // same with ops, just for search
  std::map<std::string, std::shared_ptr<PatternOp>> opMap;

  // output PatternOp id of pattern
  std::string outputOpId;

  bool hasError = false;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_FUSION_PATTERN_H
