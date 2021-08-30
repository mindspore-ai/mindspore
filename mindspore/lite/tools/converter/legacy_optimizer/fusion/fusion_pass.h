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

#ifndef MINDSPORE_PREDICT_FUSION_PASS_H
#define MINDSPORE_PREDICT_FUSION_PASS_H

#include <unordered_map>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "tools/common/node_util.h"
#include "tools/converter/optimizer.h"
#include "tools/converter/legacy_optimizer/fusion/fusion_pattern.h"

namespace mindspore {
namespace lite {
class FusionPass : public GraphPass {
 public:
  FusionPass() = default;

  ~FusionPass() override;

  virtual STATUS DefinePattern() = 0;

  STATUS Run(schema::MetaGraphT *graph) override;

 protected:
  virtual STATUS DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                          std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) = 0;

  STATUS MatchPatterns(schema::MetaGraphT *graph);

  STATUS MatchOnePattern(schema::MetaGraphT *graph, FusionPattern *pattern);

  bool MatchTree(schema::MetaGraphT *graph, size_t nodeIdx, const std::shared_ptr<PatternOp> &target,
                 std::vector<size_t> &sinkIdes, std::vector<size_t> &pathSinkIdes);

  bool CheckMatchParams(schema::MetaGraphT *graph, size_t nodeIdx, const std::shared_ptr<PatternOp> &target,
                        std::vector<size_t> &sinkIdes, std::vector<size_t> &pathSinkIdes);
  static bool CheckMatch(schema::MetaGraphT *graph, const std::shared_ptr<PatternOp> &patternOp);

  STATUS Fuse(schema::MetaGraphT *graph);

 protected:
  std::vector<FusionPattern *> patterns{};
  std::map<std::string, std::vector<std::shared_ptr<PatternOp>>> matchedPaths{};
  // {name of pattern, vector<{name of pattern node, path}>}
  std::map<std::string, std::vector<std::unordered_map<std::string, std::shared_ptr<Path>>>> mapedMatchedPaths{};
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_FUSION_PASS_H
