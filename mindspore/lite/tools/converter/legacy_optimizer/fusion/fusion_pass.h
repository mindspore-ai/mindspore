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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_FUSION_FUSION_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_FUSION_FUSION_PASS_H_

#include <unordered_map>
#include <map>
#include <memory>
#include <string>
#include <vector>
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
                          const std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) = 0;

  STATUS MatchPatterns(const schema::MetaGraphT &graph);

  STATUS MatchOnePattern(const schema::MetaGraphT &graph, const FusionPattern &pattern);

  bool MatchTree(const schema::MetaGraphT &graph, size_t nodeIdx, const std::shared_ptr<PatternOp> &target,
                 std::vector<size_t> *sinkIdes, std::vector<size_t> *pathSinkIdes);

  bool CheckMatchParams(const schema::MetaGraphT &graph, size_t nodeIdx, const std::shared_ptr<PatternOp> &target,
                        const std::vector<size_t> &sinkIdes, const std::vector<size_t> &pathSinkIdes);
  static bool CheckMatch(const schema::MetaGraphT &graph, const std::shared_ptr<PatternOp> &patternOp);

  STATUS Fuse(schema::MetaGraphT *graph);

  std::vector<FusionPattern *> patterns{};
  std::map<std::string, std::vector<std::shared_ptr<PatternOp>>> matchedPaths{};
  // {name of pattern, vector<{name of pattern node, path}>}
  std::map<std::string, std::vector<std::unordered_map<std::string, std::shared_ptr<Path>>>> mapedMatchedPaths{};
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_FUSION_FUSION_PASS_H_
