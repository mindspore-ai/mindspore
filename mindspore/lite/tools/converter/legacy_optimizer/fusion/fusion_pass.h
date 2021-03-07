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
#define CONV_OP_NO_BIAS_WEIGHT_NUM 1
#define CONV_OP_HAS_BIAS_WEIGHT_NUM 2
#define CONV_OP_NO_BIAS_INPUT_NUM 2
#define CONV_OP_HAS_BIAS_INPUT_NUM 3

#define CONV_OP_FILTER_INDEX_IN_WEIGHT 0
#define CONV_OP_BIAS_INDEX_IN_WEIGHT 1
#define CONV_OP_FILTER_INDEX_IN_INPUT 1
#define CONV_OP_BIAS_INDEX_IN_INPUT 2

#define CONV_FILTER_SHAPE_SIZE 4

// PatternOp Ids
constexpr const char *kConvName = "CONVOLUTION";
constexpr const char *DST_NAME = "DESTINATION";
constexpr const char *ACTIVATION_NAME = "ACTIVATION";
constexpr const char *BIASADD_NAME = "BIASADD";

class FusionPass : public GraphPass {
 public:
  FusionPass() = default;

  ~FusionPass() override;

  virtual STATUS DefinePattern() = 0;

  STATUS Run(schema::MetaGraphT *graph) override;

  virtual STATUS DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                          std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) = 0;

 protected:
  STATUS MatchPatterns(schema::MetaGraphT *graph);

  STATUS MatchOnePattern(schema::MetaGraphT *graph, FusionPattern *pattern);

  bool MatchTree(schema::MetaGraphT *graph, size_t nodeIdx, const std::shared_ptr<PatternOp> &target,
                 std::vector<size_t> &sinkIdes, std::vector<size_t> &pathSinkIdes);

  bool CheckMatchParams(schema::MetaGraphT *graph, size_t nodeIdx, const std::shared_ptr<PatternOp> &target,
                        std::vector<size_t> &sinkIdes, std::vector<size_t> &pathSinkIdes);
  static bool CheckMatch(schema::MetaGraphT *graph, const std::shared_ptr<PatternOp> &patternOp);

  void MergeNodeAttrFromPost(std::unique_ptr<schema::CNodeT> &dstOp, std::unique_ptr<schema::CNodeT> &postOp,
                             size_t dstOpOutIdx = 0);

  STATUS Fuse(schema::MetaGraphT *graph);

 protected:
  std::vector<FusionPattern *> patterns;
  std::map<std::string, std::vector<std::shared_ptr<PatternOp>>> matchedPaths;
  // {name of pattern, vector<{name of pattern node, path}>}
  std::map<std::string, std::vector<std::unordered_map<std::string, std::shared_ptr<Path>>>> mapedMatchedPaths;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_FUSION_PASS_H
