/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/parallel/split_strategy.h"
#include <vector>
#include <unordered_map>
#include <string>

namespace mindspore {
namespace opt {
std::unordered_map<std::string, opt::SplitStrategy> ParserSplitStrategy() {
  std::unordered_map<std::string, opt::SplitStrategy> split_strategys;
  if (kSplitRatio.empty() || kSplitDefaultRatio.empty() || kSplitDevTypes.empty()) {
    return split_strategys;
  }
  if (kSplitRatio.size() != kSplitDevTypes.size()) {
    return split_strategys;
  }
  std::vector<std::vector<int64_t>> split_feature_map;
  std::vector<std::vector<int64_t>> split_weight;
  switch (kParallelMode) {
    case SplitN:
      split_feature_map = {kSplitRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      split_weight = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      break;
    case SplitH:
      split_feature_map = {kSplitDefaultRatio, kSplitRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      split_weight = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      break;
    case SplitCIN:
      split_feature_map = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitRatio};
      split_weight = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitRatio};
      break;
    case SplitCOUT:
      split_feature_map = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      split_weight = {kSplitRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      break;
    default:
      return split_strategys;
  }
  opt::Strategys strategys = {split_feature_map, split_weight};

  for (const auto &supported_parallel_op : kParallelOpNames) {
    split_strategys[supported_parallel_op.second] = {strategys, kSplitDevTypes, kSplitDevTypes.size(), kParallelMode};
  }

  return split_strategys;
}
}  // namespace opt
}  // namespace mindspore
