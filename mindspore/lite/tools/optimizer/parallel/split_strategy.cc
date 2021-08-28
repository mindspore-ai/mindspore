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
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {

int64_t ApproximateFLOPs(const std::vector<int64_t> &strides, const std::vector<int64_t> &input_shape,
                         const std::vector<int64_t> &weight_shape) {
  MS_CHECK_GT(strides.size(), 1, 0);
  MS_CHECK_GT(input_shape.size(), kInputSize2, 0);
  MS_CHECK_GT(weight_shape.size(), kInputSize1, 0);
  int64_t input_h = input_shape.at(kShapeH);
  int64_t input_w = input_shape.at(kShapeW);
  int64_t in_c = input_shape.at(kShapeC);
  int64_t out_c = weight_shape.at(kShapeN);
  int64_t k_h = weight_shape.at(kShapeH);
  int64_t k_w = weight_shape.at(kShapeW);
  int64_t stride_h = strides.at(kIndexH);
  int64_t stride_w = strides.at(kIndexW);
  if (stride_h == 0 || stride_w == 0) {
    MS_LOG(ERROR) << "divisor is zero.";
    return 0;
  }
  return (input_h / stride_h) * (input_w / stride_w) * in_c * k_h * k_w * out_c / kPerFlops;
}

std::unordered_map<std::string, opt::SplitStrategy> ParserSplitStrategy(const std::vector<int64_t> &split_ratio,
                                                                        const std::vector<std::string> &split_device,
                                                                        SplitMode split_mode) {
  std::unordered_map<std::string, opt::SplitStrategy> split_strategys;
  if (split_ratio.empty() || kSplitDefaultRatio.empty() || split_device.empty()) {
    return split_strategys;
  }
  if (split_ratio.size() != kSplitDevTypes.size()) {
    return split_strategys;
  }
  std::vector<std::vector<int64_t>> split_feature_map;
  std::vector<std::vector<int64_t>> split_weight;
  switch (split_mode) {
    case SplitN:
      split_feature_map = {split_ratio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      split_weight = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      break;
    case SplitH:
      split_feature_map = {kSplitDefaultRatio, split_ratio, kSplitDefaultRatio, kSplitDefaultRatio};
      split_weight = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      break;
    case SplitCIN:
      split_feature_map = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, split_ratio};
      split_weight = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, split_ratio};
      break;
    case SplitCOUT:
      split_feature_map = {kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      split_weight = {split_ratio, kSplitDefaultRatio, kSplitDefaultRatio, kSplitDefaultRatio};
      break;
    default:
      return split_strategys;
  }
  opt::Strategys strategys = {split_feature_map, split_weight};
  for (const auto &supported_parallel_op : kParallelOpNames) {
    split_strategys[supported_parallel_op.second] = {strategys, kSplitDevTypes, kSplitDevTypes.size(), split_mode};
  }

  return split_strategys;
}
}  // namespace opt
}  // namespace mindspore
