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

#include <vector>
#include <string>
#include <unordered_map>
#ifndef MINDSPORE_LITE_SRC_PASS_PARALLEL_SPLIT_STRATEGY_H_
#define MINDSPORE_LITE_SRC_PASS_PARALLEL_SPLIT_STRATEGY_H_

namespace mindspore {
namespace opt {
constexpr auto OP = "op";
constexpr auto STRATEGY = "strategy";
constexpr auto DEV_TYPE = "dev_type";

constexpr auto PARALLEL_NAME_SUFFIX = "_parallel";

constexpr auto kSplitOp = "Conv2D";

const std::vector<int64_t> kSplitRatio = {1, 1};

const std::vector<int64_t> kSplitDefaultRatio = {0, 0};

const std::vector<std::string> kSplitDevTypes = {"CPU", "GPU"};

using Strategys = std::vector<std::vector<std::vector<int64_t>>>;

constexpr auto kDeviceTypeNone = -1;
// strategy format is NHWC-KHWC
constexpr int32_t kAxisN = 0;
constexpr int32_t kAxisCIn = 3;
constexpr int32_t kAxisCOut = 0;
constexpr int32_t kAxisH = 1;
constexpr int32_t kAxisW = 2;

constexpr auto kIndexH = 0;
constexpr auto kIndexW = 1;

constexpr auto kPadUp = 0;
constexpr auto kPadDown = 1;
constexpr auto kPadLeft = 2;
constexpr auto kPadRight = 3;

enum SplitMode {
  NoSplit = 0,
  SplitN = 1,
  SplitH = 2,
  SplitCIN = 3,
  SplitCOUT = 4,
};

struct SplitStrategy {
  Strategys strategys;
  std::vector<std::string> dev_types;
  size_t dev_num;
};

std::unordered_map<std::string, opt::SplitStrategy> ParserSplitStrategy(SplitMode parallel_mode);

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PASS_PARALLEL_SPLIT_STRATEGY_H_
