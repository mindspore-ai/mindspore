/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_STRATEGY_H_
#define MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_STRATEGY_H_

#include <cstdint>
#include <list>
#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "optimizer/parallel/status.h"

namespace mindspore {
namespace parallel {

#define MIN_SLICE_NUM 1

using Dimensions = std::vector<int32_t>;

class Strategy;
using StrategyPtr = std::shared_ptr<Strategy>;

class Strategy {
 public:
  Strategy(int32_t stage, std::vector<Dimensions> inputs) : stage_(stage), inputs_(std::move(inputs)) {}
  ~Strategy() = default;
  size_t GetInputNumber() const { return inputs_.size(); }
  std::vector<Dimensions> GetInputDim() const { return inputs_; }
  int32_t GetInputStage() const { return stage_; }
  void ExpandInputDimFromOneToTwo() {
    if (inputs_.size() == 1) {
      inputs_.push_back(inputs_[0]);
    }
  }
  void ResetInputs(const std::vector<Dimensions>& input) { inputs_ = input; }

 private:
  const int32_t stage_;

  // The size of Dimensions must equal to inputs_ tensor dimension.
  std::vector<Dimensions> inputs_;
};

inline StrategyPtr NewStrategy(const int32_t stage, const std::vector<Dimensions>& inputs) {
  return std::make_shared<Strategy>(stage, inputs);
}
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_STRATEGY_H_
