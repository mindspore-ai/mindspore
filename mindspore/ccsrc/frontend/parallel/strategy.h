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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/status.h"

namespace mindspore {
namespace parallel {
#define MIN_SLICE_NUM 1

using Dimensions = Shape;
using Strategys = std::vector<Dimensions>;
class Strategy;
using StrategyPtr = std::shared_ptr<Strategy>;

class Strategy {
 public:
  Strategy(int64_t stage, Strategys inputs)
      : stage_(stage), inputs_(std::move(inputs)), internal_size_(0), internal_stragies_() {}

  Strategy(const Strategy &another_stra) : stage_(another_stra.GetInputStage()) {
    inputs_ = another_stra.GetInputDim();
    internal_size_ = another_stra.GetInternalSize();
    if (internal_size_ != 0) {
      internal_stragies_ = another_stra.GetInternalStrategies();
    } else {
      internal_stragies_ = {};
    }
  }

  ~Strategy() = default;
  size_t GetInputNumber() const { return inputs_.size(); }
  Strategys GetInputDim() const { return inputs_; }
  int64_t GetInputStage() const { return stage_; }
  void ExpandInputDimFromOneToTwo() {
    if (inputs_.size() == 1) {
      inputs_.push_back(inputs_[0]);
    }
  }
  void ResetInputs(const Strategys &input) { inputs_ = input; }
  std::vector<StrategyPtr> GetInternalStrategies() const { return internal_stragies_; }
  size_t GetInternalSize() const { return internal_size_; }

  // TODO(Xiaoda): need fix for adapting 'CoverStrategy'
  bool IsEqual(const StrategyPtr &another_stra) {
    if (another_stra == nullptr) {
      return false;
    }
    if ((stage_ != another_stra->GetInputStage()) || (inputs_ != another_stra->GetInputDim())) {
      return false;
    }
    return true;
  }

  // Include 'another_stra' into this strategy
  void CoverStrategy(const StrategyPtr &another_stra) {
    internal_stragies_.push_back(another_stra);
    internal_size_++;
  }

 private:
  const int64_t stage_;

  // The size of Dimensions must equal to inputs_ tensor dimension.
  Strategys inputs_;
  size_t internal_size_ = 0;
  std::vector<StrategyPtr> internal_stragies_;
};

inline StrategyPtr NewStrategy(const int64_t stage, const Strategys &inputs) {
  return std::make_shared<Strategy>(stage, inputs);
}
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_H_
