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

#ifndef MINDSPORE_CCSRC_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_
#define MINDSPORE_CCSRC_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_

#include <string>
#include <unordered_map>
#include "parallel/ops_info/ops_utils.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
constexpr char DEFAULT_CHECKPOINT_PATH[] = "./strategys.ckpt";

using StrategyMap = std::unordered_map<std::string, StrategyPtr>;
class StrategyCheckpoint {
 public:
  StrategyCheckpoint() : path_(DEFAULT_CHECKPOINT_PATH), current_train_time_(1) {
    train_times_ = 1;
    checkpoint_on_ = false;
    const char *train_times_str = std::getenv("PARALLEL_TRAIN_TIMES");
    if (train_times_str != nullptr && std::stoi(train_times_str) > 0) {
      train_times_ = std::stoi(train_times_str);
    }
    const char *checkpoint_on_str = std::getenv("PARALLEL_CHECKPOINT_ON");
    if (checkpoint_on_str != nullptr) {
      checkpoint_on_ = (std::string(checkpoint_on_str) == "on");
    }
  }
  ~StrategyCheckpoint() = default;
  bool CheckPointExit() const;
  Status RemoveCheckPoint() const;
  Status Load(StrategyMap *strategy_map);
  Status Save(const StrategyMap &strategy_map);

  static StrategyCheckpoint &GetInstance();
  int32_t GetTrainTimes() const { return train_times_; }
  int32_t GetCurrentTrainTime() const { return current_train_time_; }
  bool CheckPointOn() const { return checkpoint_on_; }

 private:
  std::string path_;
  bool checkpoint_on_;
  // total train times for a train, get from Environmental variable:TRAIN_TIME, please export it
  int32_t train_times_;
  int32_t current_train_time_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_
