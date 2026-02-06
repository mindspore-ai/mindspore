/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_INCLUDE_COMMON_UTILS_CONFIG_MANAGER_H_
#define MINDSPORE_CORE_INCLUDE_COMMON_UTILS_CONFIG_MANAGER_H_

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <sstream>

#include "utils/overload.h"
#include "mindapi/base/macros.h"

namespace mindspore {

enum ParallelStrategy {
  ONE_DEVICE = 0,
  DISTRIBUTION,
};

enum DatasetMode { DS_NORMAL_MODE = 0, DS_SINK_MODE };

class MS_CORE_API ConfigManager {
 public:
  ConfigManager(const ConfigManager &) = delete;
  ConfigManager &operator=(const ConfigManager &) = delete;
  static ConfigManager &GetInstance() noexcept;

  ParallelStrategy parallel_strategy() const { return parallel_strategy_; }
  void set_parallel_strategy(ParallelStrategy strategy) { parallel_strategy_ = strategy; }

  DatasetMode dataset_mode() const { return dataset_mode_; }
  void set_dataset_mode(DatasetMode mode) { dataset_mode_ = mode; }
  int64_t iter_num() const {
    if (dataset_mode_ == DS_NORMAL_MODE) {
      return 1;
    }
    return iter_num_;
  }

  void set_iter_num(const std::string &queue_name, const int64_t num) {
    queue_name_ = queue_name;
    iter_num_ = num;
    queue_info_map[queue_name_] = num;
  }

  std::string dataset_phase() const { return dataset_phase_; }
  void set_dataset_phase(const std::string &phase) { dataset_phase_ = phase; }

  static void SetDatasetModeConfig(const std::string &mode);

  void ResetConfig() noexcept;

  void ResetIterNum() noexcept;

  void ResetQueue(const std::string &queue_name) noexcept;
  std::string QueueName() const { return queue_name_; }

 private:
  ConfigManager() = default;
  ~ConfigManager() = default;

  ParallelStrategy parallel_strategy_{ONE_DEVICE};
  DatasetMode dataset_mode_{DS_NORMAL_MODE};
  int64_t iter_num_{1};
  std::string queue_name_{""};
  // now only save iter_num_ in the map
  std::map<std::string, int64_t> queue_info_map;
  std::string dataset_phase_{""};
};

}  // namespace mindspore

#endif  // MINDSPORE_CORE_INCLUDE_COMMON_UTILS_CONFIG_MANAGER_H_
