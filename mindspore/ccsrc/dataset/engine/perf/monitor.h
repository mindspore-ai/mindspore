/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MONITOR_H
#define MINDSPORE_MONITOR_H

#include <memory>
#include <unordered_map>
#include <vector>
#include "dataset/util/status.h"
#include "dataset/engine/perf/profiling.h"

namespace mindspore {
namespace dataset {
class ExecutionTree;
class Monitor {
 public:
  // Monitor object constructor
  explicit Monitor(ExecutionTree *tree);

  Monitor() = default;

  ~Monitor() = default;

  // Functor for Perf Monitor main loop.
  // This function will be the entry point of Mindspore::Dataset::Task
  Status operator()();

  int64_t GetSamplingInterval() { return sampling_interval_; }

 private:
  int64_t cur_row_;
  int64_t max_samples_;
  int64_t sampling_interval_;
  ExecutionTree *tree_;
  std::vector<std::shared_ptr<Sampling>> sampling_list_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_MONITOR_H
