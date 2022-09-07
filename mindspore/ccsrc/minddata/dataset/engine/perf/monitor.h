/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <mutex>
#include <unordered_map>
#include <vector>
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/util/cond_var.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class ExecutionTree;
class ProfilingManager;
class ConfigManager;

class Monitor {
 public:
  // Monitor object constructor
  explicit Monitor(ProfilingManager *profiling_manager);

  ~Monitor();

  // Functor for Perf Monitor main loop.
  // This function will be the entry point of mindspore::Dataset::Task
  Status operator()();

  // Setter for execution tree pointer
  void SetTree(ExecutionTree *tree) { tree_ = tree; }

 private:
  // private constructor
  Monitor(ProfilingManager *profiling_manager, const std::shared_ptr<ConfigManager> &cfg);

  ProfilingManager *profiling_manager_;
  int64_t sampling_interval_;
  ExecutionTree *tree_;
  std::mutex mux_;
  CondVar cv_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_MONITOR_H
