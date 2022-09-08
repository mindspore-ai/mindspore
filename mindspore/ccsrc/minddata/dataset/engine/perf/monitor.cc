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
#include "minddata/dataset/engine/perf/monitor.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"

namespace mindspore {
namespace dataset {

Monitor::Monitor(ProfilingManager *profiling_manager) : Monitor(profiling_manager, GlobalContext::config_manager()) {}

Monitor::Monitor(ProfilingManager *profiling_manager, const std::shared_ptr<ConfigManager> &cfg)
    : profiling_manager_(profiling_manager), sampling_interval_(cfg->monitor_sampling_interval()) {
  if (profiling_manager_ != nullptr) {
    tree_ = profiling_manager_->tree_;
  }
}

Monitor::~Monitor() {
  // just set the pointer to nullptr, it's not be released here
  if (profiling_manager_) {
    profiling_manager_ = nullptr;
  }

  if (tree_) {
    tree_ = nullptr;
  }
}

Status Monitor::operator()() {
  // Register this thread with TaskManager to receive proper interrupt signal.
  TaskManager::FindMe()->Post();
  std::unique_lock<std::mutex> _lock(mux_);

  // Keep sampling if
  // 1) Monitor Task is not interrupted by TaskManager AND
  // 2) Iterator has not received EOF

  while (!this_thread::is_interrupted() && !(tree_->isFinished())) {
    if (tree_->IsEpochEnd()) {
      tree_->SetExecuting();
    }
    for (auto &node : profiling_manager_->GetSamplingNodes()) {
      RETURN_IF_NOT_OK(node.second->Sample());
    }
    RETURN_IF_NOT_OK(cv_.WaitFor(&_lock, sampling_interval_));
  }
  MS_LOG(INFO) << "Monitor Thread terminating...";
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
