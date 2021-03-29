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
#include "minddata/dataset/engine/perf/monitor.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {

Monitor::Monitor(ExecutionTree *tree) : tree_(tree) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  sampling_interval_ = cfg->monitor_sampling_interval();
  max_samples_ = 0;
  cur_row_ = 0;
}
Status Monitor::operator()() {
  // Register this thread with TaskManager to receive proper interrupt signal.
  TaskManager::FindMe()->Post();
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  cfg->set_profiler_file_status(false);

  // Keep sampling if
  // 1) Monitor Task is not interrupted by TaskManager AND
  // 2) Iterator has not received EOF
  while (!this_thread::is_interrupted() && !(tree_->isFinished()) && !(cfg->stop_profiler_status())) {
    if (tree_->IsEpochEnd()) {
      RETURN_IF_NOT_OK(tree_->GetProfilingManager()->SaveProfilingData());
      tree_->SetExecuting();
    }
    for (auto &node : tree_->GetProfilingManager()->GetSamplingNodes()) {
      RETURN_IF_NOT_OK(node.second->Sample());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sampling_interval_));
  }

  // Output all profiling data upon request.
  RETURN_IF_NOT_OK(tree_->GetProfilingManager()->Analyze());
  RETURN_IF_NOT_OK(tree_->GetProfilingManager()->SaveProfilingData());
  RETURN_IF_NOT_OK(tree_->GetProfilingManager()->ChangeFileMode());

  cfg->set_profiler_file_status(true);
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
