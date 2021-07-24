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
}
Status Monitor::operator()() {
  // Register this thread with TaskManager to receive proper interrupt signal.
  TaskManager::FindMe()->Post();
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  cfg->set_profiler_file_status(false);

  // Keep sampling if
  // 1) Monitor Task is not interrupted by TaskManager AND
  // 2) Iterator has not received EOF

  // this will trigger a save on 2min, 4min, 8min, 16min ... mark on top of the save per_epoch
  // The idea is whenever training is interrupted, you will get at least half of the sampling data during training
  constexpr int64_t num_ms_in_two_minutes = 120000;
  int64_t save_interval = 1 + (num_ms_in_two_minutes / sampling_interval_);
  int64_t loop_cnt = 1;
  constexpr int64_t geometric_series_ratio = 2;
  while (!this_thread::is_interrupted() && !(tree_->isFinished()) && !(cfg->stop_profiler_status())) {
    if (tree_->IsEpochEnd()) {
      RETURN_IF_NOT_OK(tree_->GetProfilingManager()->SaveProfilingData());
      tree_->SetExecuting();
    } else if (loop_cnt % save_interval == 0) {
      RETURN_IF_NOT_OK(tree_->GetProfilingManager()->SaveProfilingData());
    }
    for (auto &node : tree_->GetProfilingManager()->GetSamplingNodes()) {
      RETURN_IF_NOT_OK(node.second->Sample());
    }
    if (loop_cnt % save_interval == 0) save_interval *= geometric_series_ratio;
    loop_cnt += 1;
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
