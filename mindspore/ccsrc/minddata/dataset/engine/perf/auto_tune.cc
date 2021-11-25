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

#include "minddata/dataset/engine/perf/auto_tune.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
AutoTune::AutoTune(TreeAdapter *tree_adap, ProfilingManager *profiling_mgr)
    : tree_adapter_(tree_adap), profiling_manager_(profiling_mgr), leaf_op_id_(-1), cur_epoch_(1) {
  tree_modifier_ = std::make_unique<TreeModifier>(tree_adapter_);
  max_workers_ = GlobalContext::config_manager()->num_cpu_threads();
}
Status AutoTune::Main() {
  TaskManager::FindMe()->Post();
  MS_LOG(INFO) << "AutoTune thread has started.";
  std::unique_lock<std::mutex> _lock(mux_);
  cur_epoch_ = 1;
  Status rc;
  while (!this_thread::is_interrupted() && !(tree_adapter_->tree_->isFinished())) {
    rc = RunIteration();
    if (rc.IsError()) {
      MS_LOG(ERROR) << "AutoTune failed and will exit with the following error: " << rc;
      break;
    }
    rc = cv_.WaitFor(&_lock, GlobalContext::config_manager()->autotune_interval());
    // the thread may be interrupted for tree termination when waiting (we should not report error in this case)
    if (rc.IsError() && rc != StatusCode::kMDInterrupted) {
      return rc;
    }
  }
  RETURN_IF_NOT_OK(profiling_manager_->Stop());
  MS_LOG(INFO) << "AutoTune thread is finished.";
  return Status::OK();
}
Status AutoTune::LaunchThread() {
  MS_LOG(INFO) << "Launching AutoTune thread";
  RETURN_IF_NOT_OK(CollectOpsInfo());
  RETURN_IF_NOT_OK(cv_.Register(tree_adapter_->AllTasks()->GetIntrpService()));
  RETURN_IF_NOT_OK(tree_adapter_->AllTasks()->CreateAsyncTask("AutoTune Thread", std::bind(&AutoTune::Main, this)));
  return Status::OK();
}
Status AutoTune::CollectOpsInfo() {
  ExecutionTree *tree = tree_adapter_->tree_.get();
  RETURN_UNEXPECTED_IF_NULL(tree);
  for (auto itr = tree->begin(); itr != tree->end(); ++itr) {
    ops_[itr->id()] = itr.get();
    // get all parallel ops (num_workers>0) except leaf nodes (no children)
    if (itr->NumWorkers() > 0) {
      parallel_ops_ids_.push_back(itr->id());
    }
  }
  // sort parallel ops in reverse order of IDs (i.e., bottommost op is first)
  std::sort(parallel_ops_ids_.begin(), parallel_ops_ids_.end(), std::greater<>());
  leaf_op_id_ = ops_.size() - 1;

  if (parallel_ops_ids_.size() != 0) {
    CHECK_FAIL_RETURN_UNEXPECTED(parallel_ops_ids_[parallel_ops_ids_.size() - 1] != 0,
                                 "Non-sink pipeline, root node is a ParallelOp. AutoTune is not supported.");
  }

  return Status::OK();
}

Status AutoTune::GetOpConnectorCapacity(int32_t op_id, int64_t *capacity) {
  auto item = ops_.find(op_id);
  CHECK_FAIL_RETURN_UNEXPECTED(item != ops_.end(), "Invalid Operator ID");
  *capacity = item->second->ConnectorCapacity();
  return Status::OK();
}

Status AutoTune::GetOpsCpuUtil(std::map<int32_t, double> *ops_cpu_util) {
  // loop over all itr keys and get avg cpu usage
  for (auto itr = ops_.begin(); itr != ops_.end(); ++itr) {
    std::vector<uint16_t> sys_util;
    std::vector<uint16_t> user_util;
#ifndef ENABLE_ANDROID
    RETURN_IF_NOT_OK(profiling_manager_->GetSysCpuUtilByEpoch(itr->first, cur_epoch_, &sys_util));
    RETURN_IF_NOT_OK(profiling_manager_->GetUserCpuUtilByEpoch(itr->first, cur_epoch_, &user_util));
#endif
    double sys_cpu_util = Mean(sys_util);
    double user_cpu_util = Mean(user_util);
    (*ops_cpu_util)[itr->first] = sys_cpu_util + user_cpu_util;
  }
  return Status::OK();
}
Status AutoTune::GetOpsQueueUtil(std::map<int32_t, double> *ops_queue_util) {
  // loop over all itr keys in the ops_ and get output_queue usage
  for (auto itr = ops_.begin(); itr != ops_.end(); ++itr) {
    std::vector<int32_t> sizes;
    RETURN_IF_NOT_OK(profiling_manager_->GetConnectorSizeByEpoch(itr->first, cur_epoch_, &sizes));
    double avg_size = Mean(sizes);
    int64_t capacity = itr->second->ConnectorCapacity();
    CHECK_FAIL_RETURN_UNEXPECTED(capacity != 0, "Capacity of connector should not be 0");
    (*ops_queue_util)[itr->first] = avg_size / capacity;
  }
  return Status::OK();
}
Status AutoTune::GetOpsNumWorker(std::map<int32_t, int32_t> *ops_num_workers) {
  for (auto itr = ops_.begin(); itr != ops_.end(); ++itr) {
    (*ops_num_workers)[itr->first] = itr->second->NumWorkers();
  }
  return Status::OK();
}
bool AutoTune::IsSink() {
  std::shared_ptr<Tracing> node;
  return profiling_manager_->GetTracingNode(kDeviceQueueTracingName, &node).IsOk();
}
template <typename T>
double AutoTune::Mean(std::vector<T> items) {
  if (items.size() == 0) {
    return 0;
  }
  return std::accumulate(items.begin(), items.end(), 0.0) / static_cast<double>(items.size());
}

Status AutoTune::RunIteration() {
  // Run every epoch
  if ((profiling_manager_->GetNumOfProfiledEpochs()) >= cur_epoch_) {
    MS_LOG(INFO) << "Run AutoTune at epoch #" << cur_epoch_;
    RETURN_IF_NOT_OK(RunIterationEpoch());
    ++cur_epoch_;
  }
  return Status::OK();
}
Status AutoTune::RecordPipelineTime() {
  std::vector<int32_t> times;
  RETURN_IF_NOT_OK(profiling_manager_->GetPipelineTimeByEpoch(cur_epoch_, &times));
  double avg_time = Mean(times);
  avg_pipeline_times_.push_back(avg_time);
  MS_LOG(INFO) << "Epoch #" << cur_epoch_ << ", Average Pipeline time is " << avg_time
               << " ms. The avg pipeline time for all epochs is " << Mean(avg_pipeline_times_) << "ms";
  return Status::OK();
}
Status AutoTune::RunIterationEpoch() {
  RETURN_IF_NOT_OK(RecordPipelineTime());
  bool isBottleneck = false;
  RETURN_IF_NOT_OK(IsDSaBottleneck(&isBottleneck));
  if (isBottleneck) {
    RETURN_IF_NOT_OK(Analyse());
  }
  return Status::OK();
}

Status AutoTune::IsDSaBottleneck(bool *isBottleneck) {
  std::vector<int32_t> sizes;
  RETURN_IF_NOT_OK(profiling_manager_->GetConnectorSizeByEpoch(cur_epoch_, &sizes));
  double avg_size = Mean(sizes);
  std::vector<int32_t> capacities;
  RETURN_IF_NOT_OK(profiling_manager_->GetConnectorCapacityByEpoch(cur_epoch_, &capacities));
  double avg_capacity = Mean(capacities);
  CHECK_FAIL_RETURN_UNEXPECTED(avg_capacity != 0, "Capacities of connectors should not be 0");
  double usage_avg_last = (avg_size / avg_capacity);
  float empty_freq = 0;
  RETURN_IF_NOT_OK(profiling_manager_->GetEmptyQueueFrequencyByEpoch(cur_epoch_, &empty_freq));

  // Reporting values
  MS_LOG(INFO) << "Epoch #" << cur_epoch_ << ", Device Connector Size: " << avg_size
               << ", Connector Capacity: " << avg_capacity << ", Utilization: " << (usage_avg_last * TO_PERCENT) << "%"
               << ", Empty Freq: " << (empty_freq * TO_PERCENT) << "% ";
  // Decision
  if (usage_avg_last < DEVICE_CONNECTOR_UTIL_THRESHOLD) {
    MS_LOG(WARNING) << "Utilization: " << (usage_avg_last * TO_PERCENT) << "% < "
                    << (DEVICE_CONNECTOR_UTIL_THRESHOLD * TO_PERCENT)
                    << "% threshold, dataset pipeline performance needs tuning.";
    *isBottleneck = true;
  } else {
    MS_LOG(INFO) << "Utilization: " << (usage_avg_last * TO_PERCENT) << "% > "
                 << (DEVICE_CONNECTOR_UTIL_THRESHOLD * TO_PERCENT)
                 << "% threshold, dataset pipeline performance is OK.";
    *isBottleneck = false;
  }
  return Status::OK();
}

Status AutoTune::RequestNumWorkerChange(int32_t op_id, int32_t old_workers, int32_t new_workers) {
  new_workers = std::min(new_workers, max_workers_);
  new_workers = std::max(new_workers, MIN_NUM_WORKERS);
  RETURN_IF_NOT_OK(tree_modifier_->AddChangeRequest(op_id, std::make_shared<ChangeNumWorkersRequest>(new_workers)));
  MS_LOG(WARNING) << "Added request to change number of workers of Operator: " << ops_[op_id]->NameWithID()
                  << " New value: " << new_workers << " Old value: " << old_workers;
  return Status::OK();
}
Status AutoTune::RequestConnectorCapacityChange(int32_t op_id, int32_t old_size, int32_t new_size) {
  new_size = std::min(new_size, MAX_QUEUE_SIZE);
  new_size = std::max(new_size, MIN_QUEUE_SIZE);

  RETURN_IF_NOT_OK(tree_modifier_->AddChangeRequest(op_id, std::make_shared<ResizeConnectorRequest>(new_size)));
  MS_LOG(WARNING) << "Added request to change Connector capacity of Operator: " << ops_[op_id]->NameWithID()
                  << " New value: " << new_size << " Old value: " << old_size;
  return Status::OK();
}

Status AutoTune::Analyse() {
  // collect stats
  std::map<int32_t, int32_t> ops_num_workers;
  RETURN_IF_NOT_OK(GetOpsNumWorker(&ops_num_workers));
  std::map<int32_t, double> ops_queue_util;
  RETURN_IF_NOT_OK(GetOpsQueueUtil(&ops_queue_util));
  std::map<int32_t, double> ops_cpu_util;
  RETURN_IF_NOT_OK(GetOpsCpuUtil(&ops_cpu_util));

  // check parallel ops in loop
  for (const auto &op_id : parallel_ops_ids_) {
    // op specifics
    double output_queue_util = ops_queue_util[op_id];
    double input_queue_util = 1;  // assume that leaf op has 100% input queue util
    if (op_id + 1 < ops_.size()) {
      input_queue_util = ops_queue_util[op_id + 1];
    }
    double cpu_util = ops_cpu_util[op_id];
    int32_t num_workers = ops_num_workers[op_id];
    CHECK_FAIL_RETURN_UNEXPECTED(num_workers != 0, "ParallelOp with num_workers=0");
    // derived metrics
    double queue_diff = input_queue_util - output_queue_util;
    int64_t queue_capacity = 0;
    RETURN_IF_NOT_OK(GetOpConnectorCapacity(op_id, &queue_capacity));

    MS_LOG(DEBUG) << "Op (" << ops_[op_id]->NameWithID() << ") CPU=" << cpu_util / num_workers
                  << ", in=" << input_queue_util << "out=" << output_queue_util;
    // map decisions - queue
    if (queue_diff > INPUT_OUTPUT_QUEUE_DIFF_THRESHOLD) {
      MS_LOG(WARNING) << "Op (" << ops_[op_id]->NameWithID()
                      << ") is slow, input connector utilization=" << input_queue_util
                      << ", output connector utilization=" << output_queue_util << ", diff= " << queue_diff << " > "
                      << INPUT_OUTPUT_QUEUE_DIFF_THRESHOLD << " threshold.";
      RETURN_IF_NOT_OK(RequestNumWorkerChange(op_id, num_workers, num_workers + INCREMENT_WORKER));
    } else if ((cpu_util / num_workers) > MAP_OP_WORKER_HIGH_THRESHOLD) {
      MS_LOG(WARNING) << "Op (" << ops_[op_id]->NameWithID() << ") getting high average worker cpu utilization "
                      << (cpu_util / num_workers) << "% > " << MAP_OP_WORKER_HIGH_THRESHOLD << "% threshold.";
      RETURN_IF_NOT_OK(RequestNumWorkerChange(op_id, num_workers, num_workers + INCREMENT_WORKER));
    }
    if ((cpu_util / num_workers) < MAP_OP_WORKER_LOW_THRESHOLD &&
        ((input_queue_util < INPUT_QUEUE_LOW) || (-1 * queue_diff > INPUT_OUTPUT_QUEUE_DIFF_THRESHOLD))) {
      MS_LOG(WARNING) << "Op (" << ops_[op_id]->NameWithID() << ") getting low average worker cpu utilization "
                      << (cpu_util / num_workers) << "% < " << MAP_OP_WORKER_LOW_THRESHOLD << "% threshold.";
      RETURN_IF_NOT_OK(RequestConnectorCapacityChange(op_id, queue_capacity, queue_capacity + INCREMENT_QUEUE_SIZE));
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
