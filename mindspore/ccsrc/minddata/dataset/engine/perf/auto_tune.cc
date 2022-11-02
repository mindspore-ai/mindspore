/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/source/nonmappable_leaf_op.h"
#include "minddata/dataset/engine/serdes.h"
#endif
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
AutoTune::AutoTune(TreeAdapter *tree_adap, ProfilingManager *profiling_mgr)
    : tree_adapter_(tree_adap),
      profiling_manager_(profiling_mgr),
      tree_modifier_(std::make_unique<TreeModifier>(tree_adapter_)),
      leaf_op_id_(-1),
      cur_epoch_running_(1),
      last_epoch_autotuned_(0),
      cur_step_running_(1),
      last_step_autotuned_(0),
      mode_(0),
      step_gap_(GlobalContext::config_manager()->autotune_interval()),
      skip_flag_(true),
      AT_phase_(AutoTunePhase::kAutoTunePhaseTime),
      AT_change_(false),
      phase_1_best_time_(-1),
      phase_1_no_improve_count_(0),
      count_down_(0),
      phase_3_state_(AutoTuneMemPhase::kAutoTuneMemInit),
      phase_3_ID_(0),
      avg_batch_time(0.0),
      phase_3_prev_avg_(0.0),
      save_autoconfig_(GlobalContext::config_manager()->save_autoconfig()) {
  max_workers_ = GlobalContext::config_manager()->num_cpu_threads();
  autotune_json_filepath_ = GlobalContext::config_manager()->get_autotune_json_filepath();
}

Status AutoTune::Main() {
  TaskManager::FindMe()->Post();
  MS_LOG(INFO) << "Dataset AutoTune thread has started.";
  if (step_gap_ != 0) {
    mode_ = AutoTuneMode::kAutoTuneModeStep;
  } else {
    mode_ = AutoTuneMode::kAutoTuneModeEpoch;
  }
  const bool nodes_offloaded = !tree_adapter_->GetOffloadJson().empty();
  if (nodes_offloaded) {
    // When nodes are offloaded they are removed from the optimized IR tree.
    // Serializing the optimized IR Tree and then deserializing will not work.
    MS_LOG(WARNING) << "Some nodes have been offloaded. AutoTune is unable to write the autotune configuration to "
                       "disk. Disable offload to prevent this from happening.";
  }
  bool output_final_config = save_autoconfig_ && !nodes_offloaded;
  bool output_intermediate_config = save_intermediate_autoconfig_ && output_final_config;
  RETURN_IF_NOT_OK(ATMainLoop(output_intermediate_config));
  RETURN_IF_NOT_OK(profiling_manager_->Stop());
  PostMainLogging();
#ifndef ENABLE_ANDROID
  if (output_final_config &&
      (SaveAutotuneConfig(autotune_json_filepath_ + "_" + profiling_manager_->GetRankID() + ".json").IsError())) {
    MS_LOG(WARNING) << "Failed to write the final autotune configuration to disk";
  }
#endif
  return Status::OK();
}

Status AutoTune::ATMainLoop(bool output_intermediate_config) {
  std::unique_lock<std::mutex> _lock(mux_);
  int loop_cnt = 0;
  Status rc;
  while (!this_thread::is_interrupted() && !(tree_adapter_->tree_->isFinished())) {
#ifndef ENABLE_ANDROID
    auto last_epoch = cur_epoch_running_;
    auto last_step = cur_step_running_;
#endif
    RETURN_IF_NOT_OK(UpdateCurrentRunInfo());
    if (!WarmupSkipCheck()) {
      // Warm up complete - AT normally
      if (mode_ == AutoTuneMode::kAutoTuneModeEpoch) {
        rc = RunIterationEpoch();
      } else if (mode_ == AutoTuneMode::kAutoTuneModeStep) {
        rc = RunIterationStep();
      }
      if (rc.IsError()) {
        if (rc.StatusCode() != StatusCode::kMDInterrupted) {
          MS_LOG(ERROR) << "Dataset AutoTune failed and will exit with the following error: " << rc;
        }
        RETURN_IF_NOT_OK(profiling_manager_->Stop());
        break;
      }
#ifndef ENABLE_ANDROID
      if (last_epoch != cur_epoch_running_ || last_step != cur_step_running_) {
        if (output_intermediate_config &&
            (SaveAutotuneConfig(autotune_json_filepath_ + "_" + profiling_manager_->GetRankID() + "_" +
                                std::to_string(loop_cnt) + ".json")
               .IsError())) {
          MS_LOG(WARNING) << "Failed to write the current iteration autotune configuration to disk";
        }
        ++loop_cnt;
      }
#endif
      if (AT_phase_ == AutoTunePhase::kAutoTuneEnd) {
        MS_LOG(INFO) << "Dataset AutoTune stop, optimization complete.";
        break;
      }
    }
    rc = cv_.WaitFor(&_lock, GlobalContext::config_manager()->monitor_sampling_interval());
    // The thread may be interrupted for tree termination when waiting (we should not report error in this case)
    if (rc.IsError() && rc != StatusCode::kMDInterrupted) {
      return rc;
    }
  }
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status AutoTune::SaveAutotuneConfig(const std::string &file_name) {
  Path jsonpath(file_name);

  std::string parent_dir = jsonpath.ParentPath();
  if (access(parent_dir.c_str(), R_OK) == -1) {
    std::string err_msg = "AutoTune has no access to specified path: " + parent_dir + ", check permission.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (jsonpath.Exists()) {
    std::string err_msg = "File: <" + file_name +
                          "> already exists. File will be overwritten with the AutoTuned data pipeline configuration.";
    MS_LOG(WARNING) << err_msg;
  }

  RETURN_IF_NOT_OK(SetAutotuneConfigJson());
  // The Execution Tree is built by visiting the optimized IR Tree in DFS order.
  // So we visit the optimized IR tree in DFS order and try to match each IR node with its corresponding dataset op.
  RETURN_IF_NOT_OK(Serdes::UpdateOptimizedIRTreeJSON(&autotune_config_json_, ops_));
  std::vector<std::string> summary;
  RETURN_IF_NOT_OK(SummarizeTreeConfiguration(&summary));
  nlohmann::json out_json;
  out_json["summary"] = summary;
  out_json["tree"] = autotune_config_json_;
  std::string remark_value = "The following file has been auto-generated by the Dataset AutoTune.";
  if (tree_modifier_->GetRequestsCount() == 0) {
    remark_value += " Dataset Pipeline is not the bottleneck. No configuration changes were made by Dataset AutoTune.";
  }
  out_json["remark"] = remark_value;
  RETURN_IF_NOT_OK(Serdes::SaveJSONToFile(out_json, file_name, true));
  return Status::OK();
}

Status AutoTune::SetAutotuneConfigJson() {
  if (autotune_config_json_.empty()) {
    nlohmann::json out_json;
    RETURN_IF_NOT_OK(Serdes::SaveToJSON(tree_adapter_->RootIRNode(), "", &out_json));
    // We do not want to serialize DataQueueNode/DataQueueOp
    if (out_json["op_type"] == kTransferNode) {
      CHECK_FAIL_RETURN_UNEXPECTED(
        out_json["children"].size() == 1,
        "Expected Transfer node to have exactly 1 child but it has " + std::to_string(out_json["children"].size()));
      out_json = out_json["children"][0];
    }
    autotune_config_json_ = std::move(out_json);
  }
  return Status::OK();
}
#endif

Status AutoTune::SummarizeTreeConfiguration(std::vector<std::string> *out) {
  constexpr int op_name_width = 20;
  constexpr int val_width = 2;
  for (int i = static_cast<int>(ops_.size()) - 1; i >= 0; --i) {
    const auto op = ops_[i];
    if (!op->inlined() && op->Name() != "DataQueueOp") {
      std::stringstream s;
      s << std::left << std::setw(op_name_width) << op->NameWithID() << "(num_parallel_workers:" << std::right
        << std::setw(val_width) << (op->NumWorkers() == 0 ? "NA" : std::to_string(op->NumWorkers()))
        << ", prefetch_size:" << std::setw(val_width) << op->ConnectorCapacity() << ")";
      (void)out->emplace_back(s.str());
    }
  }
  return Status::OK();
}

void AutoTune::PostMainLogging() const {
  MS_LOG(INFO) << "Dataset AutoTune thread is finished.";
  MS_LOG(INFO) << "Printing the final tree configuration";
  PrintTreeConfiguration();
  // Print the suggestion in logs only if autotune requested some changes
  if (tree_modifier_->GetRequestsCount() > 0) {
    MS_LOG(INFO) << "Suggest to set proper num_parallel_workers for each Operation or use global setting API: "
                 << "mindspore.dataset.config.set_num_parallel_workers";
    MS_LOG(INFO) << "Suggest to choose maximum prefetch_size from tuned result and set by global setting API: "
                 << "mindspore.dataset.config.set_prefetch_size";
  }
}

void AutoTune::PrintTreeConfiguration() const {
  ExecutionTree const *tree = tree_adapter_->tree_.get();
  for (auto itr = tree->begin(); itr != tree->end(); (void)itr++) {
    if (!itr->inlined() && itr->Name() != "DataQueueOp") {
      MS_LOG(INFO) << itr->NameWithID() << " num_parallel_workers: " << itr->NumWorkers()
                   << " prefetch_size: " << itr->ConnectorCapacity();
    }
  }
}

Status AutoTune::LaunchThread() {
  MS_LOG(INFO) << "Launching Dataset AutoTune thread";
  Status rc = CollectOpsInfo();
  if (rc.IsError()) {
    if (rc.StatusCode() != StatusCode::kMDInterrupted) {
      MS_LOG(ERROR) << "Dataset AutoTune failed and will exit with the following error: " << rc;
    }
    RETURN_IF_NOT_OK(profiling_manager_->Stop());
    return Status::OK();
  }
  RETURN_IF_NOT_OK(cv_.Register(tree_adapter_->AllTasks()->GetIntrpService()));
  RETURN_IF_NOT_OK(tree_adapter_->AllTasks()->CreateAsyncTask("AutoTune Thread", std::bind(&AutoTune::Main, this)));
  return Status::OK();
}

Status AutoTune::CollectOpsInfo() {
  ExecutionTree const *tree = tree_adapter_->tree_.get();
  RETURN_UNEXPECTED_IF_NULL(tree);
  for (auto itr = tree->begin(); itr != tree->end(); ++itr) {
    ops_[itr->id()] = itr.get();
    // Get all parallel ops (num_workers>0) except leaf nodes (no children)
    if (itr->NumWorkers() > 0) {
      parallel_ops_ids_.push_back(itr->id());
    }
  }
  // Sort parallel ops in reverse order of IDs (i.e., bottommost op is first)
  std::sort(parallel_ops_ids_.begin(), parallel_ops_ids_.end(), std::greater<>());
  leaf_op_id_ = static_cast<int32_t>(ops_.size()) - 1;
  return Status::OK();
}

Status AutoTune::GetOpConnectorCapacity(int32_t op_id, int64_t *capacity) {
  auto item = ops_.find(op_id);
  CHECK_FAIL_RETURN_UNEXPECTED(item != ops_.end(), "Invalid Operator ID.");
  *capacity = item->second->ConnectorCapacity();
  return Status::OK();
}

Status AutoTune::GetOpsCpuUtil(std::map<int32_t, double> *ops_cpu_util) {
  // Loop over all itr keys and get avg cpu usage
  for (auto itr = ops_.begin(); itr != ops_.end(); ++itr) {
    std::vector<uint16_t> sys_util;
    std::vector<uint16_t> user_util;
#ifndef ENABLE_ANDROID
    if (mode_ == AutoTuneMode::kAutoTuneModeEpoch) {
      RETURN_IF_NOT_OK(profiling_manager_->GetSysCpuUtilByEpoch(itr->first, cur_epoch_running_, &sys_util));
      RETURN_IF_NOT_OK(profiling_manager_->GetUserCpuUtilByEpoch(itr->first, cur_epoch_running_, &user_util));
    } else if (mode_ == AutoTuneMode::kAutoTuneModeStep) {
      RETURN_IF_NOT_OK(
        profiling_manager_->GetSysCpuUtilByStep(itr->first, last_step_autotuned_, cur_step_running_ - 1, &sys_util));
      RETURN_IF_NOT_OK(
        profiling_manager_->GetUserCpuUtilByStep(itr->first, last_step_autotuned_, cur_step_running_ - 1, &user_util));
    }
#endif
    double sys_cpu_util = Mean(sys_util);
    double user_cpu_util = Mean(user_util);
    (*ops_cpu_util)[itr->first] = sys_cpu_util + user_cpu_util;
  }
  return Status::OK();
}

Status AutoTune::GetOpsQueueUtil(std::map<int32_t, double> *out_ops_queue_util,
                                 std::map<int32_t, double> *in_ops_queue_util) {
  // Loop over all itr keys in the ops_ and get output_queue usage
  for (auto itr = ops_.begin(); itr != ops_.end(); ++itr) {
    if (itr->second->inlined()) {
      (*out_ops_queue_util)[itr->first] = -1;
      continue;
    }
    std::vector<int32_t> sizes;
    if (mode_ == AutoTuneMode::kAutoTuneModeEpoch) {
      RETURN_IF_NOT_OK(profiling_manager_->GetConnectorSizeByEpoch(itr->first, cur_epoch_running_, &sizes));
    } else if (mode_ == AutoTuneMode::kAutoTuneModeStep) {
      RETURN_IF_NOT_OK(
        profiling_manager_->GetConnectorSizeByStep(itr->first, last_step_autotuned_, cur_step_running_ - 1, &sizes));
    }
    double avg_size = Mean(sizes);
    int64_t capacity = itr->second->ConnectorCapacity();
    CHECK_FAIL_RETURN_UNEXPECTED(capacity != 0, "Capacity of connector should not be 0");
    (*out_ops_queue_util)[itr->first] = avg_size / capacity;
  }
  for (auto itr = ops_.rbegin(); itr != ops_.rend(); ++itr) {
    // Assume that leaf op has 100% input queue util
    if (itr->first + 1 == ops_.size()) {
      (*in_ops_queue_util)[itr->first] = 1;
      continue;
    }
    // Input queue is the output queue of the child
    (*in_ops_queue_util)[itr->first] = (*out_ops_queue_util)[itr->first + 1];
    // If the child is an inlined op, use the prev known utilization
    if ((*in_ops_queue_util)[itr->first] == -1.0) {
      (*in_ops_queue_util)[itr->first] = (*in_ops_queue_util)[itr->first + 1];
    }
  }
  for (const auto &op : ops_) {
    if (op.second->inlined()) {
      (*in_ops_queue_util)[op.first] = -1;
    }
  }
  return Status::OK();
}

Status AutoTune::GetOpsNumWorker(std::map<int32_t, int32_t> *ops_num_workers) {
  for (const auto &op : ops_) {
    (*ops_num_workers)[op.first] = op.second->NumWorkers();
  }
  return Status::OK();
}

bool AutoTune::IsSink() const {
  std::shared_ptr<Tracing> node;
  return profiling_manager_->GetTracingNode(kDeviceQueueTracingName, &node).IsOk();
}

template <typename T>
double AutoTune::Mean(const std::vector<T> &items) const {
  if (items.size() == 0) {
    return 0;
  }
  return std::accumulate(items.begin(), items.end(), 0.0) / static_cast<double>(items.size());
}

Status AutoTune::UpdateCurrentRunInfo() {
  // Get current running epoch
  cur_epoch_running_ = profiling_manager_->GetNumOfProfiledEpochs();
  // Get current running step
  int32_t step_temp = 0;
  RETURN_IF_NOT_OK(profiling_manager_->GetNumberOfProfiledSteps(&step_temp));
  cur_step_running_ = step_temp;
  return Status::OK();
}

bool AutoTune::WarmupSkipCheck() {
  if (skip_flag_ == false) {
    return false;
  }
  if (mode_ == AutoTuneMode::kAutoTuneModeEpoch) {
    if (cur_epoch_running_ > EPOCH_WARMUP) {
      skip_flag_ = false;
      return false;
    }
  } else if (mode_ == AutoTuneMode::kAutoTuneModeStep) {
    int64_t skip_value = std::max(STEP_WARMUP, step_gap_);
    if (cur_step_running_ > skip_value) {
      last_step_autotuned_ = std::min(STEP_WARMUP, step_gap_);
      skip_flag_ = false;
      return false;
    }
  }
  return true;
}

Status AutoTune::RunIterationEpoch() {
  // Run every epoch
  if (last_epoch_autotuned_ < cur_epoch_running_ - 1) {
    MS_LOG(INFO) << "Run Dataset AutoTune at epoch # " << cur_epoch_running_;
    RETURN_IF_NOT_OK(RunIteration());
    last_epoch_autotuned_ = cur_epoch_running_ - 1;
  }
  return Status::OK();
}

Status AutoTune::RunIterationStep() {
  // Run at autotune step interval
  if (cur_step_running_ - last_step_autotuned_ >= step_gap_) {
    MS_LOG(INFO) << "Run AutoTune at step # " << cur_step_running_;
    RETURN_IF_NOT_OK(RunIteration());
    last_step_autotuned_ = cur_step_running_;
  }
  return Status::OK();
}

Status AutoTune::RegisterWorkersQueue() {
  ExecutionTree *tree = tree_adapter_->tree_.get();
  for (auto itr = tree->begin(); itr != tree->end(); (void)itr++) {
    if (!itr->inlined() && itr->Name() != "DataQueueOp") {
      (void)phase_1_best_workers.push_back(itr->NumWorkers());
      (void)phase_1_best_queue.push_back(itr->ConnectorCapacity());
    }
  }
  return Status::OK();
}

Status AutoTune::ResetWorkersQueue() {
  if (phase_1_best_workers.size() == 0 || phase_1_best_queue.size() == 0) {
    return Status::OK();
  }
  ExecutionTree *tree = tree_adapter_->tree_.get();
  int counter = 0;
  for (auto itr = tree->begin(); itr != tree->end(); (void)itr++) {
    if (!itr->inlined() && itr->Name() != "DataQueueOp") {
      int32_t target_workers = phase_1_best_workers[counter];
      int32_t target_queue = phase_1_best_queue[counter];
      RETURN_IF_NOT_OK(RequestNumWorkerChange(itr->id(), -1, &target_workers));
      RETURN_IF_NOT_OK(RequestConnectorCapacityChange(itr->id(), -1, target_queue));
      counter++;
    }
  }
  return Status::OK();
}

Status AutoTune::TrackPipelineTime() {
  std::vector<int32_t> pipeline_times;
  std::vector<int32_t> batch_times;
  // Select early stop threshold based on running mode
  int early_stop_threshold_mode;
  if (mode_ == AutoTuneMode::kAutoTuneModeEpoch) {
    RETURN_IF_NOT_OK(profiling_manager_->GetPipelineTimeByEpoch(cur_epoch_running_, &pipeline_times));
    RETURN_IF_NOT_OK(profiling_manager_->GetBatchTimeByEpoch(cur_epoch_running_ - 1, &batch_times));
    early_stop_threshold_mode = EARLY_STOP_TRIAL_THRESHOLD_EPOCH;
  } else if (mode_ == AutoTuneMode::kAutoTuneModeStep) {
    RETURN_IF_NOT_OK(
      profiling_manager_->GetPipelineTimeByStep(last_step_autotuned_, cur_step_running_ - 1, &pipeline_times));
    RETURN_IF_NOT_OK(profiling_manager_->GetBatchTimeByStep(last_step_autotuned_, cur_step_running_ - 1, &batch_times));
    early_stop_threshold_mode = EARLY_STOP_TRIAL_THRESHOLD_STEP;
  }
  double avg_time_pipeline = Mean(pipeline_times);
  double avg_time_batch = Mean(batch_times);
  (void)avg_pipeline_times_.push_back(avg_time_pipeline);
  MS_LOG(INFO) << "Average Pipeline time is " << avg_time_pipeline << " ms. The avg pipeline time for all epochs is "
               << Mean(avg_pipeline_times_) << "ms";
  // Time phase (phase 1) improvement tracking
  if (AT_phase_ == AutoTunePhase::kAutoTunePhaseTime) {
    if (phase_1_best_time_ < 0) {
      phase_1_best_time_ = avg_time_batch;  // set first value
    } else if (avg_time_batch < phase_1_best_time_) {
      phase_1_no_improve_count_ = 0;
      phase_1_best_time_ = avg_time_batch;
      // Trigger save process
      if (AT_change_) {
        AT_change_ = false;  // Reset for next analysis run
        RETURN_IF_NOT_OK(RegisterWorkersQueue());
      }
    } else {
      phase_1_no_improve_count_++;
    }
    if (phase_1_no_improve_count_ > early_stop_threshold_mode) {
      // Reset best config and exit
      AT_phase_ = AutoTunePhase::kAutoTunePhaseMemory;
      RETURN_IF_NOT_OK(ResetWorkersQueue());
    }
  }
  return Status::OK();
}

Status AutoTune::RunIteration() {
  RETURN_IF_NOT_OK(TrackPipelineTime());
  if (AT_phase_ == AutoTunePhase::kAutoTunePhaseTime) {
    RETURN_IF_NOT_OK(AnalyseTime());
  } else if (AT_phase_ == AutoTunePhase::kAutoTunePhaseMemory) {
    RETURN_IF_NOT_OK(AnalyseMemory());
  }
  return Status::OK();
}

Status AutoTune::GetConnectorSize(std::vector<int32_t> *sizes) const {
  if (mode_ == AutoTuneMode::kAutoTuneModeEpoch) {
    RETURN_IF_NOT_OK(profiling_manager_->GetConnectorSizeByEpoch(cur_epoch_running_, sizes));
  } else if (mode_ == AutoTuneMode::kAutoTuneModeStep) {
    RETURN_IF_NOT_OK(profiling_manager_->GetConnectorSizeByStep(last_step_autotuned_, cur_step_running_ - 1, sizes));
  }
  return Status::OK();
}

Status AutoTune::GetConnectorCapacity(std::vector<int32_t> *capacities) const {
  if (mode_ == AutoTuneMode::kAutoTuneModeEpoch) {
    RETURN_IF_NOT_OK(profiling_manager_->GetConnectorCapacityByEpoch(cur_epoch_running_, capacities));
  } else if (mode_ == AutoTuneMode::kAutoTuneModeStep) {
    RETURN_IF_NOT_OK(
      profiling_manager_->GetConnectorCapacityByStep(last_step_autotuned_, cur_step_running_ - 1, capacities));
  }
  return Status::OK();
}

Status AutoTune::GetConnectorUtil(double *usage_avg_last, double *avg_size, double *avg_capacity) {
  std::vector<int32_t> sizes;
  RETURN_IF_NOT_OK(GetConnectorSize(&sizes));
  *avg_size = Mean(sizes);
  std::vector<int32_t> capacities;
  RETURN_IF_NOT_OK(GetConnectorCapacity(&capacities));
  *avg_capacity = Mean(capacities);
  CHECK_FAIL_RETURN_UNEXPECTED(*avg_capacity != 0.0, "Capacities of connectors should not be 0.0");
  // size here means size of queue utilized
  *usage_avg_last = (*avg_size / *avg_capacity);
  return Status::OK();
}

Status AutoTune::GetEmptyQueueFrequency(float *empty_freq) const {
  if (mode_ == AutoTuneMode::kAutoTuneModeEpoch) {
    RETURN_IF_NOT_OK(profiling_manager_->GetEmptyQueueFrequencyByEpoch(cur_epoch_running_, empty_freq));
  } else if (mode_ == AutoTuneMode::kAutoTuneModeStep) {
    RETURN_IF_NOT_OK(
      profiling_manager_->GetEmptyQueueFrequencyByStep(last_step_autotuned_, cur_step_running_ - 1, empty_freq));
  }
  return Status::OK();
}

Status AutoTune::IsDSaBottleneck(bool *isBottleneck) {
  double usage_avg_last, avg_size, avg_capacity;
  RETURN_IF_NOT_OK(GetConnectorUtil(&usage_avg_last, &avg_size, &avg_capacity));
  float empty_freq = 0;
  RETURN_IF_NOT_OK(GetEmptyQueueFrequency(&empty_freq));
  if (mode_ == AutoTuneMode::kAutoTuneModeStep) {
    MS_LOG(INFO) << "Step # " << cur_step_running_ << ". Status:";
  } else {
    MS_LOG(INFO) << "Epoch # " << cur_epoch_running_ << ". Status:";
  }
  // Reporting values
  MS_LOG(INFO) << "Device Connector Size: " << avg_size << ", Connector Capacity: " << avg_capacity
               << ", Utilization: " << (usage_avg_last * TO_PERCENT) << "%"
               << ", Empty Freq: " << (empty_freq * TO_PERCENT) << "% ";
  // Decision
  if (usage_avg_last < DEVICE_CONNECTOR_UTIL_THRESHOLD) {
    MS_LOG(INFO) << "Utilization: " << (usage_avg_last * TO_PERCENT) << "% < "
                 << (DEVICE_CONNECTOR_UTIL_THRESHOLD * TO_PERCENT)
                 << "% threshold, dataset pipeline performance may benefit from tuning.";
    *isBottleneck = true;
  } else {
    MS_LOG(INFO) << "Utilization: " << (usage_avg_last * TO_PERCENT) << "% > "
                 << (DEVICE_CONNECTOR_UTIL_THRESHOLD * TO_PERCENT)
                 << "% threshold, dataset pipeline performance is OK.";
    *isBottleneck = false;
  }
  return Status::OK();
}

Status AutoTune::RequestNumWorkerChange(int32_t op_id, int32_t old_workers, int32_t *num_workers_requested) {
  AT_change_ = true;
  int new_workers = std::min(*num_workers_requested, max_workers_);
  new_workers = std::max(new_workers, MIN_NUM_WORKERS);
  RETURN_IF_NOT_OK(tree_modifier_->AddChangeRequest(op_id, std::make_shared<ChangeNumWorkersRequest>(new_workers)));
  if (old_workers == -1) {
    MS_LOG(INFO) << "Added request to change \"num_parallel_workers\" of Operator: " << ops_[op_id]->NameWithID()
                 << "to value: [" << new_workers << "].";
  } else {
    MS_LOG(INFO) << "Added request to change \"num_parallel_workers\" of Operator: " << ops_[op_id]->NameWithID()
                 << "From old value: [" << old_workers << "] to new value: [" << new_workers << "].";
  }
  *num_workers_requested = new_workers;
  return Status::OK();
}

Status AutoTune::RequestConnectorCapacityChange(int32_t op_id, int32_t old_size, int32_t new_size) {
  AT_change_ = true;
  new_size = std::min(new_size, MAX_QUEUE_SIZE);
  new_size = std::max(new_size, MIN_QUEUE_SIZE);
  RETURN_IF_NOT_OK(tree_modifier_->AddChangeRequest(op_id, std::make_shared<ResizeConnectorRequest>(new_size)));
  if (old_size == -1) {
    MS_LOG(INFO) << "Added request to change \"prefetch_size\" of Operator: " << ops_[op_id]->NameWithID()
                 << "to value: [" << new_size << "].";
  } else {
    MS_LOG(INFO) << "Added request to change \"prefetch_size\" of Operator: " << ops_[op_id]->NameWithID()
                 << "From old value: [" << old_size << "] to new value: [" << new_size << "].";
  }
  return Status::OK();
}

bool AutoTune::SkipOpsCheck(int op_id) {
  // Skip Generator op
  if (ops_[op_id]->Name() == "GeneratorOp") {
    return true;
  }
  //  NonMappableDataset is not supported in AutoTune
#ifndef ENABLE_ANDROID
  if (std::dynamic_pointer_cast<NonMappableLeafOp>(ops_[op_id]) != nullptr) {
    return true;
  }
#endif
  return false;
}

Status AutoTune::AnalyseTime() {
  // check for connector queue bottleneck
  bool isBottleneck = false;
  RETURN_IF_NOT_OK(IsDSaBottleneck(&isBottleneck));
  if (!isBottleneck) {
    return Status::OK();
  }
  // collect stats
  std::map<int32_t, int32_t> ops_num_workers;
  RETURN_IF_NOT_OK(GetOpsNumWorker(&ops_num_workers));
  std::map<int32_t, double> out_ops_queue_util;
  std::map<int32_t, double> in_ops_queue_util;
  RETURN_IF_NOT_OK(GetOpsQueueUtil(&out_ops_queue_util, &in_ops_queue_util));
  std::map<int32_t, double> ops_cpu_util;
  RETURN_IF_NOT_OK(GetOpsCpuUtil(&ops_cpu_util));
  // check parallel ops in loop
  for (const auto &op_id : parallel_ops_ids_) {
    if (SkipOpsCheck(op_id)) {
      continue;
    }
    // op specifics
    double output_queue_util = out_ops_queue_util[op_id];
    double input_queue_util = in_ops_queue_util[op_id];
    double cpu_util = ops_cpu_util[op_id];
    int32_t num_workers = ops_num_workers[op_id];
    CHECK_FAIL_RETURN_UNEXPECTED(num_workers != 0, "ParallelOp with num_workers=0");
    // derived metrics
    double queue_diff = input_queue_util - output_queue_util;
    int64_t queue_capacity;
    RETURN_IF_NOT_OK(GetOpConnectorCapacity(op_id, &queue_capacity));
    int64_t new_queue_capacity = queue_capacity;
    int32_t requested_workers = 0;
    MS_LOG(DEBUG) << "Op (" << ops_[op_id]->NameWithID() << ") CPU=" << cpu_util / num_workers
                  << ", in=" << input_queue_util << "out=" << output_queue_util;
    // map decisions - queue
    if (queue_diff > INPUT_OUTPUT_QUEUE_DIFF_THRESHOLD) {
      MS_LOG(INFO) << "Op (" << ops_[op_id]->NameWithID()
                   << ") is slow, input connector utilization=" << input_queue_util
                   << ", output connector utilization=" << output_queue_util << ", diff= " << queue_diff << " > "
                   << INPUT_OUTPUT_QUEUE_DIFF_THRESHOLD << " threshold.";
      requested_workers = num_workers + INCREMENT_WORKER;
      RETURN_IF_NOT_OK(RequestNumWorkerChange(op_id, num_workers, &requested_workers));
    } else if ((cpu_util / num_workers) > MAP_OP_WORKER_HIGH_THRESHOLD) {
      MS_LOG(INFO) << "Op (" << ops_[op_id]->NameWithID() << ") getting high average worker cpu utilization "
                   << (cpu_util / num_workers) << "% > " << MAP_OP_WORKER_HIGH_THRESHOLD << "% threshold.";
      requested_workers = num_workers + INCREMENT_WORKER;
      RETURN_IF_NOT_OK(RequestNumWorkerChange(op_id, num_workers, &requested_workers));
    }
    if ((cpu_util / num_workers) < MAP_OP_WORKER_LOW_THRESHOLD &&
        ((input_queue_util < INPUT_QUEUE_LOW) || (-1 * queue_diff > INPUT_OUTPUT_QUEUE_DIFF_THRESHOLD))) {
      MS_LOG(INFO) << "Op (" << ops_[op_id]->NameWithID() << ") getting low average worker cpu utilization "
                   << (cpu_util / num_workers) << "% < " << MAP_OP_WORKER_LOW_THRESHOLD << "% threshold.";
      new_queue_capacity = queue_capacity + INCREMENT_QUEUE_SIZE;
      if (requested_workers == 0) {
        requested_workers = num_workers;
      }
      new_queue_capacity = std::max(new_queue_capacity, static_cast<int64_t>(requested_workers));
      RETURN_IF_NOT_OK(RequestConnectorCapacityChange(op_id, queue_capacity, new_queue_capacity));
    }
  }
  return Status::OK();
}

bool AutoTune::MemoryPhaseCompareMetric(double prev_avg, double cur_avg) {
  double lower_bound = prev_avg - (prev_avg * MEMORY_COMPARISON_LOWER_BOUND_PERCENT);
  // If cur_avg worse than lower bound - negative impact on performance
  // lower bound set to account for expected fluctuations in util numbers
  if (cur_avg < lower_bound) {
    return false;
  } else {
    return true;
  }
}

Status AutoTune::AnalyseMemory() {
  double prev_avg = 0;
  double cur_avg = 0;
  bool comp_flag = true;
  double connector_avg_size;
  double connector_avg_capacity;
  int total = parallel_ops_ids_.size();
  if (total == 0) {
    AT_phase_ = AutoTunePhase::kAutoTuneEnd;
    return Status::OK();
  }
  std::map<int32_t, int32_t> ops_num_workers;
  RETURN_IF_NOT_OK(GetOpsNumWorker(&ops_num_workers));
  double reduce_percent_mode;
  // Decrease queue sizes faster in epoch mode
  if (mode_ == AutoTuneMode::kAutoTuneModeEpoch) {
    reduce_percent_mode = QUEUE_REDUCTION_PERCENTAGE_EPOCH;
  } else if (AutoTuneMode::kAutoTuneModeStep) {
    reduce_percent_mode = QUEUE_REDUCTION_PERCENTAGE_STEP;
  }
  // Init state on first call
  if (phase_3_state_ == AutoTuneMemPhase::kAutoTuneMemInit) {
    count_down_ = 0;
    for (auto op_id : parallel_ops_ids_) {
      if ((SkipOpsCheck(op_id)) || (ops_[op_id]->Name() == "DataQueueOp")) {
        // Op not supported - ignore throughout AT
        (void)OP_values.push_back(-1);
        continue;
      }
      // Op supported - attempt memory reduction on this
      (void)OP_values.push_back(0);
      count_down_++;
    }
    phase_3_state_ = AutoTuneMemPhase::kAutoTuneMemSet;
    phase_3_ID_ = 0;
  }

  // Exit when all viable ops have been tested
  // Or if none found
  if (count_down_ == 0) {
    AT_phase_ = AutoTunePhase::kAutoTuneEnd;
    return Status::OK();
  }

  if (phase_3_state_ == AutoTuneMemPhase::kAutoTuneMemSet) {
    RETURN_IF_NOT_OK(GetConnectorUtil(&phase_3_prev_avg_, &connector_avg_size, &connector_avg_capacity));
    // Search for next viable op that can be tested
    while (OP_values[phase_3_ID_] == -1) {
      phase_3_ID_ = ((phase_3_ID_ + 1) % total);
    }
    int64_t current_queue_size;
    RETURN_IF_NOT_OK(GetOpConnectorCapacity(parallel_ops_ids_[phase_3_ID_], &current_queue_size));
    int op_workers = ops_num_workers[parallel_ops_ids_[phase_3_ID_]];
    int target_mem = current_queue_size * reduce_percent_mode;
    if (std::max(target_mem, op_workers) == op_workers) {
      // Lower bound on range of queue size testing
      OP_values[phase_3_ID_] = -1;
      count_down_--;
      target_mem = op_workers;
    } else {
      // Save current queue size for possible recovery and switch to compare mode
      OP_values[phase_3_ID_] = current_queue_size;
      phase_3_state_ = AutoTuneMemPhase::kAutotTuneMemCompare;
    }
    RequestConnectorCapacityChange(parallel_ops_ids_[phase_3_ID_], -1, target_mem);
  } else if (phase_3_state_ == AutoTuneMemPhase::kAutotTuneMemCompare) {
    // Analyse impact on model from previous change made
    RETURN_IF_NOT_OK(GetConnectorUtil(&cur_avg, &connector_avg_size, &connector_avg_capacity));
    prev_avg = phase_3_prev_avg_;
    comp_flag = MemoryPhaseCompareMetric(prev_avg, cur_avg);
    // Compare current avg against pre-change avg
    if (comp_flag == false) {
      int reset_val = OP_values[phase_3_ID_];
      RequestConnectorCapacityChange(parallel_ops_ids_[phase_3_ID_], -1, reset_val);
      // Op queue size reduction caused performance decrease - ignore onwards
      OP_values[phase_3_ID_] = -1;
      count_down_--;
    }
    phase_3_state_ = AutoTuneMemPhase::kAutoTuneMemSet;
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
