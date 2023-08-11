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
#include "minddata/dataset/engine/perf/profiling.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/perf/connector_size.h"
#include "minddata/dataset/engine/perf/cpu_sampler.h"
#include "minddata/dataset/engine/perf/monitor.h"
#include "minddata/dataset/engine/tree_adapter.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/path.h"
#ifdef WITH_BACKEND
#include "utils/ms_context.h"
#endif
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
constexpr int32_t PUSH_TIME_OFFSET = 0;
constexpr int32_t BATCH_TIME_OFFSET = 1;
constexpr int32_t PIPELINE_TIME_OFFSET = 2;
constexpr int32_t CONNECTOR_DEPTH_OFFSET = 3;

Status Profiling::Start() {
  CHECK_FAIL_RETURN_UNEXPECTED(active_ == false, "Profiling node is already active.");
  active_ = true;
  return Status::OK();
}

Status Profiling::Stop() {
  CHECK_FAIL_RETURN_UNEXPECTED(active_ == true, "Profiling node is already deactivated.");
  active_ = false;
  return Status::OK();
}

Status Tracing::SaveToFile(const std::string &dir_path, const std::string &rank_id) {
  if (value_.empty()) {
    return Status::OK();
  }

  Path path = GetFileName(dir_path, rank_id);
  // Remove the file if it exists (from prior profiling usage)
  RETURN_IF_NOT_OK(path.Remove());
  std::string file_path = path.ToString();

  MS_LOG(INFO) << "Start to save profiling data for a tracing node.";
  std::ofstream handle(file_path, std::ios::trunc);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Profiling file can not be opened.");
  }
  for (const auto &value : value_) {
    handle << value << "\n";
  }
  handle.close();

  return Status::OK();
}

Status Tracing::ChangeFileMode(const std::string &dir_path, const std::string &rank_id) {
  if (value_.empty()) {
    return Status::OK();
  }

  Path path = GetFileName(dir_path, rank_id);
  std::string file_path = path.ToString();
  if (chmod(common::SafeCStr(file_path), S_IRUSR | S_IWUSR) == -1) {
    std::string err_str = "Change file mode failed," + file_path;
    return Status(StatusCode::kMDUnexpectedError, err_str);
  }
  return Status::OK();
}

void Tracing::Record(const int32_t type, const int32_t extra_info, const int32_t batch_num, const int32_t value,
                     const uint64_t time_stamp) {
  // Format: "type extra-info batch-num value"
  // type: 0: time,  1: connector size
  // extra-info: if type is 0 - 0: pipeline time, 1: push tdt time, 2: batch time
  //             if type is 1 - connector capacity
  // batch-num: batch number
  // value: if type is 0 - value is time(ms)
  //        if type is 1 - value is connector size
  // time-stamp: time stamp
  // Examples:
  // 0 0 20 10 xxx- The 20th batch took 10ms to get data from pipeline.
  // 1 64 20 5 xxx- Connector size is 5 when get the 20th batch.Connector capacity is 64.
  if (!active_) {
    return;
  }
  TracingRecord record = {type, extra_info, batch_num, value, time_stamp};
  std::lock_guard<std::mutex> guard(lock_);
  (void)records_.emplace_back(record);
  (void)value_.emplace_back(record.ToString());
  // save timestamp per batch
  const constexpr int32_t RECORDS_PER_STEP = 4;
  if (records_.size() % RECORDS_PER_STEP == 0) {
    (void)ts_.emplace_back(time_stamp);
  }
}

Status Tracing::TimeIntervalForStepRange(int32_t start_step, int32_t end_step, uint64_t *start_ts, uint64_t *end_ts) {
  RETURN_UNEXPECTED_IF_NULL(start_ts);
  RETURN_UNEXPECTED_IF_NULL(end_ts);
  std::lock_guard<std::mutex> guard(lock_);
  MS_LOG(DEBUG) << "start_step: " << start_step << " end_step: " << end_step;
  CHECK_FAIL_RETURN_UNEXPECTED(start_step > 0,
                               "Expected start_step > 0. Got start_step: " + std::to_string(start_step));
  CHECK_FAIL_RETURN_UNEXPECTED(end_step >= start_step,
                               "Expected end_step >= start_step. Got start_step: " + std::to_string(start_step) +
                                 " end_step: " + std::to_string(end_step));
  CHECK_FAIL_RETURN_UNEXPECTED(end_step < static_cast<int32_t>(ts_.size()),
                               "Expected end_step < ts_.size(). Got end_step: " + std::to_string(end_step) +
                                 " ts_.size: " + std::to_string(ts_.size()));
  // end timestamp of (start_step - 1) step
  *start_ts = ts_[start_step - 1];
  *end_ts = ts_[end_step];
  return Status::OK();
}

Status Tracing::StepIntervalForTimeRange(uint64_t start_ts, uint64_t end_ts, int32_t *start_step, int32_t *end_step) {
  RETURN_UNEXPECTED_IF_NULL(start_step);
  RETURN_UNEXPECTED_IF_NULL(end_step);
  CHECK_FAIL_RETURN_UNEXPECTED(start_ts < end_ts, "Expected start_ts < end_ts. Got start_ts: " +
                                                    std::to_string(start_ts) + " end_ts: " + std::to_string(end_ts));
  std::lock_guard<std::mutex> guard(lock_);
  CHECK_FAIL_RETURN_UNEXPECTED(ts_.size() > 1, "No tracing data available yet.");
  // find first ts that is not less than start_ts
  auto lower = std::lower_bound(ts_.begin(), ts_.end(), start_ts);
  CHECK_FAIL_RETURN_UNEXPECTED(lower != ts_.end(),
                               "No data available for time >= start_ts. start_ts: " + std::to_string(start_ts));
  // there is no 0th step. If start_ts == 0, then lower == ts_.begin()
  *start_step = std::max(1, static_cast<int32_t>(std::distance(ts_.begin(), lower)));
  // find first ts that is greater than end_ts
  auto upper = std::upper_bound(ts_.begin(), ts_.end(), end_ts);
  if (upper == ts_.end()) {
    *end_step = std::max(1, static_cast<int32_t>(std::distance(ts_.begin(), upper) - 1));
  } else {
    *end_step = std::max(1, static_cast<int32_t>(std::distance(ts_.begin(), upper)));
  }
  return Status::OK();
}

Status Tracing::GetRecordEntryFieldValue(int32_t start_step, int32_t end_step, int32_t record_offset,
                                         const std::string &field, std::vector<int32_t> *result) {
  RETURN_UNEXPECTED_IF_NULL(result);
  std::lock_guard<std::mutex> guard(lock_);
  const constexpr int32_t RECORDS_PER_STEP = 4;
  auto total_steps = records_.size() / RECORDS_PER_STEP;
  MS_LOG(DEBUG) << "start_step: " << start_step << " end_step: " << end_step;
  CHECK_FAIL_RETURN_UNEXPECTED(start_step <= static_cast<int32_t>(total_steps),
                               "Expected start_step <= total_steps. Got start_step: " + std::to_string(start_step) +
                                 " total_steps: " + std::to_string(total_steps));
  CHECK_FAIL_RETURN_UNEXPECTED(end_step <= static_cast<int32_t>(total_steps),
                               "Expected end_step <= total_steps. Got end_step: " + std::to_string(end_step) +
                                 " total_steps: " + std::to_string(total_steps));
  CHECK_FAIL_RETURN_UNEXPECTED(start_step <= end_step,
                               "Expected start_step <= end_step. Got start_step: " + std::to_string(start_step) +
                                 " end_step: " + std::to_string(end_step));

  for (auto step_num = start_step; step_num <= end_step; step_num++) {
    auto idx = (step_num - 1) * RECORDS_PER_STEP + record_offset;
    CHECK_FAIL_RETURN_UNEXPECTED(idx >= 0, "Expected idx >= 0. Got idx: " + std::to_string(idx));
    if (field == "value") {
      (void)result->emplace_back(records_[static_cast<size_t>(idx)].value);
    } else if (field == "extra_info") {
      (void)result->emplace_back(records_[static_cast<size_t>(idx)].extra_info);
    } else {
      return {StatusCode::kMDUnexpectedError,
              "Received unexpected field: " + field + R"(. Expected: ["value", "extra_info"].)"};
    }
  }
  return Status::OK();
}

Status Tracing::GetPipelineTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, PIPELINE_TIME_OFFSET, "value", result);
}

Status Tracing::GetPushTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, PUSH_TIME_OFFSET, "value", result);
}

Status Tracing::GetBatchTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, BATCH_TIME_OFFSET, "value", result);
}

Status Tracing::GetConnectorSize(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, CONNECTOR_DEPTH_OFFSET, "value", result);
}

Status Tracing::GetConnectorCapacity(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, CONNECTOR_DEPTH_OFFSET, "extra_info", result);
}

Status Tracing::GetEmptyQueueFrequency(int32_t start_step, int32_t end_step, float_t *empty_queue_freq) {
  RETURN_UNEXPECTED_IF_NULL(empty_queue_freq);
  std::vector<int32_t> sizes;
  RETURN_IF_NOT_OK(GetConnectorSize(start_step, end_step, &sizes));
  int32_t total = end_step - start_step + 1;
  CHECK_FAIL_RETURN_UNEXPECTED(total > 0, "Start step is greater than end step.");
  uint32_t count = static_cast<uint32_t>(std::count(sizes.begin(), sizes.end(), 0));
  *empty_queue_freq = static_cast<float_t>(count) / static_cast<float_t>(total);
  return Status::OK();
}

Status Tracing::Init() {
  (void)ts_.emplace_back(0);
  return Status::OK();
}

size_t Tracing::GetNumberSteps() { return ts_.size(); }

void Tracing::Clear() {
  value_.clear();
  records_.clear();
  ts_.clear();
}

// Constructor
ProfilingManager::ProfilingManager()
    : profiling_state_(ProfilingState::kProfilingStateUnBegun), tree_(nullptr), autotuning_(false), profiling_(false) {}

bool ProfilingManager::IsProfilingEnable(const ExecutionTree *tree) const {
  auto external_state = GetProfilerTreeState(tree);
  return (external_state == kEnabledTreeNotRegistered || external_state == kEnabledTreeRegistered);
}

Status ProfilingManager::RegisterTree(const TreeAdapter *tree_adapter) {
  RETURN_UNEXPECTED_IF_NULL(tree_adapter);
  CHECK_FAIL_RETURN_UNEXPECTED(tree_ == nullptr, "Another tree is already registered.");
  CHECK_FAIL_RETURN_UNEXPECTED((autotuning_ || profiling_) == true,
                               "MD Profiler is disabled. Cannot register the tree.");
  tree_ = tree_adapter->tree_.get();
  MS_LOG(INFO) << "Registering tree: " + tree_->GetUniqueId();
  perf_monitor_ = std::make_unique<Monitor>(this);
  // Register all sampling nodes here.
  // Tracing node registration is the responsibility of the Consumer
  std::shared_ptr<Sampling> connector_size_sampling = std::make_shared<ConnectorSize>(tree_);
  RETURN_IF_NOT_OK(RegisterSamplingNode(connector_size_sampling));

#ifndef ENABLE_ANDROID
  std::shared_ptr<Sampling> cpu_sampler = std::make_shared<CpuSampler>(tree_);
  RETURN_IF_NOT_OK(RegisterSamplingNode(cpu_sampler));
#endif
  // can insert a correct timestamp so that we can ignore the samples that were taken
  // during start up of the pipeline.
  (void)epoch_end_ts_.emplace_back(0);
  (void)epoch_end_step_.emplace_back(0);
  return Status::OK();
}

// Launch monitoring thread.
Status ProfilingManager::LaunchMonitor() {
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask("Monitor Thread launched", std::ref(*perf_monitor_)));
  return Status::OK();
}

// Profiling node registration
Status ProfilingManager::RegisterTracingNode(const std::shared_ptr<Tracing> &node) {
  // Check if node with the same name has already been registered.
  auto exist = tracing_nodes_.find(node->Name());
  if (exist != tracing_nodes_.end()) {
    return Status(StatusCode::kMDProfilingError, "Profiling node already exist: " + node->Name());
  }
  // Register the node with its name as key.
  RETURN_IF_NOT_OK(node->Init());
  tracing_nodes_[node->Name()] = node;

  // the user may have already started profiling.
  if (profiling_state_ == ProfilingState::kProfilingStateRunning) {
    RETURN_IF_NOT_OK(node->Start());
  }
  return Status::OK();
}

// Profiling node getter
Status ProfilingManager::GetTracingNode(const std::string &name, std::shared_ptr<Tracing> *node) {
  // Check if node with the same name has already been registered.
  auto exist = tracing_nodes_.find(name);
  if (exist == tracing_nodes_.end()) {
    return Status(StatusCode::kMDProfilingError, "Profiling node does not exist: " + name);
  }
  // Fetch node.
  *node = tracing_nodes_[name];
  return Status::OK();
}

// Profiling node registration
Status ProfilingManager::RegisterSamplingNode(const std::shared_ptr<Sampling> &node) {
  // Check if node with the same name has already been registered.
  auto exist = sampling_nodes_.find(node->Name());
  if (exist != sampling_nodes_.end()) {
    return Status(StatusCode::kMDProfilingError, "Profiling node already exist: " + node->Name());
  }
  // Register the node with its name as key.
  RETURN_IF_NOT_OK(node->Init());
  sampling_nodes_[node->Name()] = node;

  // the user may have already started profiling.
  if (profiling_state_ == ProfilingState::kProfilingStateRunning) {
    RETURN_IF_NOT_OK(node->Start());
  }
  return Status::OK();
}

// Profiling node getter
Status ProfilingManager::GetSamplingNode(const std::string &name, std::shared_ptr<Sampling> *node) {
  // Check if node with the same name has already been registered.
  auto exist = sampling_nodes_.find(name);
  if (exist == sampling_nodes_.end()) {
    return Status(StatusCode::kMDProfilingError, "Profiling node does not exist: " + name);
  }
  // Fetch node.
  *node = sampling_nodes_[name];
  return Status::OK();
}

Status ProfilingManager::SaveProfilingData(const std::string &dir_path, const std::string &rank_id) {
  MS_LOG(INFO) << "Start to save profiling data.";
  for (const auto &node : tracing_nodes_) {
    RETURN_IF_NOT_OK(node.second->SaveToFile(dir_path, rank_id));
  }
  for (const auto &node : sampling_nodes_) {
    RETURN_IF_NOT_OK(node.second->SaveToFile(dir_path, rank_id));
  }
  MS_LOG(INFO) << "Save profiling data end.";
  return Status::OK();
}

Status ProfilingManager::ChangeFileMode(const std::string &dir_path, const std::string &rank_id) {
  MS_LOG(INFO) << "Start to change file mode.";
  for (const auto &node : tracing_nodes_) {
    RETURN_IF_NOT_OK(node.second->ChangeFileMode(dir_path, rank_id));
  }
  for (const auto &node : sampling_nodes_) {
    RETURN_IF_NOT_OK(node.second->ChangeFileMode(dir_path, rank_id));
  }
  MS_LOG(INFO) << "Change file mode end.";
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status ProfilingManager::GetUserCpuUtilByEpoch(int32_t epoch_num, std::vector<uint8_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetUserCpuUtilByTime(start_ts, end_ts, result);
}

Status ProfilingManager::GetUserCpuUtilByStep(int32_t start_step, int32_t end_step, std::vector<uint8_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(StepToTimeInterval(start_step, end_step, &start_ts, &end_ts));
  return GetUserCpuUtilByTime(start_ts, end_ts, result);
}

Status ProfilingManager::GetUserCpuUtilByTime(uint64_t start_ts, uint64_t end_ts, std::vector<uint8_t> *result) {
  std::shared_ptr<Sampling> sampling_node;
  RETURN_IF_NOT_OK(GetSamplingNode(kCpuSamplerName, &sampling_node));
  auto node = std::dynamic_pointer_cast<CpuSampler>(sampling_node);
  return node->GetSystemUserCpuUtil(start_ts, end_ts, result);
}

Status ProfilingManager::GetSysCpuUtilByEpoch(int32_t epoch_num, std::vector<uint8_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetSysCpuUtilByTime(start_ts, end_ts, result);
}

Status ProfilingManager::GetSysCpuUtilByStep(int32_t start_step, int32_t end_step, std::vector<uint8_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(StepToTimeInterval(start_step, end_step, &start_ts, &end_ts));
  return GetSysCpuUtilByTime(start_ts, end_ts, result);
}

Status ProfilingManager::GetSysCpuUtilByTime(uint64_t start_ts, uint64_t end_ts, std::vector<uint8_t> *result) {
  std::shared_ptr<Sampling> sampling_node;
  RETURN_IF_NOT_OK(GetSamplingNode(kCpuSamplerName, &sampling_node));
  auto node = std::dynamic_pointer_cast<CpuSampler>(sampling_node);
  return node->GetSystemSysCpuUtil(start_ts, end_ts, result);
}

Status ProfilingManager::GetUserCpuUtilByEpoch(int32_t op_id, int32_t epoch_num, std::vector<uint16_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetUserCpuUtilByTime(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetUserCpuUtilByStep(int32_t op_id, int32_t start_step, int32_t end_step,
                                              std::vector<uint16_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(StepToTimeInterval(start_step, end_step, &start_ts, &end_ts));
  return GetUserCpuUtilByTime(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetUserCpuUtilByTime(int32_t op_id, uint64_t start_ts, uint64_t end_ts,
                                              std::vector<uint16_t> *result) {
  std::shared_ptr<Sampling> sampling_node;
  RETURN_IF_NOT_OK(GetSamplingNode(kCpuSamplerName, &sampling_node));
  auto node = std::dynamic_pointer_cast<CpuSampler>(sampling_node);
  return node->GetOpUserCpuUtil(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetSysCpuUtilByEpoch(int32_t op_id, int32_t epoch_num, std::vector<uint16_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetSysCpuUtilByTime(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetSysCpuUtilByStep(int32_t op_id, int32_t start_step, int32_t end_step,
                                             std::vector<uint16_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(StepToTimeInterval(start_step, end_step, &start_ts, &end_ts));
  return GetSysCpuUtilByTime(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetSysCpuUtilByTime(int32_t op_id, uint64_t start_ts, uint64_t end_ts,
                                             std::vector<uint16_t> *result) {
  std::shared_ptr<Sampling> sampling_node;
  RETURN_IF_NOT_OK(GetSamplingNode(kCpuSamplerName, &sampling_node));
  auto node = std::dynamic_pointer_cast<CpuSampler>(sampling_node);
  return node->GetOpSysCpuUtil(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetMainProcessMemoryInfoByEpoch(ProcessMemoryMetric metric, int32_t epoch_num,
                                                         std::vector<float> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetMainProcessMemoryInfoByTime(metric, start_ts, end_ts, result);
}

Status ProfilingManager::GetMainProcessMemoryInfoByStep(ProcessMemoryMetric metric, int32_t start_step,
                                                        int32_t end_step, std::vector<float> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(StepToTimeInterval(start_step, end_step, &start_ts, &end_ts));
  return GetMainProcessMemoryInfoByTime(metric, start_ts, end_ts, result);
}

Status ProfilingManager::GetMainProcessMemoryInfoByTime(ProcessMemoryMetric metric, uint64_t start_ts, uint64_t end_ts,
                                                        std::vector<float> *result) {
  std::shared_ptr<Sampling> sampling_node;
  RETURN_IF_NOT_OK(GetSamplingNode(kCpuSamplerName, &sampling_node));
  auto node = std::dynamic_pointer_cast<CpuSampler>(sampling_node);
  return node->GetProcessMemoryInfo(metric, start_ts, end_ts, result);
}

Status ProfilingManager::GetSystemMemoryInfoByEpoch(SystemMemoryMetric metric, int32_t epoch_num,
                                                    std::vector<float> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetSystemMemoryInfoByTime(metric, start_ts, end_ts, result);
}

Status ProfilingManager::GetSystemMemoryInfoByStep(SystemMemoryMetric metric, int32_t start_step, int32_t end_step,
                                                   std::vector<float> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(StepToTimeInterval(start_step, end_step, &start_ts, &end_ts));
  return GetSystemMemoryInfoByTime(metric, start_ts, end_ts, result);
}

Status ProfilingManager::GetSystemMemoryInfoByTime(SystemMemoryMetric metric, uint64_t start_ts, uint64_t end_ts,
                                                   std::vector<float> *result) {
  std::shared_ptr<Sampling> sampling_node;
  RETURN_IF_NOT_OK(GetSamplingNode(kCpuSamplerName, &sampling_node));
  auto node = std::dynamic_pointer_cast<CpuSampler>(sampling_node);
  return node->GetSystemMemoryInfo(metric, start_ts, end_ts, result);
}
#endif

Status ProfilingManager::EpochToTimeInterval(int32_t epoch_num, uint64_t *start_ts, uint64_t *end_ts) {
  RETURN_UNEXPECTED_IF_NULL(start_ts);
  RETURN_UNEXPECTED_IF_NULL(end_ts);
  if (epoch_num <= 0 || epoch_num >= static_cast<int32_t>(epoch_end_ts_.size())) {
    std::string err = "Epoch: " + std::to_string(epoch_num) + " is invalid.";
    MS_LOG(INFO) << err;
    return {StatusCode::kMDUnexpectedError, err};
  }
  *start_ts = epoch_end_ts_[epoch_num - 1];
  *end_ts = epoch_end_ts_[epoch_num];
  return Status::OK();
}

Status ProfilingManager::EpochToStepInterval(int32_t epoch_num, uint32_t *start_step, uint32_t *end_step) {
  RETURN_UNEXPECTED_IF_NULL(start_step);
  RETURN_UNEXPECTED_IF_NULL(end_step);
  if (epoch_num <= 0 || epoch_num >= static_cast<int32_t>(epoch_end_step_.size())) {
    std::string err = "Epoch: " + std::to_string(epoch_num) + " is invalid.";
    return {StatusCode::kMDUnexpectedError, err};
  }
  *start_step = epoch_end_step_[epoch_num - 1] + 1;
  *end_step = epoch_end_step_[epoch_num];
  return Status::OK();
}

Status ProfilingManager::StepToTimeInterval(int32_t start_step, int32_t end_step, uint64_t *start_ts,
                                            uint64_t *end_ts) {
  std::shared_ptr<Tracing> node;
  if (GetTracingNode(kDeviceQueueTracingName, &node).IsOk() ||
      GetTracingNode(kDatasetIteratorTracingName, &node).IsOk()) {
    return node->TimeIntervalForStepRange(start_step, end_step, start_ts, end_ts);
  } else {
    return {StatusCode::kMDUnexpectedError,
            "Cannot find appropriate tracing node to convert step range to time interval."};
  }
}

Status ProfilingManager::TimeToStepInterval(uint64_t start_ts, uint64_t end_ts, int32_t *start_step,
                                            int32_t *end_step) {
  std::shared_ptr<Tracing> node;
  if (GetTracingNode(kDeviceQueueTracingName, &node).IsOk() ||
      GetTracingNode(kDatasetIteratorTracingName, &node).IsOk()) {
    return node->StepIntervalForTimeRange(start_ts, end_ts, start_step, end_step);
  } else {
    return {StatusCode::kMDUnexpectedError,
            "Cannot find appropriate tracing node to convert time interval to step range."};
  }
}

Status ProfilingManager::GetConnectorSizeByEpoch(int32_t op_id, int32_t epoch_num, std::vector<int32_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetConnectorSizeByTime(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetConnectorSizeByStep(int32_t op_id, int32_t start_step, int32_t end_step,
                                                std::vector<int32_t> *result) {
  uint64_t start_ts = 0, end_ts = 0;
  RETURN_IF_NOT_OK(StepToTimeInterval(start_step, end_step, &start_ts, &end_ts));
  return GetConnectorSizeByTime(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetConnectorSizeByTime(int32_t op_id, uint64_t start_ts, uint64_t end_ts,
                                                std::vector<int32_t> *result) {
  std::shared_ptr<Sampling> node;
  RETURN_IF_NOT_OK(GetSamplingNode(kConnectorSizeSamplingName, &node));
  auto connector_node = std::dynamic_pointer_cast<ConnectorSize>(node);
  return connector_node->GetOpConnectorSize(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetPipelineTimeByEpoch(int32_t epoch_num, std::vector<int32_t> *result) {
  uint32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(EpochToStepInterval(epoch_num, &start_step, &end_step));
  return GetPipelineTimeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetPipelineTimeByStep(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  std::shared_ptr<Tracing> node;
  if (GetTracingNode(kDeviceQueueTracingName, &node).IsOk() ||
      GetTracingNode(kDatasetIteratorTracingName, &node).IsOk()) {
    return node->GetPipelineTime(start_step, end_step, result);
  } else {
    return {StatusCode::kMDUnexpectedError, "Cannot find appropriate tracing node"};
  }
}

Status ProfilingManager::GetPipelineTimeByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result) {
  int32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetPipelineTimeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetPushTimeByEpoch(int32_t epoch_num, std::vector<int32_t> *result) {
  uint32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(EpochToStepInterval(epoch_num, &start_step, &end_step));
  return GetPushTimeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetPushTimeByStep(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  std::shared_ptr<Tracing> node;
  if (GetTracingNode(kDeviceQueueTracingName, &node).IsOk() ||
      GetTracingNode(kDatasetIteratorTracingName, &node).IsOk()) {
    return node->GetPushTime(start_step, end_step, result);
  } else {
    return {StatusCode::kMDUnexpectedError, "Cannot find appropriate tracing node"};
  }
}

Status ProfilingManager::GetPushTimeByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result) {
  int32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetPushTimeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetBatchTimeByEpoch(int32_t epoch_num, std::vector<int32_t> *result) {
  uint32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(EpochToStepInterval(epoch_num, &start_step, &end_step));
  return GetBatchTimeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetBatchTimeByStep(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  std::shared_ptr<Tracing> node;
  if (GetTracingNode(kDeviceQueueTracingName, &node).IsOk() ||
      GetTracingNode(kDatasetIteratorTracingName, &node).IsOk()) {
    return node->GetBatchTime(start_step, end_step, result);
  } else {
    return {StatusCode::kMDUnexpectedError, "Cannot find appropriate tracing node"};
  }
}

Status ProfilingManager::GetBatchTimeByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result) {
  int32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetBatchTimeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetConnectorSizeByEpoch(int32_t epoch_num, std::vector<int32_t> *result) {
  uint32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(EpochToStepInterval(epoch_num, &start_step, &end_step));
  return GetConnectorSizeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetConnectorSizeByStep(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  std::shared_ptr<Tracing> node;
  if (GetTracingNode(kDeviceQueueTracingName, &node).IsOk() ||
      GetTracingNode(kDatasetIteratorTracingName, &node).IsOk()) {
    return node->GetConnectorSize(start_step, end_step, result);
  } else {
    return {StatusCode::kMDUnexpectedError, "Cannot find appropriate tracing node"};
  }
}

Status ProfilingManager::GetConnectorSizeByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result) {
  int32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetConnectorSizeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetEmptyQueueFrequencyByEpoch(int32_t epoch_num, float_t *result) {
  uint32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(EpochToStepInterval(epoch_num, &start_step, &end_step));
  return GetEmptyQueueFrequencyByStep(start_step, end_step, result);
}

Status ProfilingManager::GetEmptyQueueFrequencyByStep(int32_t start_step, int32_t end_step, float_t *result) {
  std::shared_ptr<Tracing> node;
  if (GetTracingNode(kDeviceQueueTracingName, &node).IsOk() ||
      GetTracingNode(kDatasetIteratorTracingName, &node).IsOk()) {
    return node->GetEmptyQueueFrequency(start_step, end_step, result);
  } else {
    return {StatusCode::kMDUnexpectedError, "Cannot find appropriate tracing node"};
  }
}

Status ProfilingManager::GetEmptyQueueFrequencyByTime(uint64_t start_ts, uint64_t end_ts, float_t *result) {
  int32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetEmptyQueueFrequencyByStep(start_step, end_step, result);
}

Status ProfilingManager::GetConnectorCapacityByEpoch(int32_t epoch_num, std::vector<int32_t> *result) {
  uint32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(EpochToStepInterval(epoch_num, &start_step, &end_step));
  return GetConnectorCapacityByStep(start_step, end_step, result);
}

Status ProfilingManager::GetConnectorCapacityByStep(int32_t start_step, int32_t end_step,
                                                    std::vector<int32_t> *result) {
  std::shared_ptr<Tracing> node;
  if (GetTracingNode(kDeviceQueueTracingName, &node).IsOk() ||
      GetTracingNode(kDatasetIteratorTracingName, &node).IsOk()) {
    return node->GetConnectorCapacity(start_step, end_step, result);
  } else {
    return {StatusCode::kMDUnexpectedError, "Cannot find appropriate tracing node"};
  }
}

Status ProfilingManager::GetConnectorCapacityByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result) {
  int32_t start_step = 0, end_step = 0;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetConnectorCapacityByStep(start_step, end_step, result);
}

Status ProfilingManager::GetNumberOfProfiledSteps(int32_t *steps) {
  std::shared_ptr<Tracing> node;
  if (GetTracingNode(kDeviceQueueTracingName, &node).IsOk() ||
      GetTracingNode(kDatasetIteratorTracingName, &node).IsOk()) {
    *steps = node->GetNumberSteps();
    return Status::OK();
  } else {
    return {StatusCode::kMDUnexpectedError, "Cannot find appropriate tracing node"};
  }
}

void ProfilingManager::RecordEndOfEpoch(uint32_t step_num) {
  if (profiling_state_ != ProfilingState::kProfilingStateRunning) {
    return;
  }
  MS_LOG(INFO) << "Recording end of epoch. step_num: " << step_num;
  (void)epoch_end_ts_.emplace_back(ProfilingTime::GetCurMilliSecond());
  (void)epoch_end_step_.emplace_back(step_num);
}

Status ProfilingManager::Reset() {
  for (const auto &node : tracing_nodes_) {
    node.second->Clear();
  }
  for (const auto &node : sampling_nodes_) {
    node.second->Clear();
  }
  epoch_end_ts_.clear();
  epoch_end_step_.clear();
  profiling_state_ = ProfilingState::kProfilingStateUnBegun;
  autotuning_ = false;
  profiling_ = false;
  return Status::OK();
}

Status ProfilingManager::Init(const bool for_autotune) {
  // Reinitialization should only be done in case of UT with sequential pipelines and should not be used externally.
  // Reinitialization with parallel data pipelines can have unexpected consequences.
  CHECK_FAIL_RETURN_UNEXPECTED(!autotuning_, "Stop MD Autotune before initializing the MD Profiler.");
  CHECK_FAIL_RETURN_UNEXPECTED(!profiling_, "Stop MD Profiler before initializing it.");
  CHECK_FAIL_RETURN_UNEXPECTED(profiling_state_ != ProfilingState::kProfilingStateRunning,
                               "Stop MD Profiler before reinitializing it.");
  RETURN_IF_NOT_OK(Reset());
  tracing_nodes_.clear();
  sampling_nodes_.clear();
  tree_ = nullptr;
  CHECK_FAIL_RETURN_UNEXPECTED(profiling_state_ == ProfilingState::kProfilingStateUnBegun,
                               "MD Profiler is in an unexpected state.");
  if (for_autotune) {
    autotuning_ = true;
    MS_LOG(INFO) << "MD profiler is initialized successfully for autotuning.";
  } else {
    profiling_ = true;
    MS_LOG(INFO) << "MD profiler is initialized successfully for profiling.";
  }
  return Status::OK();
}

Status ProfilingManager::Start() {
  CHECK_FAIL_RETURN_UNEXPECTED(profiling_state_ != ProfilingState::kProfilingStateRunning,
                               "MD ProfilingManager is already running.");
  if (profiling_state_ == ProfilingState::kProfilingStateFinished) {
    // This scenario (start, stop, and then start again) only happens in profiling, not autotune.
    MS_LOG(INFO) << "MD ProfilingManager had already stopped. Resetting...";
    RETURN_IF_NOT_OK(Reset());
    for (const auto &node : sampling_nodes_) {
      RETURN_IF_NOT_OK(node.second->Init());
    }
    for (const auto &node : tracing_nodes_) {
      RETURN_IF_NOT_OK(node.second->Init());
    }
    profiling_ = true;
    MS_LOG(INFO) << "MD profiler is reset successfully for profiling.";
  }

  profiling_state_ = ProfilingState::kProfilingStateRunning;
  for (const auto &node : tracing_nodes_) {
    RETURN_IF_NOT_OK(node.second->Start());
  }
  for (const auto &node : sampling_nodes_) {
    RETURN_IF_NOT_OK(node.second->Start());
  }
  MS_LOG(INFO) << "MD profiler is started.";
  return Status::OK();
}

Status ProfilingManager::Stop() {
  CHECK_FAIL_RETURN_UNEXPECTED(profiling_state_ != ProfilingState::kProfilingStateUnBegun,
                               "MD ProfilingManager has not started yet.");
  // It's OK if we are in kProfilingStateFinished state. We allow user to call Stop twice.
  if (profiling_state_ == ProfilingState::kProfilingStateFinished) {
    MS_LOG(WARNING) << "MD ProfilingManager had already stopped.";
    return Status::OK();
  }

  for (const auto &node : tracing_nodes_) {
    RETURN_IF_NOT_OK(node.second->Stop());
  }
  for (const auto &node : sampling_nodes_) {
    RETURN_IF_NOT_OK(node.second->Stop());
  }
  profiling_state_ = ProfilingState::kProfilingStateFinished;
  if (autotuning_) {
    autotuning_ = false;
    MS_LOG(INFO) << "MD Autotune is stopped.";
  }
  if (profiling_) {
    profiling_ = false;
    MS_LOG(INFO) << "MD Profiler is stopped.";
  }
  return Status::OK();
}

Status ProfilingManager::Save(const std::string &profile_data_path) {
  // Validate input profile data path
  CHECK_FAIL_RETURN_UNEXPECTED(!profile_data_path.empty(), "Invalid parameter, Profiling directory is not set.");
  CHECK_FAIL_RETURN_UNEXPECTED(profile_data_path.size() < PATH_MAX, "Invalid file, Profiling directory is invalid.");

  //  profiling file: <profile_data_path>/filename_rank_id.suffix
  char real_path[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path, common::SafeCStr(profile_data_path), PATH_MAX) == nullptr) {
    RETURN_STATUS_UNEXPECTED("Profiling dir is invalid.");
  }
#else
  if (realpath(common::SafeCStr(profile_data_path), real_path) == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid file, can not get realpath of Profiling directory.");
  }
#endif

  std::string rank_id = GetRankID();
  // Output all profiling data upon request.
  RETURN_IF_NOT_OK(SaveProfilingData(std::string(profile_data_path), rank_id));
  RETURN_IF_NOT_OK(ChangeFileMode(std::string(profile_data_path), rank_id));
  return Status::OK();
}

ProfilingManager::ProfilingRegistrationState ProfilingManager::GetProfilerTreeState(const ExecutionTree *tree) const {
  auto enabled = (profiling_ || autotuning_);
  if (!enabled) {
    return kNotEnabled;
  }
  if (tree_ == nullptr) {
    return kEnabledTreeNotRegistered;
  } else {
    return tree_ == tree ? kEnabledTreeRegistered : kEnabledDifferentTreeRegistered;
  }
}

std::string ProfilingManager::GetRankID() const {
  std::string rank_id = common::GetEnv("RANK_ID");
#ifdef WITH_BACKEND
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice) {
    std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
    int32_t rank_id_int = cfg->rank_id();
    // If DEVICE_ID is not set, default value is 0
    if (rank_id_int < 0) {
      rank_id = common::GetEnv("DEVICE_ID");
    } else {
      rank_id = std::to_string(rank_id_int);
    }
  }
#endif
  // If RANK_ID is not set, default value is 0
  if (rank_id.empty()) {
    rank_id = "0";
  }
  return rank_id;
}

uint64_t ProfilingTime::GetCurMilliSecond() {
  // because cpplint does not allow using namespace
  using std::chrono::duration_cast;
  using std::chrono::milliseconds;
  using std::chrono::steady_clock;
  return static_cast<uint64_t>(duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}
}  // namespace dataset
}  // namespace mindspore
