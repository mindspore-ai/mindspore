/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include "utils/ms_utils.h"
#include "minddata/dataset/util/path.h"
#ifdef ENABLE_GPUQUE
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#endif
#include "minddata/dataset/engine/perf/monitor.h"
#include "minddata/dataset/engine/perf/device_queue_tracing.h"
#include "minddata/dataset/engine/perf/connector_size.h"
#include "minddata/dataset/engine/perf/cpu_sampler.h"
#include "minddata/dataset/engine/perf/dataset_iterator_tracing.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {

Status Tracing::SaveToFile() {
  if (value_.empty()) {
    return Status::OK();
  }

  std::ofstream handle(file_path_, std::ios::trunc);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Profiling file can not be opened.");
  }
  for (auto value : value_) {
    handle << value << "\n";
  }
  handle.close();

  return Status::OK();
}

Status Tracing::ChangeFileMode() {
  if (value_.empty()) {
    return Status::OK();
  }

  if (chmod(common::SafeCStr(file_path_), S_IRUSR | S_IWUSR) == -1) {
    std::string err_str = "Change file mode failed," + file_path_;
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
  TracingRecord record = {type, extra_info, batch_num, value, time_stamp};
  std::lock_guard<std::mutex> guard(lock_);
  (void)records_.emplace_back(record);
  (void)value_.emplace_back(record.ToString());
  // save timestamp per batch
  if (records_.size() % records_per_step_ == 0) {
    (void)ts_.emplace_back(time_stamp);
  }
}

Status Tracing::TimeIntervalForStepRange(int32_t start_step, int32_t end_step, uint64_t *start_ts, uint64_t *end_ts) {
  std::lock_guard<std::mutex> guard(lock_);
  MS_LOG(DEBUG) << "start_step: " << start_step << " end_step: " << end_step;
  CHECK_FAIL_RETURN_UNEXPECTED(start_step > 0,
                               "Expected start_step > 0. Got start_step: " + std::to_string(start_step));
  CHECK_FAIL_RETURN_UNEXPECTED(end_step >= start_step,
                               "Expected end_step >= start_step. Got start_step: " + std::to_string(start_step) +
                                 " end_step: " + std::to_string(end_step));
  CHECK_FAIL_RETURN_UNEXPECTED(end_step < ts_.size(),
                               "Expected end_step < ts_.size(). Got end_step: " + std::to_string(end_step) +
                                 " ts_.size: " + std::to_string(ts_.size()));
  // end timestamp of (start_step - 1) step
  *start_ts = ts_[start_step - 1];
  *end_ts = ts_[end_step];
  return Status::OK();
}

Status Tracing::StepIntervalForTimeRange(uint64_t start_ts, uint64_t end_ts, int32_t *start_step, int32_t *end_step) {
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
                                         const std::string field, std::vector<int32_t> *result) {
  std::lock_guard<std::mutex> guard(lock_);
  auto total_steps = records_.size() / records_per_step_;
  MS_LOG(DEBUG) << "start_step: " << start_step << " end_step: " << end_step;
  CHECK_FAIL_RETURN_UNEXPECTED(start_step <= total_steps,
                               "Expected start_step <= total_steps. Got start_step: " + std::to_string(start_step) +
                                 " total_steps: " + std::to_string(total_steps));
  CHECK_FAIL_RETURN_UNEXPECTED(end_step <= total_steps,
                               "Expected end_step <= total_steps. Got end_step: " + std::to_string(end_step) +
                                 " total_steps: " + std::to_string(total_steps));
  CHECK_FAIL_RETURN_UNEXPECTED(start_step <= end_step,
                               "Expected start_step <= end_step. Got start_step: " + std::to_string(start_step) +
                                 " end_step: " + std::to_string(end_step));

  for (auto step_num = start_step; step_num <= end_step; step_num++) {
    auto idx = (step_num - 1) * records_per_step_ + record_offset;
    if (field == "value") {
      (void)result->emplace_back(records_[idx].value);
    } else if (field == "extra_info") {
      (void)result->emplace_back(records_[idx].extra_info);
    } else {
      return {StatusCode::kMDUnexpectedError,
              "Received unexpected field: " + field + R"(. Expected: ["value", "extra_info"].)"};
    }
  }
  return Status::OK();
}

Tracing::Tracing(int32_t records_per_step) : records_per_step_(records_per_step) {}

Status Sampling::ReadJson(nlohmann::json *output) {
  RETURN_UNEXPECTED_IF_NULL(output);
  Path path = Path(file_path_);
  if (path.Exists()) {
    MS_LOG(DEBUG) << file_path_ << " exists";
    try {
      std::ifstream file(file_path_);
      file >> (*output);
    } catch (const std::exception &err) {
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to open json file: " + file_path_ +
                               ", please delete it and try again!");
    }
  } else {
    (*output)["sampling_interval"] = GlobalContext::config_manager()->monitor_sampling_interval();
  }
  return Status::OK();
}

// Constructor
ProfilingManager::ProfilingManager(TreeConsumer *tree_consumer) : tree_consumer_(tree_consumer), enabled_(true) {
  perf_monitor_ = std::make_unique<Monitor>(tree_consumer_);
}

bool ProfilingManager::IsProfilingEnable() const { return common::GetEnv("PROFILING_MODE") == "true" && enabled_; }

Status ProfilingManager::Initialize(ExecutionTree *tree) {
  // Initialize execution tree pointer
  tree_ = tree;
  // Initialize execution tree pointer in Monitor
  perf_monitor_->SetTree(tree);

  // Register nodes based on config
  std::string dir = common::GetEnv("MINDDATA_PROFILING_DIR");
  if (dir.empty()) {
    RETURN_STATUS_UNEXPECTED("Invalid parameter, Profiling directory is not set.");
  }
  char real_path[PATH_MAX] = {0};
  if (dir.size() >= PATH_MAX) {
    RETURN_STATUS_UNEXPECTED("Invalid file, Profiling directory is invalid.");
  }
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path, common::SafeCStr(dir), PATH_MAX) == nullptr) {
    RETURN_STATUS_UNEXPECTED("Profiling dir is invalid.");
  }
#else
  if (realpath(common::SafeCStr(dir), real_path) == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid file, can not get realpath of Profiling directory.");
  }
#endif
  dir_path_ = real_path;

#ifdef ENABLE_GPUQUE
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t rank_id = cfg->rank_id();
  // If DEVICE_ID is not set, default value is 0
  if (rank_id < 0) {
    device_id_ = common::GetEnv("DEVICE_ID");
    // If DEVICE_ID is not set, default value is 0
    if (device_id_.empty()) {
      device_id_ = "0";
    }
  } else {
    device_id_ = std::to_string(rank_id);
  }
#else
  device_id_ = common::GetEnv("RANK_ID");
  // If RANK_ID is not set, default value is 0
  if (device_id_.empty()) {
    device_id_ = "0";
  }
#endif

  // Register all profiling node.
  // device_queue node is used for graph mode
  std::shared_ptr<Tracing> device_queue_tracing = std::make_shared<DeviceQueueTracing>();
  RETURN_IF_NOT_OK(RegisterTracingNode(device_queue_tracing));

  // dataset_iterator node is used for graph mode
  std::shared_ptr<Tracing> dataset_iterator_tracing = std::make_shared<DatasetIteratorTracing>();
  RETURN_IF_NOT_OK(RegisterTracingNode(dataset_iterator_tracing));

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
Status ProfilingManager::RegisterTracingNode(std::shared_ptr<Tracing> node) {
  // Check if node with the same name has already been registered.
  auto exist = tracing_nodes_.find(node->Name());
  if (exist != tracing_nodes_.end()) {
    return Status(StatusCode::kMDProfilingError, "Profiling node already exist: " + node->Name());
  }
  // Register the node with its name as key.
  RETURN_IF_NOT_OK(node->Init(dir_path_, device_id_));
  tracing_nodes_[node->Name()] = node;
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
Status ProfilingManager::RegisterSamplingNode(std::shared_ptr<Sampling> node) {
  // Check if node with the same name has already been registered.
  auto exist = sampling_nodes_.find(node->Name());
  if (exist != sampling_nodes_.end()) {
    return Status(StatusCode::kMDProfilingError, "Profiling node already exist: " + node->Name());
  }
  // Register the node with its name as key.
  RETURN_IF_NOT_OK(node->Init(dir_path_, device_id_));
  sampling_nodes_[node->Name()] = node;
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

Status ProfilingManager::SaveProfilingData() {
  if (!IsProfilingEnable()) {
    return Status::OK();
  }
  MS_LOG(INFO) << "Start to save profiling data.";
  for (auto node : tracing_nodes_) {
    RETURN_IF_NOT_OK(node.second->SaveToFile());
  }
  for (auto node : sampling_nodes_) {
    RETURN_IF_NOT_OK(node.second->SaveToFile());
  }
  MS_LOG(INFO) << "Save profiling data end.";
  return Status::OK();
}

Status ProfilingManager::Analyze() {
  if (!IsProfilingEnable()) {
    return Status::OK();
  }
  MS_LOG(INFO) << "Start to analyze profiling data.";
  for (auto node : sampling_nodes_) {
    RETURN_IF_NOT_OK(node.second->Analyze());
  }
  return Status::OK();
}

Status ProfilingManager::ChangeFileMode() {
  if (!IsProfilingEnable()) {
    return Status::OK();
  }
  MS_LOG(INFO) << "Start to change file mode.";
  for (auto node : tracing_nodes_) {
    RETURN_IF_NOT_OK(node.second->ChangeFileMode());
  }
  for (auto node : sampling_nodes_) {
    RETURN_IF_NOT_OK(node.second->ChangeFileMode());
  }
  MS_LOG(INFO) << "Change file mode end.";
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status ProfilingManager::GetUserCpuUtilByEpoch(int32_t epoch_num, std::vector<uint8_t> *result) {
  uint64_t start_ts, end_ts;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetUserCpuUtilByTime(start_ts, end_ts, result);
}

Status ProfilingManager::GetUserCpuUtilByStep(int32_t start_step, int32_t end_step, std::vector<uint8_t> *result) {
  uint64_t start_ts, end_ts;
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
  uint64_t start_ts, end_ts;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetSysCpuUtilByTime(start_ts, end_ts, result);
}

Status ProfilingManager::GetSysCpuUtilByStep(int32_t start_step, int32_t end_step, std::vector<uint8_t> *result) {
  uint64_t start_ts, end_ts;
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
  uint64_t start_ts, end_ts;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetUserCpuUtilByTime(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetUserCpuUtilByStep(int32_t op_id, int32_t start_step, int32_t end_step,
                                              std::vector<uint16_t> *result) {
  uint64_t start_ts, end_ts;
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
  uint64_t start_ts, end_ts;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetSysCpuUtilByTime(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetSysCpuUtilByStep(int32_t op_id, int32_t start_step, int32_t end_step,
                                             std::vector<uint16_t> *result) {
  uint64_t start_ts, end_ts;
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
#endif

Status ProfilingManager::EpochToTimeInterval(int32_t epoch_num, uint64_t *start_ts, uint64_t *end_ts) {
  if (epoch_num <= 0 || epoch_num >= epoch_end_ts_.size()) {
    std::string err = "Epoch: " + std::to_string(epoch_num) + " is invalid.";
    MS_LOG(INFO) << err;
    return {StatusCode::kMDUnexpectedError, err};
  }
  *start_ts = epoch_end_ts_[epoch_num - 1];
  *end_ts = epoch_end_ts_[epoch_num];
  return Status::OK();
}

Status ProfilingManager::EpochToStepInterval(int32_t epoch_num, uint32_t *start_step, uint32_t *end_step) {
  if (epoch_num <= 0 || epoch_num >= epoch_end_step_.size()) {
    std::string err = "Epoch: " + std::to_string(epoch_num) + " is invalid.";
    MS_LOG(INFO) << err;
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
            "Cannot find appropriate tracing node to convert step range to time interval."};
  }
}

Status ProfilingManager::GetConnectorSizeByEpoch(int32_t op_id, int32_t epoch_num, std::vector<int32_t> *result) {
  uint64_t start_ts, end_ts;
  RETURN_IF_NOT_OK(EpochToTimeInterval(epoch_num, &start_ts, &end_ts));
  return GetConnectorSizeByTime(op_id, start_ts, end_ts, result);
}

Status ProfilingManager::GetConnectorSizeByStep(int32_t op_id, int32_t start_step, int32_t end_step,
                                                std::vector<int32_t> *result) {
  uint64_t start_ts, end_ts;
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
  uint32_t start_step, end_step;
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
  int32_t start_step, end_step;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetPipelineTimeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetPushTimeByEpoch(int32_t epoch_num, std::vector<int32_t> *result) {
  uint32_t start_step, end_step;
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
  int32_t start_step, end_step;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetPushTimeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetBatchTimeByEpoch(int32_t epoch_num, std::vector<int32_t> *result) {
  uint32_t start_step, end_step;
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
  int32_t start_step, end_step;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetBatchTimeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetConnectorSizeByEpoch(int32_t epoch_num, std::vector<int32_t> *result) {
  uint32_t start_step, end_step;
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
  int32_t start_step, end_step;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetConnectorSizeByStep(start_step, end_step, result);
}

Status ProfilingManager::GetEmptyQueueFrequencyByEpoch(int32_t epoch_num, float_t *result) {
  uint32_t start_step, end_step;
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
  int32_t start_step, end_step;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetEmptyQueueFrequencyByStep(start_step, end_step, result);
}

Status ProfilingManager::GetConnectorCapacityByEpoch(int32_t epoch_num, std::vector<int32_t> *result) {
  uint32_t start_step, end_step;
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
  int32_t start_step, end_step;
  RETURN_IF_NOT_OK(TimeToStepInterval(start_ts, end_ts, &start_step, &end_step));
  return GetConnectorCapacityByStep(start_step, end_step, result);
}

void ProfilingManager::RecordEndOfEpoch(uint32_t step_num) {
  MS_LOG(INFO) << "Recording end of epoch. step_num: " << step_num;
  (void)epoch_end_ts_.emplace_back(ProfilingTime::GetCurMilliSecond());
  (void)epoch_end_step_.emplace_back(step_num);
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
