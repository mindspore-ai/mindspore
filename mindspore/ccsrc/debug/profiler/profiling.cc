/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "include/backend/debug/profiler/profiling.h"
#include <chrono>
#include <cmath>
#include <sstream>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#ifdef __linux__
#include <unistd.h>
#include "utils/profile.h"
#include "include/backend/debug/common/csv_writer.h"
#endif

namespace mindspore {
namespace profiler {
#ifdef __linux__
constexpr auto HostDataHeader =
  "tid,pid,parent_pid,module_name,event,stage,level,start_end,custom_info,memory_usage(kB),time_stamp(us)\n";
const auto kVmRSS = "VmRSS";
std::mutex file_line_mutex;
static bool log_once = false;
static bool first_open_file = true;
const int profile_all = 0;
const int profile_memory = 1;
const int profile_time = 2;
#endif
std::map<std::string, std::shared_ptr<Profiler>> &Profiler::GetInstanceMap() {
  static std::map<std::string, std::shared_ptr<Profiler>> instance_map = {};
  return instance_map;
}

std::shared_ptr<Profiler> Profiler::GetInstance(const std::string &name) noexcept {
  if (auto iter = GetInstanceMap().find(name); iter != GetInstanceMap().end()) {
    return iter->second;
  }

  return nullptr;
}

void Profiler::SyncEnable(const bool enable_flag) {
  MS_LOG(INFO) << "Profiler synchronous enable flag:" << enable_flag;
  sync_enable_flag_ = enable_flag;
}

void Profiler::DataProcessEnable(const bool enable_flag) {
  MS_LOG(INFO) << "Profiler data process enable flag:" << enable_flag;
  data_process_enable_ = enable_flag;
}

bool Profiler::Register(const std::string &name, const std::shared_ptr<Profiler> &instance) {
  if (GetInstanceMap().find(name) != GetInstanceMap().end()) {
    MS_LOG(WARNING) << name << " has been registered.";
  } else {
    (void)GetInstanceMap().emplace(name, instance);
  }

  return true;
}

void Profiler::Clear() { GetInstanceMap().clear(); }

uint64_t Profiler::GetHostMonoTimeStamp() const {
  auto now = std::chrono::steady_clock::now();
  int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  return static_cast<uint64_t>(ns);
}

uint64_t Profiler::GetRealTimeStamp() const {
  auto now = std::chrono::steady_clock::now();
  int64_t ms = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  return static_cast<uint64_t>(ms);
}

void Profiler::SetRunTimeData(const std::string &op_name, const float time_elapsed) {
  std::shared_lock<std::shared_mutex> lock(op_map_mutex_);
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.op_host_cost_time += time_elapsed;
  }
}

void Profiler::SetRunTimeData(const std::string &op_name, const uint64_t start, const float duration) {
  std::shared_lock<std::shared_mutex> lock(op_map_mutex_);
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.start_duration.emplace_back(StartDuration({start, duration}));
  }
}

void Profiler::RecordOneStepStartEndInfo() {
  // Multi-graph dotting data is not supported.
  std::lock_guard<std::mutex> locker(record_mutex_);
  uint32_t vector_size = static_cast<uint32_t>(step_start_end_info_vector_.size());
  if (vector_size == 0) {
    return;
  }
  step_start_end_info_.iter_start_op_name = step_start_end_info_vector_[0];
  step_start_end_info_.fp_start_op_name = step_start_end_info_vector_[0];
  // If is the first step, the step_start_end_info_vector_ length is 1.
  if (vector_size > 1) {
    FindOneStepFpStartOp(vector_size);
    FindOneStepIterEndOp(vector_size);
    if (has_find_) {
      all_step_start_end_info_.push_back(step_start_end_info_);
      // Delete the operator of the current step.
      (void)step_start_end_info_vector_.erase(step_start_end_info_vector_.begin(),
                                              step_start_end_info_vector_.begin() + iter_end_op_index_ + 1);
    } else {
      all_step_start_end_info_.push_back(step_start_end_info_);
      step_start_end_info_vector_.clear();
    }
  } else {
    all_step_start_end_info_.push_back(step_start_end_info_);
    step_start_end_info_vector_.clear();
  }
  step_start_end_info_.iter_start_op_name = "";
  step_start_end_info_.fp_start_op_name = "";
  step_start_end_info_.iter_end_op_name = "";
  has_find_ = false;
  iter_end_op_index_ = 0;
}

void Profiler::FindOneStepFpStartOp(uint32_t vector_size) {
  std::string type = "";
  bool find_fp_start = false;
  for (uint32_t index = 0; index < vector_size; ++index) {
    std::string op_name = step_start_end_info_vector_[index];
    auto begin_iter = op_name.rfind('/') + 1;
    auto end_iter = op_name.rfind('-');
    if (begin_iter != std::string::npos && end_iter != std::string::npos && begin_iter < end_iter) {
      type = op_name.substr(begin_iter, end_iter - begin_iter);
    }
    if (type == op_type_) {
      find_fp_start = true;
      if (index == 0) {
        // If the type of the first operator is GetNext, the next operator of it is the fp_start operator.
        step_start_end_info_.fp_start_op_name = step_start_end_info_vector_[index + 1];
      } else {
        // If the data processing operator is iter_start, the type of the fp_start operator should be GetNext.
        step_start_end_info_.fp_start_op_name = op_name;
      }
      break;
    }
  }
  if (!find_fp_start) {
    step_start_end_info_.fp_start_op_name = step_start_end_info_vector_[fp_start_op_index_];
  }
}

void Profiler::FindOneStepIterEndOp(uint32_t vector_size) {
  // Iterate through step_start_end_info_vector_ for the repeat operator, which is the operator of the next step and
  // is preceded by iter_end_op of the current step.
  std::string step_end_op_name;
  for (uint32_t rindex = vector_size - 1; rindex > 0; --rindex) {
    step_end_op_name = step_start_end_info_vector_[rindex];
    uint32_t lindex = 0;
    for (; lindex < rindex; ++lindex) {
      if (step_end_op_name == step_start_end_info_vector_[lindex]) {
        has_find_ = true;
        iter_end_op_index_ = rindex - 1;
        break;
      }
    }
    if (rindex == lindex) {
      break;
    }
  }
  if (has_find_) {
    step_start_end_info_.iter_end_op_name = step_start_end_info_vector_[iter_end_op_index_];
  } else {
    step_start_end_info_.iter_end_op_name = step_start_end_info_vector_[step_start_end_info_vector_.size() - 1];
  }
}

void Profiler::RecordOneStepStartEndInfo(const std::string op_name) {
  std::lock_guard<std::mutex> locker(record_mutex_);
  if (step_start_end_info_.iter_start_op_name.empty()) {
    step_start_end_info_.iter_start_op_name = op_name;
    step_start_end_info_.fp_start_op_name = op_name;
  }

  std::string fp_start_op_name = step_start_end_info_.fp_start_op_name;

  auto op_type_begin_iter = fp_start_op_name.rfind('/') + 1;
  auto op_type_end_iter = fp_start_op_name.rfind('-');
  auto op_type = fp_start_op_name.substr(op_type_begin_iter, op_type_end_iter - op_type_begin_iter);
  if (op_type == "InitDataSetQueue" || op_type == "GetNext") {
    step_start_end_info_.fp_start_op_name = op_name;
  }
  step_start_end_info_.iter_end_op_name = op_name;
  step_start_end_info_vector_.push_back(op_name);
}

std::shared_ptr<ProfilerManager> &ProfilerManager::GetInstance() {
  MS_EXCEPTION_IF_NULL(profiler_manager_inst_);
  return profiler_manager_inst_;
}

bool ProfilerManager::GetProfilingEnableFlag() const {
  if (auto gpu_instance = Profiler::GetInstance(kGPUDevice); gpu_instance != nullptr) {
    return gpu_instance->GetEnableFlag();
  }

  if (auto ascend_instance = Profiler::GetInstance(kAscendDevice); ascend_instance != nullptr) {
    return ascend_instance->GetEnableFlag();
  }

  return false;
}

void ProfilerManager::RecordOneStepStartEndInfo() const {
  if (auto gpu_instance = Profiler::GetInstance(kGPUDevice); gpu_instance != nullptr && gpu_instance->GetEnableFlag()) {
    gpu_instance->RecordOneStepStartEndInfo();
  }
}

std::string ProfilerManager::GetProfilingOptions() const {
  if (auto ascend_instance = Profiler::GetInstance(kAscendDevice); ascend_instance != nullptr) {
    return ascend_instance->GetProfilingOptions();
  }

  return "";
}

std::string ProfilerManager::ProfileDataPath() const {
  if (auto gpu_instance = Profiler::GetInstance(kGPUDevice); gpu_instance != nullptr) {
    return gpu_instance->ProfileDataPath();
  }

  if (auto ascend_instance = Profiler::GetInstance(kAscendDevice); ascend_instance != nullptr) {
    return ascend_instance->ProfileDataPath();
  }

  return "";
}

void ProfilerManager::SetProfileFramework(const std::string &profile_framework) {
  profile_framework_ = profile_framework;
}

bool ProfilerManager::NeedCollectHostTime() const {
  return profile_framework_ == "all" || profile_framework_ == "time";
}

bool ProfilerManager::NeedCollectHostMemory() const {
  return profile_framework_ == "memory" || profile_framework_ == "all";
}

bool ProfilerManager::EnableCollectHost() const { return profile_framework_ != "NULL"; }

void CollectHostInfo(const std::string &module_name, const std::string &event, const std::string &stage, int level,
                     int profile_framework, int start_end, const std::map<std::string, std::string> &custom_info) {
#ifndef ENABLE_SECURITY
#ifndef __linux__
  return;
#else
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  if (!log_once && !profiler_manager->GetProfilingEnableFlag()) {
    MS_LOG(DEBUG) << "Profiler is not enabled, no need to record Host info.";
    log_once = true;
    return;
  } else if (!profiler_manager->GetProfilingEnableFlag()) {
    return;
  }
  if (!log_once && !profiler_manager->EnableCollectHost()) {
    MS_LOG(DEBUG) << "Profiler profile_framework is not enabled, no need to record Host info.";
    log_once = true;
    return;
  } else if (!profiler_manager->EnableCollectHost()) {
    return;
  }
  auto output_path = profiler_manager->ProfileDataPath();
  if (output_path.empty()) {
    MS_LOG(ERROR) << "The output path is empty, skip collect host info.";
    return;
  }
  HostProfileData host_profile_data;
  host_profile_data.module_name = module_name;
  host_profile_data.event = event;
  host_profile_data.stage = stage;
  host_profile_data.level = level;
  host_profile_data.start_end = start_end;
  host_profile_data.custom_info = custom_info;
  auto tid = std::this_thread::get_id();
  host_profile_data.tid = tid;

  host_profile_data.pid = getpid();
  host_profile_data.parent_pid = getppid();

  // Collect Host info.
  if (profiler_manager->NeedCollectHostTime()) {
    auto now = std::chrono::system_clock::now();
    int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    uint64_t time_stamp = static_cast<uint64_t>(us);
    host_profile_data.time_stamp = time_stamp;
  }
  if ((profile_framework == profile_all || profile_framework == profile_memory) &&
      profiler_manager->NeedCollectHostMemory()) {
    ProcessStatus process_status = ProcessStatus::GetInstance();
    int64_t memory_usage = process_status.GetMemoryCost(kVmRSS);
    host_profile_data.memory_usage = memory_usage;
  }
  std::lock_guard<std::mutex> lock(file_line_mutex);
  WriteHostDataToFile(host_profile_data, output_path);
  return;
#endif
#endif
}
#ifdef __linux__
void WriteHostDataToFile(const HostProfileData &host_profile_data, const std::string &output_path) {
  if (host_profile_data.memory_usage == 0 && host_profile_data.time_stamp == 0) {
    return;
  }
  std::string file_name = "host_info_0.csv";
  std::string rank_id = common::GetEnv("RANK_ID");
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice) {
    rank_id = std::to_string(context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  }
  if (!rank_id.empty()) {
    file_name = "host_info_" + rank_id + ".csv";
  }
  std::string csv_file = output_path + "/host_info/" + file_name;
  // try to open file
  CsvWriter &csv = CsvWriter::GetInstance();
  int retry = 2;
  while (retry > 0) {
    if (first_open_file && csv.OpenFile(csv_file, HostDataHeader, true)) {
      first_open_file = false;
      break;
    } else if (csv.OpenFile(csv_file, HostDataHeader)) {
      break;
    }
    retry--;
  }
  if (retry == 0) {
    MS_LOG(WARNING) << "Open csv file failed, skipping saving host info to file.";
    return;
  }
  std::string row = "";
  const std::string kSeparator = ",";
  std::ostringstream ss_tid;
  ss_tid << host_profile_data.tid;
  row.append(ss_tid.str() + kSeparator);
  row.append(std::to_string(host_profile_data.pid) + kSeparator);
  row.append(std::to_string(host_profile_data.parent_pid) + kSeparator);
  row.append(host_profile_data.module_name + kSeparator);
  row.append(host_profile_data.event + kSeparator);
  row.append(host_profile_data.stage + kSeparator);
  row.append(std::to_string(host_profile_data.level) + kSeparator);
  row.append(std::to_string(host_profile_data.start_end) + kSeparator);
  if (!host_profile_data.custom_info.empty()) {
    std::string custom_info_str = "{";
    for (auto it = host_profile_data.custom_info.cbegin(); it != host_profile_data.custom_info.cend(); ++it) {
      custom_info_str += "{" + it->first + ":" + it->second + "}";
    }
    custom_info_str += "}";
    if (custom_info_str.find(",") != std::string::npos) {
      custom_info_str = "\"" + custom_info_str + "\"";
    }
    row.append(custom_info_str + kSeparator);
  } else {
    row.append(kSeparator);
  }
  row.append(std::to_string(host_profile_data.memory_usage) + kSeparator);
  row.append(std::to_string(host_profile_data.time_stamp));

  csv.WriteToCsv(row, true);
}
#endif
}  // namespace profiler
}  // namespace mindspore
