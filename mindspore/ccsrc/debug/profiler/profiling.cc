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

#include "include/backend/debug/profiler/profiling.h"
#include <chrono>
#include <cmath>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace profiler {
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
}  // namespace profiler
}  // namespace mindspore
