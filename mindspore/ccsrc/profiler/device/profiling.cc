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

#include "profiler/device/profiling.h"

#include <cxxabi.h>
#include <cmath>
#include <ctime>
#include "pybind_api/api_register.h"
#include "utils/log_adapter.h"
#include "utils/utils.h"
#if ENABLE_GPU
#include "profiler/device/gpu/gpu_profiling.h"
#endif
#if ENABLE_D
#include "profiler/device/ascend/ascend_profiling.h"
#endif

namespace mindspore {
namespace profiler {
std::shared_ptr<ProfilerManager> ProfilerManager::profiler_manager_inst_ = std::make_shared<ProfilerManager>();

uint64_t Profiler::GetHostMonoTimeStamp() const {
  struct timespec ts;
#if defined(_WIN32) || defined(_WIN64)
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    MS_LOG(ERROR) << "Get host timestamp failed";
    return 0;
  }
#else
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    MS_LOG(ERROR) << "Get host timestamp failed";
    return 0;
  }
#endif
  constexpr uint64_t kNSecondInSecond = 1000000000;
  uint64_t cur_time_stamp = ts.tv_sec * kNSecondInSecond + ts.tv_nsec;
  return cur_time_stamp;
}

void Profiler::SetRunTimeData(const std::string &op_name, const float time_elapsed) {
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.op_host_cost_time += time_elapsed;
  }
}

void Profiler::SetRunTimeData(const std::string &op_name, const uint64_t start, const float duration) {
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.start_duration.emplace_back(StartDuration({start, duration}));
  }
}

void Profiler::RecordOneStepStartEndInfo() {
  std::lock_guard<std::mutex> locker(record_mutex_);
  all_step_start_end_info_.push_back(step_start_end_info_);
  step_start_end_info_.iter_start_op_name = "";
  step_start_end_info_.fp_start_op_name = "";
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
}

std::shared_ptr<ProfilerManager> &ProfilerManager::GetInstance() {
  MS_EXCEPTION_IF_NULL(profiler_manager_inst_);
  return profiler_manager_inst_;
}

bool ProfilerManager::GetProfilingEnableFlag() const {
#if ENABLE_GPU
  return profiler::gpu::GPUProfiler::GetInstance()->GetEnableFlag();
#endif
#if ENABLE_D
  auto ascend_instance = profiler::ascend::AscendProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(ascend_instance);
  return ascend_instance->GetProfilingEnableFlag();
#endif
  return false;
}

void ProfilerManager::RecordOneStepStartEndInfo() const {
#if ENABLE_GPU
  auto gpu_profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  if (gpu_profiler_inst->GetEnableFlag()) {
    gpu_profiler_inst->RecordOneStepStartEndInfo();
  }
#endif
}

std::string ProfilerManager::GetProfilingOptions() const {
#if ENABLE_D
  auto ascend_instance = profiler::ascend::AscendProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(ascend_instance);
  return ascend_instance->GetProfilingOptions();
#endif
  return "";
}
}  // namespace profiler
}  // namespace mindspore
