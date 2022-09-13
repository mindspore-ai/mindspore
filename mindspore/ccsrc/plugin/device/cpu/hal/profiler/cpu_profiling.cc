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

#include "plugin/device/cpu/hal/profiler/cpu_profiling.h"
#include "plugin/device/cpu/hal/profiler/cpu_data_saver.h"
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace profiler {
namespace cpu {
namespace {
PROFILER_REG(kCPUDevice, CPUProfiler);
}  // namespace
std::shared_ptr<CPUProfiler> CPUProfiler::GetInstance() {
  auto instance = Profiler::GetInstance(kCPUDevice);
  MS_EXCEPTION_IF_NULL(instance);
  return std::dynamic_pointer_cast<CPUProfiler>(instance);
}

void CPUProfiler::Init(const std::string &profiling_path, uint32_t, const std::string &) {
  MS_LOG(INFO) << "Initialize CPU Profiling";
  base_time_ = GetHostMonoTimeStamp();
  profile_data_path_ = profiling_path;
  MS_LOG(INFO) << " Host start time(ns): " << base_time_ << " profile data path: " << profile_data_path_;
}

void CPUProfiler::StepProfilingEnable(const bool enable_flag) {
  MS_LOG(INFO) << "CPU Profiler enable flag: " << enable_flag;
  enable_flag_ = enable_flag;
}

void CPUProfiler::SetRunTimeData(const std::string &op_name, const uint32_t pid, bool is_parallel) {
  if (!is_parallel) {
    op_name_ = op_name;
    pid_ = pid;
  }
  {
    std::shared_lock<std::shared_mutex> lock(op_map_mutex_);
    auto iter = op_info_map_.find(op_name);
    if (iter != op_info_map_.end()) {
      iter->second.op_count += 1;
      return;
    }
  }
  std::unique_lock<std::shared_mutex> lock(op_map_mutex_);
  OpInfo op_info;
  op_info.op_name = op_name;
  op_info.pid = pid;
  op_info.op_count = 1;
  op_info_map_[op_name] = op_info;
}

void CPUProfiler::SetRuntimeStart(const std::string op_name, const uint64_t start_timestamp) {
  std::shared_lock<std::shared_mutex> lock(op_map_mutex_);
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.tmp_start_duration.start_timestamp = start_timestamp;
    auto actor_manager = ActorMgr::GetActorMgrRef();
    MS_EXCEPTION_IF_NULL(actor_manager);
    auto thread_pool = actor_manager->GetActorThreadPool();
    auto worker_ids_map = thread_pool->GetWorkerIdMap();
    auto id_iter = worker_ids_map.find(std::this_thread::get_id());
    if (id_iter != worker_ids_map.end()) {
      iter->second.tmp_start_duration.tid = id_iter->second;
    }
  }
}

float CPUProfiler::SetRuntimeEnd(const std::string op_name, const uint64_t stop_timestamp) {
  float op_time_elapsed = 0;
  std::shared_lock<std::shared_mutex> lock(op_map_mutex_);
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.tmp_start_duration.duration =
      (stop_timestamp - iter->second.tmp_start_duration.start_timestamp) / kNanosecondToMillisecond;
    auto actor_manager = ActorMgr::GetActorMgrRef();
    MS_EXCEPTION_IF_NULL(actor_manager);
    auto thread_pool = actor_manager->GetActorThreadPool();
    auto worker_ids_map = thread_pool->GetWorkerIdMap();
    auto id_iter = worker_ids_map.find(std::this_thread::get_id());
    if (id_iter != worker_ids_map.end()) {
      if (iter->second.tmp_start_duration.tid != id_iter->second) {
        MS_LOG(EXCEPTION) << "Op " << op_name << " start time thread id must be equal to end thread id.";
      }
    }
    (void)iter->second.start_duration.emplace_back(iter->second.tmp_start_duration);
    op_time_elapsed = iter->second.tmp_start_duration.duration;
  }
  return op_time_elapsed;
}

void CPUProfiler::OpDataProducerBeginParallel(const std::string op_name, const uint32_t pid) {
  auto start_timestamp = GetHostMonoTimeStamp();
  SetRunTimeData(op_name, pid, true);
  SetRuntimeStart(op_name, start_timestamp);

  RecordGpuOneStepStartEndInfo();
}

void CPUProfiler::RecordFrameWorkInfo(const CNodePtr &kernel) {
  CurKernelInputInfo cur_kernel_input_info;
  CurKernelInfo cur_kernel_info;
  auto op_name = kernel->fullname_with_scope();
  auto begin_iter = op_name.rfind('/') + 1;
  auto end_iter = op_name.rfind('-');
  if (begin_iter != std::string::npos && end_iter != std::string::npos && begin_iter < end_iter) {
    cur_kernel_info.op_type = op_name.substr(begin_iter, end_iter - begin_iter);
    cur_kernel_info.op_name = op_name.substr(begin_iter, op_name.length() - begin_iter);
  }
  uint32_t input_size = static_cast<uint32_t>(kernel->inputs().size());
  for (uint32_t i = 0; i < input_size; ++i) {
    if (kernel->input(i)->Shape() != nullptr) {
      cur_kernel_input_info.input_id = i;
      cur_kernel_input_info.shape = kernel->input(i)->Shape()->DumpText();
      cur_kernel_info.cur_kernel_all_inputs_info.push_back(cur_kernel_input_info);
    }
  }
  std::lock_guard<std::mutex> locker(kernel_mutex_);
  all_kernel_info_.push_back(cur_kernel_info);
}

void CPUProfiler::OpDataProducerEndParallel(const std::string op_name) {
  auto stop_timestamp = GetHostMonoTimeStamp();
  float op_time_elapsed = SetRuntimeEnd(op_name, stop_timestamp);
  MS_LOG(DEBUG) << "Host Time Elapsed(ms)," << op_name << "," << op_time_elapsed;
  Profiler::SetRunTimeData(op_name, op_time_elapsed);
}

void CPUProfiler::OpDataProducerBegin(const std::string op_name, const uint32_t pid) {
  op_time_start_ = GetHostMonoTimeStamp();
  op_time_mono_start_ = GetHostMonoTimeStamp();
  SetRunTimeData(op_name, pid);

  RecordGpuOneStepStartEndInfo();
}

void CPUProfiler::OpDataProducerEnd() {
  float op_time_elapsed = 0;
  op_time_stop_ = GetHostMonoTimeStamp();
  op_time_elapsed = (op_time_stop_ - op_time_start_) / kNanosecondToMillisecond;
  MS_LOG(DEBUG) << "Host Time Elapsed(ms)," << op_name_ << "," << op_time_elapsed;
  Profiler::SetRunTimeData(op_name_, op_time_elapsed);
  Profiler::SetRunTimeData(op_name_, op_time_mono_start_, op_time_elapsed);
}

void CPUProfiler::Stop() {
  MS_LOG(INFO) << "Stop CPU Profiling";
  SaveProfileData();
  ClearInst();
}

void CPUProfiler::SaveProfileData() {
  if (profile_data_path_.empty()) {
    MS_LOG(WARNING) << "Profile data path is empty, skip save profile data.";
  } else {
    auto cpu_data_saver_inst = profiler::cpu::CpuDataSaver::GetInstance();
    MS_EXCEPTION_IF_NULL(cpu_data_saver_inst);
    cpu_data_saver_inst->ParseOpInfo(op_info_map_);
    cpu_data_saver_inst->WriteFile(profile_data_path_);
    if (!all_kernel_info_.empty()) {
      cpu_data_saver_inst->WriteFrameWork(profile_data_path_, all_kernel_info_);
    }
  }
}

void CPUProfiler::SetGpuHeteroStatus() {
  if (!is_gpu_hetero_.has_value()) {
    is_gpu_hetero_ = Profiler::GetInstance(kGPUDevice) != nullptr;
  }
}

void CPUProfiler::ClearInst() {
  op_info_map_.clear();
  all_step_start_end_info_.clear();
  step_start_end_info_vector_.clear();
  all_kernel_info_.clear();
  init_flag_ = false;
  enable_flag_ = false;
  has_find_ = false;
}

void CPUProfiler::RecordGpuOneStepStartEndInfo() {
  SetGpuHeteroStatus();
  if (!is_gpu_hetero_.value()) {
    return;
  }

  if (auto gpu_instance = Profiler::GetInstance(kGPUDevice);
      gpu_instance != nullptr && MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT) &&
      gpu_instance->GetEnableFlag()) {
    gpu_instance->RecordOneStepStartEndInfo();
  }
}
}  // namespace cpu
}  // namespace profiler
}  // namespace mindspore
