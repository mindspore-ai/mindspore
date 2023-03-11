/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/profiler/pynative_profiling.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include "include/common/utils/utils.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/common/pybind_api/api_register.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
namespace profiler {
namespace ascend {
namespace {
constexpr auto kPyNativeName = "PyNative";

PROFILER_REG(kPyNativeName, PynativeProfiler);
}  // namespace

std::shared_ptr<PynativeProfiler> PynativeProfiler::GetInstance() {
  auto instance = Profiler::GetInstance(kPyNativeName);
  MS_EXCEPTION_IF_NULL(instance);
  return std::dynamic_pointer_cast<PynativeProfiler>(instance);
}

void PynativeProfiler::Init(const std::string &profiling_path, uint32_t, const std::string &) {
  MS_LOG(INFO) << "Initialize pynatiave Ascend Profiling";
  profile_data_path_ = profiling_path;
  enable_flag_ = true;
  std::string device_id = common::GetEnv("RANK_ID");
  if (device_id.empty()) {
    rank_id_ = 0;
  } else {
    rank_id_ = atoi(device_id.c_str());
  }
  WriteStartTime();
}

void PynativeProfiler::WriteStartTime() {
  std::string file_path = profile_data_path_ + "/start_time_" + std::to_string(rank_id_) + ".txt";
  std::ofstream ofs(file_path);
  // check if the file is writable
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }

  // write start time info into file
  try {
    uint64_t device_timestamp = GetRealTimeStamp();
    uint64_t host_monotonic_raw_time = GetHostMonoTimeStamp();
    ofs << "host_monotonic_raw_time(ns): " << host_monotonic_raw_time << std::endl;
    ofs << "device_start_time(ns): " << device_timestamp << std::endl;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Write " << file_path << "failed:" << e.what();
  }
  ofs.close();
  ChangeFileMode(file_path, S_IRUSR | S_IWUSR);
  MS_LOG(INFO) << "Write profiler start time infos into file: " << file_path;
}

void PynativeProfiler::SaveProfileData() { WriteOpDetail(profile_data_path_); }

void PynativeProfiler::ClearInst() {
  pynative_op_info_.clear();
  thread_op_info_map_.clear();
}

void PynativeProfiler::OpDataProducerEnd() {}

void PynativeProfiler::OpDataProducerBegin(AscendKernelRuntime *runtime_instance_, void *stream,
                                           std::thread::id thread_id, const std::string &op_name,
                                           bool is_dynamic_shape) {
  if (!enable_flag_) {
    return;
  }
  if (is_dynamic_shape) {
    MS_LOG(EXCEPTION) << "Dynamic shape is not supported in pynative mode.";
    return;
  }
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  MS_EXCEPTION_IF_NULL(stream);

  std::shared_ptr<DeviceEvent> start = runtime_instance_->CreateDeviceTimeEvent();
  std::shared_ptr<DeviceEvent> end = runtime_instance_->CreateDeviceTimeEvent();
  MS_EXCEPTION_IF_NULL(start);
  MS_EXCEPTION_IF_NULL(end);
  start->set_record_stream(stream);
  end->set_record_stream(stream);
  start->RecordEvent();

  PynativeOpInfo op_info;
  op_info.start = start;
  op_info.end = end;
  op_info.op_name = op_name;
  op_info.stream = stream;
  if (thread_op_info_map_.find(thread_id) == thread_op_info_map_.end()) {
    op_info.thread_index = NewThreadIndex();
  } else {
    op_info.thread_index = thread_op_info_map_[thread_id].thread_index;
  }
  std::unique_lock<std::shared_mutex> lock(op_map_mutex_);
  thread_op_info_map_[thread_id] = op_info;
}

void PynativeProfiler::StepProfilingEnable(const bool enable_flag) { enable_flag_ = enable_flag; }

void PynativeProfiler::OpDataProducerEnd(std::thread::id thread_id, bool is_dynamic_shape) {
  if (!enable_flag_) {
    return;
  }
  if (is_dynamic_shape) {
    MS_LOG(EXCEPTION) << "Dynamic shape is not supported in pynative mode.";
    return;
  }
  if (thread_op_info_map_.find(thread_id) == thread_op_info_map_.end()) {
    MS_LOG(WARNING) << "Pynative profiling, the start time of op is null"
                    << ", please call the OpDataProducerBegin function first.";
    return;
  }

  PynativeOpInfo op_info = thread_op_info_map_[thread_id];
  float cost_time = 0;
  // Operator asynchronous execution changed to synchronous
  op_info.end->RecordEvent();
  op_info.start->SyncEvent();
  op_info.end->SyncEvent();
  op_info.start->ElapsedTime(&cost_time, op_info.end.get());

  op_info.duration = cost_time;
  constexpr int64_t milli_second_ratio = 1000;
  int64_t end_timestamp = GetRealTimeStamp();
  int64_t start_timestamp = end_timestamp - static_cast<int64_t>(cost_time * milli_second_ratio);
  double_t start_t = static_cast<double_t>(start_timestamp) / milli_second_ratio;
  op_info.start_timestamp = start_t;

  std::unique_lock<std::shared_mutex> lock(op_map_mutex_);
  pynative_op_info_.push_back(op_info);
}

void PynativeProfiler::Stop() {
  SaveProfileData();
  ClearInst();
  enable_flag_ = false;
}

void PynativeProfiler::WriteOpDetail(const std::string &out_path_dir) {
  std::string file_path = out_path_dir + "/output_timeline_data_" + std::to_string(rank_id_) + ".txt";
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }
  try {
    ofs << "op_name, stream_id, start_time(ms), duration(ms)" << std::endl;
    std::sort(pynative_op_info_.begin(), pynative_op_info_.end(),
              [](const auto &op1, const auto &op2) { return op1.start_timestamp < op2.start_timestamp; });
    for (PynativeOpInfo op_info : pynative_op_info_) {
      ofs << op_info.op_name << "," << op_info.thread_index << "," << std::to_string(op_info.start_timestamp) << ","
          << op_info.duration << std::endl;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Write " << file_path << "failed: " << e.what();
  }
  ofs.close();
  ChangeFileMode(file_path, S_IRUSR | S_IWUSR);
  MS_LOG(INFO) << "Write " << pynative_op_info_.size() << " op detail infos into file: " << file_path;
}

int PynativeProfiler::NewThreadIndex() { return thread_op_info_map_.size() + 1; }
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
