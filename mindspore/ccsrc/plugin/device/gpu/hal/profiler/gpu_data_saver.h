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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_PROFILER_GPU_DATA_SAVER_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_PROFILER_GPU_DATA_SAVER_H
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include "plugin/device/gpu/hal/profiler/gpu_profiling.h"
#include "plugin/device/cpu/hal/profiler/cpu_data_saver.h"
#include "include/backend/debug/profiler/data_saver.h"
namespace mindspore {
namespace profiler {
namespace gpu {
struct ActivityData {
  std::shared_ptr<Event> basic_info_{nullptr};
  std::string block_dim_;
  std::string grid_dim_;
  int count_{0};
  float total_duration_{0};
  float avg_duration_{0};
  float max_duration_{0};
  float min_duration_{0};
  std::vector<StartDuration> start_duration;

  ActivityData() = default;

  explicit ActivityData(std::shared_ptr<Event> data);

  std::string GetHeader() const {
    return "name,type,op_full_name,stream_id,block_dim,grid_dim,occurrences,"
           "total_duration(us),avg_duration(us),max_duration(us),min_duration(us)";
  }

  friend std::ostream &operator<<(std::ostream &os, const ActivityData &event) {
    os << "\"" << event.basic_info_->kernel_name << "\"," << event.basic_info_->kernel_type << ','
       << event.basic_info_->op_name << ',' << event.basic_info_->stream_id << ',' << event.block_dim_ << ','
       << event.grid_dim_ << ',' << event.count_ << ',' << event.total_duration_ << ',' << event.avg_duration_ << ','
       << event.max_duration_ << ',' << event.min_duration_;
    return os;
  }

  ActivityData &operator+=(const ActivityData &other);
};

using DeviceActivityInfos = std::unordered_map<std::string, ActivityData>;   // <device_id, ActivityData>
using AllActivityInfos = std::unordered_map<uint32_t, DeviceActivityInfos>;  // <device_id, ActivityData>

class GpuDataSaver : public DataSaver {
 public:
  GpuDataSaver(ProfilingTraceInfo step_trace_op_name, const std::vector<OneStepStartEndInfo> &all_step_start_end_info)
      : step_trace_op_name_(step_trace_op_name), all_step_start_end_info_(all_step_start_end_info) {
    step_trace_op_name_from_graph_ = step_trace_op_name;
  }

  ~GpuDataSaver() = default;

  GpuDataSaver(const GpuDataSaver &) = delete;

  GpuDataSaver &operator=(const GpuDataSaver &) = delete;

  void ParseEvent(const std::vector<Event> &events);

  void WriteFile(std::string out_path, const BaseTime &start_time);

 private:
  void AddKernelEvent(const Event &event);

  void AddKernelEventToDevice(const Event &event, DeviceActivityInfos *device_activity_infos);

  void WriteActivity(const std::string &saver_base_dir);

  void WriteStepTrace(const std::string &saver_base_dir);

  void WriteStepTraceAsyncLaunchKernel(const std::string &saver_base_dir);

  void WriteStartTime(const std::string &saver_base_dir, const BaseTime &start_time);

  void CpuProfilingTimeSynchronizedToGpu(const BaseTime &start_time);

  AllActivityInfos activity_infos_;
  ProfilingTraceInfo step_trace_op_name_from_graph_;
  ProfilingTraceInfo step_trace_op_name_;
  const std::vector<OneStepStartEndInfo> &all_step_start_end_info_;
};
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_PROFILER_GPU_DATA_SAVER_H
