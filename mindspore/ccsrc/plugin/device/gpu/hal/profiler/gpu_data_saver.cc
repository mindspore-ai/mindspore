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
#include "plugin/device/gpu/hal/profiler/gpu_data_saver.h"
#include <fstream>
#include <numeric>
#include "sys/stat.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
namespace profiler {
namespace gpu {
ActivityData::ActivityData(std::shared_ptr<Event> data) : basic_info_(data) {
  grid_dim_ = basic_info_->activity_type == ActivityType::kKernel
                ? "\"" + std::to_string(basic_info_->kernel_info.grid_x) + ',' +
                    std::to_string(basic_info_->kernel_info.grid_y) + ',' +
                    std::to_string(basic_info_->kernel_info.grid_z) + "\""
                : "";
  block_dim_ = basic_info_->activity_type == ActivityType::kKernel
                 ? "\"" + std::to_string(basic_info_->kernel_info.block_x) + ',' +
                     std::to_string(basic_info_->kernel_info.block_y) + ',' +
                     std::to_string(basic_info_->kernel_info.block_z) + "\""
                 : "";
  count_ = 1;
  total_duration_ = (basic_info_->end_time_stamp - basic_info_->start_time_stamp) / kTimeUnit;
  avg_duration_ = total_duration_;
  max_duration_ = total_duration_;
  min_duration_ = total_duration_;
  start_duration.emplace_back(StartDuration({basic_info_->start_time_stamp, total_duration_}));
}

ActivityData &ActivityData::operator+=(const ActivityData &other) {
  this->count_ += other.count_;
  this->total_duration_ += other.total_duration_;
  // update max or min duration
  if (other.total_duration_ > this->max_duration_) {
    this->max_duration_ = other.total_duration_;
  } else if (other.max_duration_ < this->min_duration_) {
    this->min_duration_ = other.total_duration_;
  }
  return *this;
}

void GpuDataSaver::ParseEvent(const std::vector<Event> &events) {
  // Put Kernel activity events into activity_infos_
  for (const auto &event : events) {
    if (event.op_name.empty() || event.api_type != CUPTIApiType::kActivity ||
        event.activity_type != ActivityType::kKernel) {
      continue;
    }
    AddKernelEvent(event);
  }
  // update average time of kernel op cost
  for (auto &device_infos : activity_infos_) {
    // device_infos: <device_id, DeviceActivityInfos>
    for (auto &activity_info : device_infos.second) {
      if (activity_info.second.count_ == 0) {
        MS_LOG(ERROR) << "The number of activity operations can not be 0.";
        return;
      }
      // activity_info: <kernel_name, Activity>
      activity_info.second.avg_duration_ = activity_info.second.total_duration_ / activity_info.second.count_;
    }
    MS_LOG(DEBUG) << "Get " << device_infos.second.size() << " activity items for device:" << device_infos.first;
  }
}

void GpuDataSaver::AddKernelEvent(const Event &event) {
  // Put kernel event to activity_infos according to device id
  uint32_t device_id = event.device_id;
  auto iter = activity_infos_.find(device_id);
  if (iter == activity_infos_.end()) {
    auto res_flag = activity_infos_.emplace(device_id, DeviceActivityInfos());
    AddKernelEventToDevice(event, &res_flag.first->second);
  } else {
    AddKernelEventToDevice(event, &iter->second);
  }
}

void GpuDataSaver::AddKernelEventToDevice(const Event &event, DeviceActivityInfos *device_activity_infos) {
  if (device_activity_infos == nullptr) {
    MS_LOG(ERROR) << "Device activity information is invalid.";
    return;
  }
  // Combine kernel activity with same kernel name
  auto event_ptr = std::make_shared<Event>(event);
  if (event_ptr == nullptr) {
    MS_LOG(WARNING) << "Create event failed when add event to device.";
    return;
  }
  ActivityData activity_data = ActivityData(event_ptr);
  std::string kernel_name = event.kernel_name;
  auto iter = device_activity_infos->find(kernel_name);
  if (iter == device_activity_infos->end()) {
    device_activity_infos->emplace(kernel_name, activity_data);
  } else {
    iter->second += activity_data;
    iter->second.start_duration.emplace_back(StartDuration({event.start_time_stamp, activity_data.total_duration_}));
  }
}

void GpuDataSaver::CpuProfilingTimeSynchronizedToGpu(const BaseTime &start_time) {
  auto cpu_data_saver_inst = profiler::cpu::CpuDataSaver::GetInstance();
  MS_EXCEPTION_IF_NULL(cpu_data_saver_inst);
  auto &cpu_op_timestamps_map = cpu_data_saver_inst->GetOpTimeStampInfo();
  auto cpu_op_iter = cpu_op_timestamps_map.begin();
  while (cpu_op_iter != cpu_op_timestamps_map.end()) {
    for (auto &time_iter : cpu_op_iter->second) {
      time_iter.start_timestamp =
        time_iter.start_timestamp - start_time.host_start_monotonic_raw_time + start_time.gpu_start_time;
      // time unit from ms to us.
      time_iter.duration *= kTimeUnit;
    }
    cpu_op_iter++;
  }
}

void GpuDataSaver::WriteFile(std::string out_path_dir, const BaseTime &start_time) {
  if (out_path_dir.empty()) {
    MS_LOG(WARNING) << "Output directory. Ignore the writing data.";
    return;
  }
  if (op_detail_infos_.empty() || op_type_infos_.empty() || activity_infos_.empty()) {
    MS_LOG(WARNING) << "No operation detail infos to write.";
    return;
  }
  // not support multi-device for operator info per process yet
  device_id_ = std::to_string(activity_infos_.begin()->first);
  op_side_ = "gpu";
  WriteOpDetail(out_path_dir);
  WriteOpType(out_path_dir);
  WriteActivity(out_path_dir);
  WriteOpTimestamp(out_path_dir);
  WriteStartTime(out_path_dir, start_time);
  CpuProfilingTimeSynchronizedToGpu(start_time);
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    WriteStepTraceAsyncLaunchKernel(out_path_dir);
  } else {
    WriteStepTrace(out_path_dir);
  }
}

void GpuDataSaver::WriteActivity(const std::string &saver_base_dir) {
  std::string file_path_base = saver_base_dir + "/gpu_activity_data_";
  std::string timestamp_file_path_base = saver_base_dir + "/activity_execute_timestamp_";
  for (auto device_info : activity_infos_) {
    // write activity result csv
    std::string file_path = file_path_base + std::to_string(device_info.first) + ".csv";
    std::ofstream ofs(file_path);
    if (!ofs.is_open()) {
      MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
      return;
    }
    // write activity timestamp txt
    std::string timestamp_file_path = timestamp_file_path_base + std::to_string(device_info.first) + ".txt";
    std::ofstream activity_timestamp_ofs(timestamp_file_path);
    if (!activity_timestamp_ofs.is_open()) {
      ofs.close();
      MS_LOG(WARNING) << "Open file '" << timestamp_file_path << "' failed!";
      return;
    }
    // write activity data into file
    ofs << ActivityData().GetHeader() << std::endl;
    for (auto activity_data : device_info.second) {
      ofs << activity_data.second << std::endl;
      for (auto start_duration : activity_data.second.start_duration) {
        activity_timestamp_ofs << activity_data.second.basic_info_->kernel_name << ";";
        activity_timestamp_ofs << activity_data.second.basic_info_->stream_id << ";";
        activity_timestamp_ofs << start_duration.start_timestamp << ";";
        activity_timestamp_ofs << start_duration.duration << std::endl;
      }
    }
    ofs.close();
    ChangeFileMode(file_path);
    activity_timestamp_ofs.close();
    ChangeFileMode(timestamp_file_path);
    MS_LOG(INFO) << "Write " << device_info.second.size() << " activity infos into file: " << file_path;
  }
}

void GpuDataSaver::WriteStepTraceAsyncLaunchKernel(const std::string &saver_base_dir) {
  std::string file_path = saver_base_dir + "/step_trace_profiling_" + device_id_ + ".txt";
  std::ofstream ofs(file_path);
  // check if the file is writable
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }

  // cpu profiler information.
  auto cpu_data_saver_inst = profiler::cpu::CpuDataSaver::GetInstance();
  MS_EXCEPTION_IF_NULL(cpu_data_saver_inst);
  auto &cpu_op_timestamps_map = cpu_data_saver_inst->GetOpTimeStampInfo();

  // write step trace time info into file
  uint32_t step = 0;
  uint64_t duration;
  for (auto step_start_end : all_step_start_end_info_) {
    auto iter_start_op_name = step_start_end.iter_start_op_name;
    auto fp_op_name = step_start_end.fp_start_op_name;
    auto iter_end_op_name = step_start_end.iter_end_op_name;
    auto iter_start_op_timestamp = op_timestamps_map_.find(iter_start_op_name);
    auto fp_op_timestamp = op_timestamps_map_.find(fp_op_name);
    auto bp_end_op_timestamp = op_timestamps_map_.find(step_trace_op_name_.trace_bp_end);
    auto iter_end_op_timestamp = op_timestamps_map_.find(iter_end_op_name);

    // if iter_start/fp_start/iter_end op is executed on cpu, update it.
    if (iter_start_op_timestamp == op_timestamps_map_.end()) {
      iter_start_op_timestamp = cpu_op_timestamps_map.find(iter_start_op_name);
    }
    if (fp_op_timestamp == op_timestamps_map_.end()) {
      fp_op_timestamp = cpu_op_timestamps_map.find(fp_op_name);
    }
    if (iter_end_op_timestamp == op_timestamps_map_.end()) {
      iter_end_op_timestamp = cpu_op_timestamps_map.find(iter_end_op_name);
    }

    if (iter_end_op_name == "Default/InitDataSetQueue-op0") {
      continue;
    }

    if (iter_start_op_timestamp == op_timestamps_map_.end() || fp_op_timestamp == op_timestamps_map_.end() ||
        iter_end_op_timestamp == op_timestamps_map_.end() || bp_end_op_timestamp == op_timestamps_map_.end()) {
      MS_LOG(ERROR) << "[profiling step trace] failed, do not find \"" << fp_op_name << "\" or \"" << iter_end_op_name
                    << "\" or \"" << step_trace_op_name_.trace_bp_end << "\"";
      ofs.close();
      return;
    }
    if (iter_start_op_timestamp->second.size() <= step || fp_op_timestamp->second.size() <= step ||
        iter_end_op_timestamp->second.size() <= step || bp_end_op_timestamp->second.size() <= step) {
      MS_LOG(WARNING) << "[profiling step trace] insufficient number of timestamps for fp/bp/iter_end operators.";
      ofs.close();
      return;
    }

    try {
      // write fp,bp and iter_end timestamp.
      duration = iter_end_op_timestamp->second[step].duration * kTimeUnit;
      uint64_t iter_end_timestamp = iter_end_op_timestamp->second[step].start_timestamp + duration;
      ofs << iter_start_op_name << "," << iter_start_op_timestamp->second[step].start_timestamp << " " << fp_op_name
          << "," << fp_op_timestamp->second[step].start_timestamp << " " << step_trace_op_name_.trace_bp_end << ","
          << bp_end_op_timestamp->second[step].start_timestamp << " " << iter_end_op_name << "," << iter_end_timestamp;

      // write communication op info
      for (auto op_name : step_trace_op_name_.trace_custom_node) {
        // convert the time unit from 1ns to 10ns (keep the same with ascend)
        auto iter_op_timestamp = op_timestamps_map_.find(op_name);
        if (iter_op_timestamp == op_timestamps_map_.end()) {
          MS_LOG(ERROR) << "[profiling step trace] failed, do not find \"" << fp_op_name << "\" or " << iter_end_op_name
                        << "\" or \"" << step_trace_op_name_.trace_bp_end << "\"";
          ofs.close();
          return;
        }

        if (iter_op_timestamp->second.size() <= step) {
          MS_LOG(WARNING) << "[profiling step trace] insufficient number of timestamps for communication operators.";
          ofs.close();
          return;
        }

        duration = iter_op_timestamp->second[step].duration * kTimeUnit;
        uint64_t end_timestamp = (duration + iter_op_timestamp->second[step].start_timestamp);
        uint64_t start_timestamp = iter_op_timestamp->second[step].start_timestamp;
        ofs << " " << op_name << "," << start_timestamp << "," << end_timestamp;
      }
      ofs << std::endl;
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Write " << file_path << "failed:" << e.what();
      ofs.close();
      return;
    }
    step++;
  }

  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write step trace infos into file: " << file_path;
  op_timestamps_map_.clear();
}

void GpuDataSaver::WriteStepTrace(const std::string &saver_base_dir) {
  std::string file_path = saver_base_dir + "/step_trace_profiling_" + device_id_ + ".txt";
  std::ofstream ofs(file_path);
  // check if the file is writable
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }

  // write step trace time info into file
  const uint32_t factor = 10;
  std::vector<std::string> op_name_arr;
  op_name_arr.push_back(step_trace_op_name_from_graph_.trace_fp_start);
  op_name_arr.push_back(step_trace_op_name_from_graph_.trace_bp_end);
  op_name_arr.push_back(step_trace_op_name_from_graph_.trace_iter_end);
  if (!step_trace_op_name_from_graph_.trace_custom_node.empty()) {
    auto start = step_trace_op_name_from_graph_.trace_custom_node.begin();
    auto end = step_trace_op_name_from_graph_.trace_custom_node.end();
    std::copy(start, end, std::back_inserter(op_name_arr));
  }
  for (auto op_name : op_name_arr) {
    auto iter_op_timestamp = op_timestamps_map_.find(op_name);
    if (iter_op_timestamp != op_timestamps_map_.end()) {
      try {
        ofs << op_name << " ";
        for (auto start_end : iter_op_timestamp->second) {
          // convert the time unit from 1ns to 10ns (keep the same with ascend)
          uint64_t duration = start_end.duration * kTimeUnit;
          uint64_t end_timestamp = (duration + start_end.start_timestamp) / factor;
          uint64_t start_timestamp = start_end.start_timestamp / factor;
          ofs << start_timestamp << "," << end_timestamp << " ";
        }
        ofs << std::endl;
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Write " << file_path << "failed:" << e.what();
        ofs.close();
        return;
      }
    }
  }

  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write step trace infos into file: " << file_path;
  op_timestamps_map_.clear();
}

void GpuDataSaver::WriteStartTime(const std::string &saver_base_dir, const BaseTime &start_time) {
  std::string file_path = saver_base_dir + "/start_time_" + device_id_ + ".txt";
  std::ofstream ofs(file_path);
  // check if the file is writable
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }

  // write start time info into file
  try {
    ofs << "host_monotonic_raw_time(ns): " << start_time.host_start_monotonic_raw_time << std::endl;
    ofs << "gpu_start_time(ns): " << start_time.gpu_start_time << std::endl;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Write " << file_path << "failed:" << e.what();
    ofs.close();
    return;
  }

  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write profiler start time infos into file: " << file_path;
}
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore
