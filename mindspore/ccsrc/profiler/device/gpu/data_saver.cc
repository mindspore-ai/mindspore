/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "profiler/device/gpu/data_saver.h"
#include <fstream>
#include <numeric>
#include "sys/stat.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace profiler {
namespace gpu {
OpDetailInfo::OpDetailInfo(std::shared_ptr<OpInfo> op_info, float proportion)
    : op_info_(op_info), proportion_(proportion) {
  // op_full_name is like 'xxx/xxx/{op_type}-op{node_id}'
  op_full_name_ = op_info->op_name;
  auto op_type_begin_iter = op_full_name_.rfind('/') + 1;
  auto op_type_end_iter = op_full_name_.rfind('-');
  op_type_ = op_full_name_.substr(op_type_begin_iter, op_type_end_iter - op_type_begin_iter);
  op_name_ = op_full_name_.substr(op_type_begin_iter);
  op_avg_time_ = op_info->op_host_cost_time / op_info->op_count;
}

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

void DataSaver::ParseOpInfo(const OpInfoMap &op_info_maps) {
  op_detail_infos_.reserve(op_info_maps.size());
  float total_time_sum = GetTotalOpTime(op_info_maps);
  for (auto item : op_info_maps) {
    op_timestamps_map_[item.first] = item.second.start_duration;
    float proportion = item.second.op_host_cost_time / total_time_sum;
    auto op_info = std::make_shared<OpInfo>(item.second);
    OpDetailInfo op_detail_info = OpDetailInfo(op_info, proportion);
    op_detail_infos_.emplace_back(op_detail_info);
    AddOpDetailInfoForType(op_detail_info);
  }
  // update average time of op type
  for (auto &op_type : op_type_infos_) {
    // device_infos: <type_name, op_type_info>
    op_type.second.avg_time_ = op_type.second.total_time_ / op_type.second.count_;
  }
  MS_LOG(DEBUG) << "Get " << op_detail_infos_.size() << " operation items.";
  MS_LOG(DEBUG) << "Get " << op_type_infos_.size() << " operation type items.";
}

void DataSaver::AddOpDetailInfoForType(const OpDetailInfo &op_detail_info) {
  // Construct OpType object according to op detail info
  OpType op_type = OpType{op_detail_info.op_type_, op_detail_info.op_info_->op_count,
                          op_detail_info.op_info_->op_host_cost_time, 0, op_detail_info.proportion_};
  // Set the OpType into op_type_infos_ map
  std::string type_name = op_detail_info.op_type_;
  auto iter = op_type_infos_.find(type_name);
  if (iter == op_type_infos_.end()) {
    op_type_infos_.emplace(type_name, op_type);
  } else {
    iter->second += op_type;
  }
}

float DataSaver::GetTotalOpTime(const OpInfoMap &op_info_maps) {
  float sum = 0;
  sum = std::accumulate(op_info_maps.begin(), op_info_maps.end(), sum,
                        [](float i, auto iter) { return i + iter.second.op_host_cost_time; });
  MS_LOG(DEBUG) << "The total op time is " << sum;
  return sum;
}

void DataSaver::ParseEvent(const std::vector<Event> &events) {
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
      // activity_info: <kernel_name, Activity>
      activity_info.second.avg_duration_ = activity_info.second.total_duration_ / activity_info.second.count_;
    }
    MS_LOG(DEBUG) << "Get " << device_infos.second.size() << " activity items for device:" << device_infos.first;
  }
}

void DataSaver::AddKernelEvent(const Event &event) {
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

void DataSaver::AddKernelEventToDevice(const Event &event, DeviceActivityInfos *device_activity_infos) {
  // Combine kernel activity with same kernel name
  auto event_ptr = std::make_shared<Event>(event);
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

void DataSaver::WriteFile(std::string out_path_dir, const BaseTime &start_time) {
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
  WriteOpDetail(out_path_dir);
  WriteOpType(out_path_dir);
  WriteActivity(out_path_dir);
  WriteOpTimestamp(out_path_dir);
  WriteStepTrace(out_path_dir);
  WriteStartTime(out_path_dir, start_time);
}

void DataSaver::WriteOpType(const std::string &saver_base_dir) {
  std::string file_path = saver_base_dir + "/gpu_op_type_info_" + device_id_ + ".csv";
  std::ofstream ofs(file_path);
  // check if the file is writable
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }
  // write op type info into file
  ofs << OpType().GetHeader() << std::endl;
  for (auto op_type_info : op_type_infos_) {
    ofs << op_type_info.second << std::endl;
  }
  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write " << op_type_infos_.size() << " op type infos into file: " << file_path;
}

void DataSaver::WriteOpDetail(const std::string &saver_base_dir) {
  std::string file_path = saver_base_dir + "/gpu_op_detail_info_" + device_id_ + ".csv";
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }
  // write op detail info into file
  ofs << OpDetailInfo().GetHeader() << std::endl;
  for (auto op_detail : op_detail_infos_) {
    ofs << op_detail << std::endl;
  }
  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write " << op_detail_infos_.size() << " op detail infos into file: " << file_path;
}

void DataSaver::WriteActivity(const std::string &saver_base_dir) {
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

void DataSaver::WriteOpTimestamp(const std::string &saver_base_dir) {
  std::string file_path = saver_base_dir + "/op_execute_timestamp_" + device_id_ + ".txt";
  std::ofstream ofs(file_path);
  // check if the file is writable
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }
  // write op timestamp info into file
  for (const auto &op_timestamp_info : op_timestamps_map_) {
    ofs << op_timestamp_info.first << ";Ops;";
    for (auto start_end : op_timestamp_info.second) {
      ofs << start_end.start_timestamp << "," << start_end.duration << " ";
    }
    ofs << std::endl;
  }
  ofs.close();
  ChangeFileMode(file_path);
}

void DataSaver::WriteStepTrace(const std::string &saver_base_dir) {
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
  op_name_arr.push_back(step_trace_op_name.trace_fp_start);
  op_name_arr.push_back(step_trace_op_name.trace_bp_end);
  op_name_arr.push_back(step_trace_op_name.trace_iter_end);
  if (!step_trace_op_name.trace_custom_node.empty()) {
    auto start = step_trace_op_name.trace_custom_node.begin();
    auto end = step_trace_op_name.trace_custom_node.end();
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
      }
    }
  }

  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write step trace infos into file: " << file_path;
}

void DataSaver::WriteStartTime(const std::string &saver_base_dir, const BaseTime &start_time) {
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
  }

  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write profiler start time infos into file: " << file_path;
}

void DataSaver::SetStepTraceOpName(ProfilingTraceInfo trace_op_name) { step_trace_op_name = trace_op_name; }

void DataSaver::ChangeFileMode(const std::string &file_path) {
  if (chmod(common::SafeCStr(file_path), S_IRUSR) == -1) {
    MS_LOG(WARNING) << "Modify file:" << file_path << " to rw fail.";
    return;
  }
}
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore
