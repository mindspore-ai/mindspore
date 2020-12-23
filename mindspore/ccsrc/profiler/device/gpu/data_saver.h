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

#ifndef MINDSPORE_DATA_SAVER_H
#define MINDSPORE_DATA_SAVER_H
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include "profiler/device/gpu/gpu_profiling.h"
namespace mindspore {
namespace profiler {
namespace gpu {
struct OpDetailInfo {
  std::string op_type_;
  std::string op_name_;
  std::string op_full_name_;
  std::shared_ptr<OpInfo> op_info_{nullptr};
  float op_avg_time_{0};
  float proportion_{0};

  OpDetailInfo() = default;

  OpDetailInfo(std::shared_ptr<OpInfo> op_info, float proportion);

  std::string GetHeader() const {
    return "op_side,op_type,op_name,op_full_name,op_occurrences,op_total_time(us),op_avg_time(us),total_proportion,"
           "cuda_activity_cost_time(us),cuda_activity_call_count";
  }

  friend std::ostream &operator<<(std::ostream &os, const OpDetailInfo &event) {
    os << "Device," << event.op_type_ << ',' << event.op_name_ << ',' << event.op_full_name_ << ','
       << event.op_info_->op_count << ',' << event.op_info_->op_host_cost_time << ',' << event.op_avg_time_ << ','
       << event.proportion_ << ',' << event.op_info_->cupti_activity_time << ',' << event.op_info_->op_kernel_count;
    return os;
  }
};

struct OpType {
  std::string op_type_;
  int count_{0};
  float total_time_{0};
  float avg_time_{0};
  float proportion_{0};

  std::string GetHeader() const { return "op_type,type_occurrences,total_time(us),total_proportion,avg_time(us)"; }

  friend std::ostream &operator<<(std::ostream &os, const OpType &event) {
    os << event.op_type_ << ',' << event.count_ << ',' << event.total_time_ << ',' << event.proportion_ << ','
       << event.avg_time_;
    return os;
  }

  OpType &operator+=(const OpType &other) {
    this->count_ += other.count_;
    this->total_time_ += other.total_time_;
    this->proportion_ += other.proportion_;
    return *this;
  }
};

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

using OpInfoMap = std::unordered_map<std::string, OpInfo>;
using DeviceActivityInfos = std::unordered_map<std::string, ActivityData>;   // <device_id, ActivityData>
using AllActivityInfos = std::unordered_map<uint32_t, DeviceActivityInfos>;  // <device_id, ActivityData>
using OpTypeInfos = std::unordered_map<std::string, OpType>;                 // <op_full_name, Optype>
using OpDetailInfos = std::vector<OpDetailInfo>;
// <op_full_name, StartDuration>
using OpTimestampInfo = std::unordered_map<std::string, std::vector<StartDuration>>;

class DataSaver {
 public:
  DataSaver() = default;

  ~DataSaver() = default;

  DataSaver(const DataSaver &) = delete;

  DataSaver &operator=(const DataSaver &) = delete;

  void ParseOpInfo(const OpInfoMap &op_info_maps);

  void SetStepTraceOpName(ProfilingTraceInfo trace_op_name);

  void ParseEvent(const std::vector<Event> &events);

  void WriteFile(std::string out_path, const BaseTime &start_time);

 private:
  void AddOpDetailInfoForType(const OpDetailInfo &op_detail_info);

  float GetTotalOpTime(const OpInfoMap &op_info_maps);

  void AddKernelEvent(const Event &event);

  void AddKernelEventToDevice(const Event &event, DeviceActivityInfos *device_activity_infos);

  void WriteOpType(const std::string &saver_base_dir);

  void WriteOpDetail(const std::string &saver_base_dir);

  void WriteActivity(const std::string &saver_base_dir);

  void WriteOpTimestamp(const std::string &saver_base_dir);

  void WriteStepTrace(const std::string &saver_base_dir);

  void WriteStartTime(const std::string &saver_base_dir, const BaseTime &start_time);

  void ChangeFileMode(const std::string &file_path);

  std::string device_id_;
  AllActivityInfos activity_infos_;
  OpTypeInfos op_type_infos_;
  OpDetailInfos op_detail_infos_;
  OpTimestampInfo op_timestamps_map_;
  ProfilingTraceInfo step_trace_op_name;
};
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_DATA_SAVER_H
