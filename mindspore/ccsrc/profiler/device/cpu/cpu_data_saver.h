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

#ifndef MINDSPORE_CPU_DATA_SAVER_H
#define MINDSPORE_CPU_DATA_SAVER_H
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include "profiler/device/cpu/cpu_profiling.h"
namespace mindspore {
namespace profiler {
namespace cpu {
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
    return "op_side,op_type,op_name,full_op_name,op_occurrences,compute_time(ms),"
           "avg_execution_time(ms),total_proportion,subgraph,pid";
  }

  friend std::ostream &operator<<(std::ostream &os, const OpDetailInfo &event) {
    os << "Host," << event.op_type_ << ',' << event.op_name_ << ',' << event.op_full_name_ << ','
       << event.op_info_->op_count << ',' << event.op_info_->op_cost_time << ',' << event.op_avg_time_ << ','
       << event.proportion_ << ",Default," << event.op_info_->pid;
    return os;
  }
};

struct OpType {
  std::string op_type_;
  int count_{0};
  int step_{0};
  float total_time_{0};
  float avg_time_{0};
  float proportion_{0};

  std::string GetHeader() const {
    return "op_type,total_called_times,called_times(per-step),"
           "total_compute_time,compute_time(ms per-step),percent";
  }

  friend std::ostream &operator<<(std::ostream &os, const OpType &event) {
    os << event.op_type_ << ',' << event.count_ << ',' << event.count_ / event.step_ << ',' << event.total_time_ << ','
       << event.total_time_ / event.step_ << ',' << event.proportion_;
    return os;
  }

  OpType &operator+=(const OpType &other) {
    this->count_ += other.count_;
    this->total_time_ += other.total_time_;
    this->proportion_ += other.proportion_;
    return *this;
  }
};

using OpInfoMap = std::unordered_map<std::string, OpInfo>;
using OpTypeInfos = std::unordered_map<std::string, OpType>;  // <op_full_name, Optype>
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

  void WriteFile(std::string out_path);

 private:
  void AddOpDetailInfoForType(const OpDetailInfo &op_detail_info);

  float GetTotalOpTime(const OpInfoMap &op_info_maps);

  void WriteOpType(const std::string &saver_base_dir);

  void WriteOpDetail(const std::string &saver_base_dir);

  void WriteOpTimestamp(const std::string &saver_base_dir);

  void ChangeFileMode(const std::string &file_path);

  std::string device_id_;
  OpTypeInfos op_type_infos_;
  OpDetailInfos op_detail_infos_;
  OpTimestampInfo op_timestamps_map_;
};
}  // namespace cpu
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CPU_DATA_SAVER_H
