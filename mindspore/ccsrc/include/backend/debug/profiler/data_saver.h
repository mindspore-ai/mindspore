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

#ifndef MINDSPORE_CCSRC_PROFILER_DEVICE_DATA_SAVER_H
#define MINDSPORE_CCSRC_PROFILER_DEVICE_DATA_SAVER_H
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include "include/backend/debug/profiler/profiling.h"
#include "utils/log_adapter.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
namespace profiler {
struct CurKernelInputInfo {
  uint32_t input_id;
  std::string shape;
};
struct CurKernelInfo {
  std::string op_type;
  std::string op_name;
  std::vector<CurKernelInputInfo> cur_kernel_all_inputs_info;
  uint32_t graph_id;
};
struct OpDetailInfo {
  std::string op_type_;
  std::string op_name_;
  std::string op_full_name_;
  std::shared_ptr<OpInfo> op_info_{nullptr};
  float op_avg_time_{0};
  float proportion_{0};

  OpDetailInfo() = default;
  OpDetailInfo(const std::shared_ptr<OpInfo> op_info, float proportion);

  std::string GetCpuHeader() const {
    return "op_side,op_type,op_name,full_op_name,op_occurrences,op_total_time(ms),"
           "op_avg_time(ms),total_proportion,subgraph,pid";
  }
  std::string GetGpuHeader() const {
    return "op_side,op_type,op_name,op_full_name,op_occurrences,op_total_time(us),op_avg_time(us),total_proportion,"
           "cuda_activity_cost_time(us),cuda_activity_call_count";
  }

  void OutputCpuOpDetailInfo(std::ostream &os) const {
    os << "Host," << op_type_ << ',' << op_name_ << ',' << op_full_name_ << ',' << op_info_->op_count << ','
       << op_info_->op_host_cost_time << ',' << op_avg_time_ << ',' << proportion_ << ",Default," << op_info_->pid
       << std::endl;
  }

  void OutputGpuOpDetailInfo(std::ostream &os) const {
    os << "Device," << op_type_ << ',' << op_name_ << ',' << op_full_name_ << ',' << op_info_->op_count << ','
       << op_info_->op_host_cost_time << ',' << op_avg_time_ << ',' << proportion_ << ','
       << op_info_->cupti_activity_time << ',' << op_info_->op_kernel_count << std::endl;
  }
};

struct OpType {
  std::string op_type_;
  int count_{0};
  int step_{0};
  float total_time_{0};
  float avg_time_{0};
  float proportion_{0};

  std::string GetCpuHeader() const {
    return "op_type,type_occurrences,execution_frequency(per-step),"
           "total_compute_time,avg_time(ms),percent";
  }
  std::string GetGpuHeader() const { return "op_type,type_occurrences,total_time(us),total_proportion,avg_time(us)"; }

  void OutputCpuOpTypeInfo(std::ostream &os) const {
    if (step_ == 0) {
      MS_LOG(ERROR) << "The run step can not be 0.";
      return;
    }
    if (count_ == 0) {
      MS_LOG(ERROR) << "The num of operation type can not be 0.";
      return;
    }
    os << op_type_ << ',' << count_ << ',' << count_ / step_ << ',' << total_time_ << ',' << total_time_ / count_ << ','
       << proportion_ << std::endl;
  }

  void OutputGpuOpTypeInfo(std::ostream &os) const {
    os << op_type_ << ',' << count_ << ',' << total_time_ << ',' << proportion_ << ',' << avg_time_ << std::endl;
  }

  OpType &operator+=(const OpType &other) {
    this->count_ += other.count_;
    this->total_time_ += other.total_time_;
    this->proportion_ += other.proportion_;
    return *this;
  }
};

using OpTimestampInfo = std::unordered_map<std::string, std::vector<StartDuration>>;  // <op_full_name, StartDuration>
using OpInfoMap = std::unordered_map<std::string, OpInfo>;
using OpTypeInfos = std::unordered_map<std::string, OpType>;  // <op_full_name, Optype>
using OpDetailInfos = std::vector<OpDetailInfo>;

class BACKEND_EXPORT DataSaver {
 public:
  DataSaver() = default;

  virtual ~DataSaver() = default;

  void ParseOpInfo(const OpInfoMap &op_info_maps);

  void WriteFrameWork(const std::string &base_dir, const std::vector<CurKernelInfo> &all_kernel_info_);

  OpTimestampInfo op_timestamps_map_;

 protected:
  void AddOpDetailInfoForType(const OpDetailInfo &op_detail_info);

  float GetTotalOpTime(const OpInfoMap &op_info_maps) const;

  void WriteOpType(const std::string &saver_base_dir);

  void WriteOpDetail(const std::string &saver_base_dir);

  void WriteOpTimestamp(const std::string &saver_base_dir);

  void ChangeFileMode(const std::string &file_path) const;

  OpTypeInfos op_type_infos_;
  OpDetailInfos op_detail_infos_;
  std::string op_side_;
  std::string device_id_;
};
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PROFILER_DEVICE_DATA_SAVER_H
