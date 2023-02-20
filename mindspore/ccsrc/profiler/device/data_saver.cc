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
#include "profiler/device/data_saver.h"
#include <fstream>
#include <numeric>
#include "sys/stat.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace profiler {
OpDetailInfo::OpDetailInfo(const std::shared_ptr<OpInfo> op_info, float proportion)
    : op_info_(op_info), proportion_(proportion) {
  // op_full_name is like 'xxx/xxx/{op_type}-op{node_id}'
  op_full_name_ = op_info->op_name;
  auto op_type_begin_iter = op_full_name_.rfind('/') + 1;
  auto op_type_end_iter = op_full_name_.rfind('-');
  op_type_ = op_full_name_.substr(op_type_begin_iter, op_type_end_iter - op_type_begin_iter);
  op_name_ = op_full_name_.substr(op_type_begin_iter);
  if (op_info->op_count == 0) {
    MS_LOG(ERROR) << "The num of operations can not be 0.";
    return;
  }
  op_avg_time_ = op_info->op_host_cost_time / op_info->op_count;
}

void DataSaver::ParseOpInfo(const OpInfoMap &op_info_maps) {
  op_detail_infos_.reserve(op_info_maps.size());
  float total_time_sum = GetTotalOpTime(op_info_maps);
  for (auto item : op_info_maps) {
    op_timestamps_map_[item.first] = item.second.start_duration;
    if (common::IsFloatEqual(total_time_sum, 0.0)) {
      MS_LOG(ERROR) << "The total operation times can not be 0.";
      return;
    }
    float proportion = item.second.op_host_cost_time / total_time_sum;
    auto op_info = std::make_shared<OpInfo>(item.second);
    if (op_info == nullptr) {
      MS_LOG(ERROR) << "Create Operation information node failed when parse operation information.";
      return;
    }
    OpDetailInfo op_detail_info = OpDetailInfo(op_info, proportion);
    op_detail_infos_.emplace_back(op_detail_info);
    AddOpDetailInfoForType(op_detail_info);
  }
  // update average time of op type
  for (auto &op_type : op_type_infos_) {
    // device_infos: <type_name, op_type_info>
    if (op_type.second.count_ == 0) {
      MS_LOG(ERROR) << "The num of operation type can not be 0.";
      return;
    }
    op_type.second.avg_time_ = op_type.second.total_time_ / op_type.second.count_;
  }
  MS_LOG(DEBUG) << "Get " << op_detail_infos_.size() << " operation items.";
  MS_LOG(DEBUG) << "Get " << op_type_infos_.size() << " operation type items.";
}

void DataSaver::AddOpDetailInfoForType(const OpDetailInfo &op_detail_info) {
  // Construct OpType object according to op detail info
  OpType op_type = OpType{op_detail_info.op_type_,
                          op_detail_info.op_info_->op_count,
                          op_detail_info.op_info_->op_count,
                          op_detail_info.op_info_->op_host_cost_time,
                          0,
                          op_detail_info.proportion_};
  // Set the OpType into op_type_infos_ map
  std::string type_name = op_detail_info.op_type_;
  auto iter = op_type_infos_.find(type_name);
  if (iter == op_type_infos_.end()) {
    op_type_infos_.emplace(type_name, op_type);
  } else {
    iter->second += op_type;
  }
}

float DataSaver::GetTotalOpTime(const OpInfoMap &op_info_maps) const {
  float sum = 0;
  sum = std::accumulate(op_info_maps.begin(), op_info_maps.end(), sum,
                        [](float i, auto iter) { return i + iter.second.op_host_cost_time; });
  MS_LOG(DEBUG) << "The total op time is " << sum;
  return sum;
}

void DataSaver::WriteOpType(const std::string &saver_base_dir) {
  std::string file_path = saver_base_dir + "/" + op_side_ + "_op_type_info_" + device_id_ + ".csv";
  std::ofstream ofs(file_path);
  // check if the file is writable
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }
  try {
    // write op type info into file
    if (op_side_ == "cpu") {
      ofs << OpType().GetCpuHeader() << std::endl;
      for (auto op_type_info : op_type_infos_) {
        op_type_info.second.OutputCpuOpTypeInfo(ofs);
      }
    }
    if (op_side_ == "gpu") {
      ofs << OpType().GetGpuHeader() << std::endl;
      for (auto op_type_info : op_type_infos_) {
        op_type_info.second.OutputGpuOpTypeInfo(ofs);
      }
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Write " << file_path << "failed: " << e.what();
  }
  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write " << op_type_infos_.size() << " op type infos into file: " << file_path;
  op_type_infos_.clear();
}

void DataSaver::WriteOpDetail(const std::string &saver_base_dir) {
  std::string file_path = saver_base_dir + "/" + op_side_ + "_op_detail_info_" + device_id_ + ".csv";
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }
  try {
    // write op detail info into file
    if (op_side_ == "cpu") {
      ofs << OpDetailInfo().GetCpuHeader() << std::endl;
      for (auto op_detail : op_detail_infos_) {
        op_detail.OutputCpuOpDetailInfo(ofs);
      }
    }
    if (op_side_ == "gpu") {
      ofs << OpDetailInfo().GetGpuHeader() << std::endl;
      for (auto op_detail : op_detail_infos_) {
        op_detail.OutputGpuOpDetailInfo(ofs);
      }
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Write " << file_path << "failed: " << e.what();
  }
  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write " << op_detail_infos_.size() << " op detail infos into file: " << file_path;
  op_detail_infos_.clear();
}

void DataSaver::WriteOpTimestamp(const std::string &saver_base_dir) {
  std::string file_path = saver_base_dir + "/" + op_side_ + "_op_execute_timestamp_" + device_id_ + ".txt";
  std::ofstream ofs(file_path);
  // check if the file is writable
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }
  try {
    // write op timestamp info into file
    for (const auto &op_timestamp_info : op_timestamps_map_) {
      if (op_side_ == "cpu") {
        ofs << op_timestamp_info.first << ";HostCpuOps;";
        for (auto start_end : op_timestamp_info.second) {
          ofs << start_end.start_timestamp << "," << start_end.duration << "," << start_end.tid << " ";
        }
      } else {
        ofs << op_timestamp_info.first << ";GpuOps;";
        for (auto start_end : op_timestamp_info.second) {
          ofs << start_end.start_timestamp << "," << start_end.duration << " ";
        }
      }
      ofs << std::endl;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Write " << file_path << "failed: " << e.what();
  }
  ofs.close();
  ChangeFileMode(file_path);
  if (op_side_ == "cpu") {
    op_timestamps_map_.clear();
  }
}

void DataSaver::WriteFrameWork(const std::string &base_dir, const std::vector<CurKernelInfo> &all_kernel_info) {
  std::string file_path = base_dir + "/" + op_side_ + "_framework_" + device_id_ + ".txt";
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }
  for (auto kernel_info : all_kernel_info) {
    auto op_name = kernel_info.op_name;
    auto op_type = kernel_info.op_type;
    auto graph_id = kernel_info.graph_id;
    auto cur_kernel_all_inputs_info = kernel_info.cur_kernel_all_inputs_info;
    try {
      ofs << op_type << ";" << op_name << ";" << graph_id << ";";
      for (auto cur_kernel_input_info : cur_kernel_all_inputs_info) {
        ofs << "input_" << cur_kernel_input_info.input_id << ":" << cur_kernel_input_info.shape << ";";
      }
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Write " << file_path << "failed:" << e.what();
      ofs.close();
      return;
    }
    ofs << std::endl;
  }
  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write framework infos into file: " << file_path;
}

void DataSaver::ChangeFileMode(const std::string &file_path) const {
  if (chmod(common::SafeCStr(file_path), S_IRUSR | S_IWUSR) == -1) {
    MS_LOG(WARNING) << "Modify file: " << file_path << " to rw fail.";
    return;
  }
}
}  // namespace profiler
}  // namespace mindspore
