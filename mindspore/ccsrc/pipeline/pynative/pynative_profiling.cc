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

#include "pipeline/pynative/pynative_profiling.h"
#include "utils/profile.h"
#include "utils/ms_context.h"
#include "utils/utils.h"
#include "profiler/device/profiling.h"

namespace mindspore {
constexpr int kDeviceInfoCoutWidth = 25;
constexpr int kHostTimePointCoutWidth = 35;
constexpr int kHostTimeCoutWidth = 30;

void PynativeProfiler::SetEnableProfilingFlag() {
  static bool flag = false;
  if (flag) {
    return;
  }
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  enable_profiler_flag_ = profiler_manager->GetProfilingEnableFlag();
  flag = true;
}

void PynativeProfiler::Reset() {
  stage_time_point_vec_.clear();
  stage_stat_time_vec_.clear();
  op_name_launch_time_point_vec_.clear();
  op_name_launch_time_vec_.clear();
}

void PynativeProfiler::SetDeviceOpNameAndLaunchTimePoint(
  const std::pair<std::string, std::pair<double, double>> &name_start_end) {
  if (!enable_profiler_flag_ || op_name_launch_time_point_vec_.size() > kMaxVectorSize) {
    return;
  }
  op_name_launch_time_point_vec_.push_back(name_start_end);
}

void PynativeProfiler::SetDeviceOpNameAndLaunchCostTime(const std::pair<std::string, double> &name_time) {
  if (!enable_profiler_flag_ || op_name_launch_time_vec_.size() > kMaxVectorSize) {
    return;
  }
  op_name_launch_time_vec_.push_back(name_time);
}

void PynativeProfiler::ExportDeviceInfoToFile() {
  MS_LOG(DEBUG) << "Op name launch time point vec size: " << op_name_launch_time_point_vec_.size();
  if (!enable_profiler_flag_ || op_name_launch_time_point_vec_.empty()) {
    return;
  }
  static std::ofstream of_device("device_profiling_data.csv", std::ios::app);
  of_device.setf(std::ios::fixed, std::ios::floatfield);
  of_device << "DeviceIndex" << ',' << "op_name" << ',' << "LaunchStartTime(s)" << ',' << "LaunchEndTime(s)" << ','
            << "LaunchCostTime(ms)" << std::endl;
  if (op_name_launch_time_point_vec_.size() != op_name_launch_time_vec_.size()) {
    MS_LOG(EXCEPTION) << "The size of the two vector is not equal, the two vector size is "
                      << op_name_launch_time_point_vec_.size() << " and " << op_name_launch_time_vec_.size();
  }
  for (size_t i = 1; i <= op_name_launch_time_point_vec_.size(); ++i) {
    of_device << i << ',' << op_name_launch_time_point_vec_[i - 1].first << ','
              << op_name_launch_time_point_vec_[i - 1].second.first << ','
              << op_name_launch_time_point_vec_[i - 1].second.second << ','
              << op_name_launch_time_vec_[i - 1].second * kBasicTimeTransferUnit << std::endl;
  }
  op_name_launch_time_point_vec_.clear();
  op_name_launch_time_vec_.clear();
}

void PynativeProfiler::ExportDeviceInfoToScreen() {
  MS_LOG(DEBUG) << "Op name launch time point vec size: " << op_name_launch_time_point_vec_.size();
  if (!enable_profiler_flag_ || op_name_launch_time_point_vec_.empty()) {
    return;
  }
  if (op_name_launch_time_point_vec_.size() != op_name_launch_time_vec_.size()) {
    MS_LOG(EXCEPTION) << "The size of the two vector is not equal, the two vector size is "
                      << op_name_launch_time_point_vec_.size() << " and " << op_name_launch_time_vec_.size();
  }
  std::cout << "====================================DeviceInfo===================================" << std::endl;
  std::vector<std::string> head_str = {"DeviceIndex", "op_name", "LaunchStartTime(s)", "LaunchEndTime(s)",
                                       "LaunchCostTime(ms)"};
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.setf(std::ios::left);
  for (const auto &str : head_str) {
    std::cout.width(kDeviceInfoCoutWidth);
    std::cout << str;
  }
  std::cout << std::endl;
  for (size_t i = 1; i <= op_name_launch_time_point_vec_.size(); ++i) {
    std::cout.width(kDeviceInfoCoutWidth);
    std::cout << i;
    std::cout.width(kDeviceInfoCoutWidth);
    std::cout << op_name_launch_time_point_vec_[i - 1].first;
    std::cout.width(kDeviceInfoCoutWidth);
    std::cout << op_name_launch_time_point_vec_[i - 1].second.first;
    std::cout.width(kDeviceInfoCoutWidth);
    std::cout << op_name_launch_time_point_vec_[i - 1].second.second;
    std::cout.width(kDeviceInfoCoutWidth);
    std::cout << op_name_launch_time_vec_[i - 1].second * kBasicTimeTransferUnit;
    std::cout << std::endl;
  }
  std::cout << "==============================================================================" << std::endl;
}

void PynativeProfiler::SetStageTimePoint(const std::string &stage_name, const std::string &flag) {
  if (!enable_profiler_flag_ || stage_time_point_vec_.size() > kMaxVectorSize) {
    return;
  }
  stage_time_point_vec_.emplace_back(stage_name, std::make_pair(flag, GetTime()));
}

double PynativeProfiler::SetStageTimePointWithReturn(const std::string &stage_name, const std::string &flag) {
  if (!enable_profiler_flag_ || stage_time_point_vec_.size() > kMaxVectorSize) {
    return 0;
  }
  double tmp_time = GetTime();
  stage_time_point_vec_.emplace_back(stage_name, std::make_pair(flag, GetTime()));
  return tmp_time;
}

void PynativeProfiler::SetStageStatTime(const std::string &stage_name, double cost_time) {
  if (!enable_profiler_flag_ || stage_stat_time_vec_.size() > kMaxVectorSize) {
    return;
  }
  stage_stat_time_vec_.emplace_back(stage_name, cost_time);
}

void PynativeProfiler::ExportStageTimePointToFile() {
  if (!enable_profiler_flag_ || stage_time_point_vec_.empty()) {
    return;
  }
  static std::ofstream of_host("host_stage_time_point_profiling_data.csv", std::ios::app);
  of_host.setf(std::ios::fixed, std::ios::floatfield);
  for (size_t i = 0; i < stage_time_point_vec_.size(); ++i) {
    if (i == 0) {
      of_host << stage_time_point_vec_[i].first + stage_time_point_vec_[i].second.first + "Time(s)";
      continue;
    }
    of_host << ',' << stage_time_point_vec_[i].first + stage_time_point_vec_[i].second.first + "Time(s)";
  }
  of_host << std::endl;
  for (size_t i = 0; i < stage_time_point_vec_.size(); ++i) {
    if (i == 0) {
      of_host << stage_time_point_vec_[i].second.second;
      continue;
    }
    of_host << ',' << stage_time_point_vec_[i].second.second;
  }
  of_host << std::endl;
  stage_time_point_vec_.clear();
}

void PynativeProfiler::ExportStageTimePointToScreen() {
  if (!enable_profiler_flag_ || stage_time_point_vec_.empty()) {
    return;
  }
  std::cout << "===============================StageTimePoint=================================" << std::endl;
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.setf(std::ios::left);
  for (const auto &i : stage_time_point_vec_) {
    std::cout.width(kHostTimePointCoutWidth);
    std::cout << i.first + i.second.first + "Time(s)";
  }
  std::cout << std::endl;
  for (const auto &i : stage_time_point_vec_) {
    std::cout.width(kHostTimePointCoutWidth);
    std::cout << i.second.second * kBasicTimeTransferUnit;
  }
  std::cout << std::endl;
  std::cout << "==============================================================================" << std::endl;
}

void PynativeProfiler::ExportStageStatTimeToFile() {
  if (!enable_profiler_flag_ || stage_stat_time_vec_.empty()) {
    return;
  }
  static std::ofstream of_host("host_stage_stat_time_profiling_data.csv", std::ios::app);
  of_host.setf(std::ios::fixed, std::ios::floatfield);
  for (size_t i = 0; i < stage_stat_time_vec_.size(); ++i) {
    if (i == 0) {
      of_host << stage_stat_time_vec_[i].first + "Time(ms)";
      continue;
    }
    of_host << ',' << stage_stat_time_vec_[i].first + "Time(ms)";
  }
  of_host << std::endl;
  for (size_t i = 0; i < stage_stat_time_vec_.size(); ++i) {
    if (i == 0) {
      of_host << stage_stat_time_vec_[i].second * kBasicTimeTransferUnit;
      continue;
    }
    of_host << ',' << stage_stat_time_vec_[i].second * kBasicTimeTransferUnit;
  }
  of_host << std::endl;
  stage_stat_time_vec_.clear();
}

void PynativeProfiler::ExportStageStatTimeToScreen() {
  if (!enable_profiler_flag_ || stage_stat_time_vec_.empty()) {
    return;
  }
  std::cout << "================================StageStatTime=================================" << std::endl;
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.setf(std::ios::left);
  for (const auto &i : stage_stat_time_vec_) {
    std::cout.width(kHostTimeCoutWidth);
    std::cout << i.first + "Time(ms)";
  }
  std::cout << std::endl;
  for (const auto &i : stage_stat_time_vec_) {
    std::cout.width(kHostTimeCoutWidth);
    std::cout << i.second * kBasicTimeTransferUnit;
  }
  std::cout << std::endl;
  std::cout << "==============================================================================" << std::endl;
}
}  // namespace mindspore
