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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_PROFILING_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_PROFILING_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

namespace mindspore {
class PynativeProfiler {
 public:
  PynativeProfiler(const PynativeProfiler &) = delete;
  PynativeProfiler &operator=(const PynativeProfiler &) = delete;
  static void SetEnableProfilingFlag();
  static void Reset();
  static void SetDeviceOpNameAndLaunchTimePoint(
    const std::pair<std::string, std::pair<double, double>> &name_start_end);
  static void SetDeviceOpNameAndLaunchCostTime(const std::pair<std::string, double> &name_time);
  static void ExportDeviceInfoToFile();
  static void ExportDeviceInfoToScreen();
  static void SetStageTimePoint(const std::string &stage_name, const std::string &flag);
  static double SetStageTimePointWithReturn(const std::string &stage_name, const std::string &flag);
  static void SetStageStatTime(const std::string &stage_name, double cost_time);
  static void ExportStageTimePointToFile();
  static void ExportStageTimePointToScreen();
  static void ExportStageStatTimeToFile();
  static void ExportStageStatTimeToScreen();

 private:
  PynativeProfiler() = default;
  ~PynativeProfiler() = default;
  inline static bool enable_profiler_flag_ = false;
  inline static std::vector<std::pair<std::string, std::pair<std::string, double>>> stage_time_point_vec_;
  inline static std::vector<std::pair<std::string, double>> stage_stat_time_vec_;
  inline static std::vector<std::pair<std::string, std::pair<double, double>>> op_name_launch_time_point_vec_;
  inline static std::vector<std::pair<std::string, double>> op_name_launch_time_vec_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_PROFILING_H_
