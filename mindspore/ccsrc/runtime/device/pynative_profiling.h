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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_PYNATIVE_PROFILING_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_PYNATIVE_PROFILING_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

namespace mindspore {
namespace device {
class PynativeProfiler {
 public:
  static PynativeProfiler &GetInstance() {
    static PynativeProfiler instance;
    return instance;
  }

  void AddRealRunOpIndex() { ++real_run_op_index_; }
  void SetRealRunOpName(const std::string &name) { real_run_op_name_ = name; }
  void SetRealRunOpTime(const std::pair<double, double> &start_end) { real_run_op_start_time_end_time_ = start_end; }
  void SetOpNameAndLaunchTime(const std::pair<std::string, std::pair<double, double>> &name_start_end) {
    op_name_launch_start_time_end_time_vec_.push_back(name_start_end);
  }
  void SingleOpProfilingData();

 private:
  PynativeProfiler();
  ~PynativeProfiler() = default;
  PynativeProfiler(const PynativeProfiler &) = delete;
  PynativeProfiler &operator=(const PynativeProfiler &) = delete;
  bool enable_profiler_flag = false;
  int real_run_op_index_ = 0;
  std::string real_run_op_name_;
  std::pair<double, double> real_run_op_start_time_end_time_;
  std::vector<std::pair<std::string, std::pair<double, double>>> op_name_launch_start_time_end_time_vec_;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_PYNATIVE_PROFILING_H_
