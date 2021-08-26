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
class PynativeProfiler {
 public:
  static void SetForwardTimePoint(std::string stage_name, std::string flag);
  static void SetRealRunOpName(const std::string &name);
  static void SetBackwardTimePoint(std::string stage_name, std::string flag);
  static void SetBackwardRunOpImplOpName(const std::string &name);
  static void SetOpNameAndLaunchTime(const std::pair<std::string, std::pair<double, double>> &name_start_end) {
    op_name_launch_time_vec_.push_back(name_start_end);
  }

  static void SetEnableProfilingFlag();
  static void SingleOpForwardHostProfilingData();
  static void SingleOpBackwardHostProfilingData();
  static void SingleOpDeviceProfilingData();

 private:
  PynativeProfiler() = default;
  ~PynativeProfiler() = default;
  PynativeProfiler(const PynativeProfiler &) = delete;
  PynativeProfiler &operator=(const PynativeProfiler &) = delete;
  inline static bool enable_profiler_flag_ = false;
  inline static int real_run_op_index_ = 0;
  inline static std::string real_run_op_name_;
  inline static std::vector<std::pair<std::string, std::pair<std::string, double>>> forward_data_;
  inline static int backward_run_grad_graph_index_ = 0;
  inline static std::vector<std::string> backward_run_op_impl_op_name_;
  inline static std::vector<std::pair<std::string, std::pair<std::string, double>>> backward_data_;
  inline static std::vector<std::pair<std::string, std::pair<double, double>>> op_name_launch_time_vec_;
};
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_PYNATIVE_PROFILING_H_
