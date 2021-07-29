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
#include <iostream>
#include <utility>
#include <memory>
#include <string>
#include "utils/profile.h"
#include "utils/ms_context.h"

namespace mindspore {
void PynativeProfiler::SetEnableProfilingFlag() {
  static bool flag = false;
  if (flag) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  enable_profiler_flag_ = ms_context->get_param<bool>(MS_CTX_ENABLE_PROFILING);
  flag = true;
}

void PynativeProfiler::SetForwardTimePoint(const std::string &stage_name, const std::string &flag) {
  if (!enable_profiler_flag_) {
    return;
  }
  forward_data_.emplace_back(stage_name, std::make_pair(flag, GetTime()));
}

void PynativeProfiler::SetRealRunOpName(const std::string &name) {
  if (!enable_profiler_flag_) {
    return;
  }
  real_run_op_name_ = name;
}

void PynativeProfiler::SetBackwardTimePoint(const std::string &stage_name, const std::string &flag) {
  if (!enable_profiler_flag_) {
    return;
  }
  backward_data_.emplace_back(stage_name, std::make_pair(flag, GetTime()));
}

void PynativeProfiler::SetBackwardRunOpImplOpName(const std::string &name) {
  if (!enable_profiler_flag_) {
    return;
  }
  backward_run_op_impl_op_name_.push_back(name);
}

void PynativeProfiler::SingleOpForwardHostProfilingData() {
  if (!enable_profiler_flag_ || forward_data_.empty()) {
    return;
  }
  static std::ofstream of_host("pynative_forward_host_profiling_data.csv");
  of_host.setf(std::ios::fixed, std::ios::floatfield);
  ++real_run_op_index_;
  of_host << "RealRunOpIndex" << ',' << "RealRunOpName";
  for (const auto &i : forward_data_) {
    of_host << ',' << i.first + i.second.first + "Time(s)";
  }
  of_host << std::endl;
  of_host << real_run_op_index_ << ',' << real_run_op_name_;
  for (const auto &i : forward_data_) {
    of_host << ',' << i.second.second;
  }
  of_host << std::endl;
  forward_data_.clear();
}

void PynativeProfiler::SingleOpBackwardHostProfilingData() {
  if (!enable_profiler_flag_ || backward_data_.empty()) {
    return;
  }
  static std::ofstream of_host("pynative_backward_host_profiling_data.csv");
  of_host.setf(std::ios::fixed, std::ios::floatfield);
  ++backward_run_grad_graph_index_;
  of_host << "BackwardIndex";
  for (const auto &i : backward_data_) {
    if (i.first == "BackwardRunOpImpl" && i.second.first == "Start") {
      of_host << ',' << "BackwardRunOpImplOpName" << ',' << i.first + i.second.first + "Time(s)";
      continue;
    }
    of_host << ',' << i.first + i.second.first + "Time(s)";
  }
  of_host << std::endl;
  of_host << backward_run_grad_graph_index_;
  size_t backward_run_op_impl_op_name_index = 0;
  size_t backward_run_op_impl_op_name_size = backward_run_op_impl_op_name_.size();
  for (const auto &i : backward_data_) {
    if (i.first == "BackwardRunOpImpl" && i.second.first == "Start") {
      if (backward_run_op_impl_op_name_index >= backward_run_op_impl_op_name_size) {
        MS_LOG(EXCEPTION) << "backward_run_op_impl_op_name_index is bigger than backward_run_op_impl_op_name_size";
      }
      of_host << ',' << backward_run_op_impl_op_name_[backward_run_op_impl_op_name_index++] << ',' << i.second.second;
      continue;
    }
    of_host << ',' << i.second.second;
  }
  of_host << std::endl;
  backward_data_.clear();
  backward_run_op_impl_op_name_.clear();
}

void PynativeProfiler::SingleOpDeviceProfilingData() {
  if (!enable_profiler_flag_ || op_name_launch_time_vec_.empty()) {
    return;
  }
  static std::ofstream of_device("pynative_device_profiling_data.csv");
  of_device.setf(std::ios::fixed, std::ios::floatfield);
  of_device << "DeviceIndex" << ',' << "op_name" << ',' << "LaunchStartTime(s)" << ',' << "LaunchEndTime(s)"
            << std::endl;
  for (size_t i = 1; i <= op_name_launch_time_vec_.size(); ++i) {
    of_device << i << ',' << op_name_launch_time_vec_[i - 1].first << ','
              << op_name_launch_time_vec_[i - 1].second.first << ',' << op_name_launch_time_vec_[i - 1].second.second
              << std::endl;
  }
  op_name_launch_time_vec_.clear();
}
}  // namespace mindspore
