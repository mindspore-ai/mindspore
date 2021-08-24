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

#include "runtime/device/pynative_profiling.h"
#include <iostream>
#include <fstream>
#include <utility>
#include <memory>
#include <string>
#include "utils/profile.h"
#include "utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
PynativeProfiler::PynativeProfiler() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  enable_profiler_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PROFILING);
}

void PynativeProfiler::SingleOpProfilingData() {
  if (!enable_profiler_flag) {
    return;
  }
  static std::ofstream of("pynative_forward_profiling_data.csv");
  of.setf(std::ios::fixed, std::ios::floatfield);
  if (real_run_op_index_ == 1) {
    of << "RealRunOpIndex" << ',' << "RealRunOpName" << ',' << "OpName" << ',' << "RealRunOpStartTime(s)" << ','
       << "OpDeviceStartTime(s)" << ',' << "OpDeviceEndTime(s)" << ',' << "RealRunOpEndTime(s)" << std::endl;
  }
  if (op_name_launch_start_time_end_time_vec_.empty()) {
    of << real_run_op_index_ << ',' << real_run_op_name_ << ',' << "nopnode" << ','
       << real_run_op_start_time_end_time_.first << ',' << "nopnode" << ',' << "nopnode" << ','
       << real_run_op_start_time_end_time_.second << std::endl;
    return;
  }
  for (const auto &i : op_name_launch_start_time_end_time_vec_) {
    of << real_run_op_index_ << ',' << real_run_op_name_ << ',' << i.first << ','
       << real_run_op_start_time_end_time_.first << ',' << i.second.first << ',' << i.second.second << ','
       << real_run_op_start_time_end_time_.second << std::endl;
  }
  op_name_launch_start_time_end_time_vec_.clear();
}
}  // namespace device
}  // namespace mindspore
