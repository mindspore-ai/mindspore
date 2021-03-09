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

#include "profiler/device/profiling.h"

#include <time.h>
#include <cxxabi.h>
#include <cmath>
#include "profiler/device/cpu/cpu_data_saver.h"
#include "pybind_api/api_register.h"
#include "utils/log_adapter.h"
#include "utils/utils.h"

namespace mindspore {
namespace profiler {
uint64_t Profiler::GetHostMonoTimeStamp() {
  struct timespec ts;
#if defined(_WIN32) || defined(_WIN64)
  clock_gettime(CLOCK_MONOTONIC, &ts);
#else
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#endif
  constexpr uint64_t kNSecondInSecond = 1000000000;
  uint64_t cur_time_stamp = ts.tv_sec * kNSecondInSecond + ts.tv_nsec;
  return cur_time_stamp;
}

void Profiler::SetRunTimeData(const std::string &op_name, const float time_elapsed) {
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.op_host_cost_time += time_elapsed;
  }
}

void Profiler::SetRunTimeData(const std::string &op_name, const uint64_t start, const float duration) {
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.start_duration.emplace_back(StartDuration({start, duration}));
  }
}
}  // namespace profiler
}  // namespace mindspore
