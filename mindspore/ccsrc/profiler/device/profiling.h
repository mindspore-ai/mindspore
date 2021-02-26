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

#ifndef MINDSPORE_CCSRC_PROFILER_DEVICE_PROFILING_H
#define MINDSPORE_CCSRC_PROFILER_DEVICE_PROFILING_H
#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mindspore {
namespace profiler {
struct StartDuration {
  uint64_t start_timestamp = 0l;
  float duration = 0l;
};

struct OpInfo {
  std::string op_name;
  float cupti_api_call_time = 0l;
  float cupti_activity_time = 0l;
  float op_host_cost_time = 0;
  int op_kernel_api_count = 0;
  int op_kernel_count = 0;
  int op_count = 0;
  std::vector<StartDuration> start_duration;
  void *stream;
  uint32_t pid;
};

class Profiler {
 public:
  Profiler() = default;
  virtual ~Profiler() = default;

  virtual void Init(const std::string &profileDataPath) = 0;
  virtual void Stop() = 0;
  virtual void StepProfilingEnable(const bool enable_flag) = 0;
  virtual void OpDataProducerEnd() = 0;
  bool GetEnableFlag() const { return enable_flag_; }
  std::string ProfileDataPath() const { return profile_data_path_; }

 protected:
  void SetRunTimeData(const std::string &op_name, const float time_elapsed);
  void SetRunTimeData(const std::string &op_name, const uint64_t start, const float duration);
  uint64_t GetHostMonoTimeStamp();
  virtual void SaveProfileData() = 0;
  virtual void ClearInst() = 0;
  bool enable_flag_ = false;
  std::string profile_data_path_;
  std::unordered_map<std::string, OpInfo> op_info_map_;
};
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PROFILER_DEVICE_PROFILING_H
