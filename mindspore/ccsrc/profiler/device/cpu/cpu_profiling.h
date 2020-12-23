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

#ifndef MINDSPORE_CPU_PROFILING_H
#define MINDSPORE_CPU_PROFILING_H
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
namespace cpu {
struct StartDuration {
  uint64_t start_timestamp = 0l;
  float duration = 0l;
};

struct OpInfo {
  std::string op_name;
  float op_cost_time = 0;
  int op_count = 0;
  std::vector<StartDuration> start_duration;
  uint32_t pid;
};

const float kTimeUnit = 1000;

class CPUProfiler {
 public:
  static std::shared_ptr<CPUProfiler> GetInstance();
  ~CPUProfiler() = default;
  CPUProfiler(const CPUProfiler &) = delete;
  CPUProfiler &operator=(const CPUProfiler &) = delete;

  void Init(const std::string &profileDataPath);
  void Stop();
  void StepProfilingEnable(const bool enable_flag);
  bool GetEnableFlag() const { return enable_flag_; }
  void OpDataProducerBegin(const std::string op_name, const uint32_t pid);
  void OpDataProducerEnd();
  std::string ProfileDataPath() const { return profile_data_path_; }

 private:
  CPUProfiler() = default;
  void ClearInst();
  void SetRunTimeData(const std::string &op_name, const uint32_t pid);
  void SetRunTimeData(const std::string &op_name, const float time_elapsed);
  void SetRunTimeData(const std::string &op_name, const uint64_t start, const float duration);

  static std::shared_ptr<CPUProfiler> profiler_inst_;
  bool enable_flag_ = false;
  std::unordered_map<std::string, OpInfo> op_info_map_;
  uint64_t base_time_;
  std::string op_name_;
  uint32_t pid_;
  void SaveProfileData();

  uint64_t op_time_start_;
  uint64_t op_time_mono_start_;
  uint64_t op_time_stop_;
  std::string profile_data_path_;
};
}  // namespace cpu
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CPU_PROFILING_H
