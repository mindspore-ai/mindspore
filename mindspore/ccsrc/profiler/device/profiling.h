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
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "utils/hash_map.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace profiler {
struct StartDuration {
  uint64_t start_timestamp = 0l;
  float duration = 0l;
  size_t tid = 0;
};

struct OneStepStartEndInfo {
  std::string iter_start_op_name;
  std::string fp_start_op_name;
  std::string iter_end_op_name;
};

struct OpInfo {
  std::string op_name;
  float cupti_api_call_time = 0l;
  float cupti_activity_time = 0l;
  float op_host_cost_time = 0;
  int op_kernel_api_count = 0;
  int op_kernel_count = 0;
  int op_count = 0;
  StartDuration tmp_start_duration;
  std::vector<StartDuration> start_duration;
  void *stream;
  uint32_t pid;
};

class BACKEND_EXPORT ProfilerManager {
 public:
  static std::shared_ptr<ProfilerManager> &GetInstance();
  ProfilerManager() = default;
  ~ProfilerManager() = default;
  ProfilerManager(const ProfilerManager &) = delete;
  ProfilerManager &operator=(const ProfilerManager &) = delete;
  bool GetProfilingEnableFlag() const;
  void RecordOneStepStartEndInfo() const;
  std::string GetProfilingOptions() const;
  bool GetNetDynamicShapeStatus() const { return is_dynamic_shape_net_; }
  void SetNetDynamicShapeStatus() { is_dynamic_shape_net_ = true; }

 private:
  inline static std::shared_ptr<ProfilerManager> profiler_manager_inst_ = std::make_shared<ProfilerManager>();
  bool is_dynamic_shape_net_ = 0;
};

class BACKEND_EXPORT Profiler {
 public:
  static std::shared_ptr<Profiler> GetInstance(const std::string &name) noexcept;
  static bool Register(const std::string &name, const std::shared_ptr<Profiler> &instance);
  static void Clear();

  Profiler() = default;
  virtual ~Profiler() = default;

  virtual void Init(const std::string &profiling_path, uint32_t device_id, const std::string &profiling_options) = 0;
  virtual void Finalize() = 0;
  bool IsInitialized() const { return init_flag_; }
  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual void StepProfilingEnable(const bool enable_flag) = 0;
  virtual void OpDataProducerEnd() = 0;
  void RecordOneStepStartEndInfo();
  bool GetEnableFlag() const { return enable_flag_; }
  std::string GetProfilingOptions() const { return profiling_options_; }
  std::string ProfileDataPath() const { return profile_data_path_; }
  void RecordOneStepStartEndInfo(std::string op_name);
  std::pair<double, double> GetSingleOpLaunchTime() { return single_op_launch_start_time_end_time_; }
  void SetSingleOpLaunchTime(const std::pair<double, double> &launch_start_end) {
    single_op_launch_start_time_end_time_ = launch_start_end;
  }
  bool GetParallelStrategyEnableFlag() const { return is_parallel_strategy; }
  void SyncEnable(const bool enable_flag);
  void DataProcessEnable(const bool enable_flag);

 protected:
  void SetRunTimeData(const std::string &op_name, const float time_elapsed);
  void SetRunTimeData(const std::string &op_name, const uint64_t start, const float duration);
  void FindOneStepFpStartOp(uint32_t vector_size);
  void FindOneStepIterEndOp(uint32_t vector_size);
  uint64_t GetHostMonoTimeStamp() const;
  // Get timestamp in us
  uint64_t GetRealTimeStamp() const;
  virtual void SaveProfileData() = 0;
  virtual void ClearInst() = 0;
  std::pair<double, double> single_op_launch_start_time_end_time_;
  bool enable_flag_ = false;
  bool has_find_ = false;
  bool is_parallel_strategy = false;
  bool init_flag_ = false;
  std::string profile_data_path_;
  std::unordered_map<std::string, OpInfo> op_info_map_;
  OneStepStartEndInfo step_start_end_info_;
  std::vector<OneStepStartEndInfo> all_step_start_end_info_;
  std::vector<std::string> step_start_end_info_vector_;
  std::shared_mutex op_map_mutex_;
  std::mutex record_mutex_;
  std::string profiling_options_;
  uint32_t iter_end_op_index_ = 0;
  uint32_t fp_start_op_index_ = 1;
  bool sync_enable_flag_ = true;
  bool data_process_enable_ = false;
  std::string op_type_ = "GetNext";

 private:
  static std::map<std::string, std::shared_ptr<Profiler>> &GetInstanceMap();
};
}  // namespace profiler
}  // namespace mindspore

#define PROFILER_REG(NAME, CLAZZ) \
  static bool g_Profiler_##NAME##_reg_result = mindspore::profiler::Profiler::Register(NAME, std::make_shared<CLAZZ>())

#endif  // MINDSPORE_CCSRC_PROFILER_DEVICE_PROFILING_H
