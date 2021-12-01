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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_CPU_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_CPU_SAMPLER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"

namespace mindspore {
namespace dataset {

class ExecutionTree;

typedef struct SystemStat_s {
  uint64_t user_stat;
  uint64_t sys_stat;
  uint64_t io_stat;
  uint64_t idle_stat;
  uint64_t total_stat;
} SystemStat;

typedef struct SystemUtil_s {
  uint8_t user_utilization;
  uint8_t sys_utilization;
  uint8_t io_utilization;
  uint8_t idle_utilization;
} SystemUtil;

typedef struct TaskStat_s {
  uint64_t user_stat;
  uint64_t sys_stat;
} TaskStat;

struct TaskUtil_s {
  float user_utilization;
  float sys_utilization;
};

typedef struct TaskUtil_s TaskUtil;
typedef struct TaskUtil_s OpUtil;

class SystemCpuInfo {
 public:
  SystemCpuInfo() : first_sample_(true), prev_context_switch_count_(0) {}
  // Read in current stats and return previous and currently read stats
  Status SampleAndGetCurrPrevStat(SystemStat *current_stat, SystemStat *previous_stat);
  static int32_t num_cpu_;
  const std::vector<uint32_t> &GetRunningProcess() const { return running_process_; }
  const std::vector<uint64_t> &GetContextSwitchCount() const { return context_switch_count_; }
  Status GetUserCpuUtil(uint64_t start_index, uint64_t end_index, std::vector<uint8_t> *result) const;
  Status GetSysCpuUtil(uint64_t start_index, uint64_t end_index, std::vector<uint8_t> *result) const;
  std::vector<uint8_t> GetIOCpuUtil() const;
  std::vector<uint8_t> GetIdleCpuUtil() const;

 private:
  Status ParseCpuInfo(const std::string &str);
  Status ParseCtxt(const std::string &str);
  Status ParseRunningProcess(const std::string &str);
  SystemStat prev_sys_stat_{};                  // last read data /proc/stat file
  std::vector<SystemUtil> sys_cpu_util_;        // vector of system cpu utilization
  std::vector<uint32_t> running_process_;       // vector of running processes in system
  std::vector<uint64_t> context_switch_count_;  // vector of number of context switches between two sampling points
  bool first_sample_;                           // flag to indicate first time sampling
  uint64_t prev_context_switch_count_;          // last read context switch count from /proc/stat file
};

class TaskCpuInfo {
 public:
  explicit TaskCpuInfo(pid_t pid) : pid_(pid), first_sample_(true), last_sampling_failed_(false) {}
  virtual ~TaskCpuInfo() = default;
  virtual Status Sample(uint64_t total_time_elapsed) = 0;
  virtual pid_t GetId() = 0;
  TaskUtil GetLatestCpuUtil() const;
  std::vector<uint16_t> GetSysCpuUtil() const;
  std::vector<uint16_t> GetUserCpuUtil() const;

 protected:
  pid_t pid_;
  TaskStat prev_task_stat_;
  std::vector<TaskUtil> task_cpu_util_;
  bool first_sample_;
  bool last_sampling_failed_;
};

class ProcessCpuInfo : public TaskCpuInfo {
 public:
  explicit ProcessCpuInfo(pid_t pid) : TaskCpuInfo(pid) {}
  ~ProcessCpuInfo() override = default;
  Status Sample(uint64_t total_time_elapsed) override;
  pid_t GetId() override { return pid_; }
};

class ThreadCpuInfo : public TaskCpuInfo {
 public:
  explicit ThreadCpuInfo(pid_t pid, pid_t tid) : TaskCpuInfo(pid), tid_(tid) {}
  ~ThreadCpuInfo() override = default;
  Status Sample(uint64_t total_time_elapsed) override;
  pid_t GetId() override { return tid_; }

 private:
  pid_t tid_;
};

class MDOperatorCpuInfo {
 public:
  void AddTask(const std::shared_ptr<TaskCpuInfo> &task_ptr);
  bool TaskExists(pid_t id) const;
  explicit MDOperatorCpuInfo(const int32_t op_id) : id_(op_id) {}
  void CalculateOperatorUtilization();
  Status GetUserCpuUtil(uint64_t start_index, uint64_t end_index, std::vector<uint16_t> *result) const;
  Status GetSysCpuUtil(uint64_t start_index, uint64_t end_index, std::vector<uint16_t> *result) const;

 private:
  int32_t id_;
  // tid is key for threadinfo, pid is key for processinfo
  std::unordered_map<pid_t, std::shared_ptr<TaskCpuInfo>> task_by_id_;
  std::vector<OpUtil> op_cpu_util_;
};

class CpuSampler : public Sampling {
  using Timestamps = std::vector<uint64_t>;

 public:
  explicit CpuSampler(ExecutionTree *tree) : fetched_all_python_multiprocesses_(false), tree(tree) {}
  ~CpuSampler() = default;
  Status Sample() override;
  Status Init() override;
  Status ChangeFileMode(const std::string &dir_path, const std::string &rank_id) override;
  Status SaveToFile(const std::string &dir_path, const std::string &rank_id) override;
  std::string Name() const override { return kCpuSamplerName; }
  Status GetSystemUserCpuUtil(uint64_t start_ts, uint64_t end_ts, std::vector<uint8_t> *result);
  Status GetSystemSysCpuUtil(uint64_t start_ts, uint64_t end_ts, std::vector<uint8_t> *result);
  Status GetOpUserCpuUtil(int32_t op_id, uint64_t start_ts, uint64_t end_ts, std::vector<uint16_t> *result);
  Status GetOpSysCpuUtil(int32_t op_id, uint64_t start_ts, uint64_t end_ts, std::vector<uint16_t> *result);

  // Clear all collected data
  void Clear() override;

 private:
  Status UpdateTaskList();
  bool fetched_all_python_multiprocesses_{};
  ExecutionTree *tree = nullptr;
  pid_t main_pid_{};
  Timestamps ts_;
  SystemCpuInfo sys_cpu_info_;                       // store the system cpu utilization
  std::vector<std::shared_ptr<TaskCpuInfo>> tasks_;  // vector of all process and thread tasks
  std::shared_ptr<ThreadCpuInfo> main_thread_cpu_info_;
  std::shared_ptr<ProcessCpuInfo> main_process_cpu_info_;
  std::unordered_map<int32_t, MDOperatorCpuInfo> op_info_by_id_;
  Path GetFileName(const std::string &dir_path, const std::string &rank_id) override;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_CPU_SAMPLER_H_
