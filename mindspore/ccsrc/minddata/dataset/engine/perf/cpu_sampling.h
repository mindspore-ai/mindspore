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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CPU_SAMPLING_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CPU_SAMPLING_H

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

// CPU information from /proc/stat or /proc/pid/stat file
typedef struct CpuStat_s {
  uint64_t user_stat_;
  uint64_t sys_stat_;
  uint64_t io_stat_;
  uint64_t idle_stat_;
  uint64_t total_stat_;
} CpuStat;

// Cpu utilization
typedef struct CpuInfo_s {
  uint8_t user_utilization_;
  uint8_t sys_utilization_;
  uint8_t io_utilization_;
  uint8_t idle_utilization_;
} CpuUtil;

// CPU utilization of operator
typedef struct CpuOpInfo_s {
  float user_utilization_;
  float sys_utilization_;
  int32_t op_id;
} CpuOpUtil;

// CPU utilization of process
typedef struct CpuProcessInfo_s {
  float user_utilization_;
  float sys_utilization_;
} CpuProcessUtil;

// CPU stat of operator
typedef struct CpuOpStat_s {
  uint64_t user_stat_;
  uint64_t sys_stat_;
} CpuOpStat;

class BaseCpu {
 public:
  BaseCpu();
  ~BaseCpu() = default;
  // Collect CPU information
  virtual Status Collect(const ExecutionTree *tree) = 0;
  virtual Status SaveToFile(const std::string &file_path) = 0;
  virtual Status Analyze(std::string *name, double *utilization, std::string *extra_message) = 0;

 protected:
  std::vector<CpuUtil> cpu_util_;
  CpuStat pre_cpu_stat_;
  static bool fetched_all_process_shared;
  static std::unordered_map<int32_t, std::vector<pid_t>> op_process_shared;
  bool fetched_all_process;
  bool pre_fetched_state;
  std::unordered_map<int32_t, std::vector<pid_t>> op_process;
  int32_t cpu_processor_num_;
};

// Collect device CPU information
class DeviceCpu : public BaseCpu {
 public:
  DeviceCpu() : pre_running_process_(0), pre_context_switch_count_(0), first_collect_(true) {}
  ~DeviceCpu() = default;
  Status Collect(const ExecutionTree *tree) override;
  Status SaveToFile(const std::string &file_path) override;
  Status Analyze(std::string *name, double *utilization, std::string *extra_message) override;

 private:
  // Get CPU information, include use/sys/idle/io utilization
  Status ParseCpuInfo(const std::string &str);

  // Get context switch count
  Status ParseCtxt(const std::string &str);

  // Get running process count
  Status ParseRunningProcess(const std::string &str);

  std::vector<uint32_t> running_process_;
  std::vector<uint64_t> context_switch_count_;
  uint32_t pre_running_process_;
  uint64_t pre_context_switch_count_;
  bool first_collect_;
};

// Collect operator CPU information
class OperatorCpu : public BaseCpu {
 public:
  OperatorCpu() : first_collect_(true), pre_total_stat_(0), id_count_(0) {}
  ~OperatorCpu() = default;
  Status Collect(const ExecutionTree *tree) override;
  Status SaveToFile(const std::string &file_path) override;
  // Analyze will output the name of the metric, the avg utiliization of highest
  // object within the class and any extra message that would be useful for the user.
  // The Higher level CPUSampling class will combine information from different classes
  // to decide if warning should be output.
  Status Analyze(std::string *name, double *utilization, std::string *extra_message) override;

 private:
  // Get cpu information, include use/sys/idle/io utilization
  Status ParseCpuInfo(int32_t op_id, int64_t thread_id,
                      std::unordered_map<int32_t, std::unordered_map<int64_t, CpuOpStat>> *op_stat);

  // Get the total CPU time of device
  Status GetTotalCpuTime(uint64_t *total_stat);

  // Store the CPU utilization of each operator
  std::vector<std::vector<CpuOpUtil>> cpu_op_util_;

  bool first_collect_;

  // Store the id and its corresponding threads.
  std::unordered_map<int32_t, std::vector<pid_t>> op_thread;
  std::unordered_map<int32_t, std::string> op_name;
  std::unordered_map<int32_t, int32_t> op_parallel_workers;
  std::unordered_map<int32_t, std::unordered_map<int64_t, CpuOpStat>> pre_op_stat_;
  uint64_t pre_total_stat_;
  int32_t id_count_;
};

// Collect operator CPU information
class ProcessCpu : public BaseCpu {
 public:
  ProcessCpu() : first_collect_(true), pre_total_stat_(0) {}
  ~ProcessCpu() = default;
  Status Collect(const ExecutionTree *tree) override;
  Status SaveToFile(const std::string &file_path) override;
  Status Analyze(std::string *name, double *utilization, std::string *extra_message) override;

 private:
  // Get CPU information, include use/sys/idle/io utilization
  Status ParseCpuInfo();

  // Get the total CPU time of device
  Status GetTotalCpuTime(uint64_t *total_stat);

  bool first_collect_;
  std::vector<CpuProcessUtil> process_util_;
  uint64_t pre_total_stat_;
  std::unordered_map<int64_t, CpuOpStat> pre_process_stat_;
  std::vector<pid_t> process_id;
};

// Sampling CPU information
// It support JSON serialization for external usage.
class CpuSampling : public Sampling {
  using TimeStamp = std::vector<uint32_t>;

 public:
  explicit CpuSampling(ExecutionTree *tree) : tree_(tree) {}

  ~CpuSampling() = default;

  // Driver function for CPU sampling.
  // This function samples the CPU information of device/process/op
  Status Sample() override;

  std::string Name() const override { return kCpuSamplingName; }

  // Save sampling data to file
  // @return Status - The error code return
  Status SaveToFile() override;

  Status Init(const std::string &dir_path, const std::string &device_id) override;

  // Change file mode after save CPU data
  Status ChangeFileMode() override;

  // Analyze sampling data and print message to log
  Status Analyze() override;

 private:
  Status CollectTimeStamp();

  Status SaveTimeStampToFile();

  Status SaveSamplingItervalToFile();

  ExecutionTree *tree_ = nullptr;              // ExecutionTree pointer
  std::vector<std::shared_ptr<BaseCpu>> cpu_;  // CPU information of device/process/op
  TimeStamp time_stamp_;                       // Time stamp
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CPU_SAMPLING_H
