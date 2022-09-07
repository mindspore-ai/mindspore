/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_PROFILING_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_PROFILING_H_

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/perf/monitor.h"

namespace mindspore {
namespace dataset {

class Monitor;
class ExecutionTree;
class TreeConsumer;
class CpuSampler;
class TreeAdapter;

const char kDeviceQueueTracingName[] = "Device_Queue_Tracing";
const char kDatasetIteratorTracingName[] = "Dataset_Iterator_Tracing";
const char kConnectorSizeSamplingName[] = "Connector_Size_Sampling";
const char kCpuSamplerName[] = "Cpu_Sampler";

// Values for process memory metrics - common for profiling and cpu_sampler
enum ProcessMemoryMetric { kPSS, kRSS, kVSS };

// Values for system memory metrics - common for profiling and cpu_sampler
enum SystemMemoryMetric { kMemoryAvailable, kMemoryTotal, kMemoryUsed };

// Profiling is a class of basic unit of profiling action
// This base class encapsulate the serialization output logic
class Profiling : public std::enable_shared_from_this<Profiling> {
 public:
  // Constructor
  Profiling() : active_(false) {}

  // Destructor
  virtual ~Profiling() = default;

  virtual Status Init() = 0;

  // Default serialization file generator
  virtual Status SaveToFile(const std::string &dir_path, const std::string &rank_id) = 0;

  // Profiling name
  virtual std::string Name() const = 0;

  virtual Status ChangeFileMode(const std::string &dir_path, const std::string &rank_id) = 0;

  // Start collecting data
  Status Start();

  // Stop collecting data
  Status Stop();

  // Clear all collected data
  virtual void Clear() = 0;

 protected:
  bool active_;  // show current state of ProfilingManager (running, or paused)
  std::mutex lock_;
  virtual Path GetFileName(const std::string &dir_path, const std::string &rank_id) = 0;
};

// Sampling is a class of profiling which generate samples periodically.
class Sampling : public Profiling {
 public:
  // Sampling action function. This function will be invoked by performance monitor thread.
  virtual Status Sample() = 0;

  ~Sampling() override = default;
};

typedef struct TracingRecord_s {
  int32_t type;
  int32_t extra_info;
  int32_t batch_num;
  int32_t value;
  uint64_t ts;

  std::string ToString() const {
    return std::to_string(type) + " " + std::to_string(extra_info) + " " + std::to_string(batch_num) + " " +
           std::to_string(value) + " " + std::to_string(ts);
  }
} TracingRecord;

// Tracing is class of profiling which record samples upon request.
class Tracing : public Profiling {
 public:
  // Tracing has minimal interface to provide flexible on data recording.
  // It only includes some common routines.
  Status SaveToFile(const std::string &dir_path, const std::string &rank_id) override;
  Status ChangeFileMode(const std::string &dir_path, const std::string &rank_id) override;
  Status Init() override;
  Status GetPipelineTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);
  Status GetPushTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);
  Status GetBatchTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);
  Status GetConnectorSize(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);
  Status GetConnectorCapacity(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);
  Status GetEmptyQueueFrequency(int32_t start_step, int32_t end_step, float_t *empty_queue_freq);
  void Record(const int32_t type, const int32_t extra_info, const int32_t batch_num, const int32_t value,
              const uint64_t time_stamp);
  Status TimeIntervalForStepRange(int32_t start_step, int32_t end_step, uint64_t *start_ts, uint64_t *end_ts);
  Status StepIntervalForTimeRange(uint64_t start_ts, uint64_t end_ts, int32_t *start_step, int32_t *end_step);
  size_t GetNumberSteps();

  // Clear all collected data
  void Clear() override;

 protected:
  Tracing() = default;
  std::vector<std::string> value_;
  std::vector<TracingRecord> records_;
  std::vector<uint64_t> ts_;  // End time of each step or batch
  Status GetRecordEntryFieldValue(int32_t start_step, int32_t end_step, int32_t record_offset, const std::string &field,
                                  std::vector<int32_t> *result);
};

// ProfilingManager is a class manages all profiling infrastructure
// It serves the following purposes:
// 1) Fetch profiling configs from global contexts
// 2) Setup all profiling node based on config
// 3) Provide access of profiling nodes for profiling actions
// 4) Manage profiling data serialization process
class ProfilingManager {
  friend Monitor;

 public:
  ProfilingManager();

  ~ProfilingManager() = default;

  /// Register the given tree to be profiled.
  /// This method should be called once, calling it for another tree without resetting the ProfilingManager would fail.
  /// \param tree_adapter pointer the adapter that owns the ExecutionTree
  /// \return Status the status code returned
  Status RegisterTree(const TreeAdapter *tree_adapter);

  /// Reset the ProfilingManager. This method is sued when we want to profile another tree in the same process.
  /// \return Status the status code returned
  Status Reset();

  // Save profile data to file
  // @param dir_path_ The path to the directory where the profiling data will be saved.
  // @return Status The status code returned
  Status SaveProfilingData(const std::string &dir_path, const std::string &rank_id);

  // Sampling node getter
  // @param name - The name of the requested node
  // @param node - Pointer to the shared pointer for the Sampling node
  // @return Status The status code returned
  Status GetSamplingNode(const std::string &name, std::shared_ptr<Sampling> *node);

  // Tracing node getter
  // @param name - The name of the requested node
  // @param node - Pointer to the shared pointer for the Tracing node
  // @return Status The status code returned
  Status GetTracingNode(const std::string &name, std::shared_ptr<Tracing> *node);

  // return true if enabled_ is set to true, namely if Init() has been called successfully
  // @param tree - Execution tree pointer
  bool IsProfilingEnable(const ExecutionTree *tree = nullptr) const;

  // Record end of epoch information
  // @param step_num - The number of steps
  void RecordEndOfEpoch(uint32_t step_num);

  const std::unordered_map<std::string, std::shared_ptr<Sampling>> &GetSamplingNodes() const { return sampling_nodes_; }

  // Launch monitoring thread.
  Status LaunchMonitor();

  // @return Status The status code returned
  Status ChangeFileMode(const std::string &dir_path, const std::string &rank_id);

#ifndef ENABLE_ANDROID
  /// \brief API to get User CPU utilization for the system
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with the sampled User CPU Utilization for the entire system
  /// \return Status object with the error code
  Status GetUserCpuUtilByEpoch(int32_t epoch_num, std::vector<uint8_t> *result);

  /// \brief API to get User CPU utilization for the system
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with the sampled User CPU Utilization for the entire system
  /// \return Status object with the error code
  Status GetUserCpuUtilByStep(int32_t start_step, int32_t end_step, std::vector<uint8_t> *result);

  /// \brief API to get User CPU utilization for the system
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with the sampled User CPU Utilization for the entire system
  /// \return Status object with the error code
  Status GetUserCpuUtilByTime(uint64_t start_ts, uint64_t end_ts, std::vector<uint8_t> *result);

  /// \brief API to get System CPU utilization for the system
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with the sampled System CPU Utilization for the entire system
  /// \return Status object with the error code
  Status GetSysCpuUtilByEpoch(int32_t epoch_num, std::vector<uint8_t> *result);

  /// \brief API to get System CPU utilization for the system
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with the sampled System CPU Utilization for the entire system
  /// \return Status object with the error code
  Status GetSysCpuUtilByStep(int32_t start_step, int32_t end_step, std::vector<uint8_t> *result);

  /// \brief API to get System CPU utilization for the system
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with the sampled System CPU Utilization for the entire system
  /// \return Status object with the error code
  Status GetSysCpuUtilByTime(uint64_t start_ts, uint64_t end_ts, std::vector<uint8_t> *result);

  /// \brief API to get User CPU Utilization of an MD operator
  /// \param [in] op_id The id of the operator
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with the sampled User CPU Utilization of the operator.
  /// \return Status object with the error code
  Status GetUserCpuUtilByEpoch(int32_t op_id, int32_t epoch_num, std::vector<uint16_t> *result);

  /// \brief API to get User CPU Utilization of an MD operator
  /// \param [in] op_id The id of the operator
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with the sampled User CPU Utilization of the operator.
  /// \return Status object with the error code
  Status GetUserCpuUtilByStep(int32_t op_id, int32_t start_step, int32_t end_step, std::vector<uint16_t> *result);

  /// \brief API to get User CPU Utilization of an MD operator
  /// \param [in] op_id The id of the operator
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with the sampled User CPU Utilization of the operator.
  /// \return Status object with the error code
  Status GetUserCpuUtilByTime(int32_t op_id, uint64_t start_ts, uint64_t end_ts, std::vector<uint16_t> *result);

  /// \brief API to get System CPU Utilization of an MD operator
  /// \param [in] op_id The id of the operator
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with the sampled System CPU Utilization of the operator.
  /// \return Status object with the error code
  Status GetSysCpuUtilByEpoch(int32_t op_id, int32_t epoch_num, std::vector<uint16_t> *result);

  /// \brief API to get System CPU Utilization of an MD operator
  /// \param [in] op_id The id of the operator
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with the sampled System CPU Utilization of the operator.
  /// \return Status object with the error code
  Status GetSysCpuUtilByStep(int32_t op_id, int32_t start_step, int32_t end_step, std::vector<uint16_t> *result);

  /// \brief API to get System CPU Utilization of an MD operator
  /// \param [in] op_id The id of the operator
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with the sampled System CPU Utilization of the operator.
  /// \return Status object with the error code
  Status GetSysCpuUtilByTime(int32_t op_id, uint64_t start_ts, uint64_t end_ts, std::vector<uint16_t> *result);

  /// \brief API to get information on main process memory usage
  /// \param [in] metric The requested memory set usage.  One of these values:
  ///     - ProcessMemoryMetric::kVSS - virtual set size, virtual memory usage
  ///     - ProcessMemoryMetric::kPSS - proportional set size, physical memory usage with proportional allocation of
  ///     shared libraries
  ///     - ProcessMemoryMetric::kRSS - resident set size, physical memory usage (includes shared libraries)
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result The desired value in MB
  /// \return Status object with the error code
  Status GetMainProcessMemoryInfoByEpoch(ProcessMemoryMetric metric, int32_t epoch_num, std::vector<float> *result);

  /// \brief API to get information on main process memory usage
  /// \param [in] metric The requested memory set usage.  One of these values:
  ///     - ProcessMemoryMetric::kVSS - virtual set size, virtual memory usage
  ///     - ProcessMemoryMetric::kPSS - proportional set size, physical memory usage with proportional allocation of
  ///     shared libraries
  ///     - ProcessMemoryMetric::kRSS - resident set size, physical memory usage (includes shared libraries)
  /// \param [in] end_ts The time interval end range in ms
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result The desired value in MB
  /// \return Status object with the error code
  Status GetMainProcessMemoryInfoByStep(ProcessMemoryMetric metric, int32_t start_step, int32_t end_step,
                                        std::vector<float> *result);

  /// \brief API to get information on main process memory usage
  /// \param [in] metric The requested memory set usage.  One of these values:
  ///     - ProcessMemoryMetric::kVSS - virtual set size, virtual memory usage
  ///     - ProcessMemoryMetric::kPSS - proportional set size, physical memory usage with proportional allocation of
  ///     shared libraries
  ///     - ProcessMemoryMetric::kRSS - resident set size, physical memory usage (includes shared libraries)
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result The desired value in MB
  /// \return Status object with the error code
  Status GetMainProcessMemoryInfoByTime(ProcessMemoryMetric metric, uint64_t start_ts, uint64_t end_ts,
                                        std::vector<float> *result);

  /// \brief API to get information on system memory usage
  /// \param [in] metric The requested memory metric.  One of these values:
  ///     - SystemMemoryMetric::kMemoryAvailable
  ///     - SystemMemoryMetric::kMemoryTotal
  ///     - SystemMemoryMetric::kMemoryUsed
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result The desired value in MB
  /// \return Status object with the error code
  Status GetSystemMemoryInfoByEpoch(SystemMemoryMetric metric, int32_t epoch_num, std::vector<float> *result);

  /// \brief API to get information on system memory usage
  /// \param [in] metric The requested memory metric.  One of these values:
  ///     - SystemMemoryMetric::kMemoryAvailable
  ///     - SystemMemoryMetric::kMemoryTotal
  ///     - SystemMemoryMetric::kMemoryUsed
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result The desired value in MB
  /// \return Status object with the error code
  Status GetSystemMemoryInfoByStep(SystemMemoryMetric metric, int32_t start_step, int32_t end_step,
                                   std::vector<float> *result);

  /// \brief API to get information on system memory usage
  /// \param [in] metric The requested memory metric.  One of these values:
  ///     - SystemMemoryMetric::kMemoryAvailable
  ///     - SystemMemoryMetric::kMemoryTotal
  ///     - SystemMemoryMetric::kMemoryUsed
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result The desired value in MB
  /// \return Status object with the error code
  Status GetSystemMemoryInfoByTime(SystemMemoryMetric metric, uint64_t start_ts, uint64_t end_ts,
                                   std::vector<float> *result);
#endif

  /// \brief API to get the connector size of an MD operator
  /// \param [in] op_id The id of the operator
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with the sampled connector sizes of the operator
  /// \return Status object with the error code
  Status GetConnectorSizeByEpoch(int32_t op_id, int32_t epoch_num, std::vector<int32_t> *result);

  /// \brief API to get the connector size of an MD operator
  /// \param [in] op_id The id of the operator
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with the sampled connector sizes of the operator
  /// \return Status object with the error code
  Status GetConnectorSizeByStep(int32_t op_id, int32_t start_step, int32_t end_step, std::vector<int32_t> *result);

  /// \brief API to get the connector size of an MD operator
  /// \param [in] op_id The id of the operator
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with the sampled connector sizes of the operator
  /// \return Status object with the error code
  Status GetConnectorSizeByTime(int32_t op_id, uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result);

  /// \brief API to get the connector size of DatasetIterator or DataQueueOp
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with connector size at each step
  /// \return Status object with the error code
  Status GetConnectorSizeByEpoch(int32_t epoch_num, std::vector<int32_t> *result);

  /// \brief API to get the connector size of DatasetIterator or DataQueueOp
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with connector size at each step
  /// \return Status object with the error code
  Status GetConnectorSizeByStep(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);

  /// \brief API to get the connector size of DatasetIterator or DataQueueOp
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with connector size at each step
  /// \return Status object with the error code
  Status GetConnectorSizeByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result);

  /// \brief API to get the connector capacity of DatasetIterator or DataQueueOp
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with connector capacity at each step
  /// \return Status object with the error code
  Status GetConnectorCapacityByEpoch(int32_t epoch_num, std::vector<int32_t> *result);

  /// \brief API to get the connector capacity of DatasetIterator or DataQueueOp
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with connector capacity at each step
  /// \return Status object with the error code
  Status GetConnectorCapacityByStep(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);

  /// \brief API to get the connector capacity of DatasetIterator or DataQueueOp
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with connector capacity for steps in the given time range
  /// \return Status object with the error code
  Status GetConnectorCapacityByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result);

  /// \brief API to get the pipeline time of batches
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with the pipeline time for each step
  /// \return Status object with the error code
  Status GetPipelineTimeByEpoch(int32_t epoch_num, std::vector<int32_t> *result);

  /// \brief API to get the pipeline time of batches
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with the pipeline time for each step
  /// \return Status object with the error code
  Status GetPipelineTimeByStep(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);

  /// \brief API to get the pipeline time of batches
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with the pipeline time for steps in the given time range
  /// \return Status object with the error code
  Status GetPipelineTimeByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result);

  /// \brief API to get the push time of batches
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with the push time for each each step
  /// \return Status object with the error code
  Status GetPushTimeByEpoch(int32_t epoch_num, std::vector<int32_t> *result);

  /// \brief API to get the push time of batches
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with the push time for each each step
  /// \return Status object with the error code
  Status GetPushTimeByStep(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);

  /// \brief API to get the push time of batches
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with the push time for steps in the given time range
  /// \return Status object with the error code
  Status GetPushTimeByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result);

  /// \brief API to get the batch time of batches
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result A vector with the batch time for each step
  /// \return Status object with the error code
  Status GetBatchTimeByEpoch(int32_t epoch_num, std::vector<int32_t> *result);

  /// \brief API to get the batch time of batches
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result A vector with the batch time for each step
  /// \return Status object with the error code
  Status GetBatchTimeByStep(int32_t start_step, int32_t end_step, std::vector<int32_t> *result);

  /// \brief API to get the batch time of batches
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result A vector with the batch time for steps in the given time range
  /// \return Status object with the error code
  Status GetBatchTimeByTime(uint64_t start_ts, uint64_t end_ts, std::vector<int32_t> *result);

  /// \brief API to get fraction of steps that DatasetIterator or DataQueueOp connector was empty
  /// \param [in] epoch_num The epoch number for which results are requested
  /// \param [out] result The empty queue frequency
  /// \return Status object with the error code
  Status GetEmptyQueueFrequencyByEpoch(int32_t epoch_num, float_t *result);

  /// \brief API to get fraction of steps that DatasetIterator or DataQueueOp connector was empty
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] result The empty queue frequency
  /// \return Status object with the error code
  Status GetEmptyQueueFrequencyByStep(int32_t start_step, int32_t end_step, float_t *result);

  /// \brief API to get fraction of steps that DatasetIterator or DataQueueOp connector was empty
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] result The empty queue frequency
  /// \return Status object with the error code
  Status GetEmptyQueueFrequencyByTime(uint64_t start_ts, uint64_t end_ts, float_t *result);

  // Register profile node to tree
  // @param node - Profiling node
  // @return Status The status code returned
  Status RegisterTracingNode(const std::shared_ptr<Tracing> &node);

  /// \brief API to initialize profiling manager
  /// \param for_autotune flag to indicate if Profiler is initialized for autotuning or profiling purposes
  /// \return Status object with the error code
  Status Init(const bool for_autotune = false);

  /// \brief API to signal the profiling nodes to start collecting data
  /// \return Status object with the error code
  Status Start();

  /// \brief API to signal profiling nodes to stop collecting data
  /// \return Status object with the error code
  Status Stop();

  /// \brief API to save to file all the collected data between Start and Stop calls
  /// \return Status object with the error code
  Status Save(const std::string &profile_data_path);

  /// \brief Helper to get the rank id. Currently being used for appending rank id to files
  /// \return String The rank id
  std::string GetRankID() const;

  /// Get number of epochs that have been already profiled
  /// \return number of epochs
  int32_t GetNumOfProfiledEpochs() const { return static_cast<int32_t>(epoch_end_step_.size()) - 1; }

  // Get number of steps taken in pipeline
  /// \return number of steps
  Status GetNumberOfProfiledSteps(int32_t *steps);

  /// Determine if the Profiler is being used for autotuning.
  /// \return boolean
  bool IsAutotuning() const { return autotuning_; }

  /// Determine if the Profiler is being used for profiling.
  /// \return boolean
  bool IsProfiling() const { return profiling_; }

  // Registration state for the profiler
  enum ProfilingRegistrationState {
    kNotEnabled,
    kEnabledTreeNotRegistered,
    kEnabledTreeRegistered,
    kEnabledDifferentTreeRegistered,
  };

  /// \brief Getter for the profiling and tree registration state
  /// \param tree Execution Tree pointer
  /// \return ProfilingRegistrationState
  ProfilingRegistrationState GetProfilerTreeState(const ExecutionTree *tree) const;

 protected:
  std::unique_ptr<Monitor> perf_monitor_;

  // State flags for profiling
  enum ProfilingState {
    kProfilingStateUnBegun,
    kProfilingStateRunning,
    kProfilingStateFinished,
  };
  ProfilingState profiling_state_;  // show current state of ProfilingManager (running, or paused)
  std::unordered_map<std::string, std::shared_ptr<Tracing>> tracing_nodes_;
  std::unordered_map<std::string, std::shared_ptr<Sampling>> sampling_nodes_;
  ExecutionTree *tree_;                   // ExecutionTree pointer
  std::vector<uint64_t> epoch_end_ts_;    // End of epoch timestamp
  std::vector<uint32_t> epoch_end_step_;  // End of epoch step number
  std::atomic<bool> autotuning_;  // flag to indicate if ProfilingManager is being used for auto-tuning the pipeline
  std::atomic<bool> profiling_;   // flag to indicate if ProfilingManager is being used for profiling the pipeline

  // Register profile node to tree
  // @param node - Profiling node
  // @return Status The status code returned
  Status RegisterSamplingNode(const std::shared_ptr<Sampling> &node);

  /// \brief Helper to convert a given epoch number to a step interval
  /// \param [in] epoch_num The epoch number to be converted
  /// \param [out] start_step The corresponding start step for the given epoch
  /// \param [out] end_step The corresponding end step for the given epoch
  /// \return Status object with the error code
  Status EpochToStepInterval(int32_t epoch_num, uint32_t *start_step, uint32_t *end_step);

  /// \brief Helper to convert a given epoch number to a time interval
  /// \param [in] epoch_num The epoch number to be converted
  /// \param [out] start_ts The corresponding starting timestamp in ms for the given epoch
  /// \param [out] end_ts The corresponding ending timestamp in ms for the given epoch
  /// \return Status object with the error code
  Status EpochToTimeInterval(int32_t epoch_num, uint64_t *start_ts, uint64_t *end_ts);

  /// \brief Helper to convert step interval to a time interval
  /// \param [in] start_step The step interval start range
  /// \param [in] end_step The step interval end range
  /// \param [out] start_ts The corresponding starting timestamp in ms for the given step interval
  /// \param [out] end_ts The corresponding ending timestamp in ms for the given step interval
  /// \return Status object with the error code
  Status StepToTimeInterval(int32_t start_step, int32_t end_step, uint64_t *start_ts, uint64_t *end_ts);

  /// \brief Helper to convert time interval to a step interval
  /// \param [in] start_ts The time interval start range in ms
  /// \param [in] end_ts The time interval end range in ms
  /// \param [out] start_step The corresponding start step for the given time interval
  /// \param [out] end_step The corresponding end step for the given time interval
  /// \return Status object with the error code
  Status TimeToStepInterval(uint64_t start_ts, uint64_t end_ts, int32_t *start_step, int32_t *end_step);
};

enum ProfilingType { TIME, CONNECTOR_DEPTH };

enum ProfilingTimeSubType {
  PIPELINE_TIME,
  TDT_PUSH_TIME,
  BATCH_TIME,
  INVALID_TIME,
};

class ProfilingTime {
 public:
  static uint64_t GetCurMilliSecond();
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_PROFILING_H_
