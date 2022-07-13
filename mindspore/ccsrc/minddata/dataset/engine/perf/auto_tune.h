/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_AUTO_TUNE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_AUTO_TUNE_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/tree_adapter.h"
#include "minddata/dataset/engine/tree_modifier.h"
#include "minddata/dataset/engine/perf/profiling.h"

namespace mindspore {
namespace dataset {
class TreeModifier;
class AutoTune {
 public:
  /// AutoTune constructor
  /// \param tree_adap_  pointer to the tree adapter
  /// \param profiling_mgr_ pointer to the profiler manager
  AutoTune(TreeAdapter *tree_adap, ProfilingManager *profiling_mgr);

  ~AutoTune() = default;

  /// Function to create and launch the AutoTune thread.
  /// \return Status object
  Status LaunchThread();

 private:
  /// Main entry function for AT, triggers loop function.
  /// \return Status object
  Status Main();

  /// Primary AT loop till exit
  /// \return Status object
  Status ATMainLoop(bool output_intermediate_config);

  /// \brief Helper to print the tree configuration
  void PrintTreeConfiguration() const;

  /// \brief Helper to print the logs after/post the main loop in AutoTune
  void PostMainLogging() const;

  /// \brief Helper to summarize the execution tree
  /// \param[out] out An output vector of string to store the summary
  /// \return Status object
  Status SummarizeTreeConfiguration(std::vector<std::string> *out);

#ifndef ENABLE_ANDROID
  /// \brief Serialize the dataset and save the AT config (workers and queue size) to a json file
  /// \param file_name Name of the file
  /// \return Status object
  Status SaveAutotuneConfig(const std::string &file_name);

  /// Setter for autotune_config_json_
  /// \return Status code
  Status SetAutotuneConfigJson();
#endif

  /// Function to collect info from the tree
  /// \return Status code
  Status CollectOpsInfo();

  /// Function to check for current step and execute logic
  /// \return status code
  Status RunIterationStep();

  /// Function to check for current epoch and execute logic
  /// \return status code
  Status RunIterationEpoch();

  /// The AutoTune logic for pipelines that executes every epoch
  /// \return status code
  Status RunIteration();

  /// Fetches connector size for steps or epoch based on mode
  /// \return status code
  Status GetConnectorSize(std::vector<int32_t> *sizes) const;

  /// Fetches connector capacity for steps or epoch based on mode
  /// \return status code
  Status GetConnectorCapacity(std::vector<int32_t> *capacities) const;

  /// Computes current connector queue util percentage
  /// \param[out] usage_avg_last double return avg util percentage for connector queue
  /// \param[out] avg_size double to return avg size (usage) of connector queue
  /// \param[out] avg_capacity double to return avg capacity for connector queue
  /// \return status code
  Status GetConnectorUtil(double *usage_avg_last, double *avg_size, double *avg_capacity);

  /// Fetches Connector Queue empty frequency for steps or epoch based on mode
  /// \return status code
  Status GetEmptyQueueFrequency(float *empty_freq) const;

  /// Check if the dataset pipeline is the bottleneck
  /// \param[out] isBottleneck bool
  /// \return Status code
  Status IsDSaBottleneck(bool *isBottleneck);

  /// Returns true if the pipeline is sink or non-sink
  /// \return bool
  bool IsSink() const;

  const int32_t TO_PERCENT = 100;
  // system specifics
  int32_t max_workers_;
  const int32_t MIN_NUM_WORKERS = 1;
  const int32_t MAX_QUEUE_SIZE = 128;
  const int32_t MIN_QUEUE_SIZE = 1;
  // Warmup specifics
  const int32_t EPOCH_WARMUP = 1;
  const int64_t STEP_WARMUP = 150;
  // Worker specifics
  const int32_t INCREMENT_WORKER = 2;
  const int32_t DECREMENT_WORKER = -1;
  // Queue Specifics
  const float_t INPUT_QUEUE_LOW = 0.5;

  // Value to maintain checking for device_queue utlization at.
  const float_t DEVICE_CONNECTOR_UTIL_THRESHOLD = 0.75;

  const float_t LEAF_QUEUE_THRESHOLD = 0.9;
  const float_t INPUT_OUTPUT_QUEUE_DIFF_THRESHOLD = 0.35;
  const int64_t INCREMENT_QUEUE_SIZE = 4;
  // CPU Specifics
  const float_t MAP_OP_WORKER_HIGH_THRESHOLD = 75;
  const float_t MAP_OP_WORKER_LOW_THRESHOLD = 35;
  // Running mode specifics
  enum AutoTuneMode { kAutoTuneModeEpoch, kAutoTuneModeStep };
  enum AutoTunePhase { kAutoTunePhaseTime, kAutoTunePhaseMemory, kAutoTuneEnd };
  enum AutoTuneMemPhase { kAutoTuneMemInit, kAutoTuneMemSet, kAutotTuneMemCompare };
  // Early stop specifics
  const int32_t EARLY_STOP_TRIAL_THRESHOLD_EPOCH = 4;
  const int32_t EARLY_STOP_TRIAL_THRESHOLD_STEP = 10;
  // Memory specifics
  const float MEMORY_COMPARISON_LOWER_BOUND_PERCENT = 0.02;
  const float QUEUE_REDUCTION_PERCENTAGE_EPOCH = 0.5;
  const float QUEUE_REDUCTION_PERCENTAGE_STEP = 0.8;

  /// Get the out connector capacity of the operator
  /// \param[in] op_id operator id
  /// \param[out] capacity the capacity of the connector
  /// \return Status code
  Status GetOpConnectorCapacity(int32_t op_id, int64_t *capacity);

  /// Get the CPU usage of each operator in the pipeline
  /// \param[out] ops_cpu_util map from op_id to cpu utilization
  /// \return Status code
  Status GetOpsCpuUtil(std::map<int32_t, double> *ops_cpu_util);

  /// Get the queue utilization of each operator in the pipeline
  /// \param[out] ops_queue_util map from op_id to output queue utilization
  /// \param[out] ops_queue_util map from op_id to input queue utilization
  /// \note inline ops would report -1 in both input and output queue utilization
  /// \return Status code
  Status GetOpsQueueUtil(std::map<int32_t, double> *out_ops_queue_util, std::map<int32_t, double> *in_ops_queue_util);

  /// Get the number of workers for each operator in the pipeline
  /// \param[out] ops_num_workers map from op_id to num_workers
  /// \return Status code
  Status GetOpsNumWorker(std::map<int32_t, int32_t> *ops_num_workers);

  /// Check whether an op is an unsupported by AutoTune
  /// \param op_id ID to check
  /// \return bool to skip or not
  bool SkipOpsCheck(int op_id);

  /// Main AutoTune algorithm
  /// \return Status code
  Status AnalyseTime();

  /// AutoTune memory algorithm
  /// \return Status code
  Status AnalyseMemory();

  /// Send a ChangeRequest to the operator to update the number of workers
  /// \param op_id operator ID
  /// \param old_workers Old number of workers for logging purposes
  /// \param new_workers new number of worker
  /// \return Status code
  Status RequestNumWorkerChange(int32_t op_id, int32_t old_workers, int32_t *num_workers_requested);

  /// Send a ChangeRequest to the operator to update the connector capacity
  /// \param op_id operator ID
  /// \param old_workers Old size for logging purposes
  /// \param new_workers new size
  /// \return Status code
  Status RequestConnectorCapacityChange(int32_t op_id, int32_t old_size, int32_t new_size);

  /// Track the pipeline time of the current epoch into avg_pipeline_times_
  /// \return Status code
  Status TrackPipelineTime();

  /// Utility function to calculate the mean/average of a list of numbers
  /// \tparam T type of the vector
  /// \param items vector of T
  /// \return double the calculated mean
  template <typename T>
  double Mean(const std::vector<T> &items) const;

  /// Get and update current epoch and step counts
  /// \return Status Code
  Status UpdateCurrentRunInfo();

  /// Decide whether warmup period is complete to start AT
  /// \return the decision for skipping further or not
  bool WarmupSkipCheck();

  /// Save current worker and queue size configurations
  /// \return Status code
  Status RegisterWorkersQueue();

  /// Reset values of workers and queue sizes for ops to saved best config
  /// \return  Status code
  Status ResetWorkersQueue();

  /// Compare current and previous metrics for memory performance in memory phase of
  /// AT tuning. Logic can be changed without modification to primary function
  /// \param prev_avg previous comparison value - normally pre-change in pipeline
  /// \param cur_avg current comparison value - normally post-change in pipeline
  /// \return decision on good (True) or bad (False) change in metric
  bool MemoryPhaseCompareMetric(double prev_avg, double cur_avg);

  /// Pointer to the tree adapter to get tree info
  TreeAdapter *tree_adapter_;
  /// Pointer to the profiler manager to get statistics
  ProfilingManager *profiling_manager_;
  /// Unique_ptr to the tree modifier to handle change requests
  std::unique_ptr<TreeModifier> tree_modifier_;

  /// mux to be used to sleep
  std::mutex mux_;
  /// Conditional variable used to sleep
  CondVar cv_;

  /// a map from op_id to a pointer to the operator
  std::map<int32_t, std::shared_ptr<DatasetOp>> ops_;
  /// list of all map_ops
  std::vector<int32_t> parallel_ops_ids_;
  /// ID of the leaf op
  int32_t leaf_op_id_;
  /// vector of pipeline time per epoch
  std::vector<double> avg_pipeline_times_;

  /// the current epoch and step indices (starts from 1)
  int32_t cur_epoch_running_;
  int32_t last_epoch_autotuned_;
  // step based auto-tuning specifics
  int32_t cur_step_running_;
  int64_t last_step_autotuned_;

  int32_t mode_;
  int64_t step_gap_;
  bool skip_flag_;
  int32_t AT_phase_;
  // tracking whether AT makes a change
  bool AT_change_;

  // Phase 1 - Analyse Time
  double phase_1_best_time_;
  int32_t phase_1_no_improve_count_;
  std::vector<int32_t> phase_1_best_workers;
  std::vector<int32_t> phase_1_best_queue;

  // phase 2 - Analyse Memory
  int32_t count_down_;
  int32_t phase_3_state_;
  int32_t phase_3_ID_;
  double avg_batch_time;
  double phase_3_prev_avg_;
  std::vector<int32_t> OP_values;

  /// True if should save AutoTune configuration
  bool save_autoconfig_;

  /// Flag to enable saving of intermediate autotune config to disk
  bool save_intermediate_autoconfig_{false};

  /// Filepath name of the final AutoTune Configuration JSON file
  std::string autotune_json_filepath_;

  /// Serialized json of the optimized ir tree that holds the updated configuration (workers and queue size)
  nlohmann::json autotune_config_json_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_AUTO_TUNE_H_
