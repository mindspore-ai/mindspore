/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BATCH_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BATCH_OP_H_

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/api/python/python_mp.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/shared_memory_queue.h"
#include "minddata/dataset/core/message_queue.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/batch_info.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"

namespace mindspore {
namespace dataset {

using PadInfo = std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>>;

class BatchOp : public ParallelOp<std::pair<std::unique_ptr<TensorQTable>, CBatchInfo>, TensorRow> {
 public:
#ifdef ENABLE_PYTHON
  BatchOp(int32_t batch_size, bool drop, bool pad, int32_t op_queue_size, int32_t num_workers,
          const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
          const py::function &batch_size_func, const py::function &batch_map_func, PadInfo pad_map);
#endif

  BatchOp(int32_t batch_size, bool drop, bool pad, int32_t op_queue_size, int32_t num_workers, std::vector<std::string>,
          PadInfo pad_map);

  // BatchOp destructor
  ~BatchOp() override;

  // @param int32_t workerId
  // @return Status The status code returned
  Status EofReceived(int32_t) override;

  // @param int32_t workerId
  // @return Status The status code returned
  Status EoeReceived(int32_t) override;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param sO - reference to the BatchOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const BatchOp &bo) {
    bo.Print(out, false);
    return out;
  }

  // Main loop of batch
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kBatchOp; }

  // batch the rows in src table then put it to dest table
  // @param const std::unique_ptr<TensorQTable> *tensor_row_dequeue - table that has the rows for batching
  // @param TensorRow *batched_tensor_row - dest_table to hold batched rows
  // @param bool concat_batch - whether to keep batch to 1 row or expand dimensions
  // @param bool contains_per_batch_map - whether user has provided per_batch_map
  // @notes contains_per_batch_map is passed to this function since some callers require this function to be static
  // @return Status The status code returned
  static Status BatchRows(const std::unique_ptr<TensorQTable> *tensor_row_dequeue, TensorRow *batched_tensor_row,
                          bool concat_batch = false, bool contains_per_batch_map = false);

  // convert the rows to tensor
  // @param const std::unique_ptr<TensorQTable> *tensor_row_dequeue - table that has the rows for batching
  // @param std::shared_ptr<Tensor> *batched_tensor - dest_table to hold batched rows
  // @param int32_t size - batch_size
  // @param int32_t size - column_index
  // @param bool contains_per_batch_map - whether user has provided per_batch_map
  // @notes contains_per_batch_map is passed to this function since some callers require this function to be static
  // @return Status The status code returned
  static Status ConvertRowsToTensor(const std::unique_ptr<TensorQTable> *tensor_row_dequeue,
                                    std::shared_ptr<Tensor> *batched_tensor, dsize_t batch_size, size_t column_index,
                                    bool contains_per_batch_map);

  // @param table
  // @param const PadInfo &pad_info pad info
  // @param const std::unordered_map<std::string, int32_t>& column_name_id_map - column names to index mapping
  // @return Status The status code returned
  static Status PadColumns(const std::unique_ptr<TensorQTable> *table, const PadInfo &pad_info,
                           const std::unordered_map<std::string, int32_t> &column_name_id_map);

  int64_t GetTreeBatchSize() override;

  bool IsPython() const override {
#ifdef ENABLE_PYTHON
    if (batch_map_func_ || batch_size_func_) {
      return true;
    }
#endif
    return false;
  }

  /// Set the instance of Python multiprocessing which will passed from Python
  /// \param python_multiprocessing_runtime PythonMultiprocessingRuntime
  void SetPythonMp(std::shared_ptr<PythonMultiprocessingRuntime> python_multiprocessing_runtime);

#if !defined(_WIN32) && !defined(_WIN64)
  /// Return the list of PIDs of worker processes
  /// \return vector of int
  std::vector<int32_t> GetMPWorkerPIDs() const override;

  // @Used by independent dataset mode to stop the subprocess
  // @return Status The status code returned
  Status Terminate() override;
#endif

 private:
#if !defined(_WIN32) && !defined(_WIN64)
  void UpdateMonitorResource(int32_t num_workers);

  // Execute the per_batch_map with python multiprocessing worker
  Status ComputeWithWorker(TensorTable *input, TensorTable *output, CBatchInfo info, bool *concat_batch,
                           int32_t worker_id);
#endif

  // Execute the per_batch_map with C++ thread
  Status ComputeWithThread(TensorTable *input, TensorTable *output, CBatchInfo info, bool *concat_batch,
                           int32_t worker_id);

  // Worker thread for doing the memcpy of batch
  // @param int32_t param workerId
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  // Generate row with batched tensors
  // @return Status The status code returned
  Status MakeBatchedRow(std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> tensor_info_pair,
                        TensorRow *batched_tensor_row, int32_t worker_id = 0);

#ifdef ENABLE_PYTHON
  // Function that calls pyfunc to perform map on batch
  // @param (std::pair<std::unique_ptr<TensorQTable>, batch_stats> *table_pair - contains un-batched tensor
  // @return Status The status code returned
  Status MapColumns(std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> *table_pair, bool *concat_batch,
                    int32_t worker_id);
#endif

  // @param const PadInfo &pad_info pad info to unpack
  // @param const std::unordered_map<std::string, int32_t>& column_name_id_map - column names to index mapping
  // @param std::set<int32_t> *cols, col ids to perform pad on
  // @param std::vector<float> *vals, default padding value for each column
  // @param std::vector<std::vector<dsize_t>> *shapes, padding shape specified by user
  // @return Status The status code returned
  static Status UnpackPadInfo(const PadInfo &pad_info,
                              const std::unordered_map<std::string, int32_t> &column_name_id_map,
                              std::set<int32_t> *pad_cols, std::vector<std::shared_ptr<Tensor>> *pad_vals,
                              std::vector<std::vector<dsize_t>> *pad_shapes);

  // get the batch size for next batch
  // @return Status The status code returned
  Status GetBatchSize(int32_t *batch_size, CBatchInfo info);

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

  Status SendWaitFlagToWorker(int32_t worker_id) override;

  Status SendQuitFlagToWorker(int32_t worker_id) override;

  Status ComputeColMap() override;

#ifdef ENABLE_PYTHON
  // Invoke batch size function with current BatchInfo to generate batch size.
  // @return Status The status code returned
  Status InvokeBatchSizeFunc(int32_t *batch_size, CBatchInfo info);

  // Invoke batch map function with current BatchInfo to generate tensors to batch.
  // @return Status The status code returned
  Status InvokeBatchMapFunc(TensorTable *input, TensorTable *output, CBatchInfo info, bool *concat_batch,
                            int32_t worker_id);
#endif

  void UpdateCounterAndSendEOE(TensorRow *const row);

  int32_t start_batch_size_;
  const bool drop_;                                     // bool for whether to drop remainder or not
  const bool pad_;                                      // bool for whether to perform padding on tensor
  std::vector<std::string> in_col_names_;               // input column name for per_batch_map
  std::vector<std::string> out_col_names_;              // output column name for per_batch_map
  PadInfo pad_info_;                                    // column names to perform padding on
  std::unique_ptr<ChildIterator> child_iterator_;       // child iterator for fetching TensorRows 1 by 1
  std::unordered_map<std::string, int32_t> child_map_;  // col_name_id_map of the child node
  int64_t batch_num_;
  int64_t batch_cnt_;
#ifdef ENABLE_PYTHON
  py::function batch_size_func_;  // Function pointer of batch size function
  py::function batch_map_func_;   // Function pointer of per batch map function
#endif
  std::shared_ptr<PythonMultiprocessingRuntime> python_multiprocessing_runtime_;  // python multiprocessing instance
#if !defined(_WIN32) && !defined(_WIN64)
  std::vector<key_t> ftok_keys_;  // used to create msg queue and shm queue for per_batch_map in process mode
  std::vector<std::shared_ptr<MessageQueue>> msg_queues_;
  std::vector<std::shared_ptr<SharedMemoryQueue>> shm_queues_;
  std::vector<int32_t> worker_pids_;

  // monitor the per_batch_map in process mode
  std::vector<std::unique_ptr<std::mutex>> monitor_mtx_;
  std::vector<std::unique_ptr<std::condition_variable>> monitor_cv_;
  std::vector<std::unique_ptr<bool>> monitor_exit_flag_;
#endif

 protected:
#if !defined(_WIN32) && !defined(_WIN64)
  Status Launch() override;

  Status AddNewWorkers(int32_t num_new_workers) override;
  Status RemoveWorkers(int32_t num_workers) override;
#endif

  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  bool eoe_received_ = false;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BATCH_OP_H_
