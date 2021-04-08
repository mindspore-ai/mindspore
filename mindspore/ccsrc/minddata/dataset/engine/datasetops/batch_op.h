/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class DataBuffer;

using PadInfo = std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>>;

class BatchOp : public ParallelOp {
 public:
  class Builder {
   public:
    // Builder constructor for Batch, batch size needs to be specified
    // @param int32_t batch_size
    explicit Builder(int32_t batch_size);

    // Default destructor
    ~Builder() = default;

    // set number of parallel Workers on batch
    // @param int32_t num_workers
    // @return Builder & reference to builder class object
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
      return *this;
    }

    // set drop for batch op,default false
    // @param bool drop
    // @return Builder & reference to builder class object
    Builder &SetDrop(bool drop) {
      builder_drop_ = drop;
      return *this;
    }

    Builder &SetPaddingMap(const PadInfo &pad_map, bool pad = true) {
      builder_pad_ = pad;
      builder_pad_map_ = pad_map;
      return *this;
    }

    // set connector size for batch
    // @param int32_t op_conn_size
    // @return Builder & reference to builder class object
    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = (op_connector_size == 0 ? builder_op_connector_size_ : op_connector_size);
      return *this;
    }

    /// \param in_col_name
    /// \return Builder & reference to builder class object
    Builder &SetInColNames(const std::vector<std::string> &in_col_name) {
      builder_in_names_ = in_col_name;
      return *this;
    }

    /// \param out_col_name
    /// \return Builder & reference to builder class object
    Builder &SetOutColNames(const std::vector<std::string> &out_col_name) {
      builder_out_names_ = out_col_name;
      return *this;
    }

#ifdef ENABLE_PYTHON
    // set columns to perform map on
    // @param const std::vector<std::string> & cols_to_map - name of columns to perform map on
    // @return Builder & reference to builder class object
    Builder &SetBatchMapFunc(py::function batch_map_func) {
      builder_batch_map_func_ = batch_map_func;
      return *this;
    }

    // SetBatchSizeFunc, a function that calls to python after every batch is made
    // @param py::function batch_size_func - python function to call, GIL required before calling
    // @return Builder & reference to builder class object
    Builder &SetBatchSizeFunc(py::function batch_size_func) {
      builder_batch_size_func_ = batch_size_func;
      return *this;
    }
#endif

    // @param std::shared_ptr<BatchOp>  *ptr pointer to shared_ptr, actual return arg
    // @return Status The status code returned
    Status Build(std::shared_ptr<BatchOp> *);

   private:
    // Sanity check for builder class args
    // @return Status The status code returned
    Status SanityCheck();

    bool builder_drop_;
    bool builder_pad_;
    int32_t builder_batch_size_;
    int32_t builder_num_workers_;
    int32_t builder_op_connector_size_;
    std::vector<std::string> builder_in_names_;
    std::vector<std::string> builder_out_names_;
    PadInfo builder_pad_map_;
#ifdef ENABLE_PYTHON
    py::function builder_batch_size_func_;
    py::function builder_batch_map_func_;
#endif
  };

  enum batchCtrl : int8_t { kNoCtrl = 0, kEOE = 1, kEOF = 2, kQuit = 3 };

  // Parameters associate with one batch.
  // This struct is used for both internal control and python callback.
  // This struct is bound to python with read-only access.
  struct CBatchInfo {
    CBatchInfo(int64_t ep, int64_t bat, int64_t cur, batchCtrl ctrl)
        : epoch_num_(ep), batch_num_(bat), total_batch_num_(cur), ctrl_(ctrl) {}
    CBatchInfo(int64_t ep, int64_t bat, int64_t cur) : CBatchInfo(ep, bat, cur, batchCtrl::kNoCtrl) {}
    CBatchInfo() : CBatchInfo(0, 0, 0, batchCtrl::kNoCtrl) {}
    explicit CBatchInfo(batchCtrl ctrl) : CBatchInfo(0, 0, 0, ctrl) {}
    int64_t epoch_num_;        // i-th epoch. i starts from 0
    int64_t batch_num_;        // i-th batch since the start of current epoch. i starts from 0
    int64_t total_batch_num_;  // i-th batch since the start of first epoch. i starts from 0
    batchCtrl ctrl_;           // No control=0, EOE=1, EOF=2, Quit=3
    const int64_t get_batch_num() const { return batch_num_; }
    const int64_t get_epoch_num() const { return epoch_num_; }
  };

#ifdef ENABLE_PYTHON

  BatchOp(int32_t batch_size, bool drop, bool pad, int32_t op_queue_size, int32_t num_workers,
          const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
          py::function batch_size_func, py::function batch_map_func, PadInfo pad_map);
#else
  BatchOp(int32_t batch_size, bool drop, bool pad, int32_t op_queue_size, int32_t num_workers,
          const std::vector<std::string> &, PadInfo pad_map);
#endif

  // BatchOp destructor
  ~BatchOp() {}

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
  // @param const std::unique_ptr<TensorQTable> *src - table that has the rows for batching
  // @param const std::unique_ptr<TensorQTable> *dest - dest_table to hold batched rows
  // @param int32_t size - batch_size
  // @param const std::unordered_map<std::string, int32_t>& column_name_id_map - column names to index mapping
  // @return Status The status code returned
  static Status BatchRows(const std::unique_ptr<TensorQTable> *src, const std::unique_ptr<TensorQTable> *dest,
                          dsize_t batch_size);

  // @param table
  // @param const PadInfo &pad_info pad info
  // @param const std::unordered_map<std::string, int32_t>& column_name_id_map - column names to index mapping
  // @return Status The status code returned
  static Status PadColumns(std::unique_ptr<TensorQTable> *table, const PadInfo &pad_info,
                           const std::unordered_map<std::string, int32_t> &column_name_id_map);

  int64_t GetTreeBatchSize() override;

 protected:
  Status ComputeColMap() override;

 private:
  // Worker thread for doing the memcpy of batch
  // @param int32_t param workerId
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  // Generate buffer with batched tensors
  // @return Status The status code returned
  Status MakeBatchedBuffer(std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> table_pair,
                           std::unique_ptr<DataBuffer> *db);

#ifdef ENABLE_PYTHON
  // Function that calls pyfunc to perform map on batch
  // @param (std::pair<std::unique_ptr<TensorQTable>, batch_stats> *table_pair - contains un-batched tensor
  // @return Status The status code returned
  Status MapColumns(std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> *table_pair);
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

  // the number of thread pulling from the mOutConnector of the Op below
  // @return int32_t, 1
  int32_t num_consumers() const override { return 1; }

  // get the batch size for next batch
  // @return Status The status code returned
  Status GetBatchSize(int32_t *batch_size, CBatchInfo info);

  // Do the initialization of all queues then start all worker threads
  // @return Status The status code returned
  Status LaunchThreadsAndInitOp();

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRow(TensorRow *const row) override;

#ifdef ENABLE_PYTHON
  // Invoke batch size function with current BatchInfo to generate batch size.
  // @return Status The status code returned
  Status InvokeBatchSizeFunc(int32_t *batch_size, CBatchInfo info);

  // Invoke batch map function with current BatchInfo to generate tensors to batch.
  // @return Status The status code returned
  Status InvokeBatchMapFunc(TensorTable *input, TensorTable *output, CBatchInfo info);
#endif

  int32_t start_batch_size_;
  const bool drop_;                                     // bool for whether to drop remainder or not
  const bool pad_;                                      // bool for whether to perform padding on tensor
  const std::vector<std::string> in_col_names_;         // input column name for per_batch_map
  std::vector<std::string> out_col_names_;              // output column name for per_batch_map
  PadInfo pad_info_;                                    // column names to perform padding on
  std::unique_ptr<ChildIterator> child_iterator_;       // child iterator for fetching TensorRows 1 by 1
  std::unordered_map<std::string, int32_t> child_map_;  // col_name_id_map of the child node
  QueueList<std::pair<std::unique_ptr<TensorQTable>, CBatchInfo>> worker_queues_;  // internal queue for syncing worker
  int64_t batch_num_;
  int64_t batch_cnt_;
#ifdef ENABLE_PYTHON
  py::function batch_size_func_;  // Function pointer of batch size function
  py::function batch_map_func_;   // Function pointer of per batch map function
#endif
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BATCH_OP_H_
