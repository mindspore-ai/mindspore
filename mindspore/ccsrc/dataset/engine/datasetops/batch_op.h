/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_ENGINE_DATASETOPS_BATCH_OP_H_
#define DATASET_ENGINE_DATASETOPS_BATCH_OP_H_

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dataset/core/config_manager.h"
#include "dataset/core/tensor.h"
#include "dataset/engine/dataset_iterator.h"
#include "dataset/engine/datasetops/parallel_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class DataBuffer;

using TensorBatch = std::vector<std::shared_ptr<Tensor>>;
using TensorBatchTable = std::vector<TensorBatch>;

class BatchOp : public ParallelOp {
 public:
  class Builder {
   public:
    // Builder constructor for Batch, batch size needs to be specified
    // @param int32_t batch_size
    explicit Builder(int32_t batch_size);

    // Builder constructor for Batch, batch size function needs to be specified
    // @param py::function batch_size_func
    explicit Builder(py::function batch_size_func);

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

    // set connector size for batch
    // @param int32_t op_conn_size
    // @return Builder & reference to builder class object
    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = (op_connector_size == 0 ? builder_op_connector_size_ : op_connector_size);
      return *this;
    }

    // set columns to perform map on
    // @param const std::vector<std::string> & cols_to_map - name of columns to perform map on
    // @return Builder & reference to builder class object
    Builder &SetColumnsToMap(const std::vector<std::string> &cols_to_map) {
      builder_cols_to_map_ = cols_to_map;
      return *this;
    }

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

    // @param std::shared_ptr<BatchOp>  *ptr pointer to shared_ptr, actual return arg
    // @return Status - The error code return
    Status Build(std::shared_ptr<BatchOp> *);

   private:
    // Sanity check for builder class args
    // @return Status - The error code return
    Status SanityCheck();

    bool builder_drop_;
    int32_t builder_batch_size_;
    int32_t builder_num_workers_;
    int32_t builder_op_connector_size_;
    std::vector<std::string> builder_cols_to_map_;

    py::function builder_batch_size_func_;
    py::function builder_batch_map_func_;
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

  // BatchOp constructor
  // @param int32_t batch_size
  // @param bool drop
  // @param int32_t op_queue_size
  // @param int32_t rows_per_buf
  // @param int32_t num_workers
  BatchOp(int32_t batch_size, bool drop, int32_t op_queue_size, int32_t num_workers, const std::vector<std::string> &,
          py::function batch_size_func, py::function batch_map_func);

  // BatchOp destructor
  ~BatchOp() {}

  // @param int32_t workerId
  // @return Status - The error code return
  Status EofReceived(int32_t) override;

  // @param int32_t workerId
  // @return Status - The error code return
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
  // @return Status - The error code return
  Status operator()() override;

 private:
  // Worker thread for doing the memcpy of batch
  // @param int32_t param workerId
  // @return Status - The error code return
  Status WorkerEntry(int32_t worker_id) override;

  // Generate buffer with batched tensors
  // @return Status - The error code return
  Status MakeBatchedBuffer(std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> table_pair,
                           std::unique_ptr<DataBuffer> *db);

  // batch the rows in src table then put it to dest table
  // @param const std::unique_ptr<TensorQTable> *src - table that has the rows for batching
  // @param const std::unique_ptr<TensorQTable> *dest - dest_table to hold batched rows
  // @param int32_t size - batch_size
  // @return Status - The error code return
  Status BatchRows(const std::unique_ptr<TensorQTable> *src, const std::unique_ptr<TensorQTable> *dest, size_t size);

  // Function that calls pyfunc to perform map on batch
  // @param (std::pair<std::unique_ptr<TensorQTable>, batch_stats> *table_pair - contains un-batched tensor
  // @return Status - The error code return
  Status MapColumns(std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> *table_pair);

  // the number of thread pulling from the mOutConnector of the Op below
  // @return int32_t, 1
  int32_t num_consumers() const override { return 1; }

  // get the batch size for next batch
  // @return Status - The error code return
  Status GetBatchSize(int32_t *batch_size, CBatchInfo info);

  // Do the initialization of all queues then start all worker threads
  // @return Status - The error code return
  Status LaunchThreadsAndInitOp();

  // Invoke batch size function with current BatchInfo to generate batch size.
  // @return Status - The error code return
  Status InvokeBatchSizeFunc(int32_t *batch_size, CBatchInfo info);

  // Invoke batch map function with current BatchInfo to generate tensors to batch.
  // @return Status - The error code return
  Status InvokeBatchMapFunc(TensorTable *input, TensorTable *output, CBatchInfo info);

  int32_t start_batch_size_;
  bool drop_;
  // Name of the columns to perform map op on
  std::vector<std::string> input_column_names_;
  // Iterator for fetching
  std::unique_ptr<ChildIterator> child_iterator_;
  // Map of column_name: column_index
  std::unordered_map<std::string, int32_t> column_name_map_;
  // Internal queue for task distribution
  QueueList<std::pair<std::unique_ptr<TensorQTable>, CBatchInfo>> worker_queues_;
  // Function pointer of batch size function
  py::function batch_size_func_;
  // Function pointer of per batch map function
  py::function batch_map_func_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_BATCH_OP_H_
