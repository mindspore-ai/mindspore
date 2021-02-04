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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MINDRECORD_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MINDRECORD_OP_H_
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"
#include "minddata/mindrecord/include/shard_column.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_reader.h"
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;
class DataBuffer;

using mindrecord::ShardOperator;
using mindrecord::ShardReader;
using ShardTuple = std::vector<std::tuple<std::vector<uint8_t>, mindrecord::json>>;  // Row of data from ShardReader

const int32_t LOG_INTERVAL = 19;

class MindRecordOp : public ParallelOp {
 public:
  // The nested builder class inside of the MindRecordOp is used to help manage all of the arguments
  // for constructing it.  Use the builder by setting each argument with the provided set methods,
  // and then finally call the build method to execute the actual construction.
  class Builder {
   public:
    Builder();

    ~Builder() = default;

    Status Build(std::shared_ptr<MindRecordOp> *);

    Builder &SetRowsPerBuffer(int rows_per_buffer) {
      build_rows_per_buffer_ = rows_per_buffer;
      return *this;
    }

    Builder &SetNumMindRecordWorkers(int32_t num_mind_record_workers) {
      build_num_mind_record_workers_ = num_mind_record_workers;
      return *this;
    }

    Builder &SetOpConnectorQueueSize(int32_t queue_size) {
      build_op_connector_queue_size_ = queue_size;
      return *this;
    }

    Builder &SetDatasetFile(const std::vector<std::string> &files) {
      build_dataset_file_ = files;
      return *this;
    }

    Builder &SetColumnsToLoad(const std::vector<std::string> &columns) {
      build_columns_to_load_ = columns;
      return *this;
    }

    Builder &SetOperators(const std::vector<std::shared_ptr<ShardOperator>> &operators) {
      build_operators_ = operators;
      return *this;
    }

    Builder &SetLoadDataset(bool load_dataset) {
      build_load_dataset_ = load_dataset;
      return *this;
    }

    Builder &SetNumToPadSamples(int64_t num_padded) {
      build_num_padded_ = num_padded;
      return *this;
    }

    Builder &SetPaddedSample(const py::handle &sample) {
      build_sample_ = sample;
      return *this;
    }

    Status SanityCheck() const;

    static int32_t num_mind_record_workers() { return kDefaultMindRecordWorkers; }

    mindrecord::json ToJson(const py::handle &obj);

   private:
    static constexpr int32_t kDefaultMindRecordWorkers = 4;
    // The builder saves all MindRecordOp construction arguments internally.
    // The following are the arguments.
    int32_t build_num_mind_record_workers_;
    int32_t builder_num_workers_;
    int32_t build_rows_per_buffer_;
    int32_t build_op_connector_queue_size_;
    std::vector<std::string> build_dataset_file_;
    bool build_load_dataset_;
    std::vector<std::string> build_columns_to_load_;
    std::vector<std::shared_ptr<ShardOperator>> build_operators_;
    int64_t build_num_padded_;
    py::handle build_sample_;
    std::map<std::string, std::string> build_sample_bytes_;
  };

  // Constructor of the MindRecordOp.
  // @note The builder class should be used to call it
  // @param num_mind_record_workers - The number of workers for the op (run by ShardReader)
  // @param rows_per_buffer - The requested number of rows per buffer
  // @param dataset_file - dataset files
  // @param op_connector_queue_size - The output connector queue size
  // @param columns_to_load - The list of columns to use (column name)
  // @param operators - ShardOperators for Shuffle, Category, Sample
  MindRecordOp(int32_t num_mind_record_workers, int32_t rows_per_buffer, std::vector<std::string> dataset_file,
               bool load_dataset, int32_t op_connector_queue_size, const std::vector<std::string> &columns_to_load,
               const std::vector<std::shared_ptr<ShardOperator>> &operators, int64_t num_padded_,
               const mindrecord::json &sample_json, const std::map<std::string, std::string> &sample_bytes_);

  // Destructor
  ~MindRecordOp() override;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param op - reference to the MindRecordOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const MindRecordOp &op) {
    op.Print(out, false);
    return out;
  }

  // Worker thread pulls a number of IOBlock from IOBlock Queue, make a buffer and push it to Connector
  // @param int32_t workerId - id of each worker
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;

  // Class functor operator () override.
  // All DatasetOps operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work.
  // @return Status The status code returned
  Status operator()() override;

  // Called first when function is called
  // @return
  Status LaunchThreadAndInitOp();

  // Overrides base class reset method.  When an operator does a reset, it cleans up any state
  // info from it's previous execution and then initializes itself so that it can be executed
  // again.
  // @return Status The status code returned
  Status Reset() override;

  // Getter method
  int32_t num_rows() const { return num_rows_; }

  static Status CountTotalRows(const std::vector<std::string> dataset_path, bool load_dataset,
                               const std::shared_ptr<ShardOperator> &op, int64_t *count, int64_t num_padded);

  // Getter method
  int32_t rows_per_buffer() const { return rows_per_buffer_; }

  // Getter method
  std::vector<std::string> dataset_file() const { return dataset_file_; }

  // Getter method
  std::vector<std::string> columns_to_load() const { return columns_to_load_; }

  bool load_dataset() const { return load_dataset_; }

  Status Init();

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "MindRecordOp"; }

 private:
  Status GetBufferFromReader(std::unique_ptr<DataBuffer> *fetched_buffer, int64_t buffer_id, int32_t worker_id);

  // Parses a single cell and puts the data into a tensor
  // @param tensor_row - the tensor row to put the parsed data in
  // @param columns_blob - the blob data received from the reader
  // @param columns_json - the data for fields received from the reader
  Status LoadTensorRow(TensorRow *tensor_row, const std::vector<uint8_t> &columns_blob,
                       const mindrecord::json &columns_json, const mindrecord::TaskType task_type);

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  int32_t rows_per_buffer_;                                // The number of requested rows per buffer.
  std::vector<std::string> dataset_file_;                  // dataset files
  bool load_dataset_;                                      // load dataset from single file or not
  std::vector<std::string> columns_to_load_;               // Columns to load from dataset
  std::vector<std::shared_ptr<ShardOperator>> operators_;  // ShardOperators to use
  int32_t num_mind_record_workers_;                        // number of workers to be spawned by ShardReader
  int32_t buffers_needed_;                                 // Counter for the buffers that were fetched
  int64_t buf_cnt_;                                        // Buffer counter
  int32_t num_rows_;                                       // One more than the last row id in the range for this cache
  std::atomic<int32_t> ended_worker_;

  int64_t num_padded_;
  mindrecord::json sample_json_;
  std::map<std::string, std::string> sample_bytes_;

  std::unique_ptr<DataSchema> data_schema_;  // Data schema for column typing
  std::vector<std::string> columns_blob_;    // Blob Columns to load from dataset
  std::vector<int32_t> columns_blob_index_;  // Blob Columns to load from dataset

  std::unique_ptr<ShardReader> shard_reader_;

  std::mutex ended_worker_mutex_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MINDRECORD_OP_H_
