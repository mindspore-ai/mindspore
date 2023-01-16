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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_NONMAPPABLE_LEAF_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_NONMAPPABLE_LEAF_OP_H_

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <map>

#include "minddata/dataset/util/wait_post.h"
#include "minddata/dataset/util/auto_index.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"

namespace mindspore {
namespace dataset {
template <typename T>
class Queue;

template <class T>
class Connector;

class JaggedConnector;
class FilenameBlock;

using StringIndex = AutoIndexObj<std::string>;

class NonMappableLeafOp : public ParallelOp<std::unique_ptr<IOBlock>, TensorRow> {
 public:
  // NONE: No compression_type is used
  // GZIP: GZIP compression_type with num_samples provided
  // ZLIB: ZLIB compression_type with num_samples provided
  // GZIP_WITH_COUNT: GZIP compression_type with num_samples not provided
  // ZLIB_WITH_COUNT: ZLIB compression_type with num_samples not provided
  enum class CompressionType { NONE = 0, GZIP = 1, ZLIB = 2, GZIP_WITH_COUNT = 3, ZLIB_WITH_COUNT = 4 };

  // Constructor of TFReaderOp (2)
  // @note The builder class should be used to call this constructor.
  // @param num_workers - number of worker threads reading data from tf_file files.
  // @param worker_connector_size - size of each internal queue.
  // @param total_num_rows - Number of rows to read
  // @param dataset_files_list - list of filepaths for the dataset files.
  // @param op_connector_size - size of each queue in the connector that the child operator pulls from.
  // @param columns_to_load - the names of the columns to load data from.
  // @param shuffle_files - whether or not to shuffle the files before reading data.
  // @param equal_rows_per_shard - whether or not to get equal rows for each process.
  // @param compression_type - the compression type of the tf_file files
  NonMappableLeafOp(int32_t num_workers, int32_t worker_connector_size, int64_t total_num_rows,
                    int32_t op_connector_size, bool shuffle_files, int32_t num_devices, int32_t device_id,
                    const CompressionType &compression_type = CompressionType::NONE);

  // Default destructor
  ~NonMappableLeafOp() = default;

  // Instantiates the internal queues and connectors.
  // @return Status - the error code returned.
  virtual Status Init() = 0;

  // Class functor operator () override.
  // All dataset operators operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status - the error code returned.
  Status operator()() override;

  // Overrides base class reset method. Cleans up any state info from it's previous execution and
  // reinitializes itself so that it can be executed again, as if it was just created.
  // @return Status - the error code returned.
  Status Reset() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "NonMappableLeafOp"; }

  // \brief During tree prepare phase, operators may have specific post-operations to perform depending on
  //     their role.
  // \notes Derived versions of this function should always call their superclass version first
  //     before providing their own implementations.
  // @return Status The status code returned
  Status PrepareOperator() override;

 protected:
  // The entry point for when workers are launched.
  // @param worker_id - the id of the worker that is executing this function.
  // @return Status - the error code returned.
  Status WorkerEntry(int32_t worker_id) override;

  // Pushes a control indicator onto the IOBlockQueue for each worker to consume.
  // When the worker pops this control indicator, it will shut itself down gracefully.
  // @return Status - the error code returned.
  Status PostEndOfData();

  // Pushes a control indicator onto the IOBlockQueue for each worker to consume. When the worker
  // pops this control indicator, it will wait until the next epoch starts and then resume execution.
  // @return Status - the error code returned.
  Status PostEndOfEpoch(int32_t queue_index);

  // Called asynchronously by another thread. Will wait until notified to fill the IOBlockQueue.
  // @return Status - the error code returned.
  Status WaitToFillIOBlockQueue();

  // Notifies the thread which called WaitToFillIOBlockQueue to resume execution.
  void NotifyToFillIOBlockQueue();

  // Pops an element from a queue in IOBlockQueue.
  // @param index - the index of the queue to pop from.
  // @param out_block - the popped element.
  // @return Status - the error code returned.
  Status PopIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> *out_block);

  // Pushes an element to a queue in IOBlockQueue.
  // @param index - the index of the queue to push to.
  // @param io_block - the element to push onto the queue.
  // @return Status - the error code returned.
  Status PushIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> &&io_block);

  // Reads a tf_file file and loads the data into multiple TensorRows.
  // @param filename - the tf_file file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @return Status - the error code returned.
  virtual Status LoadFile(const std::string &filename, int64_t start_offset, int64_t end_offset, int32_t worker_id) = 0;

  // Select file and push it to the block queue.
  // @param file_name - File name.
  // @param start_file - If file contains the first sample of data.
  // @param end_file - If file contains the end sample of data.
  // @param pre_count - Total rows of previous files.
  // @return Status - the error code returned.
  bool NeedPushFileToBlockQueue(const std::string &file_name, int64_t *start_offset, int64_t *end_offset,
                                const int64_t &pre_count);

  // Calculate number of rows in each shard.
  // @return Status - the error code returned.
  virtual Status CalculateNumRowsPerShard() = 0;

  void ShuffleKeys();

  // Fill the IOBlockQueue.
  // @para i_keys - keys of file to fill to the IOBlockQueue
  // @return Status - the error code returned.
  virtual Status FillIOBlockQueue(const std::vector<int64_t> &i_keys) = 0;

  virtual bool GetLoadIoBlockQueue() {
    bool ret_load_io_block_queue = false;
    {
      std::unique_lock<std::mutex> lock(load_io_block_queue_mutex_);
      ret_load_io_block_queue = load_io_block_queue_;
    }
    return ret_load_io_block_queue;
  }

  virtual bool GetLoadJaggedConnector() {
    bool ret_load_jagged_connector = false;
    {
      std::unique_lock<std::mutex> lock(load_jagged_connector_mutex_);
      ret_load_jagged_connector = load_jagged_connector_;
    }
    return ret_load_jagged_connector;
  }

  int32_t device_id_;
  int32_t num_devices_;
  bool load_jagged_connector_;
  std::mutex load_jagged_connector_mutex_;
  std::unique_ptr<StringIndex> filename_index_;

  QueueList<std::unique_ptr<FilenameBlock>> io_block_queues_;
  std::map<std::string, int64_t> filename_numrows_;
  bool finished_reading_dataset_;
  // Note: If compression_type_ is not empty, then total_rows_ is the total rows that will be read per shard
  int64_t total_rows_;
  CompressionType compression_type_;

  WaitPost io_block_queue_wait_post_;
  bool load_io_block_queue_;
  std::mutex load_io_block_queue_mutex_;
  std::unique_ptr<JaggedConnector> jagged_rows_connector_;
  bool shuffle_files_;
  int64_t num_rows_per_shard_;
  int64_t num_rows_;

 private:
  std::vector<int64_t> shuffled_keys_;  // to store shuffled filename indices
  uint32_t seed_;                       // used to shuffle filename indices
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_NONMAPPABLE_LEAF_OP_H_
