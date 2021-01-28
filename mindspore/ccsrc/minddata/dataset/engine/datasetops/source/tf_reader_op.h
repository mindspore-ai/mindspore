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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_TF_READER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_TF_READER_OP_H_

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <utility>
#include <map>

#include "minddata/dataset/util/wait_post.h"
#include "minddata/dataset/util/auto_index.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"

namespace dataengine {
class Example;
class Feature;
class BytesList;
}  // namespace dataengine

namespace mindspore {
namespace dataset {
template <typename T>
class Queue;

template <class T>
class Connector;

class JaggedConnector;
class FilenameBlock;

using StringIndex = AutoIndexObj<std::string>;

class TFReaderOp : public ParallelOp {
 public:
  class Builder {
   public:
    // Builder constructor. Creates the builder object.
    // @note No default args
    // @return This is a constructor.
    Builder();

    // Default destructor
    ~Builder() = default;

    // Checks if the inputs of the builder is valid.
    // @return Status - the error code returned.
    Status ValidateInputs() const;

    Status Build(std::shared_ptr<TFReaderOp> *out_tf_reader_op);

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetDataSchema(std::unique_ptr<DataSchema> data_schema) {
      builder_data_schema_ = std::move(data_schema);
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetWorkerConnectorSize(int32_t size) {
      builder_worker_connector_size_ = size;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = op_connector_size;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetRowsPerBuffer(int64_t rows_per_buffer) {
      builder_rows_per_buffer_ = rows_per_buffer;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetNumDevices(int64_t num_dev) {
      builder_num_devices_ = num_dev;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetDeviceId(int64_t dev_id) {
      builder_device_id_ = dev_id;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &setTotalRows(int64_t total_rows) {
      builder_total_rows_ = total_rows;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetDatasetFilesList(const std::vector<std::string> &dataset_files_list) {
      builder_dataset_files_list_ = dataset_files_list;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetColumnsToLoad(const std::vector<std::string> &columns_to_load) {
      builder_columns_to_load_ = columns_to_load;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetShuffleFiles(bool shuffle_files) {
      builder_shuffle_files_ = shuffle_files;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetShardEqualRows(bool shard_equal_rows) {
      builder_equal_rows_per_shard_ = shard_equal_rows;
      return *this;
    }

   private:
    std::unique_ptr<DataSchema> builder_data_schema_;
    int32_t builder_device_id_;
    int32_t builder_num_devices_;
    int32_t builder_num_workers_;
    int32_t builder_worker_connector_size_;
    int32_t builder_op_connector_size_;
    int64_t builder_rows_per_buffer_;
    int64_t builder_total_rows_;
    std::vector<std::string> builder_dataset_files_list_;
    std::vector<std::string> builder_columns_to_load_;
    bool builder_shuffle_files_;
    bool builder_equal_rows_per_shard_;
  };

  // Constructor of TFReaderOp (2)
  // @note The builder class should be used to call this constructor.
  // @param num_workers - number of worker threads reading data from tf_file files.
  // @param worker_connector_size - size of each internal queue.
  // @param rows_per_buffer - number of rows that a full buffer will contain.
  // @param total_num_rows - Number of rows to read
  // @param dataset_files_list - list of filepaths for the dataset files.
  // @param data_schema - the data schema object.
  // @param op_connector_size - size of each queue in the connector that the child operator pulls from.
  // @param columns_to_load - the names of the columns to load data from.
  // @param shuffle_files - whether or not to shuffle the files before reading data.
  // @param equal_rows_per_shard - whether or not to get equal rows for each process.
  TFReaderOp(int32_t num_workers, int32_t worker_connector_size, int64_t rows_per_buffer, int64_t total_num_rows,
             std::vector<std::string> dataset_files_list, std::unique_ptr<DataSchema> data_schema,
             int32_t op_connector_size, std::vector<std::string> columns_to_load, bool shuffle_files,
             int32_t num_devices, int32_t device_id, bool equal_rows_per_shard);

  // Default destructor
  ~TFReaderOp() = default;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // Instantiates the internal queues and connectors.
  // @return Status - the error code returned.
  Status Init();

  // Class functor operator () override.
  // All dataset operators operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status - the error code returned.
  Status operator()() override;

  // Overrides base class reset method. Cleans up any state info from it's previous execution and
  // reinitializes itself so that it can be executed again, as if it was just created.
  // @return Status - the error code returned.
  Status Reset() override;

  // Getter method
  int64_t rows_per_buffer() const { return rows_per_buffer_; }

  // Reads all the provided tf_file files and counts the total number of rows. filenames will
  // first be sectioned into equal parts, then sections are read in parallel. If threads is
  // greater than the number of files, threads will be clamped to the number of files.
  // @param out_total_tows - output parameter which contains the total number of rows
  // @param filenames - a list of tf_file filenames.
  // @param threads - number of threads to use to read the tf_file files.
  // @param estimate - estimate mode, under this mode each threads will sample a single file from each chunk
  // @return Status - the error code returned.
  static Status CountTotalRows(int64_t *out_total_rows, const std::vector<std::string> &filenames, int64_t threads = 1,
                               bool estimate = false);

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "TFReaderOp"; }

  // File names getter
  // @return Vector of the input file names
  std::vector<std::string> FileNames() { return dataset_files_list_; }

  static bool ValidateFirstRowCrc(const std::string &filename);

 private:
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

  // Reads a tf_file file and loads the data into multiple buffers.
  // @param filename - the tf_file file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @return Status - the error code returned.
  Status LoadFile(const std::string &filename, const int64_t start_offset, const int64_t end_offset,
                  const int32_t &worker_id);

  // Parses a single row and puts the data into a tensor table.
  // @param tf_file - the row to be parsed.
  // @param tensor_table - the tensor table to put the parsed data in.
  // @param row - the id of the row filled in the tensor table.
  // @return Status - the error code returned.
  Status LoadExample(const dataengine::Example *tf_file, std::unique_ptr<TensorQTable> *tensor_table, int64_t row);

  // Parses a single cell and puts the data into a tensor table.
  // @param tensor_table - the tensor table to put the parsed data in.
  // @param column_values_list - the cell to parse.
  // @param current_col - the column descriptor containing the expected shape and type of the data.
  // @return Status - the error code returned.
  Status LoadFeature(const std::unique_ptr<TensorQTable> *tensor_table, const dataengine::Feature &column_values_list,
                     const ColDescriptor &current_col, int64_t row, int32_t col);

  // Reads values from a bytes list
  // @param current_col - the column descriptor containing the expected shape and type of the data.
  // @param column_values_list - the cell that contains the bytes list to read from.
  // @param elementStr - the string we read the value into.
  // @return Status - the error code returned.
  static Status LoadBytesList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                              int32_t *num_elements, std::shared_ptr<Tensor> *tensor);

  // Reads values from a float list
  // @param current_col - the column descriptor containing the expected shape and type of the data.
  // @param column_values_list - the cell that contains the float list to read from.
  // @Param numElements - number of values in the float list.
  // @param float_array - the array we read the values into.
  // @return Status - the error code returned.
  Status LoadFloatList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                       int32_t *num_elements, std::unique_ptr<float[]> *float_array);

  // Reads values from a bytes list and casts the value to type T, must be an integral
  // type compatible with int64_t
  // @param current_col - the column descriptor containing the expected shape and type of the data.
  // @param column_values_list - the cell that contains the int list to read from.
  // @Param num_elements - number of values in the int list.
  // @param tensor - the tensor we read the values into.
  // @return Status - the error code returned.
  template <typename T>
  Status LoadIntList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                     int32_t *num_elements, std::shared_ptr<Tensor> *tensor);

  // Determines which template type to use and calls LoadIntList
  // @param current_col - the column descriptor containing the expected shape and type of the data.
  // @param column_values_list - the cell that contains the int list to read from.
  // @Param numElements - number of values in the int list.
  // @param tensor - the tensor we read the values into.
  // @return Status - the error code returned.
  Status LoadIntListSwitch(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                           int32_t *num_elements, std::shared_ptr<Tensor> *tensor);

  // Reads one row of data from a tf file and creates a schema based on that row
  // @return Status - the error code returned.
  Status CreateSchema(const std::string tf_file, std::vector<std::string> columns_to_load);

  // Meant to be called async. Will read files in the range [begin, end) and return the total rows
  // @param filenames - a list of tf data filenames.
  // @param begin - index of first file to read.
  // @param end - one greater than the index of the last file to read.
  // @return int63_t - the total number of rows of files read.
  static int64_t CountTotalRowsSectioned(const std::vector<std::string> &filenames, const int64_t begin,
                                         const int64_t end);
  // Fill IO block queue if shuffle is true
  // @param i_keys - shuffle keys.
  // @return Status - the error code returned.
  Status FillIOBlockShuffle(const std::vector<int64_t> &i_keys);

  /**
   * Fill IO block queue if shuffle is false
   * @param i_keys - shuffle keys.
   * @return Status - the error code returned.
   */
  Status FillIOBlockNoShuffle();

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
  Status CalculateNumRowsPerShard();

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  int32_t device_id_;
  int32_t num_devices_;
  int64_t rows_per_buffer_;
  int64_t total_rows_;
  std::vector<std::string> dataset_files_list_;
  std::vector<std::string> columns_to_load_;
  bool finished_reading_dataset_;
  bool shuffle_files_;
  std::unique_ptr<DataSchema> data_schema_;
  std::unique_ptr<StringIndex> filename_index_;
  bool load_io_block_queue_;
  bool load_jagged_connector_;

  std::unique_ptr<JaggedConnector> jagged_buffer_connector_;
  QueueList<std::unique_ptr<FilenameBlock>> io_block_queues_;
  WaitPost io_block_queue_wait_post_;
  std::mutex load_io_block_queue_mutex_;
  std::map<std::string, int64_t> filename_numrows_;
  int64_t num_rows_;
  int64_t num_rows_per_shard_;
  bool equal_rows_per_shard_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_TF_READER_OP_H_
