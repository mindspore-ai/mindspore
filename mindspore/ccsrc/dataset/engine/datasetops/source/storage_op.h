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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_STORAGE_OP_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_STORAGE_OP_H_

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "dataset/engine/data_schema.h"
#include "dataset/engine/datasetops/parallel_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

// A type for a container of DataBuffer shared_ptr's
using DataBuffers = std::vector<std::unique_ptr<DataBuffer>>;

// A type for the queue of buffer id's for workers to fetch.
using ActionQueue = std::vector<std::unique_ptr<Queue<int32_t>>>;

// Forward declare
class DataBuffer;

class StorageClient;

class StorageOp : public ParallelOp {
 public:
  // The nested builder class inside of the StorageOp is used to help manage all of the arguments
  // for constructing it.  Use the builder by setting each argument with the provided set methods,
  // and then finally call the build method to execute the actual construction.
  class Builder {
   public:
    // Builder constructor.  Creates the builder object.
    // @note No default args
    // @return This is a constructor.
    Builder();

    // Default destructor
    ~Builder() = default;

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumRows(int num_rows) {
      build_num_rows_ = num_rows;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetRowsPerBuffer(int rows_per_buffer) {
      build_rows_per_buffer_ = rows_per_buffer;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetSchema(std::unique_ptr<DataSchema> schema) {
      build_schema_ = std::move(schema);
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      build_num_workers_ = num_workers;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetWorkerConnectorSize(int32_t connector_size) {
      build_worker_connector_size_ = connector_size;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t connector_size) {
      build_op_connector_size_ = connector_size;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetSchemaDir(const std::string &schema_dir) {
      build_schema_file_ = schema_dir + "/datasetSchema.json";
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetSchemaFile(const std::string &schema_file) {
      build_schema_file_ = schema_file;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetDatasetFilesDir(const std::string &files_dir) {
      build_dataset_files_dir_ = files_dir;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetDatasetFileList(const std::vector<std::string> &file_list) {
      build_dataset_file_list_ = file_list;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetColumnsToLoad(const std::vector<std::string> &columns) {
      build_columns_to_load_ = columns;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetDataDistributionFile(const std::string &data_distribution_file) {
      build_data_distribution_file_ = data_distribution_file;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &setLabelsFileName(const std::string &labels_file_name) {
      build_labels_file_name_ = labels_file_name;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetDatasetUsage(const std::string &dataset_usage) {
      build_dataset_usage_ = dataset_usage;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetBatchSize(int32_t batch_size) {
      build_batch_size_ = batch_size;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetDropRemainder(bool drop_remainder) {
      build_drop_remainder_ = drop_remainder;
      return *this;
    }

    // The builder "build" method creates the final object.
    // @param shared_ptr to the new StorageOp object
    // @return Status - The error code return
    Status Build(std::shared_ptr<StorageOp> *);

   private:
    // The builder saves all StorageOp construction arguments internally.
    // The following are the arguments.
    std::string build_dataset_files_dir_;
    std::string build_schema_file_;
    int32_t build_num_rows_;
    std::string build_data_distribution_file_;
    int32_t build_rows_per_buffer_;
    int32_t build_worker_connector_size_;
    int32_t build_num_workers_;
    int32_t build_op_connector_size_;
    std::unique_ptr<DataSchema> build_schema_;
    std::vector<std::string> build_dataset_file_list_;
    std::vector<std::string> build_columns_to_load_;
    std::string build_labels_file_name_;
    std::string build_dataset_usage_;
    int32_t build_batch_size_;
    bool build_drop_remainder_;
  };

  // Constructor of the StorageOp.
  // @note The builder class should be used to call it
  // @param num_workers - The number of workers for the op
  // @param worker_connector_size - The internal connector size between workers and master
  // @param rows_per_buffer - The requested number of rows per buffer
  // @param op_connector_size - The output connector queue size
  // @param columns_to_load - The list of columns to use (column name)
  StorageOp(int32_t num_workers, int32_t worker_connector_size, int32_t rows_per_buffer, int32_t op_connector_size,
            std::vector<std::string> columns_to_load, std::string data_distribution_file, int32_t batch_size,
            bool drop_remainder);

  // Init the StorageOp.  This is 1 of 3 init.
  // This version of the init does not take the schema in it's arguments. It must perform an
  // internal handshake with the dataset to produce the schema.
  // @note The builder class should be used to call it
  // @param dataset_files_dir - The directory that has the dataset files
  // @param schema_file - The schema file for providing column info
  Status InitOp(const std::string &dataset_files_dir, const std::string &schema_file,
                const std::string &labels_file_name, const std::string &dataset_usage);

  // Init the StorageOp.  This is 2 of 3 init.
  // This version of the init allows the user to input the schema and other dataset properties rather
  // than get it from the dataset itself.
  // @note The builder class should be used to call it
  // @param num_rows - The number of rows in the dataset
  // @param dataset_files_dir - The directory that has the dataset files
  // @param data_schema - The schema to use
  Status InitOp(int32_t num_rows, const std::string &dataset_files_dir, std::unique_ptr<DataSchema> data_schema,
                const std::string &labels_file_name, const std::string &dataset_usage);

  // Init the StorageOp.  This is 3 of 3 init.
  // This version of the init does not take the schema in it's arguments. It must perform an
  // internal handshake with the dataset to produce the schema.  Unlike constructor 1, it takes a
  // list of files rather than a directory.
  // @note The builder class should be used to call it
  // @param files_list - The list of files to use for the dataset
  // @param schema_file - The schema file for providing column info
  Status InitOp(const std::vector<std::string> &files_list, const std::string &schema_file);

  // Destructor
  ~StorageOp();

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param storage_op - reference to the StorageOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const StorageOp &storage_op) {
    storage_op.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All DatasetOps operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work.
  // @return Status - The error code return
  Status operator()() override;

  // The entry point code for when workers are launched.
  // @param worker_id - The worker id
  // @return Status - The error code return
  Status WorkerEntry(int32_t worker_id) override;

  // The entry point code for when workers are launched.
  // Given the input bufferId, it returns a shared_ptr to that buffer back to you by driving a
  // load operation.  This function is intended to be run by worker threads, when they are
  // populating the memory with the actual data of the buffer.
  // @param buffer_id - The buffer id to get.
  // @param ptr - Pointer to shared_ptr to the buffer that was loaded in.
  // @return Status - The error code return
  Status GetBuffer(int32_t buffer_id, std::unique_ptr<DataBuffer> *ptr);

  // Overrides base class reset method.  When an operator does a reset, it cleans up any state
  // info from it's previous execution and then initializes itself so that it can be executed
  // again.
  // @return Status - The error code return
  Status Reset() override;

  // Getter method
  int32_t num_rows() const { return num_rows_; }

  // Setter method
  void set_num_rows(int32_t num_rows) { num_rows_ = num_rows; }

  // Getter method
  int32_t rows_per_buffer() const { return rows_per_buffer_; }

  // Setter method
  void set_rows_per_buffer(int32_t rows_per_buffer) { rows_per_buffer_ = rows_per_buffer; }

  // Getter method
  std::string dataset_files_dir() const { return dataset_files_dir_; }

  // Getter method
  std::vector<std::string> dataset_file_list() const { return dataset_file_list_; }

  // Getter method
  std::string schema_file() const { return schema_file_; }

  // Getter method
  const DataSchema *schema() const;

  // Getter method
  const std::vector<std::string> columns_to_load() const { return columns_to_load_; }

  // Getter method
  std::string data_distribution_file() const { return data_distribution_file_; }

  // Getter method
  int32_t device_num() const { return device_num_; }

  // Getter method
  int32_t device_id() const { return device_id_; }

  // Getter method
  std::string shard_config() const { return shard_config_; }

  // Getter method
  uint32_t seed() const { return seed_; }

  // Getter method
  bool shuffle_config() const { return shuffle_config_; }

  // Getter method
  int32_t num_classes() const { return num_classes_; }

  // Getter method
  std::string labels_file_name() const { return labels_file_name_; }

  // Getter method
  std::string dataset_usage() const { return dataset_usage_; }

  // Getter method
  int32_t batch_size() const { return batch_size_; }

  // Getter method
  bool drop_remainder() const { return drop_remainder_; }

 private:
  // Private helper method.  This one populates the action queue with the list of buffer ids.
  // @param randomize - T/F if the id's in the action queue should be randomized or sequential.
  Status FillActionQueue(bool randomize);

  // Private helper method.  This one encapsulates some common construction/reset tasks and is
  // designed to be re-entrant so that you can re-init a previously used StorageOp without needing
  // to redo the storage client handshake.
  // @return Status - The error code return
  Status init();

  // Private helper method.  This one posts a control indicator for each worker thread to consume
  // from the action queue.  When the worker pops this msg, it will shut itself down gracefully.
  // @return Status - The error code return
  Status PostEndOfData();

  Status LoadParallelConfig();

  DataBuffers data_buffers_;                     // A vector of pointers to buffers
  std::shared_ptr<StorageClient> store_client_;  // The client for interacting with storage
  ActionQueue action_queue_;                     // The queues of buffer id's for workers to fetch.
  int32_t worker_conn_size_;                     // connector size for internal worker queue
  int32_t rows_per_buffer_;                      // The number of requested rows per buffer.
  int32_t num_rows_;                             // One more than the last row id in the range for this cache
  std::string dataset_files_dir_;                // The path for the dataset files
  std::vector<std::string> dataset_file_list_;   // List of paths to files for the dataset
  int32_t buffers_fetched_;                      // Counter for the buffers that were fetched
  std::string schema_file_;                      // Path to the schema json file
  std::vector<std::string> columns_to_load_;     // Columns to load from dataset
  std::string data_distribution_file_;           // Distribution configuration file
  int32_t device_num_;                           // All device number
  int32_t device_id_;                            // Device id
  std::string shard_config_;                     // ALL UNIQUE RANDOM
  uint32_t seed_;                                // Used for shuffle
  bool shuffle_config_;                          // True or false
  std::string labels_file_name_;                 // File name of labels
  int32_t num_classes_;                          // Label class number
  std::string dataset_usage_;                    // train/eval/inference
  int32_t batch_size_;
  bool drop_remainder_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_STORAGE_OP_H_
