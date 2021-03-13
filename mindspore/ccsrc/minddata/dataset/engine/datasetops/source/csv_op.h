/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_CSV_OP_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_CSV_OP_H_

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include <limits>

#include "minddata/dataset/util/auto_index.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"

namespace mindspore {
namespace dataset {

const size_t CSV_BUFFER_SIZE = 4096;
using StringIndex = AutoIndexObj<std::string>;
class JaggedConnector;

class CsvOp : public ParallelOp {
 public:
  enum RecordType : uint8_t { INT = 0, FLOAT, STRING };

  struct BaseRecord {
   public:
    BaseRecord() = default;
    explicit BaseRecord(RecordType t) : type(t) {}
    virtual ~BaseRecord() {}
    RecordType type;
  };

  template <typename T>
  class Record : public BaseRecord {
   public:
    Record() = default;
    Record(RecordType t, T v) : BaseRecord(t), value(v) {}
    ~Record() {}
    T value;
  };

  // CsvParser is a class that parsing CSV file.
  // We design a state machine to implement CSV syntactic analysis. It contains two state diagram,'sd' and 'sdl'.
  // The 'sd' is used for parsing CSV syntactic, it's complete and complicate.
  // The 'sdl' is used for counting the record rows, it's concise and it runs fast.
  struct CsvParser {
   public:
    CsvParser() = delete;

    CsvParser(int32_t worker_id, std::shared_ptr<JaggedConnector> connector, int64_t rows_per_buffer, char field_delim,
              std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default, std::string file_path);

    ~CsvParser() = default;

    void Reset();

    void SetStartOffset(int64_t start_offset) { start_offset_ = start_offset; }

    void SetEndOffset(int64_t end_offset) { end_offset_ = end_offset; }

    int ProcessMessage(int c);

    int CountRows(int c);

    Status InitCsvParser();

    int64_t GetTotalRows() { return total_rows_; }

    std::string GetErrorMessage() { return err_message_; }

   private:
    enum State : uint8_t {
      START_OF_FILE = 0,
      UNQUOTE,
      DELIM,
      QUOTE,
      SECOND_QUOTE,
      END_OF_LINE,
      END_OF_FILE,
      EXCEPTION
    };

    enum Message : uint8_t {
      MS_NORMAL = 0,
      MS_DELIM,
      MS_QUOTE,
      MS_END_OF_LINE,
      MS_END_OF_FILE,
    };

    typedef std::pair<State, Message> StateMessagePair;
    typedef std::pair<State, std::function<int(CsvParser &, int)>> StateActionPair;
    typedef std::map<StateMessagePair, StateActionPair> StateDiagram;

    Message GetMessage(int c);

    int NullFunc(int c) { return 0; }

    int PutChar(int c);

    int PutRecord(int c);

    int PutRow(int c);

    int EndFile(int c);

    int AddRow(int c);

    int CatchException(int c);

    int32_t worker_id_;
    std::shared_ptr<JaggedConnector> buffer_connector_;
    int64_t csv_rows_per_buffer_;
    const char csv_field_delim_;
    std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default_;
    State cur_state_;
    size_t pos_;
    int cur_row_;
    int cur_col_;
    int64_t total_rows_;
    int64_t start_offset_;
    int64_t end_offset_;
    StateDiagram sd;
    StateDiagram sdl;
    std::vector<char> str_buf_;
    std::unique_ptr<TensorQTable> tensor_table_;
    std::unique_ptr<DataBuffer> cur_buffer_;
    std::string err_message_;
    std::string file_path_;
  };

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

    // Create the final object.
    // @param op - dataset op.
    // @return - the error code return.
    Status Build(std::shared_ptr<CsvOp> *op);

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
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
    Builder &SetCsvFilesList(const std::vector<std::string> &files_list) {
      builder_csv_files_list_ = files_list;
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
    Builder &SetNumSamples(int64_t num_samples) {
      builder_num_samples_ = num_samples;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetFieldDelim(char field_delim) {
      builder_field_delim_ = field_delim;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetColumDefault(std::vector<std::shared_ptr<CsvOp::BaseRecord>> record_list) {
      builder_column_default_list_ = record_list;
      return *this;
    }

    // Setter method.
    // @return Builder - setter method returns reference to the builder.
    Builder &SetColumName(std::vector<std::string> col_name_list) {
      builder_column_name_list_ = col_name_list;
      return *this;
    }

   private:
    int32_t builder_device_id_;
    int32_t builder_num_devices_;
    int32_t builder_num_workers_;
    int32_t builder_op_connector_size_;
    int64_t builder_rows_per_buffer_;
    int64_t builder_num_samples_;
    int32_t builder_worker_connector_size_;
    std::vector<std::string> builder_csv_files_list_;
    bool builder_shuffle_files_;
    char builder_field_delim_;
    std::vector<std::shared_ptr<CsvOp::BaseRecord>> builder_column_default_list_;
    std::vector<std::string> builder_column_name_list_;
  };

  // Constructor of CsvOp
  CsvOp() = delete;

  CsvOp(const std::vector<std::string> &csv_files_list, char field_delim,
        const std::vector<std::shared_ptr<BaseRecord>> &column_default, const std::vector<std::string> &column_name,
        int32_t num_workers, int64_t rows_per_buffer, int64_t num_samples, int32_t worker_connector_size,
        int32_t op_connector_size, bool shuffle_files, int32_t num_devices, int32_t device_id);

  // Default destructor
  ~CsvOp() = default;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // Instantiates the internal queues and connectors
  // @return Status - the error code returned
  Status Init();

  // Class functor operator () override.
  // All dataset operators operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status - the error code returned.
  Status operator()() override;

  // Overrides base class reset method. Cleans up any state info from it's previous execution
  // reinitializes itself so that it can be executed again, as if it was just created.
  // @return Status - the error code returned.
  Status Reset() override;

  // Get total rows in files.
  // @param files - all csv files.
  // @param csv_header - a bool that indicates csv file include header line
  // @param count - number of rows.
  // @return Status - the error coed returned.
  static Status CountAllFileRows(const std::vector<std::string> &files, bool csv_header, int64_t *count);

  // File names getter
  // @return Vector of the input file names
  std::vector<std::string> FileNames() { return csv_files_list_; }

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "CsvOp"; }

 private:
  // The entry point for when workers are launched.
  // @param worker_id - the id of the worker that is executing this function.
  // @return Status - the error code returned.
  Status WorkerEntry(int32_t worker_id) override;

  // Parses a single row and puts the data into a tensor table.
  // @param line - the content of the row.
  // @param tensor_table - the tensor table to put the parsed data in.
  // @param row - the id of the row filled in the tensor table.
  // @return Status - the error code returned.
  Status LoadTensor(const std::string &line, std::unique_ptr<TensorQTable> *tensor_table, int64_t row);

  // Reads a csv file and loads the data into multiple buffers.
  // @param file - the file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @return Status - the error code returned.
  Status LoadFile(const std::string &file, const int64_t start_offset, const int64_t end_offset,
                  const int32_t worker_id);

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

  // Called asynchronously by another thread. Will wait until notified to fill the IOBlockQueue.
  // @return Status - the error code returned.
  Status WaitToFillIOBlockQueue();

  // Fill the IOBlockQueue.
  // @para i_keys - keys of file to fill to the IOBlockQueue
  // @return Status - the error code returned.
  Status FillIOBlockQueue(const std::vector<int64_t> &i_keys);

  // Notifies the thread which called FillIoBlockQueue to resume execution
  void NotifyToFillIOBlockQueue();

  // Select file and push it to the block queue.
  // @param file_name - File name.
  // @param start_offset - If file contains the first sample of data.
  // @param end_offset - If file contains the end sample of data.
  // @param pre_count - Total rows of previous files.
  // @return Status - the error code returned.
  bool NeedPushFileToBlockQueue(const std::string &file_name, int64_t *start_offset, int64_t *end_offset,
                                const int64_t &pre_count);

  // Pushes a control indicator onto the IOBlockQueue for each worker to consume. When the worker
  // pops this control indicator, it will wait until the next epoch starts and then resume execution.
  // @return Status - the error code returned.
  Status PostEndOfEpoch(int32_t queue_index);

  // Calculate number of rows in each shard.
  // @return Status - the error code returned.
  Status CalculateNumRowsPerShard();

  // Count number of rows in each file.
  // @param filename - csv file name.
  // @return int64_t - the total number of rows in file.
  int64_t CountTotalRows(const std::string &file);

  // Pushes a control indicator onto the IOBlockQueue for each worker to consume.
  // When the worker pops this control indicator, it will shut itself down gracefully.
  // @return Status - the error code returned.
  Status PostEndOfData();

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  // Split string based on a character delimiter
  // @param str - the input string
  // @param str - the delimiter
  // @return - the a string vector
  std::vector<std::string> split(const std::string &s, char delim);

  // Private function for analysing the column name in every CSV file
  // @return - Status
  Status ColMapAnalyse(const std::string &csv_file_name);

  // Private function for validating whether the column name set in every CSV file remain the same
  // @return bool - whether column name identical in all CSV files
  bool ColumnNameValidate();

  int32_t device_id_;
  bool shuffle_files_;
  bool finished_reading_dataset_;
  int32_t num_devices_;
  int64_t rows_per_buffer_;
  bool load_io_block_queue_;
  int64_t num_rows_per_shard_;
  int64_t all_num_rows_;
  int64_t num_samples_;
  std::map<std::string, int64_t> filename_numrows_;
  std::unique_ptr<StringIndex> filename_index_;
  std::vector<std::string> csv_files_list_;
  WaitPost io_block_queue_wait_post_;
  std::shared_ptr<JaggedConnector> jagged_buffer_connector_;
  QueueList<std::unique_ptr<FilenameBlock>> io_block_queues_;
  bool load_jagged_connector_;
  char field_delim_;
  std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default_list_;
  std::vector<std::string> column_name_list_;
  bool check_flag_ = false;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_CSV_OP_H_
