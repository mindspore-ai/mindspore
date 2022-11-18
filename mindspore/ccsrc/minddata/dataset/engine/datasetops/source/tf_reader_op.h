/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#if !defined(_WIN32) && !defined(_WIN64)
#include <zlib.h>
#endif

#include <algorithm>
#include <condition_variable>
#include <iomanip>
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
#include "minddata/dataset/engine/datasetops/source/nonmappable_leaf_op.h"
#include "minddata/dataset/engine/jagged_connector.h"

namespace dataengine {
class Example;
class Feature;
class BytesList;
}  // namespace dataengine

namespace mindspore {
namespace dataset {
const int kTFRecordRecLenSize = sizeof(int64_t);
const int kTFRecordHeadFootSize = sizeof(int32_t);  // header has same size with footer
const int kZLIBChunkSize = 16384;

template <typename T>
class Queue;

template <class T>
class Connector;

class JaggedConnector;
class FilenameBlock;

using StringIndex = AutoIndexObj<std::string>;

class TFReaderOp : public NonMappableLeafOp {
 public:
  // Constructor of TFReaderOp (2)
  // @note The builder class should be used to call this constructor.
  // @param num_workers - number of worker threads reading data from TFRecord files.
  // @param worker_connector_size - size of each internal queue.
  // @param total_num_rows - Number of rows to read
  // @param dataset_files_list - list of filepaths for the dataset files.
  // @param data_schema - the data schema object.
  // @param op_connector_size - size of each queue in the connector that the child operator pulls from.
  // @param columns_to_load - the names of the columns to load data from.
  // @param shuffle_files - whether or not to shuffle the files before reading data.
  // @param equal_rows_per_shard - whether or not to get equal rows for each process.
  // @param compression_type - the compression type of the TFRecord files
  TFReaderOp(int32_t num_workers, int32_t worker_connector_size, int64_t total_num_rows,
             std::vector<std::string> dataset_files_list, std::unique_ptr<DataSchema> data_schema,
             int32_t op_connector_size, std::vector<std::string> columns_to_load, bool shuffle_files,
             int32_t num_devices, int32_t device_id, bool equal_rows_per_shard,
             const CompressionType &compression_type = CompressionType::None);

  /// Default destructor
  ~TFReaderOp() = default;

  /// A print method typically used for debugging
  /// @param out - The output stream to write output to
  /// @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // Instantiates the internal queues and connectors.
  // @return Status - the error code returned.
  Status Init() override;

  /// Reads all the provided TFRecord files and counts the total number of rows. filenames will
  /// first be sectioned into equal parts, then sections are read in parallel. If threads is
  /// greater than the number of files, threads will be clamped to the number of files.
  /// @param out_total_rows - output parameter which contains the total number of rows
  /// @param filenames - a list of TFRecord filenames.
  /// @param threads - number of threads to use to read the TFRecord files.
  /// @param estimate - estimate mode, under this mode each threads will sample a single file from each chunk
  /// @return Status - the error code returned.
  static Status CountTotalRows(int64_t *out_total_rows, const std::vector<std::string> &filenames, int64_t threads = 1,
                               bool estimate = false);

  /// Op name getter
  /// @return Name of the current Op
  std::string Name() const override { return "TFReaderOp"; }

  /// File names getter
  /// @return Vector of the input file names
  std::vector<std::string> FileNames() { return dataset_files_list_; }

 private:
  // Reads a TFRecord file and loads the data into multiple TensorRows.
  // @param filename - the TFRecord file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @return Status - the error code returned.
  Status LoadFile(const std::string &filename, int64_t start_offset, int64_t end_offset, int32_t worker_id) override;

  // Helper function to read a non-compressed TFRecord file and loads the data into multiple TensorRows.
  // @param filename - the TFRecord file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @param realpath_value - the real path for filename.
  // @return Status - the error code returned.
  Status HelperLoadNonCompFile(const std::string &filename, int64_t start_offset, int64_t end_offset, int32_t worker_id,
                               const std::string &realpath_value);

#if !defined(_WIN32) && !defined(_WIN64)
  // ZLIBStream struct to initial ZLIB stream
  typedef struct ZLIBStreamInf {
    z_stream strm;
    char input_stream[kZLIBChunkSize];               // instream
    unsigned char record_size[kTFRecordRecLenSize];  // in order to get record_length
    unsigned char garbage[kTFRecordHeadFootSize];    // header and footer are ignored
    std::unique_ptr<unsigned char[]> content;        // for serialized example
    int64_t record_length;                           // the content's length
    int read_flag;                                   // flag to keep track what is currently being read
    int64_t left_to_read;                            // number of bytes left to read for particular read_flag
    int inflate_status;                              // in order to keep track of inflate status

    ZLIBStreamInf() {
      // allocate inflate state
      strm.zalloc = Z_NULL;
      strm.zfree = Z_NULL;
      strm.opaque = Z_NULL;
      strm.avail_in = 0;
      strm.next_in = Z_NULL;

      read_flag = ZLIBReadFlag::RecordLength;
      left_to_read = 0;
    }
  } ZLIBStreamInf;

  // Helper function to read a compressed GZIP TFRecord file and loads the data into multiple TensorRows.
  // @param filename - the TFRecord file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @param realpath_value - the real path for filename.
  // @return Status - the error code returned.
  Status HelperLoadCompGZIPFile(const std::string &filename, int64_t start_offset, int64_t end_offset,
                                int32_t worker_id, const std::string &realpath_value);

  // Helper function to read a compressed ZLIB TFRecord file and loads the data into multiple TensorRows.
  // @param filename - the TFRecord file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @param realpath_value - the real path for filename.
  // @return Status - the error code returned.
  Status HelperLoadCompZLIBFile(const std::string &filename, int64_t start_offset, int64_t end_offset,
                                int32_t worker_id, const std::string &realpath_value);

  // Helper function to convert binary data to int64_t.
  // @param str_record_size - the binary data stored in string.
  // @param str_size - length of binary data stored in string.
  // @return int64_t - the integer.
  int64_t HelperBinDataToInt(unsigned char *str_record_size, size_t str_size);

  // Helper function to inflate ZLIB stream
  // @param zlib_stream - ZLIB stream.
  // @param filename - TFRecord file name (for throwing error purposes)
  // @return Status - the error code returned.
  Status HelperInflateZLIB(ZLIBStreamInf *zlib_stream, const std::string &filename);

  // Helper function to process ZLIB data depending on the flag.
  // @param zlib_stream - ZLIB stream.
  // @param rows_read - number of rows that have been read (content only).
  // @param rows_total - number of total rows that have been read.
  // @param filename - the TFRecord file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @return Status - the error code returned.
  Status HelperProcessZLIBData(ZLIBStreamInf *zlib_stream, int64_t *rows_read, int64_t *rows_total,
                               const std::string &filename, int64_t start_offset, int64_t end_offset,
                               int32_t worker_id);
#endif

  // Helper function to get serialized example for Schema.
  // @param serialized_example - container to store the serialized_example.
  // @param realpath_value - the path for the file.
  // @param filename - TFRecord file name (for throwing error purposes)
  // @return Status - the error code returned.
  Status HelperGetExampleSchema(std::string *serialized_example, const std::string &realpath_value,
                                const std::string &filename);

  // Parses a single row and puts the data into a tensor table.
  // @param tf_record_file - the row to be parsed.
  // @param tensor_table - the tensor table to put the parsed data in.
  // @param row - the id of the row filled in the tensor table.
  // @return Status - the error code returned.
  Status LoadExample(const dataengine::Example *tf_record_file, TensorRow *out_row);

  // Parses a single cell and puts the data into a tensor table.
  // @param tensor_table - the tensor table to put the parsed data in.
  // @param column_values_list - the cell to parse.
  // @param current_col - the column descriptor containing the expected shape and type of the data.
  // @return Status - the error code returned.
  Status LoadFeature(TensorRow *tensor_row, const dataengine::Feature &column_values_list,
                     const ColDescriptor &current_col, int32_t col);

  /// Reads values from a bytes list
  /// @param current_col - the column descriptor containing the expected shape and type of the data.
  /// @param column_values_list - the cell that contains the bytes list to read from.
  /// @param elementStr - the string we read the value into.
  /// @return Status - the error code returned.
  static Status LoadBytesList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                              int32_t *num_elements, std::shared_ptr<Tensor> *tensor);

  /// Reads values from a float list
  /// @param current_col - the column descriptor containing the expected shape and type of the data.
  /// @param column_values_list - the cell that contains the float list to read from.
  /// @Param numElements - number of values in the float list.
  /// @param float_array - the array we read the values into.
  /// @return Status - the error code returned.
  Status LoadFloatList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                       int32_t *num_elements, std::unique_ptr<float[]> *float_array);

  /// Reads values from a bytes list and casts the value to type T, must be an integral
  /// type compatible with int64_t
  /// @param current_col - the column descriptor containing the expected shape and type of the data.
  /// @param column_values_list - the cell that contains the int list to read from.
  /// @Param num_elements - number of values in the int list.
  /// @param tensor - the tensor we read the values into.
  /// @return Status - the error code returned.
  template <typename T>
  Status LoadIntList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                     int32_t *num_elements, std::shared_ptr<Tensor> *tensor);

  /// Determines which template type to use and calls LoadIntList
  /// @param current_col - the column descriptor containing the expected shape and type of the data.
  /// @param column_values_list - the cell that contains the int list to read from.
  /// @Param numElements - number of values in the int list.
  /// @param tensor - the tensor we read the values into.
  /// @return Status - the error code returned.
  Status LoadIntListSwitch(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                           int32_t *num_elements, std::shared_ptr<Tensor> *tensor);

  /// Reads one row of data from a tf file and creates a schema based on that row
  /// @return Status - the error code returned.
  Status CreateSchema(const std::string tf_record_file, std::vector<std::string> columns_to_load);

  /// Meant to be called async. Will read files in the range [begin, end) and return the total rows
  /// @param filenames - a list of tf data filenames.
  /// @param begin - index of first file to read.
  /// @param end - one greater than the index of the last file to read.
  /// @return int63_t - the total number of rows of files read.
  static int64_t CountTotalRowsSectioned(const std::vector<std::string> &filenames, const int64_t begin,
                                         const int64_t end);

 protected:
  Status FillIOBlockQueue(const std::vector<int64_t> &i_keys) override;

 private:
  enum ZLIBReadFlag { RecordLength = 0, Header = 1, Content = 2, Footer = 3 };

  // Helper to fill IO block queue
  // @param queue_index - the queue index.
  // @param key_index - current key_index.
  // @param pre_count - to keep track total number of rows read so far.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param key - key to be put in the io block.
  // @param file_name - the TFRecord file name based on the current key iteration.
  // @return Status - the error code returned.
  Status HelperIOBlockFiller(int32_t *queue_index, int32_t *key_index, int64_t *pre_count, int64_t *start_offset,
                             int64_t *end_offset, int64_t key, const std::string &file_name);

  // Calculate number of rows in each shard.
  // @return Status - the error code returned.
  Status CalculateNumRowsPerShard() override;

  /// Private function for computing the assignment of the column name map.
  /// @return - Status
  Status ComputeColMap() override;

  std::vector<std::string> dataset_files_list_;
  std::vector<std::string> columns_to_load_;
  std::unique_ptr<DataSchema> data_schema_;
  bool equal_rows_per_shard_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_TF_READER_OP_H_
