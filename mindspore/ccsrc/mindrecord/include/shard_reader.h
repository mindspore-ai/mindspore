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

#ifndef MINDRECORD_INCLUDE_SHARD_READER_H_
#define MINDRECORD_INCLUDE_SHARD_READER_H_

#include <dirent.h>
#include <signal.h>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/prctl.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "mindrecord/include/common/shard_utils.h"
#include "mindrecord/include/shard_category.h"
#include "mindrecord/include/shard_error.h"
#include "mindrecord/include/shard_index_generator.h"
#include "mindrecord/include/shard_operator.h"
#include "mindrecord/include/shard_reader.h"
#include "mindrecord/include/shard_sample.h"
#include "mindrecord/include/shard_shuffle.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace mindrecord {
using ROW_GROUPS =
  std::tuple<MSRStatus, std::vector<std::vector<std::vector<uint64_t>>>, std::vector<std::vector<json>>>;
using ROW_GROUP_BRIEF =
  std::tuple<MSRStatus, std::string, int, uint64_t, std::vector<std::vector<uint64_t>>, std::vector<json>>;
using TASK_RETURN_CONTENT = std::pair<MSRStatus, std::vector<std::tuple<std::vector<uint8_t>, json>>>;
const int kNumBatchInMap = 1000;  // iterator buffer size in row-reader mode
const int kNumPageInBuffer = 16;  // page buffer size in block-reader mode

class ShardReader {
 public:
  ShardReader();

  virtual ~ShardReader();

  /// \brief open files and initialize reader, c++ API
  /// \param[in] file_path the path of ONE file, any file in dataset is fine
  /// \param[in] n_consumer number of threads when reading
  /// \param[in] selected_columns column list to be populated
  /// \param[in] operators operators applied to data, operator type is shuffle, sample or category
  /// \param[in] block_reader block-reader mode if true, otherwise row-reader mode
  /// \return MSRStatus the status of MSRStatus
  MSRStatus Open(const std::string &file_path, int n_consumer = 4,
                 const std::vector<std::string> &selected_columns = {},
                 const std::vector<std::shared_ptr<ShardOperator>> &operators = {}, const bool &block_reader = false);

  /// \brief open files and initialize reader, python API
  /// \param[in] file_path the path of ONE file, any file in dataset is fine
  /// \param[in] n_consumer number of threads when reading
  /// \param[in] selected_columns column list to be populated
  /// \param[in] operators operators applied to data, operator type is shuffle, sample or category
  /// \return MSRStatus the status of MSRStatus
  MSRStatus OpenPy(const std::string &file_path, const int &n_consumer = 4,
                   const std::vector<std::string> &selected_columns = {},
                   const std::vector<std::shared_ptr<ShardOperator>> &operators = {});

  /// \brief close reader
  /// \return null
  void Close();

  /// \brief read the file, get schema meta,statistics and index, single-thread mode
  /// \return MSRStatus the status of MSRStatus
  MSRStatus Open();

  /// \brief read the file, get schema meta,statistics and index, multiple-thread mode
  /// \return MSRStatus the status of MSRStatus
  MSRStatus Open(int n_consumer);

  /// \brief launch threads to get batches
  /// \param[in] is_simple_reader trigger threads if false; do nothing if true
  /// \return MSRStatus the status of MSRStatus
  MSRStatus Launch(bool is_simple_reader = false);

  /// \brief aim to get the meta data
  /// \return the metadata
  std::shared_ptr<ShardHeader> get_shard_header() const;

  /// \brief get the number of shards
  /// \return # of shards
  int get_shard_count() const;

  /// \brief get the number of rows in database
  /// \param[in] file_path the path of ONE file, any file in dataset is fine
  /// \param[in] op smart pointer refer to ShardCategory or ShardSample object
  /// \param[out] count # of rows
  /// \return MSRStatus the status of MSRStatus
  MSRStatus CountTotalRows(const std::string &file_path, const std::shared_ptr<ShardOperator> &op, int64_t *count);

  /// \brief shuffle task with incremental seed
  /// \return void
  void ShuffleTask();

  /// \brief get the number of rows in database
  /// \return # of rows
  int get_num_rows() const;

  /// \brief Read the summary of row groups
  /// \return the tuple of 4 elements
  ///         1. Sharding ID
  ///         2. Row group ID
  ///         3. The row ID started in row group
  ///         4. # of rows in row group
  std::vector<std::tuple<int, int, int, uint64_t>> ReadRowGroupSummary();

  /// \brief Read 1 row group data, excluding images
  /// \param[in] groupID row group ID
  /// \param[in] shard_id sharding ID
  /// \param[in] columns multi-columns retrieved
  /// \return the tuple of 5 elements
  ///         1. file name where row group is located
  ///         2. Actual row group size
  ///         3. Offset address of row group in file
  ///         4. The list of image offset in page [startOffset, endOffset)
  ///         5. The list of columns data
  ROW_GROUP_BRIEF ReadRowGroupBrief(int group_id, int shard_id,
                                    const std::vector<std::string> &columns = std::vector<std::string>());

  /// \brief Read 1 row group data, excluding images, following an index field criteria
  /// \param[in] groupID row group ID
  /// \param[in] shard_id sharding ID
  /// \param[in] column-value pair of criteria to fulfill
  /// \param[in] columns multi-columns retrieved
  /// \return the tuple of 5 elements
  ///         1. file name where row group is located
  ///         2. Actual row group size
  ///         3. Offset address of row group in file
  ///         4. The list of image offset in page [startOffset, endOffset)
  ///         5. The list of columns data
  ROW_GROUP_BRIEF ReadRowGroupCriteria(int group_id, int shard_id, const std::pair<std::string, std::string> &criteria,
                                       const std::vector<std::string> &columns = std::vector<std::string>());

  /// \brief join all created threads
  /// \return MSRStatus the status of MSRStatus
  MSRStatus Finish();

  /// \brief return a batch, given that one is ready
  /// \return a batch of images and image data
  std::vector<std::tuple<std::vector<uint8_t>, json>> GetNext();

  /// \brief return a row by id
  /// \return a batch of images and image data
  std::vector<std::tuple<std::vector<uint8_t>, json>> GetNextById(const int64_t &task_id, const int32_t &consumer_id);

  /// \brief return a batch in block-reader mode, given that one is ready
  /// \return a batch of images and image data
  std::vector<std::tuple<std::vector<uint8_t>, json>> GetBlockNext();

  /// \brief return a batch, given that one is ready, python API
  /// \return a batch of images and image data
  std::vector<std::tuple<std::vector<uint8_t>, pybind11::object>> GetNextPy();

  /// \brief  get blob filed list
  /// \return blob field list
  std::pair<ShardType, std::vector<std::string>> get_blob_fields();

  /// \brief reset reader
  /// \return null
  void Reset();

  /// \brief set flag of all-in-index
  /// \return null
  void set_all_in_index(bool all_in_index) { all_in_index_ = all_in_index; }

  /// \brief get NLP flag
  bool get_nlp_flag();

  /// \brief get all classes
  MSRStatus GetAllClasses(const std::string &category_field, std::set<std::string> &categories);

 protected:
  /// \brief sqlite call back function
  static int SelectCallback(void *p_data, int num_fields, char **p_fields, char **p_col_names);

 private:
  /// \brief wrap up labels to json format
  MSRStatus ConvertLabelToJson(const std::vector<std::vector<std::string>> &labels, std::shared_ptr<std::fstream> fs,
                               std::vector<std::vector<std::vector<uint64_t>>> &offsets, int shard_id,
                               const std::vector<std::string> &columns, std::vector<std::vector<json>> &column_values);

  /// \brief read all rows for specified columns
  ROW_GROUPS ReadAllRowGroup(std::vector<std::string> &columns);

  /// \brief read all rows in one shard
  MSRStatus ReadAllRowsInShard(int shard_id, const std::string &sql, const std::vector<std::string> &columns,
                               std::vector<std::vector<std::vector<uint64_t>>> &offsets,
                               std::vector<std::vector<json>> &column_values);

  /// \brief initialize reader
  MSRStatus Init(const std::string &file_path);

  /// \brief validate column list
  MSRStatus CheckColumnList(const std::vector<std::string> &selected_columns);

  /// \brief populate one row by task list in row-reader mode
  MSRStatus ConsumerByRow(int consumer_id);

  /// \brief populate one row by task list in block-reader mode
  MSRStatus ConsumerByBlock(int consumer_id);

  /// \brief get offset address of images within page
  std::vector<std::vector<uint64_t>> GetImageOffset(int group_id, int shard_id,
                                                    const std::pair<std::string, std::string> &criteria = {"", ""});

  /// \brief execute sqlite query with prepare statement
  MSRStatus QueryWithCriteria(sqlite3 *db, string &sql, string criteria, std::vector<std::vector<std::string>> &labels);

  /// \brief get column values
  std::pair<MSRStatus, std::vector<json>> GetLabels(int group_id, int shard_id, const std::vector<std::string> &columns,
                                                    const std::pair<std::string, std::string> &criteria = {"", ""});

  /// \brief get column values from raw data page
  std::pair<MSRStatus, std::vector<json>> GetLabelsFromPage(int group_id, int shard_id,
                                                            const std::vector<std::string> &columns,
                                                            const std::pair<std::string, std::string> &criteria = {"",
                                                                                                                   ""});

  /// \brief create task list in block-reader mode
  MSRStatus CreateTasksByBlock(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                               const std::vector<std::shared_ptr<ShardOperator>> &operators);

  /// \brief create category-applied task list
  MSRStatus CreateTasksByCategory(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                                  const std::shared_ptr<ShardOperator> &op);

  /// \brief create task list in row-reader mode
  MSRStatus CreateTasksByRow(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                             const std::vector<std::shared_ptr<ShardOperator>> &operators);

  /// \brief crate task list
  MSRStatus CreateTasks(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                        const std::vector<std::shared_ptr<ShardOperator>> &operators);

  /// \brief set NLP flag
  void CheckNlp();

  /// \brief check if all specified columns are in index table
  void CheckIfColumnInIndex(const std::vector<std::string> &columns);

  /// \brief open multiple file handle
  void FileStreamsOperator();

  /// \brief read one row by one task
  TASK_RETURN_CONTENT ConsumerOneTask(int task_id, uint32_t consumer_id);

  /// \brief get all the column names by schema
  vector<std::string> GetAllColumns();

  /// \brief get one row from buffer in block-reader mode
  std::shared_ptr<std::vector<std::tuple<std::vector<uint8_t>, json>>> GetRowFromBuffer(int bufId, int rowId);

  /// \brief get labels from binary file
  std::pair<MSRStatus, std::vector<json>> GetLabelsFromBinaryFile(
    int shard_id, const std::vector<std::string> &columns, const std::vector<std::vector<std::string>> &label_offsets);

  MSRStatus ReadBlob(const int &shard_id, const uint64_t &page_offset, const int &page_length, const int &buf_id);

  /// \brief get classes in one shard
  void GetClassesInShard(sqlite3 *db, int shard_id, const std::string sql, std::set<std::string> &categories);

  /// \brief get number of classes
  int64_t GetNumClasses(const std::string &file_path, const std::string &category_field);

  /// \brief get exactly blob fields data by indices
  std::vector<uint8_t> ExtractBlobFieldBySelectColumns(std::vector<uint8_t> &blob_fields_bytes,
                                                       std::vector<uint32_t> &ordered_selected_columns_index);

 protected:
  uint64_t header_size_;                       // header size
  uint64_t page_size_;                         // page size
  int shard_count_;                            // number of shards
  std::shared_ptr<ShardHeader> shard_header_;  // shard header
  bool nlp_ = false;                           // NLP data

  std::vector<sqlite3 *> database_paths_;                                        // sqlite handle list
  std::vector<string> file_paths_;                                               // file paths
  std::vector<std::shared_ptr<std::fstream>> file_streams_;                      // single-file handle list
  std::vector<std::vector<std::shared_ptr<std::fstream>>> file_streams_random_;  // multiple-file handle list

 private:
  int n_consumer_;                                         // number of workers (threads)
  std::vector<std::string> selected_columns_;              // columns which will be read
  std::map<string, uint64_t> column_schema_id_;            // column-schema map
  std::vector<std::shared_ptr<ShardOperator>> operators_;  // data operators, including shuffle, sample and category
  ShardTask tasks_;                                        // shard task
  std::mutex shard_locker_;                                // locker of shard

  // flags
  bool all_in_index_ = true;  // if all columns are stored in index-table
  bool interrupt_ = false;    // reader interrupted

  // Delivery/Iterator mode begin
  const std::string kThreadName = "THRD_ITER_";  // prefix of thread name
  std::vector<std::thread> thread_set_;          // thread list
  int num_rows_;                                 // number of rows
  std::mutex mtx_delivery_;                      // locker for delivery
  std::condition_variable cv_delivery_;          // conditional variable for delivery
  std::condition_variable cv_iterator_;          // conditional variable for iterator
  std::atomic<int> task_id_;                     // task ID which is working
  std::atomic<int> deliver_id_;                  // delivery ID which is picked up by iterator
  // map of delivery
  std::unordered_map<int, std::shared_ptr<std::vector<std::tuple<std::vector<uint8_t>, json>>>> delivery_map_;
  // Delivery/Iterator mode end

  // Block reader mode begin
  bool block_reader_;  // block-reader mode
  int row_id_;         // row id in one page
  int num_blocks_;     // number of pages
  // raw data page
  std::vector<std::shared_ptr<std::pair<std::vector<std::vector<uint64_t>>, std::vector<json>>>> delivery_block_;
  std::unordered_set<int> delivery_block_set_;  // set of delivered pages
  std::vector<std::vector<uint8_t>> buf_;       // page buffer
  // Block reader mode end
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_READER_H_
