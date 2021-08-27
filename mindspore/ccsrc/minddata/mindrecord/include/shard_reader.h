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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_READER_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_READER_H_

#include <dirent.h>
#include <signal.h>
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
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
#include <stack>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_category.h"
#include "minddata/mindrecord/include/shard_column.h"
#include "minddata/mindrecord/include/shard_distributed_sample.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_index_generator.h"
#include "minddata/mindrecord/include/shard_operator.h"
#include "minddata/mindrecord/include/shard_pk_sample.h"
#include "minddata/mindrecord/include/shard_reader.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "utils/log_adapter.h"

#define API_PUBLIC __attribute__((visibility("default")))

namespace mindspore {
namespace mindrecord {
using ROW_GROUPS = std::pair<std::vector<std::vector<std::vector<uint64_t>>>, std::vector<std::vector<json>>>;
using ROW_GROUP_BRIEF = std::tuple<std::string, int, uint64_t, std::vector<std::vector<uint64_t>>, std::vector<json>>;
using TASK_CONTENT = std::pair<TaskType, std::vector<std::tuple<std::vector<uint8_t>, json>>>;
const int kNumBatchInMap = 1000;  // iterator buffer size in row-reader mode

class API_PUBLIC ShardReader {
 public:
  ShardReader();

  virtual ~ShardReader();

  /// \brief open files and initialize reader, c++ API
  /// \param[in] file_paths the path of ONE file, any file in dataset is fine or file list
  /// \param[in] load_dataset load dataset from single file or not
  /// \param[in] n_consumer number of threads when reading
  /// \param[in] selected_columns column list to be populated
  /// \param[in] operators operators applied to data, operator type is shuffle, sample or category
  /// \param[in] num_padded the number of padded samples
  /// \param[in] lazy_load if the mindrecord dataset is too large, enable lazy load mode to speed up initialization
  /// \return MSRStatus the status of MSRStatus
  Status Open(const std::vector<std::string> &file_paths, bool load_dataset, int n_consumer = 4,
              const std::vector<std::string> &selected_columns = {},
              const std::vector<std::shared_ptr<ShardOperator>> &operators = {}, const int num_padded = 0,
              bool lazy_load = false);

  /// \brief close reader
  /// \return null
  void Close();

  /// \brief read the file, get schema meta,statistics and index, single-thread mode
  /// \return MSRStatus the status of MSRStatus
  Status Open();

  /// \brief read the file, get schema meta,statistics and index, multiple-thread mode
  /// \return MSRStatus the status of MSRStatus
  Status Open(int n_consumer);

  /// \brief launch threads to get batches
  /// \param[in] is_simple_reader trigger threads if false; do nothing if true
  /// \return MSRStatus the status of MSRStatus
  Status Launch(bool is_simple_reader = false);

  /// \brief aim to get the meta data
  /// \return the metadata
  std::shared_ptr<ShardHeader> GetShardHeader() const;

  /// \brief aim to get columns context
  /// \return the columns
  std::shared_ptr<ShardColumn> GetShardColumn() const;

  /// \brief get the number of shards
  /// \return # of shards
  int GetShardCount() const;

  /// \brief get the number of rows in database
  /// \param[in] file_paths the path of ONE file, any file in dataset is fine or file list
  /// \param[in] load_dataset load dataset from single file or not
  /// \param[in] op smart pointer refer to ShardCategory or ShardSample object
  /// \param[out] count # of rows
  /// \return MSRStatus the status of MSRStatus
  Status CountTotalRows(const std::vector<std::string> &file_paths, bool load_dataset,
                        const std::shared_ptr<ShardOperator> &op, int64_t *count, const int num_padded);

  /// \brief shuffle task with incremental seed
  /// \return void
  void ShuffleTask();

  /// \brief get the number of rows in database
  /// \return # of rows
  int GetNumRows() const;

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
  Status ReadRowGroupBrief(int group_id, int shard_id, const std::vector<std::string> &columns,
                           std::shared_ptr<ROW_GROUP_BRIEF> *row_group_brief_ptr);

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
  Status ReadRowGroupCriteria(int group_id, int shard_id, const std::pair<std::string, std::string> &criteria,
                              const std::vector<std::string> &columns,
                              std::shared_ptr<ROW_GROUP_BRIEF> *row_group_brief_ptr);

  /// \brief return a batch, given that one is ready
  /// \return a batch of images and image data
  std::vector<std::tuple<std::vector<uint8_t>, json>> GetNext();

  /// \brief return a row by id
  /// \return a batch of images and image data
  TASK_CONTENT GetNextById(const int64_t &task_id, const int32_t &consumer_id);
  /// \brief  get blob filed list
  /// \return blob field list
  std::pair<ShardType, std::vector<std::string>> GetBlobFields();

  /// \brief reset reader
  /// \return null
  void Reset();

  /// \brief set flag of all-in-index
  /// \return null
  void SetAllInIndex(bool all_in_index) { all_in_index_ = all_in_index; }

  /// \brief get all classes
  Status GetAllClasses(const std::string &category_field, std::shared_ptr<std::set<std::string>> category_ptr);

  /// \brief get a read-only ptr to the sampled ids for this epoch
  const std::vector<int> *GetSampleIds();

  /// \brief get the size of blob data
  Status GetTotalBlobSize(int64_t *total_blob_size);

  /// \brief extract uncompressed data based on column list
  Status UnCompressBlob(const std::vector<uint8_t> &raw_blob_data,
                        std::shared_ptr<std::vector<std::vector<uint8_t>>> *blob_data_ptr);

 protected:
  /// \brief sqlite call back function
  static int SelectCallback(void *p_data, int num_fields, char **p_fields, char **p_col_names);

 private:
  /// \brief wrap up labels to json format
  Status ConvertLabelToJson(const std::vector<std::vector<std::string>> &labels, std::shared_ptr<std::fstream> fs,
                            std::shared_ptr<std::vector<std::vector<std::vector<uint64_t>>>> offset_ptr, int shard_id,
                            const std::vector<std::string> &columns,
                            std::shared_ptr<std::vector<std::vector<json>>> col_val_ptr);

  /// \brief read all rows for specified columns
  Status ReadAllRowGroup(const std::vector<std::string> &columns, std::shared_ptr<ROW_GROUPS> *row_group_ptr);

  /// \brief read row meta by shard_id and sample_id
  Status ReadRowGroupByShardIDAndSampleID(const std::vector<std::string> &columns, const uint32_t &shard_id,
                                          const uint32_t &sample_id, std::shared_ptr<ROW_GROUPS> *row_group_ptr);

  /// \brief read all rows in one shard
  Status ReadAllRowsInShard(int shard_id, const std::string &sql, const std::vector<std::string> &columns,
                            std::shared_ptr<std::vector<std::vector<std::vector<uint64_t>>>> offset_ptr,
                            std::shared_ptr<std::vector<std::vector<json>>> col_val_ptr);

  /// \brief initialize reader
  Status Init(const std::vector<std::string> &file_paths, bool load_dataset);

  /// \brief validate column list
  Status CheckColumnList(const std::vector<std::string> &selected_columns);

  /// \brief populate one row by task list in row-reader mode
  void ConsumerByRow(int consumer_id);

  /// \brief get offset address of images within page
  std::vector<std::vector<uint64_t>> GetImageOffset(int group_id, int shard_id,
                                                    const std::pair<std::string, std::string> &criteria = {"", ""});

  /// \brief get page id by category
  Status GetPagesByCategory(int shard_id, const std::pair<std::string, std::string> &criteria,
                            std::shared_ptr<std::vector<uint64_t>> *pages_ptr);
  /// \brief execute sqlite query with prepare statement
  Status QueryWithCriteria(sqlite3 *db, const string &sql, const string &criteria,
                           std::shared_ptr<std::vector<std::vector<std::string>>> labels_ptr);
  /// \brief verify the validity of dataset
  Status VerifyDataset(sqlite3 **db, const string &file);

  /// \brief get column values
  Status GetLabels(int page_id, int shard_id, const std::vector<std::string> &columns,
                   const std::pair<std::string, std::string> &criteria, std::shared_ptr<std::vector<json>> *labels_ptr);

  /// \brief get column values from raw data page
  Status GetLabelsFromPage(int page_id, int shard_id, const std::vector<std::string> &columns,
                           const std::pair<std::string, std::string> &criteria,
                           std::shared_ptr<std::vector<json>> *labels_ptr);

  /// \brief create category-applied task list
  Status CreateTasksByCategory(const std::shared_ptr<ShardOperator> &op);

  /// \brief create task list in row-reader mode
  Status CreateTasksByRow(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                          const std::vector<std::shared_ptr<ShardOperator>> &operators);

  /// \brief create task list in row-reader mode and lazy mode
  Status CreateLazyTasksByRow(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                              const std::vector<std::shared_ptr<ShardOperator>> &operators);

  /// \brief crate task list
  Status CreateTasks(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                     const std::vector<std::shared_ptr<ShardOperator>> &operators);

  /// \brief check if all specified columns are in index table
  void CheckIfColumnInIndex(const std::vector<std::string> &columns);

  /// \brief open multiple file handle
  void FileStreamsOperator();

  /// \brief read one row by one task
  Status ConsumerOneTask(int task_id, uint32_t consumer_id, std::shared_ptr<TASK_CONTENT> *task_content_pt);

  /// \brief get labels from binary file
  Status GetLabelsFromBinaryFile(int shard_id, const std::vector<std::string> &columns,
                                 const std::vector<std::vector<std::string>> &label_offsets,
                                 std::shared_ptr<std::vector<json>> *labels_ptr);

  /// \brief get classes in one shard
  void GetClassesInShard(sqlite3 *db, int shard_id, const std::string &sql,
                         std::shared_ptr<std::set<std::string>> category_ptr);

  /// \brief get number of classes
  int64_t GetNumClasses(const std::string &category_field);

  /// \brief get meta of header
  Status GetMeta(const std::string &file_path, std::shared_ptr<json> meta_data_ptr,
                 std::shared_ptr<std::vector<std::string>> *addresses_ptr);

 protected:
  uint64_t header_size_;                       // header size
  uint64_t page_size_;                         // page size
  int shard_count_;                            // number of shards
  std::shared_ptr<ShardHeader> shard_header_;  // shard header
  std::shared_ptr<ShardColumn> shard_column_;  // shard column

  std::vector<sqlite3 *> database_paths_;                                        // sqlite handle list
  std::vector<string> file_paths_;                                               // file paths
  std::vector<std::shared_ptr<std::fstream>> file_streams_;                      // single-file handle list
  std::vector<std::vector<std::shared_ptr<std::fstream>>> file_streams_random_;  // multiple-file handle list

 private:
  int n_consumer_;                                         // number of workers (threads)
  std::vector<std::string> selected_columns_;              // columns which will be read
  std::map<string, uint64_t> column_schema_id_;            // column-schema map
  std::vector<std::shared_ptr<ShardOperator>> operators_;  // data operators, including shuffle, sample and category
  ShardTaskList tasks_;                                    // shard task list
  std::mutex shard_locker_;                                // locker of shard

  // flags
  bool all_in_index_ = true;  // if all columns are stored in index-table
  bool interrupt_ = false;    // reader interrupted

  int num_padded_;  // number of padding samples

  // Delivery/Iterator mode begin
  const std::string kThreadName = "THRD_ITER_";  // prefix of thread name
  std::vector<std::thread> thread_set_;          // thread list
  int num_rows_;                                 // number of rows
  int64_t total_blob_size_;                      // total size of blob data
  std::mutex mtx_delivery_;                      // locker for delivery
  std::condition_variable cv_delivery_;          // conditional variable for delivery
  std::condition_variable cv_iterator_;          // conditional variable for iterator
  std::atomic<int> sample_id_position_;          // index into the sample ids vector for the current sample id
  std::atomic<int> deliver_id_;                  // delivery ID which is picked up by iterator
  // map of delivery
  std::unordered_map<int, std::shared_ptr<std::vector<std::tuple<std::vector<uint8_t>, json>>>> delivery_map_;
  // Delivery/Iterator mode end

  // all metadata in the index is not loaded during initialization
  bool lazy_load_;

  // indicate shard_id : inc_count
  // 0 : 15  -  shard0 has 15 samples
  // 1 : 41  -  shard1 has 26 samples
  // 2 : 58  -  shard2 has 17 samples
  std::vector<uint32_t> shard_sample_count_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_READER_H_
