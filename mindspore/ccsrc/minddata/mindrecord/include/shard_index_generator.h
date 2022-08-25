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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_INDEX_GENERATOR_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_INDEX_GENERATOR_H_

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/shard_header.h"
#include "./sqlite3.h"

namespace mindspore {
namespace mindrecord {
using INDEX_FIELDS = std::vector<std::tuple<std::string, std::string, std::string>>;
using ROW_DATA = std::vector<std::vector<std::tuple<std::string, std::string, std::string>>>;
class MINDRECORD_API ShardIndexGenerator {
 public:
  explicit ShardIndexGenerator(const std::string &file_path, bool append = false);

  Status Build();

  static Status GenerateFieldName(const std::pair<uint64_t, std::string> &field, std::shared_ptr<std::string> *fn_ptr);

  ~ShardIndexGenerator() {}

  /// \brief fetch value in json by field name
  /// \param[in] field
  /// \param[in] input
  /// \param[in] value
  /// \return Status
  Status GetValueByField(const string &field, const json &input, std::shared_ptr<std::string> *value);

  /// \brief fetch field type in schema n by field path
  /// \param[in] field_path
  /// \param[in] schema
  /// \return the type of field
  static std::string TakeFieldType(const std::string &field_path, json &schema);  // NOLINT

  /// \brief create databases for indexes
  Status WriteToDatabase();

  static Status Finalize(const std::vector<std::string> file_names);

 private:
  static int Callback(void *not_used, int argc, char **argv, char **az_col_name);

  static Status ExecuteSQL(const std::string &statement, sqlite3 *db, const string &success_msg = "");

  static std::string ConvertJsonToSQL(const std::string &json);

  Status CreateDatabase(int shard_no, sqlite3 **db);

  Status GetSchemaDetails(const std::vector<uint64_t> &schema_lens, std::fstream &in,
                          std::shared_ptr<std::vector<json>> *detail_ptr);

  static Status GenerateRawSQL(const std::vector<std::pair<uint64_t, std::string>> &fields,
                               std::shared_ptr<std::string> *sql_ptr);

  Status CheckDatabase(const std::string &shard_address, sqlite3 **db);

  ///
  /// \param shard_no
  /// \param blob_id_to_page_id
  /// \param raw_page_id
  /// \param in
  /// \return Status
  Status GenerateRowData(int shard_no, const std::map<int, int> &blob_id_to_page_id, int raw_page_id, std::fstream &in,
                         std::shared_ptr<ROW_DATA> *row_data_ptr);
  ///
  /// \param db
  /// \param sql
  /// \param data
  /// \return
  Status BindParameterExecuteSQL(sqlite3 *db, const std::string &sql, const ROW_DATA &data);

  Status GenerateIndexFields(const std::vector<json> &schema_detail, std::shared_ptr<INDEX_FIELDS> *index_fields_ptr);

  Status ExecuteTransaction(const int &shard_no, sqlite3 *db, const std::vector<int> &raw_page_ids,
                            const std::map<int, int> &blob_id_to_page_id);

  Status CreateShardNameTable(sqlite3 *db, const std::string &shard_name);

  Status AddBlobPageInfo(std::vector<std::tuple<std::string, std::string, std::string>> &row_data,   // NOLINT
                         const std::shared_ptr<Page> cur_blob_page, uint64_t &cur_blob_page_offset,  // NOLINT
                         std::fstream &in);                                                          // NOLINT

  Status AddIndexFieldByRawData(const std::vector<json> &schema_detail,
                                std::vector<std::tuple<std::string, std::string, std::string>> &row_data);  // NOLINT

  void DatabaseWriter();  // worker thread

  std::string file_path_;
  bool append_;
  ShardHeader shard_header_;
  uint64_t page_size_;
  uint64_t header_size_;
  int schema_count_;
  std::atomic_int task_;
  std::atomic_bool write_success_;
  std::vector<std::pair<uint64_t, std::string>> fields_;
};
}  // namespace mindrecord
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_INDEX_GENERATOR_H_
