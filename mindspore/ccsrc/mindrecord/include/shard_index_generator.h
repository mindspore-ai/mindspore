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

#ifndef MINDRECORD_INCLUDE_SHARD_INDEX_GENERATOR_H_
#define MINDRECORD_INCLUDE_SHARD_INDEX_GENERATOR_H_

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "mindrecord/include/shard_header.h"
#include "./sqlite3.h"

namespace mindspore {
namespace mindrecord {
using INDEX_FIELDS = std::pair<MSRStatus, std::vector<std::tuple<std::string, std::string, std::string>>>;
using ROW_DATA = std::pair<MSRStatus, std::vector<std::vector<std::tuple<std::string, std::string, std::string>>>>;
class ShardIndexGenerator {
 public:
  explicit ShardIndexGenerator(const std::string &file_path, bool append = false);

  MSRStatus Build();

  static std::pair<MSRStatus, std::string> GenerateFieldName(const std::pair<uint64_t, std::string> &field);

  ~ShardIndexGenerator() {}

  /// \brief fetch value in json by field name
  /// \param[in] field
  /// \param[in] input
  /// \return pair<MSRStatus, value>
  std::pair<MSRStatus, std::string> GetValueByField(const string &field, json input);

  /// \brief fetch field type in schema n by field path
  /// \param[in] field_path
  /// \param[in] schema
  /// \return the type of field
  static std::string TakeFieldType(const std::string &field_path, json schema);

  /// \brief create databases for indexes
  MSRStatus WriteToDatabase();

 private:
  static int Callback(void *not_used, int argc, char **argv, char **az_col_name);

  static MSRStatus ExecuteSQL(const std::string &statement, sqlite3 *db, const string &success_msg = "");

  static std::string ConvertJsonToSQL(const std::string &json);

  std::pair<MSRStatus, sqlite3 *> CreateDatabase(int shard_no);

  std::pair<MSRStatus, std::vector<json>> GetSchemaDetails(const std::vector<uint64_t> &schema_lens, std::fstream &in);

  static std::pair<MSRStatus, std::string> GenerateRawSQL(const std::vector<std::pair<uint64_t, std::string>> &fields);

  std::pair<MSRStatus, sqlite3 *> CheckDatabase(const std::string &shard_address);

  ///
  /// \param shard_no
  /// \param blob_id_to_page_id
  /// \param raw_page_id
  /// \param in
  /// \return field name, db type, field value
  ROW_DATA GenerateRowData(int shard_no, const std::map<int, int> &blob_id_to_page_id, int raw_page_id,
                           std::fstream &in);
  ///
  /// \param db
  /// \param sql
  /// \param data
  /// \return
  MSRStatus BindParameterExecuteSQL(
    sqlite3 *db, const std::string &sql,
    const std::vector<std::vector<std::tuple<std::string, std::string, std::string>>> &data);

  INDEX_FIELDS GenerateIndexFields(const std::vector<json> &schema_detail);

  MSRStatus ExecuteTransaction(const int &shard_no, std::pair<MSRStatus, sqlite3 *> &db,
                               const std::vector<int> &raw_page_ids, const std::map<int, int> &blob_id_to_page_id);

  MSRStatus CreateShardNameTable(sqlite3 *db, const std::string &shard_name);

  MSRStatus AddBlobPageInfo(std::vector<std::tuple<std::string, std::string, std::string>> &row_data,
                            const std::shared_ptr<Page> cur_blob_page, uint64_t &cur_blob_page_offset,
                            std::fstream &in);

  void AddIndexFieldByRawData(const std::vector<json> &schema_detail,
                              std::vector<std::tuple<std::string, std::string, std::string>> &row_data);

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
#endif  // MINDRECORD_INCLUDE_SHARD_INDEX_GENERATOR_H_
