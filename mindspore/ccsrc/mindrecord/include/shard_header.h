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

#ifndef MINDRECORD_INCLUDE_SHARD_HEADER_H_
#define MINDRECORD_INCLUDE_SHARD_HEADER_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "mindrecord/include/common/shard_utils.h"
#include "mindrecord/include/shard_error.h"
#include "mindrecord/include/shard_index.h"
#include "mindrecord/include/shard_page.h"
#include "mindrecord/include/shard_schema.h"
#include "mindrecord/include/shard_statistics.h"

namespace mindspore {
namespace mindrecord {
class ShardHeader {
 public:
  ShardHeader();

  MSRStatus Build(const std::string &file_path);

  ~ShardHeader() = default;

  /// \brief add the schema and save it
  /// \param[in] schema the schema needs to be added
  /// \return the last schema's id
  int AddSchema(std::shared_ptr<Schema> schema);

  /// \brief add the statistic and save it
  /// \param[in] statistic the statistic needs to be added
  /// \return the last statistic's id
  void AddStatistic(std::shared_ptr<Statistics> statistic);

  /// \brief create index and add fields which from schema for each schema
  /// \param[in] fields the index fields needs to be added
  /// \return SUCCESS if add successfully, FAILED if not
  MSRStatus AddIndexFields(std::vector<std::pair<uint64_t, std::string>> fields);

  MSRStatus AddIndexFields(const std::vector<std::string> &fields);

  /// \brief get the schema
  /// \return the schema
  std::vector<std::shared_ptr<Schema>> get_schemas();

  /// \brief get Statistics
  /// \return the Statistic
  std::vector<std::shared_ptr<Statistics>> get_statistics();

  /// \brief get the fields of the index
  /// \return the fields of the index
  std::vector<std::pair<uint64_t, std::string>> get_fields();

  /// \brief get the index
  /// \return the index
  std::shared_ptr<Index> get_index();

  /// \brief get the schema by schemaid
  /// \param[in] schemaId the id of schema needs to be got
  /// \return the schema obtained by schemaId
  std::pair<std::shared_ptr<Schema>, MSRStatus> GetSchemaByID(int64_t schema_id);

  /// \brief get the filepath to shard by shardID
  /// \param[in] shardID the id of shard which filepath needs to be obtained
  /// \return the filepath obtained by shardID
  std::string get_shard_address_by_id(int64_t shard_id);

  /// \brief get the statistic by statistic id
  /// \param[in] statisticId the id of statistic needs to be get
  /// \return the statistics obtained by statistic id
  std::pair<std::shared_ptr<Statistics>, MSRStatus> GetStatisticByID(int64_t statistic_id);

  MSRStatus InitByFiles(const std::vector<std::string> &file_paths);

  void set_index(Index index) { index_ = std::make_shared<Index>(index); }

  std::pair<std::shared_ptr<Page>, MSRStatus> GetPage(const int &shard_id, const int &page_id);

  MSRStatus SetPage(const std::shared_ptr<Page> &new_page);

  MSRStatus AddPage(const std::shared_ptr<Page> &new_page);

  int64_t GetLastPageId(const int &shard_id);

  int GetLastPageIdByType(const int &shard_id, const std::string &page_type);

  const std::pair<MSRStatus, std::shared_ptr<Page>> GetPageByGroupId(const int &group_id, const int &shard_id);

  std::vector<std::string> get_shard_addresses() const { return shard_addresses_; }

  int get_shard_count() const { return shard_count_; }

  int get_schema_count() const { return schema_.size(); }

  uint64_t get_header_size() const { return header_size_; }

  uint64_t get_page_size() const { return page_size_; }

  void set_header_size(const uint64_t &header_size) { header_size_ = header_size; }

  void set_page_size(const uint64_t &page_size) { page_size_ = page_size; }

  const string get_version() { return version_; }

  std::vector<std::string> SerializeHeader();

  MSRStatus PagesToFile(const std::string dump_file_name);

  MSRStatus FileToPages(const std::string dump_file_name);

 private:
  MSRStatus InitializeHeader(const std::vector<json> &headers);

  /// \brief get the headers from all the shard data
  /// \param[in] the shard data real path
  /// \param[in] the headers which readed from the shard data
  /// \return SUCCESS/FAILED
  MSRStatus get_headers(const vector<string> &real_addresses, std::vector<json> &headers);

  MSRStatus ValidateField(const std::vector<std::string> &field_name, json schema, const uint64_t &schema_id);

  /// \brief check the binary file status
  MSRStatus CheckFileStatus(const std::string &path);

  std::pair<MSRStatus, json> ValidateHeader(const std::string &path);

  void ParseHeader(const json &header);

  void GetHeadersOneTask(int start, int end, std::vector<json> &headers, const vector<string> &realAddresses);

  MSRStatus ParseIndexFields(const json &index_fields);

  MSRStatus CheckIndexField(const std::string &field, const json &schema);

  void ParsePage(const json &page);

  MSRStatus ParseStatistics(const json &statistics);

  MSRStatus ParseSchema(const json &schema);

  void ParseShardAddress(const json &address);

  std::string SerializeIndexFields();

  std::vector<std::string> SerializePage();

  std::string SerializeStatistics();

  std::string SerializeSchema();

  std::string SerializeShardAddress();

  std::shared_ptr<Index> InitIndexPtr();

  MSRStatus GetAllSchemaID(std::set<uint64_t> &bucket_count);

  uint32_t shard_count_;
  uint64_t header_size_;
  uint64_t page_size_;
  string version_ = "2.0";

  std::shared_ptr<Index> index_;
  std::vector<std::string> shard_addresses_;
  std::vector<std::shared_ptr<Schema>> schema_;
  std::vector<std::shared_ptr<Statistics>> statistics_;
  std::vector<std::vector<std::shared_ptr<Page>>> pages_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_HEADER_H_
