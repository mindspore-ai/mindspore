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

#include "mindrecord/include/shard_header.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/utils.h"
#include "mindrecord/include/shard_error.h"
#include "mindrecord/include/shard_page.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
std::atomic<bool> thread_status(false);
ShardHeader::ShardHeader() : shard_count_(0), header_size_(0), page_size_(0) { index_ = std::make_shared<Index>(); }

MSRStatus ShardHeader::InitializeHeader(const std::vector<json> &headers, bool load_dataset) {
  shard_count_ = headers.size();
  int shard_index = 0;
  bool first = true;
  for (const auto &header : headers) {
    if (first) {
      first = false;
      if (ParseSchema(header["schema"]) != SUCCESS) {
        return FAILED;
      }
      if (ParseIndexFields(header["index_fields"]) != SUCCESS) {
        return FAILED;
      }
      if (ParseStatistics(header["statistics"]) != SUCCESS) {
        return FAILED;
      }
      ParseShardAddress(header["shard_addresses"]);
      header_size_ = header["header_size"].get<uint64_t>();
      page_size_ = header["page_size"].get<uint64_t>();
    }
    ParsePage(header["page"], shard_index, load_dataset);
    shard_index++;
  }
  return SUCCESS;
}

MSRStatus ShardHeader::CheckFileStatus(const std::string &path) {
  std::ifstream fin(common::SafeCStr(path), std::ios::in | std::ios::binary);
  if (!fin) {
    MS_LOG(ERROR) << "File does not exist or permission denied. path: " << path;
    return FAILED;
  }
  if (fin.fail()) {
    MS_LOG(ERROR) << "Failed to open file. path: " << path;
    return FAILED;
  }

  // fetch file size
  auto &io_seekg = fin.seekg(0, std::ios::end);
  if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
    fin.close();
    MS_LOG(ERROR) << "File seekg failed";
    return FAILED;
  }

  size_t file_size = fin.tellg();
  if (file_size < kMinFileSize) {
    fin.close();
    MS_LOG(ERROR) << "File size %d is smaller than the minimum value.";
    return FAILED;
  }
  fin.close();
  return SUCCESS;
}

std::pair<MSRStatus, json> ShardHeader::ValidateHeader(const std::string &path) {
  if (CheckFileStatus(path) != SUCCESS) {
    return {FAILED, {}};
  }

  // read header size
  json json_header;
  std::ifstream fin(common::SafeCStr(path), std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    MS_LOG(ERROR) << "File seekg failed";
    return {FAILED, json_header};
  }

  uint64_t header_size = 0;
  auto &io_read = fin.read(reinterpret_cast<char *>(&header_size), kInt64Len);
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    MS_LOG(ERROR) << "File read failed";
    fin.close();
    return {FAILED, json_header};
  }

  if (header_size > kMaxHeaderSize) {
    fin.close();
    MS_LOG(ERROR) << "Header size is illegal.";
    return {FAILED, json_header};
  }

  // read header content
  std::vector<uint8_t> header_content(header_size);
  auto &io_read_content = fin.read(reinterpret_cast<char *>(&header_content[0]), header_size);
  if (!io_read_content.good() || io_read_content.fail() || io_read_content.bad()) {
    MS_LOG(ERROR) << "File read failed";
    fin.close();
    return {FAILED, json_header};
  }

  fin.close();
  std::string raw_header_content = std::string(header_content.begin(), header_content.end());
  // parse json content
  try {
    json_header = json::parse(raw_header_content);
  } catch (json::parse_error &e) {
    MS_LOG(ERROR) << "Json parse error: " << e.what();
    return {FAILED, json_header};
  }
  return {SUCCESS, json_header};
}

std::pair<MSRStatus, json> ShardHeader::BuildSingleHeader(const std::string &file_path) {
  auto ret = ValidateHeader(file_path);
  if (SUCCESS != ret.first) {
    return {FAILED, json()};
  }
  json raw_header = ret.second;
  json header = {{"shard_addresses", raw_header["shard_addresses"]},
                 {"header_size", raw_header["header_size"]},
                 {"page_size", raw_header["page_size"]},
                 {"index_fields", raw_header["index_fields"]},
                 {"blob_fields", raw_header["schema"][0]["blob_fields"]},
                 {"schema", raw_header["schema"][0]["schema"]},
                 {"version", raw_header["version"]}};
  return {SUCCESS, header};
}

MSRStatus ShardHeader::BuildDataset(const std::vector<std::string> &file_paths, bool load_dataset) {
  uint32_t thread_num = std::thread::hardware_concurrency();
  if (thread_num == 0) thread_num = kThreadNumber;
  uint32_t work_thread_num = 0;
  uint32_t shard_count = file_paths.size();
  int group_num = ceil(shard_count * 1.0 / thread_num);
  std::vector<std::thread> thread_set(thread_num);
  std::vector<json> headers(shard_count);
  for (uint32_t x = 0; x < thread_num; ++x) {
    int start_num = x * group_num;
    int end_num = ((x + 1) * group_num > shard_count) ? shard_count : (x + 1) * group_num;
    if (start_num >= end_num) {
      continue;
    }

    thread_set[x] =
      std::thread(&ShardHeader::GetHeadersOneTask, this, start_num, end_num, std::ref(headers), file_paths);
    work_thread_num++;
  }

  for (uint32_t x = 0; x < work_thread_num; ++x) {
    thread_set[x].join();
  }
  if (thread_status) {
    thread_status = false;
    return FAILED;
  }
  if (SUCCESS != InitializeHeader(headers, load_dataset)) {
    return FAILED;
  }
  return SUCCESS;
}

void ShardHeader::GetHeadersOneTask(int start, int end, std::vector<json> &headers,
                                    const vector<string> &realAddresses) {
  if (thread_status || end > realAddresses.size()) {
    return;
  }
  for (int x = start; x < end; ++x) {
    auto ret = ValidateHeader(realAddresses[x]);
    if (SUCCESS != ret.first) {
      thread_status = true;
      return;
    }
    json header;
    header = ret.second;
    header["shard_addresses"] = realAddresses;
    if (std::find(kSupportedVersion.begin(), kSupportedVersion.end(), header["version"]) == kSupportedVersion.end()) {
      MS_LOG(ERROR) << "Version wrong, file version is: " << header["version"].dump()
                    << ", lib version is: " << kVersion;
      thread_status = true;
      return;
    }
    headers[x] = header;
  }
}

MSRStatus ShardHeader::InitByFiles(const std::vector<std::string> &file_paths) {
  std::vector<std::string> file_names(file_paths.size());
  std::transform(file_paths.begin(), file_paths.end(), file_names.begin(), [](std::string fp) -> std::string {
    if (GetFileName(fp).first == SUCCESS) {
      return GetFileName(fp).second;
    }
  });

  shard_addresses_ = std::move(file_names);
  shard_count_ = file_paths.size();
  if (shard_count_ == 0) {
    return FAILED;
  }
  if (shard_count_ <= kMaxShardCount) {
    pages_.resize(shard_count_);
  } else {
    return FAILED;
  }
  return SUCCESS;
}

void ShardHeader::ParseHeader(const json &header) {}

MSRStatus ShardHeader::ParseIndexFields(const json &index_fields) {
  std::vector<std::pair<uint64_t, std::string>> parsed_index_fields;
  for (auto &index_field : index_fields) {
    auto schema_id = index_field["schema_id"].get<uint64_t>();
    std::string field_name = index_field["index_field"].get<std::string>();
    std::pair<uint64_t, std::string> parsed_index_field(schema_id, field_name);
    parsed_index_fields.push_back(parsed_index_field);
  }
  if (!parsed_index_fields.empty() && AddIndexFields(parsed_index_fields) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

void ShardHeader::ParsePage(const json &pages, int shard_index, bool load_dataset) {
  // set shard_index when load_dataset is false
  if (pages_.empty() && shard_count_ <= kMaxShardCount) {
    pages_.resize(shard_count_);
  }
  for (auto &page : pages) {
    int page_id = page["page_id"];
    int shard_id = page["shard_id"];
    std::string page_type = page["page_type"];
    int page_type_id = page["page_type_id"];
    auto start_row_id = page["start_row_id"].get<uint64_t>();
    auto end_row_id = page["end_row_id"].get<uint64_t>();

    std::vector<std::pair<int, uint64_t>> row_group_ids(page["row_group_ids"].size());
    std::transform(page["row_group_ids"].begin(), page["row_group_ids"].end(), row_group_ids.begin(),
                   [](json rg) { return std::make_pair(rg["id"], rg["offset"].get<uint64_t>()); });

    auto page_size = page["page_size"].get<uint64_t>();

    std::shared_ptr<Page> parsed_page = std::make_shared<Page>(page_id, shard_id, page_type, page_type_id, start_row_id,
                                                               end_row_id, row_group_ids, page_size);
    if (load_dataset == true) {
      pages_[shard_id].push_back(std::move(parsed_page));
    } else {
      pages_[shard_index].push_back(std::move(parsed_page));
    }
  }
}

MSRStatus ShardHeader::ParseStatistics(const json &statistics) {
  for (auto &statistic : statistics) {
    if (statistic.find("desc") == statistic.end() || statistic.find("statistics") == statistic.end()) {
      MS_LOG(ERROR) << "Deserialize statistics failed, statistic: " << statistics.dump();
      return FAILED;
    }
    std::string statistic_description = statistic["desc"].get<std::string>();
    json statistic_body = statistic["statistics"];
    std::shared_ptr<Statistics> parsed_statistic = Statistics::Build(statistic_description, statistic_body);
    if (!parsed_statistic) {
      return FAILED;
    }
    AddStatistic(parsed_statistic);
  }
  return SUCCESS;
}

MSRStatus ShardHeader::ParseSchema(const json &schemas) {
  for (auto &schema : schemas) {
    // change how we get schemaBody once design is finalized
    if (schema.find("desc") == schema.end() || schema.find("blob_fields") == schema.end() ||
        schema.find("schema") == schema.end()) {
      MS_LOG(ERROR) << "Deserialize schema failed. schema: " << schema.dump();
      return FAILED;
    }
    std::string schema_description = schema["desc"].get<std::string>();
    std::vector<std::string> blob_fields = schema["blob_fields"].get<std::vector<std::string>>();
    json schema_body = schema["schema"];
    std::shared_ptr<Schema> parsed_schema = Schema::Build(schema_description, schema_body);
    if (!parsed_schema) {
      return FAILED;
    }
    AddSchema(parsed_schema);
  }
  return SUCCESS;
}

void ShardHeader::ParseShardAddress(const json &address) {
  std::copy(address.begin(), address.end(), std::back_inserter(shard_addresses_));
}

std::vector<std::string> ShardHeader::SerializeHeader() {
  std::vector<string> header;
  auto index = SerializeIndexFields();
  auto stats = SerializeStatistics();
  auto schema = SerializeSchema();
  auto pages = SerializePage();
  auto address = SerializeShardAddress();
  if (shard_count_ > static_cast<int>(pages.size())) {
    return std::vector<string>{};
  }
  if (shard_count_ <= kMaxShardCount) {
    for (int shardId = 0; shardId < shard_count_; shardId++) {
      string s;
      s += "{\"header_size\":" + std::to_string(header_size_) + ",";
      s += "\"index_fields\":" + index + ",";
      s += "\"page\":" + pages[shardId] + ",";
      s += "\"page_size\":" + std::to_string(page_size_) + ",";
      s += "\"schema\":" + schema + ",";
      s += "\"shard_addresses\":" + address + ",";
      s += "\"shard_id\":" + std::to_string(shardId) + ",";
      s += "\"statistics\":" + stats + ",";
      s += "\"version\":\"" + std::string(kVersion) + "\"";
      s += "}";
      header.emplace_back(s);
    }
  }
  return header;
}

std::string ShardHeader::SerializeIndexFields() {
  json j;
  auto fields = index_->GetFields();
  for (const auto &field : fields) {
    j.push_back({{"schema_id", field.first}, {"index_field", field.second}});
  }
  return j.dump();
}

std::vector<std::string> ShardHeader::SerializePage() {
  std::vector<string> pages;
  for (auto &shard_pages : pages_) {
    json j;
    for (const auto &p : shard_pages) {
      j.emplace_back(p->GetPage());
    }
    pages.emplace_back(j.dump());
  }
  return pages;
}

std::string ShardHeader::SerializeStatistics() {
  json j;
  for (const auto &stats : statistics_) {
    j.emplace_back(stats->GetStatistics());
  }
  return j.dump();
}

std::string ShardHeader::SerializeSchema() {
  json j;
  for (const auto &schema : schema_) {
    j.emplace_back(schema->GetSchema());
  }
  return j.dump();
}

std::string ShardHeader::SerializeShardAddress() {
  json j;
  for (const auto &addr : shard_addresses_) {
    j.emplace_back(GetFileName(addr).second);
  }
  return j.dump();
}

std::pair<std::shared_ptr<Page>, MSRStatus> ShardHeader::GetPage(const int &shard_id, const int &page_id) {
  if (shard_id < static_cast<int>(pages_.size()) && page_id < static_cast<int>(pages_[shard_id].size())) {
    return std::make_pair(pages_[shard_id][page_id], SUCCESS);
  } else {
    return std::make_pair(nullptr, FAILED);
  }
}

MSRStatus ShardHeader::SetPage(const std::shared_ptr<Page> &new_page) {
  if (new_page == nullptr) {
    return FAILED;
  }
  int shard_id = new_page->GetShardID();
  int page_id = new_page->GetPageID();
  if (shard_id < static_cast<int>(pages_.size()) && page_id < static_cast<int>(pages_[shard_id].size())) {
    pages_[shard_id][page_id] = new_page;
    return SUCCESS;
  } else {
    return FAILED;
  }
}

MSRStatus ShardHeader::AddPage(const std::shared_ptr<Page> &new_page) {
  if (new_page == nullptr) {
    return FAILED;
  }
  int shard_id = new_page->GetShardID();
  int page_id = new_page->GetPageID();
  if (shard_id < static_cast<int>(pages_.size()) && page_id == static_cast<int>(pages_[shard_id].size())) {
    pages_[shard_id].push_back(new_page);
    return SUCCESS;
  } else {
    return FAILED;
  }
}

int64_t ShardHeader::GetLastPageId(const int &shard_id) {
  if (shard_id >= static_cast<int>(pages_.size())) {
    return 0;
  }
  return pages_[shard_id].size() - 1;
}

int ShardHeader::GetLastPageIdByType(const int &shard_id, const std::string &page_type) {
  if (shard_id >= static_cast<int>(pages_.size())) {
    return 0;
  }
  int last_page_id = -1;
  for (uint64_t i = pages_[shard_id].size(); i >= 1; i--) {
    if (pages_[shard_id][i - 1]->GetPageType() == page_type) {
      last_page_id = pages_[shard_id][i - 1]->GetPageID();
      return last_page_id;
    }
  }
  return last_page_id;
}

const std::pair<MSRStatus, std::shared_ptr<Page>> ShardHeader::GetPageByGroupId(const int &group_id,
                                                                                const int &shard_id) {
  if (shard_id >= static_cast<int>(pages_.size())) {
    MS_LOG(ERROR) << "Shard id is more than sum of shards.";
    return {FAILED, nullptr};
  }
  for (uint64_t i = pages_[shard_id].size(); i >= 1; i--) {
    auto page = pages_[shard_id][i - 1];
    if (page->GetPageType() == kPageTypeBlob && page->GetPageTypeID() == group_id) {
      return {SUCCESS, page};
    }
  }
  MS_LOG(ERROR) << "Could not get page by group id " << group_id;
  return {FAILED, nullptr};
}

int ShardHeader::AddSchema(std::shared_ptr<Schema> schema) {
  if (schema == nullptr) {
    MS_LOG(ERROR) << "Schema is illegal";
    return -1;
  }

  if (!schema_.empty()) {
    MS_LOG(ERROR) << "Only support one schema";
    return -1;
  }

  int64_t schema_id = schema->GetSchemaID();
  if (schema_id == -1) {
    schema_id = schema_.size();
    schema->SetSchemaID(schema_id);
  }
  schema_.push_back(schema);
  return schema_id;
}

void ShardHeader::AddStatistic(std::shared_ptr<Statistics> statistic) {
  if (statistic) {
    int64_t statistics_id = statistic->GetStatisticsID();
    if (statistics_id == -1) {
      statistics_id = statistics_.size();
      statistic->SetStatisticsID(statistics_id);
    }
    statistics_.push_back(statistic);
  }
}

std::shared_ptr<Index> ShardHeader::InitIndexPtr() {
  std::shared_ptr<Index> index = index_;
  if (!index_) {
    index = std::make_shared<Index>();
    index_ = index;
  }
  return index;
}

MSRStatus ShardHeader::CheckIndexField(const std::string &field, const json &schema) {
  // check field name is or is not valid
  if (schema.find(field) == schema.end()) {
    MS_LOG(ERROR) << "Schema do not contain the field: " << field << ".";
    return FAILED;
  }

  if (schema[field]["type"] == "bytes") {
    MS_LOG(ERROR) << field << " is bytes type, can not be schema index field.";
    return FAILED;
  }

  if (schema.find(field) != schema.end() && schema[field].find("shape") != schema[field].end()) {
    MS_LOG(ERROR) << field << " array can not be schema index field.";
    return FAILED;
  }
  return SUCCESS;
}

MSRStatus ShardHeader::AddIndexFields(const std::vector<std::string> &fields) {
  // create index Object
  std::shared_ptr<Index> index = InitIndexPtr();

  if (fields.size() == kInt0) {
    MS_LOG(ERROR) << "There are no index fields";
    return FAILED;
  }

  if (GetSchemas().empty()) {
    MS_LOG(ERROR) << "No schema is set";
    return FAILED;
  }

  for (const auto &schemaPtr : schema_) {
    auto result = GetSchemaByID(schemaPtr->GetSchemaID());
    if (result.second != SUCCESS) {
      MS_LOG(ERROR) << "Could not get schema by id.";
      return FAILED;
    }

    if (result.first == nullptr) {
      MS_LOG(ERROR) << "Could not get schema by id.";
      return FAILED;
    }

    json schema = result.first->GetSchema().at("schema");

    // checkout and add fields for each schema
    std::set<std::string> field_set;
    for (const auto &item : index->GetFields()) {
      field_set.insert(item.second);
    }
    for (const auto &field : fields) {
      if (field_set.find(field) != field_set.end()) {
        MS_LOG(ERROR) << "Add same index field twice";
        return FAILED;
      }

      // check field name is or is not valid
      if (CheckIndexField(field, schema) == FAILED) {
        return FAILED;
      }
      field_set.insert(field);

      // add field into index
      index.get()->AddIndexField(schemaPtr->GetSchemaID(), field);
    }
  }

  index_ = index;
  return SUCCESS;
}

MSRStatus ShardHeader::GetAllSchemaID(std::set<uint64_t> &bucket_count) {
  // get all schema id
  for (const auto &schema : schema_) {
    auto bucket_it = bucket_count.find(schema->GetSchemaID());
    if (bucket_it != bucket_count.end()) {
      MS_LOG(ERROR) << "Schema duplication";
      return FAILED;
    } else {
      bucket_count.insert(schema->GetSchemaID());
    }
  }
  return SUCCESS;
}

MSRStatus ShardHeader::AddIndexFields(std::vector<std::pair<uint64_t, std::string>> fields) {
  // create index Object
  std::shared_ptr<Index> index = InitIndexPtr();

  if (fields.size() == kInt0) {
    MS_LOG(ERROR) << "There are no index fields";
    return FAILED;
  }

  // get all schema id
  std::set<uint64_t> bucket_count;
  if (GetAllSchemaID(bucket_count) != SUCCESS) {
    return FAILED;
  }

  // check and add fields for each schema
  std::set<std::pair<uint64_t, std::string>> field_set;
  for (const auto &item : index->GetFields()) {
    field_set.insert(item);
  }
  for (const auto &field : fields) {
    if (field_set.find(field) != field_set.end()) {
      MS_LOG(ERROR) << "Add same index field twice";
      return FAILED;
    }

    uint64_t schema_id = field.first;
    std::string field_name = field.second;

    // check schemaId is or is not valid
    if (bucket_count.find(schema_id) == bucket_count.end()) {
      MS_LOG(ERROR) << "Illegal schema id: " << schema_id;
      return FAILED;
    }

    // check field name is or is not valid
    auto result = GetSchemaByID(schema_id);
    if (result.second != SUCCESS) {
      MS_LOG(ERROR) << "Could not get schema by id.";
      return FAILED;
    }
    json schema = result.first->GetSchema().at("schema");
    if (schema.find(field_name) == schema.end()) {
      MS_LOG(ERROR) << "Schema " << schema_id << " do not contain the field: " << field_name;
      return FAILED;
    }

    if (CheckIndexField(field_name, schema) == FAILED) {
      return FAILED;
    }

    field_set.insert(field);

    // add field into index
    index.get()->AddIndexField(schema_id, field_name);
  }
  index_ = index;
  return SUCCESS;
}

std::string ShardHeader::GetShardAddressByID(int64_t shard_id) {
  if (shard_id >= shard_addresses_.size()) {
    return "";
  }
  return shard_addresses_.at(shard_id);
}

std::vector<std::shared_ptr<Schema>> ShardHeader::GetSchemas() { return schema_; }

std::vector<std::shared_ptr<Statistics>> ShardHeader::GetStatistics() { return statistics_; }

std::vector<std::pair<uint64_t, std::string>> ShardHeader::GetFields() { return index_->GetFields(); }

std::shared_ptr<Index> ShardHeader::GetIndex() { return index_; }

std::pair<std::shared_ptr<Schema>, MSRStatus> ShardHeader::GetSchemaByID(int64_t schema_id) {
  int64_t schemaSize = schema_.size();
  if (schema_id < 0 || schema_id >= schemaSize) {
    MS_LOG(ERROR) << "Illegal schema id";
    return std::make_pair(nullptr, FAILED);
  }
  return std::make_pair(schema_.at(schema_id), SUCCESS);
}

std::pair<std::shared_ptr<Statistics>, MSRStatus> ShardHeader::GetStatisticByID(int64_t statistic_id) {
  int64_t statistics_size = statistics_.size();
  if (statistic_id < 0 || statistic_id >= statistics_size) {
    return std::make_pair(nullptr, FAILED);
  }
  return std::make_pair(statistics_.at(statistic_id), SUCCESS);
}

MSRStatus ShardHeader::PagesToFile(const std::string dump_file_name) {
  // write header content to file, dump whatever is in the file before
  std::ofstream page_out_handle(dump_file_name.c_str(), std::ios_base::trunc | std::ios_base::out);
  if (page_out_handle.fail()) {
    MS_LOG(ERROR) << "Failed in opening page file";
    return FAILED;
  }

  auto pages = SerializePage();
  for (const auto &shard_pages : pages) {
    page_out_handle << shard_pages << "\n";
  }

  page_out_handle.close();
  return SUCCESS;
}

MSRStatus ShardHeader::FileToPages(const std::string dump_file_name) {
  for (auto &v : pages_) {  // clean pages
    v.clear();
  }
  // attempt to open the file contains the page in json
  std::ifstream page_in_handle(dump_file_name.c_str());

  if (!page_in_handle.good()) {
    MS_LOG(INFO) << "No page file exists.";
    return SUCCESS;
  }

  std::string line;
  while (std::getline(page_in_handle, line)) {
    ParsePage(json::parse(line), -1, true);
  }

  page_in_handle.close();
  return SUCCESS;
}
}  // namespace mindrecord
}  // namespace mindspore
