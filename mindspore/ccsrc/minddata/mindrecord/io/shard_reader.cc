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

#include <algorithm>
#include <thread>

#include "minddata/mindrecord/include/shard_distributed_sample.h"
#include "minddata/mindrecord/include/shard_reader.h"
#include "utils/ms_utils.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::DEBUG;
using mindspore::MsLogLevel::ERROR;
using mindspore::MsLogLevel::INFO;

namespace mindspore {
namespace mindrecord {
template <class Type>
// convert the string to exactly number type (int32_t/int64_t/float/double)
Type StringToNum(const std::string &str) {
  std::istringstream iss(str);
  Type num;
  iss >> num;
  return num;
}

ShardReader::ShardReader()
    : header_size_(0),
      page_size_(0),
      shard_count_(0),
      n_consumer_(0),
      num_padded_(0),
      num_rows_(0),
      total_blob_size_(0),
      task_id_(0),
      deliver_id_(0),
      lazy_load_(false),
      shard_sample_count_() {}

std::pair<MSRStatus, std::vector<std::string>> ShardReader::GetMeta(const std::string &file_path, json &meta_data) {
  if (!IsLegalFile(file_path)) {
    return {FAILED, {}};
  }
  auto ret = ShardHeader::BuildSingleHeader(file_path);
  if (ret.first != SUCCESS) {
    return {FAILED, {}};
  }
  auto header = ret.second;
  meta_data = {{"header_size", header["header_size"]}, {"page_size", header["page_size"]},
               {"version", header["version"]},         {"index_fields", header["index_fields"]},
               {"schema", header["schema"]},           {"blob_fields", header["blob_fields"]}};
  return {SUCCESS, header["shard_addresses"]};
}

MSRStatus ShardReader::Init(const std::vector<std::string> &file_paths, bool load_dataset) {
  std::string file_path = file_paths[0];
  json first_meta_data = json();
  auto ret = GetMeta(file_path, first_meta_data);
  if (ret.first != SUCCESS) {
    return FAILED;
  }
  if (file_paths.size() == 1 && load_dataset == true) {
    auto ret2 = GetParentDir(file_path);
    if (SUCCESS != ret2.first) {
      return FAILED;
    }
    std::vector<std::string> real_addresses;
    for (const auto &path : ret.second) {
      std::string abs_path = ret2.second + string(path);
      real_addresses.emplace_back(abs_path);
    }
    file_paths_ = real_addresses;
  } else if (file_paths.size() >= 1 && load_dataset == false) {
    file_paths_ = file_paths;
  } else {
    MS_LOG(ERROR) << "Error in parameter file_path or load_dataset.";
    return FAILED;
  }
  for (const auto &file : file_paths_) {
    json meta_data = json();
    auto ret1 = GetMeta(file, meta_data);
    if (ret1.first != SUCCESS) {
      return FAILED;
    }
    if (meta_data != first_meta_data) {
      MS_LOG(ERROR) << "Mindrecord files meta information is different.";
      return FAILED;
    }
    sqlite3 *db = nullptr;
    // sqlite3_open create a database if not found, use sqlite3_open_v2 instead of it
    int rc = sqlite3_open_v2(common::SafeCStr(file + ".db"), &db, SQLITE_OPEN_READONLY, nullptr);
    if (rc != SQLITE_OK) {
      MS_LOG(ERROR) << "Invalid file, failed to open database: " << file + ".db, error: " << sqlite3_errmsg(db);
      return FAILED;
    }
    MS_LOG(DEBUG) << "Opened database successfully";

    string sql = "select NAME from SHARD_NAME;";
    std::vector<std::vector<std::string>> name;
    char *errmsg = nullptr;
    rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &name, &errmsg);
    if (rc != SQLITE_OK) {
      MS_LOG(ERROR) << "Error in select statement, sql: " << sql << ", error: " << errmsg;
      sqlite3_free(errmsg);
      sqlite3_close(db);
      db = nullptr;
      return FAILED;
    } else {
      MS_LOG(DEBUG) << "Get " << static_cast<int>(name.size()) << " records from index.";
      string shardName = GetFileName(file).second;
      if (name.empty() || name[0][0] != shardName) {
        MS_LOG(ERROR) << "Invalid file, DB file can not match file: " << file;
        sqlite3_free(errmsg);
        sqlite3_close(db);
        db = nullptr;
        return FAILED;
      }
    }
    database_paths_.push_back(db);
  }
  ShardHeader sh = ShardHeader();
  if (sh.BuildDataset(file_paths_, load_dataset) == FAILED) {
    return FAILED;
  }
  shard_header_ = std::make_shared<ShardHeader>(sh);
  header_size_ = shard_header_->GetHeaderSize();
  page_size_ = shard_header_->GetPageSize();
  // version < 3.0
  if (first_meta_data["version"] < kVersion) {
    shard_column_ = std::make_shared<ShardColumn>(shard_header_, false);
  } else {
    shard_column_ = std::make_shared<ShardColumn>(shard_header_, true);
  }
  num_rows_ = 0;
  auto row_group_summary = ReadRowGroupSummary();
  for (const auto &rg : row_group_summary) {
    num_rows_ += std::get<3>(rg);
  }

  if (num_rows_ > LAZY_LOAD_THRESHOLD) {
    lazy_load_ = true;
    MS_LOG(WARNING) << "The number of samples is larger than " << LAZY_LOAD_THRESHOLD
                    << ", enable lazy load mode. If you want to speed up data loading, "
                    << "it is recommended that you save multiple samples into one record when creating mindrecord file,"
                    << " so that you can enable fast loading mode, and don't forget to adjust your batch size "
                    << "according to the current samples.";
  }

  auto disk_size = page_size_ * row_group_summary.size();
  auto compression_size = shard_header_->GetCompressionSize();
  total_blob_size_ = disk_size + compression_size;
  MS_LOG(INFO) << "Blob data size, on disk: " << disk_size << " , additional uncompression: " << compression_size
               << " , Total: " << total_blob_size_;

  MS_LOG(INFO) << "Get meta from mindrecord file & index file successfully.";

  return SUCCESS;
}

MSRStatus ShardReader::CheckColumnList(const std::vector<std::string> &selected_columns) {
  vector<int> inSchema(selected_columns.size(), 0);
  for (auto &p : GetShardHeader()->GetSchemas()) {
    auto schema = p->GetSchema()["schema"];
    for (unsigned int i = 0; i < selected_columns.size(); ++i) {
      if (schema.find(selected_columns[i]) != schema.end()) {
        inSchema[i] = 1;
      }
    }
  }
  if (std::any_of(std::begin(inSchema), std::end(inSchema), [](int x) { return x == 0; })) {
    return FAILED;
  }

  return SUCCESS;
}

MSRStatus ShardReader::Open() {
  file_streams_.clear();

  for (const auto &file : file_paths_) {
    std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
    fs->open(common::SafeCStr(file), std::ios::in | std::ios::binary);
    if (!fs->good()) {
      MS_LOG(ERROR) << "Invalid file, failed to open file: " << file;
      return FAILED;
    }
    MS_LOG(INFO) << "Open shard file successfully.";
    file_streams_.push_back(fs);
  }

  return SUCCESS;
}

MSRStatus ShardReader::Open(int n_consumer) {
  file_streams_random_ =
    std::vector<std::vector<std::shared_ptr<std::fstream>>>(n_consumer, std::vector<std::shared_ptr<std::fstream>>());
  for (const auto &file : file_paths_) {
    for (int j = 0; j < n_consumer; ++j) {
      std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
      fs->open(common::SafeCStr(file), std::ios::in | std::ios::binary);
      if (!fs->good()) {
        MS_LOG(ERROR) << "Invalid file, failed to open file: " << file;
        return FAILED;
      }
      file_streams_random_[j].push_back(fs);
    }
    MS_LOG(INFO) << "Open shard file successfully.";
  }

  return SUCCESS;
}

void ShardReader::FileStreamsOperator() {
  for (int i = static_cast<int>(file_streams_.size()) - 1; i >= 0; --i) {
    if (file_streams_[i] != nullptr) {
      file_streams_[i]->close();
    }
  }
  for (int i = static_cast<int>(file_streams_random_.size()) - 1; i >= 0; --i) {
    for (int j = static_cast<int>(file_streams_random_[i].size()) - 1; j >= 0; --j) {
      if (file_streams_random_[i][j] != nullptr) {
        file_streams_random_[i][j]->close();
      }
    }
  }
  for (int i = static_cast<int>(database_paths_.size()) - 1; i >= 0; --i) {
    if (database_paths_[i] != nullptr) {
      auto ret = sqlite3_close(database_paths_[i]);
      if (ret != SQLITE_OK) {
        MS_LOG(ERROR) << "Close db failed. Error code: " << ret << ".";
      }
      database_paths_[i] = nullptr;
    }
  }
}

ShardReader::~ShardReader() { Close(); }

void ShardReader::Close() {
  {
    std::lock_guard<std::mutex> lck(mtx_delivery_);
    interrupt_ = true;  // interrupt reading and stop threads
  }
  cv_delivery_.notify_all();

  // Wait for all threads to finish
  for (auto &i_thread : thread_set_) {
    if (i_thread.joinable()) {
      i_thread.join();
    }
  }

  FileStreamsOperator();
}

std::shared_ptr<ShardHeader> ShardReader::GetShardHeader() const { return shard_header_; }

std::shared_ptr<ShardColumn> ShardReader::GetShardColumn() const { return shard_column_; }

int ShardReader::GetShardCount() const { return shard_header_->GetShardCount(); }

int ShardReader::GetNumRows() const { return num_rows_; }

std::vector<std::tuple<int, int, int, uint64_t>> ShardReader::ReadRowGroupSummary() {
  std::vector<std::tuple<int, int, int, uint64_t>> row_group_summary;
  int shard_count = shard_header_->GetShardCount();
  if (shard_count <= 0) {
    return row_group_summary;
  }
  if (shard_count <= kMaxFileCount) {
    uint32_t total_count = 0;
    for (int shard_id = 0; shard_id < shard_count; ++shard_id) {
      // return -1 when page's size equals to 0.
      auto last_page_id = shard_header_->GetLastPageId(shard_id);
      if (static_cast<int>(last_page_id) == -1) {
        continue;
      }
      for (uint64_t page_id = 0; page_id <= last_page_id; ++page_id) {
        const auto &page_t = shard_header_->GetPage(shard_id, page_id);
        const auto &page = page_t.first;
        if (page->GetPageType() != kPageTypeBlob) continue;
        uint64_t start_row_id = page->GetStartRowID();
        if (start_row_id > page->GetEndRowID()) {
          return std::vector<std::tuple<int, int, int, uint64_t>>();
        }
        uint64_t number_of_rows = page->GetEndRowID() - start_row_id;
        total_count += number_of_rows;
        row_group_summary.emplace_back(shard_id, page->GetPageTypeID(), start_row_id, number_of_rows);
      }
      shard_sample_count_.push_back(total_count);
    }
  }
  return row_group_summary;
}

MSRStatus ShardReader::GetTotalBlobSize(int64_t *total_blob_size) {
  *total_blob_size = total_blob_size_;
  return SUCCESS;
}

MSRStatus ShardReader::ConvertLabelToJson(const std::vector<std::vector<std::string>> &labels,
                                          std::shared_ptr<std::fstream> fs,
                                          std::vector<std::vector<std::vector<uint64_t>>> &offsets, int shard_id,
                                          const std::vector<std::string> &columns,
                                          std::vector<std::vector<json>> &column_values) {
  for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
    uint64_t group_id = std::stoull(labels[i][0]);
    uint64_t offset_start = std::stoull(labels[i][1]) + kInt64Len;
    uint64_t offset_end = std::stoull(labels[i][2]);
    offsets[shard_id].emplace_back(
      std::vector<uint64_t>{static_cast<uint64_t>(shard_id), group_id, offset_start, offset_end});
    if (!all_in_index_) {
      int raw_page_id = std::stoi(labels[i][3]);
      uint64_t label_start = std::stoull(labels[i][4]) + kInt64Len;
      uint64_t label_end = std::stoull(labels[i][5]);
      auto len = label_end - label_start;
      auto label_raw = std::vector<uint8_t>(len);
      auto &io_seekg = fs->seekg(page_size_ * raw_page_id + header_size_ + label_start, std::ios::beg);
      if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
        MS_LOG(ERROR) << "File seekg failed";
        fs->close();
        return FAILED;
      }

      auto &io_read = fs->read(reinterpret_cast<char *>(&label_raw[0]), len);
      if (!io_read.good() || io_read.fail() || io_read.bad()) {
        MS_LOG(ERROR) << "File read failed";
        fs->close();
        return FAILED;
      }
      json label_json = json::from_msgpack(label_raw);
      json tmp;
      if (!columns.empty()) {
        for (auto &col : columns) {
          if (label_json.find(col) != label_json.end()) {
            tmp[col] = label_json[col];
          }
        }
      } else {
        tmp = label_json;
      }
      column_values[shard_id].emplace_back(tmp);
    } else {
      json construct_json;
      for (unsigned int j = 0; j < columns.size(); ++j) {
        // construct json "f1": value
        auto schema = shard_header_->GetSchemas()[0]->GetSchema()["schema"];

        // convert the string to base type by schema
        if (schema[columns[j]]["type"] == "int32") {
          construct_json[columns[j]] = StringToNum<int32_t>(labels[i][j + 3]);
        } else if (schema[columns[j]]["type"] == "int64") {
          construct_json[columns[j]] = StringToNum<int64_t>(labels[i][j + 3]);
        } else if (schema[columns[j]]["type"] == "float32") {
          construct_json[columns[j]] = StringToNum<float>(labels[i][j + 3]);
        } else if (schema[columns[j]]["type"] == "float64") {
          construct_json[columns[j]] = StringToNum<double>(labels[i][j + 3]);
        } else {
          construct_json[columns[j]] = std::string(labels[i][j + 3]);
        }
      }
      column_values[shard_id].emplace_back(construct_json);
    }
  }

  return SUCCESS;
}

MSRStatus ShardReader::ReadAllRowsInShard(int shard_id, const std::string &sql, const std::vector<std::string> &columns,
                                          std::vector<std::vector<std::vector<uint64_t>>> &offsets,
                                          std::vector<std::vector<json>> &column_values) {
  auto db = database_paths_[shard_id];
  std::vector<std::vector<std::string>> labels;
  char *errmsg = nullptr;
  int rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &labels, &errmsg);
  if (rc != SQLITE_OK) {
    MS_LOG(ERROR) << "Error in select statement, sql: " << sql << ", error: " << errmsg;
    sqlite3_free(errmsg);
    sqlite3_close(db);
    db = nullptr;
    return FAILED;
  }
  MS_LOG(INFO) << "Get " << static_cast<int>(labels.size()) << " records from shard " << shard_id << " index.";

  std::string file_name = file_paths_[shard_id];
  std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
  if (!all_in_index_) {
    fs->open(common::SafeCStr(file_name), std::ios::in | std::ios::binary);
    if (!fs->good()) {
      MS_LOG(ERROR) << "Invalid file, failed to open file: " << file_name;
      return FAILED;
    }
  }
  sqlite3_free(errmsg);
  return ConvertLabelToJson(labels, fs, offsets, shard_id, columns, column_values);
}

MSRStatus ShardReader::GetAllClasses(const std::string &category_field, std::set<std::string> &categories) {
  std::map<std::string, uint64_t> index_columns;
  for (auto &field : GetShardHeader()->GetFields()) {
    index_columns[field.second] = field.first;
  }
  if (index_columns.find(category_field) == index_columns.end()) {
    MS_LOG(ERROR) << "Index field " << category_field << " does not exist.";
    return FAILED;
  }
  auto ret = ShardIndexGenerator::GenerateFieldName(std::make_pair(index_columns[category_field], category_field));
  if (SUCCESS != ret.first) {
    return FAILED;
  }
  std::string sql = "SELECT DISTINCT " + ret.second + " FROM INDEXES";
  std::vector<std::thread> threads = std::vector<std::thread>(shard_count_);
  for (int x = 0; x < shard_count_; x++) {
    threads[x] = std::thread(&ShardReader::GetClassesInShard, this, database_paths_[x], x, sql, std::ref(categories));
  }

  for (int x = 0; x < shard_count_; x++) {
    threads[x].join();
  }
  return SUCCESS;
}

void ShardReader::GetClassesInShard(sqlite3 *db, int shard_id, const std::string sql,
                                    std::set<std::string> &categories) {
  if (nullptr == db) {
    return;
  }
  std::vector<std::vector<std::string>> columns;
  char *errmsg = nullptr;
  int ret = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &columns, &errmsg);
  if (ret != SQLITE_OK) {
    sqlite3_free(errmsg);
    sqlite3_close(db);
    db = nullptr;
    MS_LOG(ERROR) << "Error in select sql statement, sql: " << common::SafeCStr(sql) << ", error: " << errmsg;
    return;
  }
  MS_LOG(INFO) << "Get " << static_cast<int>(columns.size()) << " records from shard " << shard_id << " index.";
  std::lock_guard<std::mutex> lck(shard_locker_);
  for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
    categories.emplace(columns[i][0]);
  }
}

ROW_GROUPS ShardReader::ReadAllRowGroup(std::vector<std::string> &columns) {
  std::string fields = "ROW_GROUP_ID, PAGE_OFFSET_BLOB, PAGE_OFFSET_BLOB_END";
  std::vector<std::vector<std::vector<uint64_t>>> offsets(shard_count_, std::vector<std::vector<uint64_t>>{});
  std::vector<std::vector<json>> column_values(shard_count_, std::vector<json>{});
  if (all_in_index_) {
    for (unsigned int i = 0; i < columns.size(); ++i) {
      fields += ',';
      auto ret = ShardIndexGenerator::GenerateFieldName(std::make_pair(column_schema_id_[columns[i]], columns[i]));
      if (ret.first != SUCCESS) {
        return std::make_tuple(FAILED, std::move(offsets), std::move(column_values));
      }
      fields += ret.second;
    }
  } else {  // fetch raw data from Raw page while some field is not index.
    fields += ", PAGE_ID_RAW, PAGE_OFFSET_RAW, PAGE_OFFSET_RAW_END ";
  }

  std::string sql = "SELECT " + fields + " FROM INDEXES ORDER BY ROW_ID ;";

  std::vector<std::thread> thread_read_db = std::vector<std::thread>(shard_count_);
  for (int x = 0; x < shard_count_; x++) {
    thread_read_db[x] =
      std::thread(&ShardReader::ReadAllRowsInShard, this, x, sql, columns, std::ref(offsets), std::ref(column_values));
  }

  for (int x = 0; x < shard_count_; x++) {
    thread_read_db[x].join();
  }
  return std::make_tuple(SUCCESS, std::move(offsets), std::move(column_values));
}

ROW_GROUPS ShardReader::ReadRowGroupByShardIDAndSampleID(const std::vector<std::string> &columns,
                                                         const uint32_t &shard_id, const uint32_t &sample_id) {
  std::string fields = "ROW_GROUP_ID, PAGE_OFFSET_BLOB, PAGE_OFFSET_BLOB_END";
  std::vector<std::vector<std::vector<uint64_t>>> offsets(shard_count_, std::vector<std::vector<uint64_t>>{});
  std::vector<std::vector<json>> column_values(shard_count_, std::vector<json>{});
  if (all_in_index_) {
    for (unsigned int i = 0; i < columns.size(); ++i) {
      fields += ',';
      auto ret = ShardIndexGenerator::GenerateFieldName(std::make_pair(column_schema_id_[columns[i]], columns[i]));
      if (ret.first != SUCCESS) {
        return std::make_tuple(FAILED, std::move(offsets), std::move(column_values));
      }
      fields += ret.second;
    }
  } else {  // fetch raw data from Raw page while some field is not index.
    fields += ", PAGE_ID_RAW, PAGE_OFFSET_RAW, PAGE_OFFSET_RAW_END ";
  }

  std::string sql = "SELECT " + fields + " FROM INDEXES WHERE ROW_ID = " + std::to_string(sample_id);

  if (ReadAllRowsInShard(shard_id, sql, columns, offsets, column_values) != SUCCESS) {
    MS_LOG(ERROR) << "Read shard id: " << shard_id << ", sample id: " << sample_id << " from index failed.";
    return std::make_tuple(FAILED, std::move(offsets), std::move(column_values));
  }

  return std::make_tuple(SUCCESS, std::move(offsets), std::move(column_values));
}

ROW_GROUP_BRIEF ShardReader::ReadRowGroupBrief(int group_id, int shard_id, const std::vector<std::string> &columns) {
  const auto &ret = shard_header_->GetPageByGroupId(group_id, shard_id);
  if (SUCCESS != ret.first) {
    return std::make_tuple(FAILED, "", 0, 0, std::vector<std::vector<uint64_t>>(), std::vector<json>());
  }
  const std::shared_ptr<Page> &page = ret.second;
  std::string file_name = file_paths_[shard_id];
  uint64_t page_length = page->GetPageSize();
  uint64_t page_offset = page_size_ * page->GetPageID() + header_size_;
  std::vector<std::vector<uint64_t>> image_offset = GetImageOffset(page->GetPageID(), shard_id);

  auto status_labels = GetLabels(page->GetPageID(), shard_id, columns);
  if (status_labels.first != SUCCESS) {
    return std::make_tuple(FAILED, "", 0, 0, std::vector<std::vector<uint64_t>>(), std::vector<json>());
  }
  return std::make_tuple(SUCCESS, file_name, page_length, page_offset, std::move(image_offset),
                         std::move(status_labels.second));
}

ROW_GROUP_BRIEF ShardReader::ReadRowGroupCriteria(int group_id, int shard_id,
                                                  const std::pair<std::string, std::string> &criteria,
                                                  const std::vector<std::string> &columns) {
  const auto &ret = shard_header_->GetPageByGroupId(group_id, shard_id);
  if (SUCCESS != ret.first) {
    return std::make_tuple(FAILED, "", 0, 0, std::vector<std::vector<uint64_t>>(), std::vector<json>());
  }
  vector<string> criteria_list{criteria.first};
  if (CheckColumnList(criteria_list) == FAILED) {
    return std::make_tuple(FAILED, "", 0, 0, std::vector<std::vector<uint64_t>>(), std::vector<json>());
  }
  const std::shared_ptr<Page> &page = ret.second;
  std::string file_name = file_paths_[shard_id];
  uint64_t page_length = page->GetPageSize();
  uint64_t page_offset = page_size_ * page->GetPageID() + header_size_;
  std::vector<std::vector<uint64_t>> image_offset = GetImageOffset(page->GetPageID(), shard_id, criteria);

  auto status_labels = GetLabels(page->GetPageID(), shard_id, columns, criteria);
  if (status_labels.first != SUCCESS) {
    return std::make_tuple(FAILED, "", 0, 0, std::vector<std::vector<uint64_t>>(), std::vector<json>());
  }

  return std::make_tuple(SUCCESS, file_name, page_length, page_offset, std::move(image_offset),
                         std::move(status_labels.second));
}

int ShardReader::SelectCallback(void *p_data, int num_fields, char **p_fields, char **p_col_names) {
  auto *records = static_cast<std::vector<std::vector<std::string>> *>(p_data);
  if (num_fields > 0 && num_fields <= kMaxFieldCount) {
    for (int i = 0; i < num_fields; ++i)
      if (p_fields[i] == nullptr) p_fields[i] = const_cast<char *>("");
  }
  records->emplace_back(p_fields, p_fields + num_fields);
  return 0;
}

std::vector<std::vector<uint64_t>> ShardReader::GetImageOffset(int page_id, int shard_id,
                                                               const std::pair<std::string, std::string> &criteria) {
  auto db = database_paths_[shard_id];

  std::string sql =
    "SELECT PAGE_OFFSET_BLOB, PAGE_OFFSET_BLOB_END FROM INDEXES WHERE PAGE_ID_BLOB = " + std::to_string(page_id);

  // whether use index search
  if (!criteria.first.empty()) {
    auto schema = shard_header_->GetSchemas()[0]->GetSchema();

    // not number field should add '' in sql
    if (kNumberFieldTypeSet.find(schema["schema"][criteria.first]["type"]) != kNumberFieldTypeSet.end()) {
      sql +=
        " AND " + criteria.first + "_" + std::to_string(column_schema_id_[criteria.first]) + " = " + criteria.second;
    } else {
      sql += " AND " + criteria.first + "_" + std::to_string(column_schema_id_[criteria.first]) + " = '" +
             criteria.second + "'";
    }
  }
  sql += ";";
  std::vector<std::vector<std::string>> image_offsets;
  char *errmsg = nullptr;
  int rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &image_offsets, &errmsg);
  if (rc != SQLITE_OK) {
    MS_LOG(ERROR) << "Error in select statement, sql: " << sql << ", error: " << errmsg;
    sqlite3_free(errmsg);
    sqlite3_close(db);
    db = nullptr;
    return std::vector<std::vector<uint64_t>>();
  } else {
    MS_LOG(DEBUG) << "Get " << static_cast<int>(image_offsets.size()) << "records from index.";
  }
  std::vector<std::vector<uint64_t>> res;
  for (int i = static_cast<int>(image_offsets.size()) - 1; i >= 0; i--) res.emplace_back(std::vector<uint64_t>{0, 0});
  for (int i = 0; i < static_cast<int>(image_offsets.size()); i++) {
    const auto &image_offset = image_offsets[i];
    res[i][0] = std::stoull(image_offset[0]) + kInt64Len;
    res[i][1] = std::stoull(image_offset[1]);
  }
  sqlite3_free(errmsg);
  return res;
}

std::pair<ShardType, std::vector<std::string>> ShardReader::GetBlobFields() {
  std::vector<std::string> blob_fields;
  for (auto &p : GetShardHeader()->GetSchemas()) {
    // assume one schema
    const auto &fields = p->GetBlobFields();
    blob_fields.assign(fields.begin(), fields.end());
    break;
  }
  return std::make_pair(kCV, blob_fields);
}

void ShardReader::CheckIfColumnInIndex(const std::vector<std::string> &columns) {
  // assume different schemas do not contain same key.
  if (columns.empty()) {
    all_in_index_ = false;
    return;
  }
  for (auto &field : GetShardHeader()->GetFields()) {
    column_schema_id_[field.second] = field.first;
  }
  for (auto &col : columns) {
    if (column_schema_id_.find(col) == column_schema_id_.end()) {
      all_in_index_ = false;
      return;
    }
  }
}

MSRStatus ShardReader::QueryWithCriteria(sqlite3 *db, string &sql, string criteria,
                                         std::vector<std::vector<std::string>> &labels) {
  sqlite3_stmt *stmt = nullptr;
  if (sqlite3_prepare_v2(db, common::SafeCStr(sql), -1, &stmt, 0) != SQLITE_OK) {
    MS_LOG(ERROR) << "SQL error: could not prepare statement, sql: " << sql;
    return FAILED;
  }
  int index = sqlite3_bind_parameter_index(stmt, ":criteria");
  if (sqlite3_bind_text(stmt, index, common::SafeCStr(criteria), -1, SQLITE_STATIC) != SQLITE_OK) {
    MS_LOG(ERROR) << "SQL error: could not bind parameter, index: " << index << ", field value: " << criteria;
    return FAILED;
  }
  int rc = sqlite3_step(stmt);
  while (rc != SQLITE_DONE) {
    vector<string> tmp;
    int ncols = sqlite3_column_count(stmt);
    for (int i = 0; i < ncols; i++) {
      tmp.emplace_back(reinterpret_cast<const char *>(sqlite3_column_text(stmt, i)));
    }
    labels.push_back(tmp);
    rc = sqlite3_step(stmt);
  }
  (void)sqlite3_finalize(stmt);
  return SUCCESS;
}

std::pair<MSRStatus, std::vector<json>> ShardReader::GetLabelsFromBinaryFile(
  int shard_id, const std::vector<std::string> &columns, const std::vector<std::vector<std::string>> &label_offsets) {
  std::string file_name = file_paths_[shard_id];
  std::vector<json> res;
  std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
  fs->open(common::SafeCStr(file_name), std::ios::in | std::ios::binary);
  if (!fs->good()) {
    MS_LOG(ERROR) << "Invalid file, failed to open file: " << file_name;
    return {FAILED, {}};
  }

  // init the return
  for (unsigned int i = 0; i < label_offsets.size(); ++i) {
    res.emplace_back(json{});
  }

  for (unsigned int i = 0; i < label_offsets.size(); ++i) {
    const auto &labelOffset = label_offsets[i];
    uint64_t label_start = std::stoull(labelOffset[1]) + kInt64Len;
    uint64_t label_end = std::stoull(labelOffset[2]);
    int raw_page_id = std::stoi(labelOffset[0]);
    auto len = label_end - label_start;
    auto label_raw = std::vector<uint8_t>(len);
    auto &io_seekg = fs->seekg(page_size_ * raw_page_id + header_size_ + label_start, std::ios::beg);
    if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
      MS_LOG(ERROR) << "File seekg failed";
      fs->close();
      return {FAILED, {}};
    }

    auto &io_read = fs->read(reinterpret_cast<char *>(&label_raw[0]), len);
    if (!io_read.good() || io_read.fail() || io_read.bad()) {
      MS_LOG(ERROR) << "File read failed";
      fs->close();
      return {FAILED, {}};
    }

    json label_json = json::from_msgpack(label_raw);
    json tmp = label_json;
    for (auto &col : columns) {
      if (label_json.find(col) != label_json.end()) {
        tmp[col] = label_json[col];
      }
    }
    res[i] = tmp;
  }
  return {SUCCESS, res};
}

std::pair<MSRStatus, std::vector<json>> ShardReader::GetLabelsFromPage(
  int page_id, int shard_id, const std::vector<std::string> &columns,
  const std::pair<std::string, std::string> &criteria) {
  // get page info from sqlite
  auto db = database_paths_[shard_id];
  std::string sql = "SELECT PAGE_ID_RAW, PAGE_OFFSET_RAW,PAGE_OFFSET_RAW_END FROM INDEXES WHERE PAGE_ID_BLOB = " +
                    std::to_string(page_id);
  std::vector<std::vector<std::string>> label_offsets;
  if (!criteria.first.empty()) {
    sql += " AND " + criteria.first + "_" + std::to_string(column_schema_id_[criteria.first]) + " = :criteria";
    if (QueryWithCriteria(db, sql, criteria.second, label_offsets) == FAILED) {
      return {FAILED, {}};
    }
  } else {
    sql += ";";
    char *errmsg = nullptr;
    int rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &label_offsets, &errmsg);
    if (rc != SQLITE_OK) {
      MS_LOG(ERROR) << "Error in select statement, sql: " << sql << ", error: " << errmsg;
      sqlite3_free(errmsg);
      sqlite3_close(db);
      db = nullptr;
      return {FAILED, {}};
    }
    MS_LOG(DEBUG) << "Get " << label_offsets.size() << "records from index.";
    sqlite3_free(errmsg);
  }
  // get labels from binary file
  return GetLabelsFromBinaryFile(shard_id, columns, label_offsets);
}

std::pair<MSRStatus, std::vector<json>> ShardReader::GetLabels(int page_id, int shard_id,
                                                               const std::vector<std::string> &columns,
                                                               const std::pair<std::string, std::string> &criteria) {
  if (all_in_index_) {
    auto db = database_paths_[shard_id];
    std::string fields;
    for (unsigned int i = 0; i < columns.size(); ++i) {
      if (i > 0) fields += ',';
      uint64_t schema_id = column_schema_id_[columns[i]];
      fields += columns[i] + "_" + std::to_string(schema_id);
    }
    if (fields.empty()) fields = "*";
    std::vector<std::vector<std::string>> labels;
    std::string sql = "SELECT " + fields + " FROM INDEXES WHERE PAGE_ID_BLOB = " + std::to_string(page_id);
    if (!criteria.first.empty()) {
      sql += " AND " + criteria.first + "_" + std::to_string(column_schema_id_[criteria.first]) + " = " + ":criteria";
      if (QueryWithCriteria(db, sql, criteria.second, labels) == FAILED) {
        return {FAILED, {}};
      }
    } else {
      sql += ";";
      char *errmsg = nullptr;
      int rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &labels, &errmsg);
      if (rc != SQLITE_OK) {
        MS_LOG(ERROR) << "Error in select statement, sql: " << sql << ", error: " << errmsg;
        sqlite3_free(errmsg);
        sqlite3_close(db);
        db = nullptr;
        return {FAILED, {}};
      } else {
        MS_LOG(DEBUG) << "Get " << static_cast<int>(labels.size()) << "records from index.";
      }
      sqlite3_free(errmsg);
    }
    std::vector<json> ret;
    for (unsigned int i = 0; i < labels.size(); ++i) ret.emplace_back(json{});
    for (unsigned int i = 0; i < labels.size(); ++i) {
      json construct_json;
      for (unsigned int j = 0; j < columns.size(); ++j) {
        // construct json "f1": value
        auto schema = shard_header_->GetSchemas()[0]->GetSchema()["schema"];

        // convert the string to base type by schema
        if (schema[columns[j]]["type"] == "int32") {
          construct_json[columns[j]] = StringToNum<int32_t>(labels[i][j]);
        } else if (schema[columns[j]]["type"] == "int64") {
          construct_json[columns[j]] = StringToNum<int64_t>(labels[i][j]);
        } else if (schema[columns[j]]["type"] == "float32") {
          construct_json[columns[j]] = StringToNum<float>(labels[i][j]);
        } else if (schema[columns[j]]["type"] == "float64") {
          construct_json[columns[j]] = StringToNum<double>(labels[i][j]);
        } else {
          construct_json[columns[j]] = std::string(labels[i][j]);
        }
      }
      ret[i] = construct_json;
    }
    return {SUCCESS, ret};
  }
  return GetLabelsFromPage(page_id, shard_id, columns, criteria);
}

bool ResortRowGroups(std::tuple<int, int, int, int> a, std::tuple<int, int, int, int> b) {
  return std::get<1>(a) < std::get<1>(b) || (std::get<1>(a) == std::get<1>(b) && std::get<0>(a) < std::get<0>(b));
}

int64_t ShardReader::GetNumClasses(const std::string &category_field) {
  auto shard_count = file_paths_.size();
  auto index_fields = shard_header_->GetFields();

  std::map<std::string, int64_t> map_schema_id_fields;
  for (auto &field : index_fields) {
    map_schema_id_fields[field.second] = field.first;
  }

  if (map_schema_id_fields.find(category_field) == map_schema_id_fields.end()) {
    MS_LOG(ERROR) << "Field " << category_field << " does not exist.";
    return -1;
  }
  auto ret =
    ShardIndexGenerator::GenerateFieldName(std::make_pair(map_schema_id_fields[category_field], category_field));
  if (SUCCESS != ret.first) {
    return -1;
  }
  std::string sql = "SELECT DISTINCT " + ret.second + " FROM INDEXES";
  std::vector<std::thread> threads = std::vector<std::thread>(shard_count);
  std::set<std::string> categories;
  for (int x = 0; x < shard_count; x++) {
    sqlite3 *db = nullptr;
    int rc = sqlite3_open_v2(common::SafeCStr(file_paths_[x] + ".db"), &db, SQLITE_OPEN_READONLY, nullptr);
    if (SQLITE_OK != rc) {
      MS_LOG(ERROR) << "Invalid file, failed to open database: " << file_paths_[x] + ".db, error: "
                    << sqlite3_errmsg(db);
      return -1;
    }
    threads[x] = std::thread(&ShardReader::GetClassesInShard, this, db, x, sql, std::ref(categories));
  }

  for (int x = 0; x < shard_count; x++) {
    threads[x].join();
  }
  return categories.size();
}

MSRStatus ShardReader::CountTotalRows(const std::vector<std::string> &file_paths, bool load_dataset,
                                      const std::shared_ptr<ShardOperator> &ops, int64_t *count, const int num_padded) {
  if (SUCCESS != Init(file_paths, load_dataset)) {
    return FAILED;
  }
  int64_t num_samples = num_rows_;
  bool root = true;
  std::stack<std::shared_ptr<ShardOperator>> stack_ops;
  std::shared_ptr<ShardOperator> op(ops);
  while (op != nullptr) {
    stack_ops.push(op);
    op = op->GetChildOp();
  }
  while (!stack_ops.empty()) {
    op = stack_ops.top();
    stack_ops.pop();
    if (std::dynamic_pointer_cast<ShardShuffle>(op)) {
      num_samples = op->GetNumSamples(num_samples, 0);
      if (num_padded > 0 && root == true) {
        num_samples += num_padded;
        MS_LOG(DEBUG) << "Padding samples work on shuffle sampler.";
        root = false;
      }
    } else if (std::dynamic_pointer_cast<ShardCategory>(op)) {
      auto category_op = std::dynamic_pointer_cast<ShardCategory>(op);
      std::string category_field = category_op->GetCategoryField();
      auto num_classes = GetNumClasses(category_field);
      num_samples = category_op->GetNumSamples(num_samples, num_classes);
      if (std::dynamic_pointer_cast<ShardPkSample>(op)) {
        auto tmp = std::dynamic_pointer_cast<ShardPkSample>(op)->GetNumSamples();
        if (tmp != 0) {
          num_samples = std::min(num_samples, tmp);
        }
      }
    } else if (std::dynamic_pointer_cast<ShardSample>(op)) {
      if (std::dynamic_pointer_cast<ShardDistributedSample>(op)) {
        auto sampler_op = std::dynamic_pointer_cast<ShardDistributedSample>(op);
        if (root == true) {
          sampler_op->SetNumPaddedSamples(num_padded);
          num_samples = op->GetNumSamples(num_samples, 0);
          if (-1 == num_samples) {
            MS_LOG(ERROR) << "Dataset size plus number of padded samples is not divisible by number of shards.";
            return FAILED;
          }
          root = false;
        }
      } else {
        num_samples = op->GetNumSamples(num_samples, 0);
      }
    } else {
      if (num_padded > 0) num_samples += num_padded;
    }
  }
  *count = num_samples;
  return SUCCESS;
}

MSRStatus ShardReader::Open(const std::vector<std::string> &file_paths, bool load_dataset, int n_consumer,
                            const std::vector<std::string> &selected_columns,
                            const std::vector<std::shared_ptr<ShardOperator>> &operators, int num_padded,
                            bool lazy_load) {
  lazy_load_ = lazy_load;

  // Open file and set header by ShardReader
  auto ret = Init(file_paths, load_dataset);
  if (SUCCESS != ret) {
    return ret;
  }
  auto thread_limit = GetMaxThreadNum();
  if (n_consumer > thread_limit) {
    n_consumer = thread_limit;
  }
  if (n_consumer < kMinConsumerCount) {
    n_consumer = kMinConsumerCount;
  }

  selected_columns_ = selected_columns;

  if (CheckColumnList(selected_columns_) == FAILED) {
    MS_LOG(ERROR) << "Illegal column list";
    return ILLEGAL_COLUMN_LIST;
  }

  // Initialize argument
  shard_count_ = static_cast<int>(file_paths_.size());
  n_consumer_ = n_consumer;
  num_padded_ = num_padded;

  operators_ = operators;

  if (Open(n_consumer) == FAILED) {
    return FAILED;
  }
  return SUCCESS;
}

MSRStatus ShardReader::OpenPy(const std::vector<std::string> &file_paths, bool load_dataset, const int &n_consumer,
                              const std::vector<std::string> &selected_columns,
                              const std::vector<std::shared_ptr<ShardOperator>> &operators) {
  // Open file and set header by ShardReader
  if (SUCCESS != Init(file_paths, load_dataset)) {
    return FAILED;
  }
  // should remove blob field from selected_columns when call from python
  std::vector<std::string> columns(selected_columns);
  auto blob_fields = GetBlobFields().second;
  for (auto &blob_field : blob_fields) {
    auto it = std::find(selected_columns.begin(), selected_columns.end(), blob_field);
    if (it != selected_columns.end()) {
      columns.erase(columns.begin() + std::distance(selected_columns.begin(), it));
    }
  }
  if (CheckColumnList(columns) == FAILED) {
    MS_LOG(ERROR) << "Illegal column list";
    return FAILED;
  }
  if (Open(n_consumer) == FAILED) {
    return FAILED;
  }
  // Initialize argument
  shard_count_ = static_cast<int>(file_paths_.size());
  n_consumer_ = n_consumer;

  // Initialize columns which will be read
  selected_columns_ = selected_columns;
  operators_ = operators;

  return SUCCESS;
}

MSRStatus ShardReader::Launch(bool isSimpleReader) {
  // Get all row groups' info
  auto row_group_summary = ReadRowGroupSummary();

  // Sort row group by (group_id, shard_id), prepare for parallel reading
  std::sort(row_group_summary.begin(), row_group_summary.end(), ResortRowGroups);
  if (CreateTasks(row_group_summary, operators_) != SUCCESS) {
    MS_LOG(ERROR) << "Failed to launch read threads.";
    interrupt_ = true;
    return FAILED;
  }
  if (isSimpleReader) return SUCCESS;
  // Start provider consumer threads
  thread_set_ = std::vector<std::thread>(n_consumer_);
  if (n_consumer_ <= 0 || n_consumer_ > kMaxConsumerCount) {
    return FAILED;
  }

  for (int x = 0; x < n_consumer_; ++x) {
    thread_set_[x] = std::thread(&ShardReader::ConsumerByRow, this, x);
  }

  MS_LOG(INFO) << "Launch read thread successfully.";
  return SUCCESS;
}

MSRStatus ShardReader::CreateTasksByCategory(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                                             const std::shared_ptr<ShardOperator> &op) {
  CheckIfColumnInIndex(selected_columns_);
  auto category_op = std::dynamic_pointer_cast<ShardCategory>(op);
  auto categories = category_op->GetCategories();
  int64_t num_elements = category_op->GetNumElements();
  int64_t num_samples = 0;
  if (std::dynamic_pointer_cast<ShardPkSample>(op)) {
    num_samples = std::dynamic_pointer_cast<ShardPkSample>(op)->GetNumSamples();
    if (num_samples < 0) {
      MS_LOG(ERROR) << "Invalid parameter, num_samples must be greater than or equal to 0, but got " << num_samples;
      return FAILED;
    }
  }
  if (num_elements <= 0) {
    MS_LOG(ERROR) << "Invalid parameter, num_elements must be greater than 0, but got " << num_elements;
    return FAILED;
  }
  if (categories.empty() == true) {
    std::string category_field = category_op->GetCategoryField();
    int64_t num_categories = category_op->GetNumCategories();
    if (num_categories <= 0) {
      MS_LOG(ERROR) << "Invalid parameter, num_categories must be greater than 0, but got " << num_elements;
      return FAILED;
    }
    std::set<std::string> categories_set;
    auto ret = GetAllClasses(category_field, categories_set);
    if (SUCCESS != ret) {
      return FAILED;
    }
    int i = 0;
    for (auto it = categories_set.begin(); it != categories_set.end() && i < num_categories; ++it) {
      categories.emplace_back(category_field, *it);
      i++;
    }
  }
  // Generate task list, a task will create a batch
  std::vector<ShardTask> categoryTasks(categories.size());
  for (uint32_t categoryNo = 0; categoryNo < categories.size(); ++categoryNo) {
    int category_index = 0;
    for (const auto &rg : row_group_summary) {
      if (category_index >= num_elements) break;
      auto shard_id = std::get<0>(rg);
      auto group_id = std::get<1>(rg);

      auto details = ReadRowGroupCriteria(group_id, shard_id, categories[categoryNo], selected_columns_);
      if (SUCCESS != std::get<0>(details)) {
        return FAILED;
      }
      auto offsets = std::get<4>(details);

      auto number_of_rows = offsets.size();
      for (uint32_t iStart = 0; iStart < number_of_rows; iStart += 1) {
        if (category_index < num_elements) {
          categoryTasks[categoryNo].InsertTask(TaskType::kCommonTask, shard_id, group_id, std::get<4>(details)[iStart],
                                               std::get<5>(details)[iStart]);
          category_index++;
        }
      }
    }
    MS_LOG(INFO) << "Category #" << categoryNo << " has " << categoryTasks[categoryNo].Size() << " tasks";
  }
  tasks_ = ShardTask::Combine(categoryTasks, category_op->GetReplacement(), num_elements, num_samples);
  if (SUCCESS != (*category_op)(tasks_)) {
    return FAILED;
  }
  return SUCCESS;
}

MSRStatus ShardReader::CreateTasksByRow(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                                        const std::vector<std::shared_ptr<ShardOperator>> &operators) {
  CheckIfColumnInIndex(selected_columns_);

  auto ret = ReadAllRowGroup(selected_columns_);
  if (std::get<0>(ret) != SUCCESS) {
    return FAILED;
  }
  auto &offsets = std::get<1>(ret);
  auto &local_columns = std::get<2>(ret);
  if (shard_count_ <= kMaxFileCount) {
    int sample_count = 0;
    for (int shard_id = 0; shard_id < shard_count_; shard_id++) {
      sample_count += offsets[shard_id].size();
    }
    MS_LOG(DEBUG) << "There are " << sample_count << " records in the dataset.";

    // Init the tasks_ size
    tasks_.ResizeTask(sample_count);

    // Init the task threads, maybe use ThreadPool is better
    std::vector<std::thread> init_tasks_thread(shard_count_);

    uint32_t current_offset = 0;
    for (uint32_t shard_id = 0; shard_id < shard_count_; shard_id++) {
      init_tasks_thread[shard_id] = std::thread([this, &offsets, &local_columns, shard_id, current_offset]() {
        auto offset = current_offset;
        for (uint32_t i = 0; i < offsets[shard_id].size(); i += 1) {
          tasks_.InsertTask(offset, TaskType::kCommonTask, offsets[shard_id][i][0], offsets[shard_id][i][1],
                            std::vector<uint64_t>{offsets[shard_id][i][2], offsets[shard_id][i][3]},
                            local_columns[shard_id][i]);
          offset++;
        }
      });
      current_offset += offsets[shard_id].size();
    }

    for (uint32_t shard_id = 0; shard_id < shard_count_; shard_id++) {
      init_tasks_thread[shard_id].join();
    }
  } else {
    return FAILED;
  }
  return SUCCESS;
}

MSRStatus ShardReader::CreateLazyTasksByRow(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                                            const std::vector<std::shared_ptr<ShardOperator>> &operators) {
  CheckIfColumnInIndex(selected_columns_);

  if (shard_count_ <= kMaxFileCount) {
    uint32_t sample_count = shard_sample_count_[shard_sample_count_.size() - 1];
    MS_LOG(DEBUG) << "There are " << sample_count << " records in the dataset.";

    // Init the tasks_ size
    tasks_.ResizeTask(sample_count);

    // Init the task threads, maybe use ThreadPool is better
    std::vector<std::thread> init_tasks_thread(shard_count_);

    for (uint32_t shard_id = 0; shard_id < shard_count_; shard_id++) {
      // the offset indicate the shard start
      uint32_t current_offset = shard_id == 0 ? 0 : shard_sample_count_[shard_id - 1];

      // the count indicate the number of samples in the shard
      uint32_t shard_count =
        shard_id == 0 ? shard_sample_count_[0] : shard_sample_count_[shard_id] - shard_sample_count_[shard_id - 1];
      init_tasks_thread[shard_id] = std::thread([this, shard_id, current_offset, shard_count]() {
        for (uint32_t i = current_offset; i < shard_count + current_offset; ++i) {
          // here "i - current_offset" indicate the sample id in the shard
          tasks_.InsertTask(i, TaskType::kCommonTask, shard_id, i - current_offset, {}, json());
        }
      });
    }

    for (uint32_t shard_id = 0; shard_id < shard_count_; shard_id++) {
      init_tasks_thread[shard_id].join();
    }
  } else {
    return FAILED;
  }
  return SUCCESS;
}

MSRStatus ShardReader::CreateTasks(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                                   const std::vector<std::shared_ptr<ShardOperator>> &operators) {
  int category_operator = -1;
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators[i];
    if (std::dynamic_pointer_cast<ShardCategory>(op)) {
      category_operator = static_cast<int>(i);
      break;
    }
  }
  if (-1 == category_operator) {
    if (lazy_load_ == false) {
      if (SUCCESS != CreateTasksByRow(row_group_summary, operators)) {
        return FAILED;
      }
    } else {
      if (SUCCESS != CreateLazyTasksByRow(row_group_summary, operators)) {
        return FAILED;
      }
    }

    // need padded sample to the task
    if (num_padded_ > 0) {
      for (int i = 0; i < num_padded_; ++i) {
        tasks_.InsertTask(TaskType::kPaddedTask, 0, 0, {}, json());
      }
    }
  } else {
    if (SUCCESS != CreateTasksByCategory(row_group_summary, operators[category_operator])) {
      return FAILED;
    }
  }

  for (uint32_t operator_no = 0; operator_no < operators.size(); operator_no++) {
    const auto &op = operators[operator_no];
    if (std::dynamic_pointer_cast<ShardCategory>(op)) continue;
    if (SUCCESS != (*op)(tasks_)) {
      return FAILED;
    }
  }

  if (tasks_.permutation_.empty()) tasks_.MakePerm();
  num_rows_ = tasks_.Size();
  MS_LOG(INFO) << "Total rows is " << num_rows_;
  return SUCCESS;
}

TASK_RETURN_CONTENT ShardReader::ConsumerOneTask(int task_id, uint32_t consumer_id) {
  // All tasks are done
  if (task_id >= static_cast<int>(tasks_.Size())) {
    return std::make_pair(FAILED,
                          std::make_pair(TaskType::kCommonTask, std::vector<std::tuple<std::vector<uint8_t>, json>>()));
  }

  uint32_t shard_id = 0;
  uint32_t group_id = 0;
  uint32_t blob_start = 0;
  uint32_t blob_end = 0;
  json var_fields;

  // Pick up task from task list
  auto task = tasks_.GetTaskByID(tasks_.permutation_[task_id]);

  // check task type
  auto task_type = std::get<0>(task);
  if (task_type == TaskType::kPaddedTask) {
    return std::make_pair(SUCCESS,
                          std::make_pair(TaskType::kPaddedTask, std::vector<std::tuple<std::vector<uint8_t>, json>>()));
  }

  shard_id = std::get<0>(std::get<1>(task));  // shard id

  if (lazy_load_ == false) {
    group_id = std::get<1>(std::get<1>(task));  // group id
    blob_start = std::get<2>(task)[0];          // blob start
    blob_end = std::get<2>(task)[1];            // blob end
    var_fields = std::get<3>(task);             // scalar variable field
  } else {
    // get scalar variable fields by sample id
    uint32_t sample_id_in_shard = std::get<1>(std::get<1>(task));

    // read the meta from index
    auto row_meta = ReadRowGroupByShardIDAndSampleID(selected_columns_, shard_id, sample_id_in_shard);
    if (std::get<0>(row_meta) != SUCCESS) {
      return std::make_pair(
        FAILED, std::make_pair(TaskType::kCommonTask, std::vector<std::tuple<std::vector<uint8_t>, json>>()));
    }
    auto &offsets = std::get<1>(row_meta);
    auto &local_columns = std::get<2>(row_meta);

    group_id = offsets[shard_id][0][1];       // group_id
    blob_start = offsets[shard_id][0][2];     // blob start
    blob_end = offsets[shard_id][0][3];       // blob end
    var_fields = local_columns[shard_id][0];  // scalar variable field
  }

  // read the blob from data file
  const auto &ret = shard_header_->GetPageByGroupId(group_id, shard_id);
  if (SUCCESS != ret.first) {
    return std::make_pair(FAILED,
                          std::make_pair(TaskType::kCommonTask, std::vector<std::tuple<std::vector<uint8_t>, json>>()));
  }
  const std::shared_ptr<Page> &page = ret.second;

  // Pack image list
  std::vector<uint8_t> images(blob_end - blob_start);
  auto file_offset = header_size_ + page_size_ * (page->GetPageID()) + blob_start;

  auto &io_seekg = file_streams_random_[consumer_id][shard_id]->seekg(file_offset, std::ios::beg);
  if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
    MS_LOG(ERROR) << "File seekg failed";
    file_streams_random_[consumer_id][shard_id]->close();
    return std::make_pair(FAILED,
                          std::make_pair(TaskType::kCommonTask, std::vector<std::tuple<std::vector<uint8_t>, json>>()));
  }

  auto &io_read =
    file_streams_random_[consumer_id][shard_id]->read(reinterpret_cast<char *>(&images[0]), blob_end - blob_start);
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    MS_LOG(ERROR) << "File read failed";
    file_streams_random_[consumer_id][shard_id]->close();
    return std::make_pair(FAILED,
                          std::pair(TaskType::kCommonTask, std::vector<std::tuple<std::vector<uint8_t>, json>>()));
  }

  // Deliver batch data to output map
  std::vector<std::tuple<std::vector<uint8_t>, json>> batch;
  batch.emplace_back(std::move(images), std::move(var_fields));

  return std::make_pair(SUCCESS, std::make_pair(TaskType::kCommonTask, std::move(batch)));
}

MSRStatus ShardReader::ConsumerByRow(int consumer_id) {
  // Set thread name
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  auto thread_id = kThreadName + std::to_string(consumer_id);
  prctl(PR_SET_NAME, common::SafeCStr(thread_id), 0, 0, 0);
#endif

  // Loop forever
  for (;;) {
    int task_id = 0;

    // Get next task ID
    task_id = task_id_++;

    // All tasks are done
    if (task_id >= static_cast<int>(tasks_.Size())) {
      return FAILED;
    }
    const auto &ret = ConsumerOneTask(task_id, consumer_id);
    if (SUCCESS != ret.first) {
      return FAILED;
    }
    const auto &batch = (ret.second).second;
    // Hanging if maximum map size exceeded
    //   otherwise, set batch data in map
    {
      std::unique_lock<std::mutex> lck(mtx_delivery_);
      cv_delivery_.wait(lck, [task_id, this] { return interrupt_ || task_id <= deliver_id_ + kNumBatchInMap; });
      if (interrupt_) {
        return SUCCESS;
      }
      delivery_map_[task_id] = std::make_shared<std::vector<std::tuple<std::vector<uint8_t>, json>>>(std::move(batch));
    }
    cv_iterator_.notify_one();
  }
}

std::vector<std::tuple<std::vector<uint8_t>, json>> ShardReader::GetNext() {
  if (interrupt_) {
    return std::vector<std::tuple<std::vector<uint8_t>, json>>();
  }
  if (deliver_id_ >= static_cast<int>(tasks_.Size())) {
    return std::vector<std::tuple<std::vector<uint8_t>, json>>();
  }

  std::shared_ptr<std::vector<std::tuple<std::vector<uint8_t>, json>>> res;
  {
    std::unique_lock<std::mutex> lck(mtx_delivery_);
    cv_iterator_.wait(lck, [this] { return interrupt_ || (delivery_map_.count(deliver_id_) > 0); });
    if (interrupt_) {
      return std::vector<std::tuple<std::vector<uint8_t>, json>>();
    }
    res = delivery_map_[deliver_id_];
    delivery_map_.erase(deliver_id_++);
  }

  cv_delivery_.notify_all();

  return *res;
}

std::pair<TaskType, std::vector<std::tuple<std::vector<uint8_t>, json>>> ShardReader::GetNextById(
  const int64_t &task_id, const int32_t &consumer_id) {
  if (interrupt_) {
    return std::make_pair(TaskType::kCommonTask, std::vector<std::tuple<std::vector<uint8_t>, json>>());
  }
  const auto &ret = ConsumerOneTask(task_id, consumer_id);
  if (SUCCESS != ret.first) {
    return std::make_pair(TaskType::kCommonTask, std::vector<std::tuple<std::vector<uint8_t>, json>>());
  }
  return std::move(ret.second);
}

std::pair<MSRStatus, std::vector<std::vector<uint8_t>>> ShardReader::UnCompressBlob(
  const std::vector<uint8_t> &raw_blob_data) {
  auto loaded_columns = selected_columns_.size() == 0 ? shard_column_->GetColumnName() : selected_columns_;
  auto blob_fields = GetBlobFields().second;
  std::vector<std::vector<uint8_t>> blob_data;
  for (uint32_t i_col = 0; i_col < loaded_columns.size(); ++i_col) {
    if (std::find(blob_fields.begin(), blob_fields.end(), loaded_columns[i_col]) == blob_fields.end()) continue;
    const unsigned char *data = nullptr;
    std::unique_ptr<unsigned char[]> data_ptr;
    uint64_t n_bytes = 0;
    auto ret = shard_column_->GetColumnFromBlob(loaded_columns[i_col], raw_blob_data, &data, &data_ptr, &n_bytes);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "Error when get data from blob, column name is " << loaded_columns[i_col] << ".";
      return {FAILED, std::vector<std::vector<uint8_t>>(blob_fields.size(), std::vector<uint8_t>())};
    }
    if (data == nullptr) {
      data = reinterpret_cast<const unsigned char *>(data_ptr.get());
    }
    std::vector<uint8_t> column(data, data + (n_bytes / sizeof(unsigned char)));
    blob_data.push_back(column);
  }
  return {SUCCESS, blob_data};
}

std::vector<std::tuple<std::vector<std::vector<uint8_t>>, pybind11::object>> ShardReader::GetNextPy() {
  auto res = GetNext();
  vector<std::tuple<std::vector<std::vector<uint8_t>>, pybind11::object>> data;
  std::transform(res.begin(), res.end(), std::back_inserter(data),
                 [this](const std::tuple<std::vector<uint8_t>, json> &item) {
                   auto &j = std::get<1>(item);
                   pybind11::object obj = nlohmann::detail::FromJsonImpl(j);
                   auto ret = UnCompressBlob(std::get<0>(item));
                   return std::make_tuple(ret.second, std::move(obj));
                 });
  return data;
}

void ShardReader::Reset() {
  {
    std::lock_guard<std::mutex> lck(mtx_delivery_);
    task_id_ = 0;
    deliver_id_ = 0;
  }
  cv_delivery_.notify_all();
}

void ShardReader::ShuffleTask() {
  // exist shuffle and distributed sampler in ops, skip shuffle
  bool has_sharding = false;
  for (const auto &op : operators_) {
    if (std::dynamic_pointer_cast<ShardDistributedSample>(op)) {
      has_sharding = true;
    }
  }
  for (const auto &op : operators_) {
    if (std::dynamic_pointer_cast<ShardShuffle>(op) && has_sharding == false) {
      if (SUCCESS != (*op)(tasks_)) {
        MS_LOG(WARNING) << "Redo randomSampler failed.";
      }
    } else if (std::dynamic_pointer_cast<ShardDistributedSample>(op)) {
      if (SUCCESS != (*op)(tasks_)) {
        MS_LOG(WARNING) << "Redo distributeSampler failed.";
      }
    }
  }
  if (tasks_.permutation_.empty()) tasks_.MakePerm();
}

}  // namespace mindrecord
}  // namespace mindspore
