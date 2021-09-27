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

#include "minddata/mindrecord/include/shard_reader.h"

#include <algorithm>
#include <thread>

#include "utils/file_utils.h"
#include "minddata/mindrecord/include/shard_distributed_sample.h"
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
      sample_id_position_(0),
      deliver_id_(0),
      lazy_load_(false),
      shard_sample_count_() {}

Status ShardReader::GetMeta(const std::string &file_path, std::shared_ptr<json> meta_data_ptr,
                            std::shared_ptr<std::vector<std::string>> *addresses_ptr) {
  RETURN_UNEXPECTED_IF_NULL(addresses_ptr);
  CHECK_FAIL_RETURN_UNEXPECTED(IsLegalFile(file_path), "Invalid file, path: " + file_path);
  std::shared_ptr<json> header_ptr;
  RETURN_IF_NOT_OK(ShardHeader::BuildSingleHeader(file_path, &header_ptr));

  *meta_data_ptr = {{"header_size", (*header_ptr)["header_size"]}, {"page_size", (*header_ptr)["page_size"]},
                    {"version", (*header_ptr)["version"]},         {"index_fields", (*header_ptr)["index_fields"]},
                    {"schema", (*header_ptr)["schema"]},           {"blob_fields", (*header_ptr)["blob_fields"]}};
  *addresses_ptr = std::make_shared<std::vector<std::string>>((*header_ptr)["shard_addresses"]);
  return Status::OK();
}

Status ShardReader::Init(const std::vector<std::string> &file_paths, bool load_dataset) {
  std::string file_path = file_paths[0];
  auto first_meta_data_ptr = std::make_shared<json>();
  std::shared_ptr<std::vector<std::string>> addresses_ptr;
  RETURN_IF_NOT_OK(GetMeta(file_path, first_meta_data_ptr, &addresses_ptr));
  if (file_paths.size() == 1 && load_dataset == true) {
    auto ds = std::make_shared<std::vector<std::string>>();
    RETURN_IF_NOT_OK(GetDatasetFiles(file_path, *addresses_ptr, &ds));
    file_paths_ = *ds;
  } else if (file_paths.size() >= 1 && load_dataset == false) {
    file_paths_ = file_paths;
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid data, number of MindRecord files [" + std::to_string(file_paths.size()) +
                             "] or 'load_dataset' [" + std::to_string(load_dataset) + "]is invalid.");
  }
  for (const auto &file : file_paths_) {
    auto meta_data_ptr = std::make_shared<json>();
    RETURN_IF_NOT_OK(GetMeta(file, meta_data_ptr, &addresses_ptr));
    CHECK_FAIL_RETURN_UNEXPECTED(*meta_data_ptr == *first_meta_data_ptr,
                                 "Invalid data, MindRecord files meta data is not consistent.");
    sqlite3 *db = nullptr;
    RETURN_IF_NOT_OK(VerifyDataset(&db, file));
    database_paths_.push_back(db);
  }
  ShardHeader sh = ShardHeader();
  RETURN_IF_NOT_OK(sh.BuildDataset(file_paths_, load_dataset));
  shard_header_ = std::make_shared<ShardHeader>(sh);
  header_size_ = shard_header_->GetHeaderSize();
  page_size_ = shard_header_->GetPageSize();
  // version < 3.0
  if ((*first_meta_data_ptr)["version"] < kVersion) {
    shard_column_ = std::make_shared<ShardColumn>(shard_header_, false);
  } else {
    shard_column_ = std::make_shared<ShardColumn>(shard_header_, true);
  }
  num_rows_ = 0;
  auto row_group_summary = ReadRowGroupSummary();

  // clear the shard_sample_count_, because it will be insert when Launch func
  shard_sample_count_.clear();

  for (const auto &rg : row_group_summary) {
    num_rows_ += std::get<3>(rg);
  }

  if (num_rows_ > LAZY_LOAD_THRESHOLD) {
    lazy_load_ = true;
    MS_LOG(WARNING)
      << "The number of samples is larger than " << LAZY_LOAD_THRESHOLD
      << ", enable lazy load mode. If you want to speed up data loading, "
      << "it is recommended that you save multiple samples into one record when creating MindRecord files,"
      << " so that you can enable fast loading mode, and don't forget to adjust your batch size "
      << "according to the current samples.";
  }

  auto disk_size = page_size_ * row_group_summary.size();
  auto compression_size = shard_header_->GetCompressionSize();
  total_blob_size_ = disk_size + compression_size;
  MS_LOG(INFO) << "Blob data size on disk: " << disk_size << " , additional uncompression size: " << compression_size
               << " , Total blob size: " << total_blob_size_;

  MS_LOG(INFO) << "Succeed to get meta from mindrecord file & index file.";

  return Status::OK();
}

Status ShardReader::VerifyDataset(sqlite3 **db, const string &file) {
  // sqlite3_open create a database if not found, use sqlite3_open_v2 instead of it
  CHECK_FAIL_RETURN_UNEXPECTED(
    sqlite3_open_v2(common::SafeCStr(file + ".db"), db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK,
    "Invalid database file, path: " + file + ".db, " + sqlite3_errmsg(*db));
  MS_LOG(DEBUG) << "Succeed to Open database, path: " << file << ".db.";

  string sql = "SELECT NAME from SHARD_NAME;";
  std::vector<std::vector<std::string>> name;
  char *errmsg = nullptr;
  if (sqlite3_exec(*db, common::SafeCStr(sql), SelectCallback, &name, &errmsg) != SQLITE_OK) {
    std::ostringstream oss;
    oss << "Failed to execute sql [ " << sql + " ], " << errmsg;
    sqlite3_free(errmsg);
    sqlite3_close(*db);
    RETURN_STATUS_UNEXPECTED(oss.str());
  } else {
    MS_LOG(DEBUG) << "Succeed to get " << static_cast<int>(name.size()) << " records from index.";
    std::shared_ptr<std::string> fn_ptr;
    RETURN_IF_NOT_OK(GetFileName(file, &fn_ptr));
    if (name.empty() || name[0][0] != *fn_ptr) {
      sqlite3_free(errmsg);
      sqlite3_close(*db);
      RETURN_STATUS_UNEXPECTED("Invalid database file, shard name [" + *fn_ptr + "] can not match [" + name[0][0] +
                               "].");
    }
  }
  return Status::OK();
}

Status ShardReader::CheckColumnList(const std::vector<std::string> &selected_columns) {
  vector<int> inSchema(selected_columns.size(), 0);
  for (auto &p : GetShardHeader()->GetSchemas()) {
    auto schema = p->GetSchema()["schema"];
    for (unsigned int i = 0; i < selected_columns.size(); ++i) {
      if (schema.find(selected_columns[i]) != schema.end()) {
        inSchema[i] = 1;
      }
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!std::any_of(std::begin(inSchema), std::end(inSchema), [](int x) { return x == 0; }),
                               "Invalid data, column is not found in schema.");
  return Status::OK();
}

Status ShardReader::Open() {
  file_streams_.clear();
  for (const auto &file : file_paths_) {
    std::optional<std::string> dir = "";
    std::optional<std::string> local_file_name = "";
    FileUtils::SplitDirAndFileName(file, &dir, &local_file_name);
    if (!dir.has_value()) {
      dir = ".";
    }

    auto realpath = FileUtils::GetRealPath(dir.value().data());
    CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(), "Failed to get real path, path: " + file);

    std::optional<std::string> whole_path = "";
    FileUtils::ConcatDirAndFileName(&realpath, &local_file_name, &whole_path);

    std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
    fs->open(whole_path.value(), std::ios::in | std::ios::binary);
    if (!fs->good()) {
      RETURN_STATUS_UNEXPECTED(
        "Failed to open file: " + file +
        ", reach the maximum number of open files, use \"ulimit -a\" to view \"open files\" and further resize");
    }
    MS_LOG(INFO) << "Succeed to open shard file.";
    file_streams_.push_back(fs);
  }
  return Status::OK();
}

Status ShardReader::Open(int n_consumer) {
  file_streams_random_ =
    std::vector<std::vector<std::shared_ptr<std::fstream>>>(n_consumer, std::vector<std::shared_ptr<std::fstream>>());
  for (const auto &file : file_paths_) {
    for (int j = 0; j < n_consumer; ++j) {
      std::optional<std::string> dir = "";
      std::optional<std::string> local_file_name = "";
      FileUtils::SplitDirAndFileName(file, &dir, &local_file_name);
      if (!dir.has_value()) {
        dir = ".";
      }

      auto realpath = FileUtils::GetRealPath(dir.value().data());
      CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(), "Failed to get real path, path: " + file);

      std::optional<std::string> whole_path = "";
      FileUtils::ConcatDirAndFileName(&realpath, &local_file_name, &whole_path);

      std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
      fs->open(whole_path.value(), std::ios::in | std::ios::binary);
      if (!fs->good()) {
        RETURN_STATUS_UNEXPECTED(
          "Failed to open file: " + file +
          ", reach the maximum number of open files, use \"ulimit -a\" to view \"open files\" and further resize");
      }
      file_streams_random_[j].push_back(fs);
    }
    MS_LOG(INFO) << "Succeed to open file, path: " << file;
  }
  return Status::OK();
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
        MS_LOG(ERROR) << "Failed to close database, error code: " << ret << ".";
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
        std::shared_ptr<Page> page_ptr;
        (void)shard_header_->GetPage(shard_id, page_id, &page_ptr);
        if (page_ptr->GetPageType() != kPageTypeBlob) {
          continue;
        }
        uint64_t start_row_id = page_ptr->GetStartRowID();
        if (start_row_id > page_ptr->GetEndRowID()) {
          return std::vector<std::tuple<int, int, int, uint64_t>>();
        }
        uint64_t number_of_rows = page_ptr->GetEndRowID() - start_row_id;
        total_count += number_of_rows;
        row_group_summary.emplace_back(shard_id, page_ptr->GetPageTypeID(), start_row_id, number_of_rows);
      }
      shard_sample_count_.push_back(total_count);
    }
  }

  return row_group_summary;
}

Status ShardReader::ConvertLabelToJson(const std::vector<std::vector<std::string>> &labels,
                                       std::shared_ptr<std::fstream> fs,
                                       std::shared_ptr<std::vector<std::vector<std::vector<uint64_t>>>> offset_ptr,
                                       int shard_id, const std::vector<std::string> &columns,
                                       std::shared_ptr<std::vector<std::vector<json>>> col_val_ptr) {
  for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
    try {
      uint64_t group_id = std::stoull(labels[i][0]);
      uint64_t offset_start = std::stoull(labels[i][1]) + kInt64Len;
      uint64_t offset_end = std::stoull(labels[i][2]);
      (*offset_ptr)[shard_id].emplace_back(
        std::vector<uint64_t>{static_cast<uint64_t>(shard_id), group_id, offset_start, offset_end});
      if (!all_in_index_) {
        int raw_page_id = std::stoi(labels[i][3]);
        uint64_t label_start = std::stoull(labels[i][4]) + kInt64Len;
        uint64_t label_end = std::stoull(labels[i][5]);
        auto len = label_end - label_start;
        auto label_raw = std::vector<uint8_t>(len);
        auto &io_seekg = fs->seekg(page_size_ * raw_page_id + header_size_ + label_start, std::ios::beg);
        if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
          fs->close();
          RETURN_STATUS_UNEXPECTED("Failed to seekg file.");
        }
        auto &io_read = fs->read(reinterpret_cast<char *>(&label_raw[0]), len);
        if (!io_read.good() || io_read.fail() || io_read.bad()) {
          fs->close();
          RETURN_STATUS_UNEXPECTED("Failed to read file.");
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
        (*col_val_ptr)[shard_id].emplace_back(tmp);
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
        (*col_val_ptr)[shard_id].emplace_back(construct_json);
      }
    } catch (std::out_of_range &e) {
      fs->close();
      RETURN_STATUS_UNEXPECTED("Out of range exception raised in ConvertLabelToJson function, " +
                               std::string(e.what()));
    } catch (std::invalid_argument &e) {
      fs->close();
      RETURN_STATUS_UNEXPECTED("Invalid argument exception raised in ConvertLabelToJson function, " +
                               std::string(e.what()));
    } catch (...) {
      fs->close();
      RETURN_STATUS_UNEXPECTED("Unknown exception raised in ConvertLabelToJson function");
    }
  }

  fs->close();
  return Status::OK();
}

Status ShardReader::ReadAllRowsInShard(int shard_id, const std::string &sql, const std::vector<std::string> &columns,
                                       std::shared_ptr<std::vector<std::vector<std::vector<uint64_t>>>> offset_ptr,
                                       std::shared_ptr<std::vector<std::vector<json>>> col_val_ptr) {
  auto db = database_paths_[shard_id];
  std::vector<std::vector<std::string>> labels;
  char *errmsg = nullptr;
  int rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &labels, &errmsg);
  if (rc != SQLITE_OK) {
    std::ostringstream oss;
    oss << "Failed to execute sql [ " << sql + " ], " << errmsg;
    sqlite3_free(errmsg);
    sqlite3_close(db);
    db = nullptr;
    RETURN_STATUS_UNEXPECTED(oss.str());
  }
  MS_LOG(INFO) << "Succeed to get " << static_cast<int>(labels.size()) << " records from shard "
               << std::to_string(shard_id) << " index.";

  std::string file_name = file_paths_[shard_id];
  auto realpath = FileUtils::GetRealPath(file_name.data());
  if (!realpath.has_value()) {
    sqlite3_free(errmsg);
    sqlite3_close(db);
    RETURN_STATUS_UNEXPECTED("Failed to get real path, path: " + file_name);
  }

  std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
  if (!all_in_index_) {
    fs->open(realpath.value(), std::ios::in | std::ios::binary);
    if (!fs->good()) {
      sqlite3_free(errmsg);
      sqlite3_close(db);
      RETURN_STATUS_UNEXPECTED("Failed to open file, path: " + file_name);
    }
  }
  sqlite3_free(errmsg);
  return ConvertLabelToJson(labels, fs, offset_ptr, shard_id, columns, col_val_ptr);
}

Status ShardReader::GetAllClasses(const std::string &category_field,
                                  std::shared_ptr<std::set<std::string>> category_ptr) {
  std::map<std::string, uint64_t> index_columns;
  for (auto &field : GetShardHeader()->GetFields()) {
    index_columns[field.second] = field.first;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(index_columns.find(category_field) != index_columns.end(),
                               "Invalid data, index field " + category_field + " does not exist.");
  std::shared_ptr<std::string> fn_ptr;
  RETURN_IF_NOT_OK(
    ShardIndexGenerator::GenerateFieldName(std::make_pair(index_columns[category_field], category_field), &fn_ptr));
  std::string sql = "SELECT DISTINCT " + *fn_ptr + " FROM INDEXES";
  std::vector<std::thread> threads = std::vector<std::thread>(shard_count_);
  for (int x = 0; x < shard_count_; x++) {
    threads[x] = std::thread(&ShardReader::GetClassesInShard, this, database_paths_[x], x, sql, category_ptr);
  }

  for (int x = 0; x < shard_count_; x++) {
    threads[x].join();
  }
  return Status::OK();
}

void ShardReader::GetClassesInShard(sqlite3 *db, int shard_id, const std::string &sql,
                                    std::shared_ptr<std::set<std::string>> category_ptr) {
  if (db == nullptr) {
    return;
  }
  std::vector<std::vector<std::string>> columns;
  char *errmsg = nullptr;
  int ret = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &columns, &errmsg);
  if (ret != SQLITE_OK) {
    sqlite3_free(errmsg);
    sqlite3_close(db);
    db = nullptr;
    MS_LOG(ERROR) << "Failed to execute sql [ " << common::SafeCStr(sql) << " ], " << errmsg;
    return;
  }
  MS_LOG(INFO) << "Succeed to get " << static_cast<int>(columns.size()) << " records from shard "
               << std::to_string(shard_id) << " index.";
  std::lock_guard<std::mutex> lck(shard_locker_);
  for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
    category_ptr->emplace(columns[i][0]);
  }
  sqlite3_free(errmsg);
}

Status ShardReader::ReadAllRowGroup(const std::vector<std::string> &columns,
                                    std::shared_ptr<ROW_GROUPS> *row_group_ptr) {
  RETURN_UNEXPECTED_IF_NULL(row_group_ptr);
  std::string fields = "ROW_GROUP_ID, PAGE_OFFSET_BLOB, PAGE_OFFSET_BLOB_END";
  auto offset_ptr = std::make_shared<std::vector<std::vector<std::vector<uint64_t>>>>(
    shard_count_, std::vector<std::vector<uint64_t>>{});
  auto col_val_ptr = std::make_shared<std::vector<std::vector<json>>>(shard_count_, std::vector<json>{});

  if (all_in_index_) {
    for (unsigned int i = 0; i < columns.size(); ++i) {
      fields += ',';
      std::shared_ptr<std::string> fn_ptr;
      RETURN_IF_NOT_OK(
        ShardIndexGenerator::GenerateFieldName(std::make_pair(column_schema_id_[columns[i]], columns[i]), &fn_ptr));
      fields += *fn_ptr;
    }
  } else {  // fetch raw data from Raw page while some field is not index.
    fields += ", PAGE_ID_RAW, PAGE_OFFSET_RAW, PAGE_OFFSET_RAW_END ";
  }

  std::string sql = "SELECT " + fields + " FROM INDEXES ORDER BY ROW_ID ;";

  std::vector<std::thread> thread_read_db = std::vector<std::thread>(shard_count_);
  for (int x = 0; x < shard_count_; x++) {
    thread_read_db[x] = std::thread(&ShardReader::ReadAllRowsInShard, this, x, sql, columns, offset_ptr, col_val_ptr);
  }

  for (int x = 0; x < shard_count_; x++) {
    thread_read_db[x].join();
  }
  *row_group_ptr = std::make_shared<ROW_GROUPS>(std::move(*offset_ptr), std::move(*col_val_ptr));
  return Status::OK();
}

Status ShardReader::ReadRowGroupByShardIDAndSampleID(const std::vector<std::string> &columns, const uint32_t &shard_id,
                                                     const uint32_t &sample_id,
                                                     std::shared_ptr<ROW_GROUPS> *row_group_ptr) {
  RETURN_UNEXPECTED_IF_NULL(row_group_ptr);
  std::string fields = "ROW_GROUP_ID, PAGE_OFFSET_BLOB, PAGE_OFFSET_BLOB_END";
  auto offset_ptr = std::make_shared<std::vector<std::vector<std::vector<uint64_t>>>>(
    shard_count_, std::vector<std::vector<uint64_t>>{});
  auto col_val_ptr = std::make_shared<std::vector<std::vector<json>>>(shard_count_, std::vector<json>{});
  if (all_in_index_) {
    for (unsigned int i = 0; i < columns.size(); ++i) {
      fields += ',';
      std::shared_ptr<std::string> fn_ptr;
      RETURN_IF_NOT_OK(
        ShardIndexGenerator::GenerateFieldName(std::make_pair(column_schema_id_[columns[i]], columns[i]), &fn_ptr));
      fields += *fn_ptr;
    }
  } else {  // fetch raw data from Raw page while some field is not index.
    fields += ", PAGE_ID_RAW, PAGE_OFFSET_RAW, PAGE_OFFSET_RAW_END ";
  }

  std::string sql = "SELECT " + fields + " FROM INDEXES WHERE ROW_ID = " + std::to_string(sample_id);

  RETURN_IF_NOT_OK(ReadAllRowsInShard(shard_id, sql, columns, offset_ptr, col_val_ptr));
  *row_group_ptr = std::make_shared<ROW_GROUPS>(std::move(*offset_ptr), std::move(*col_val_ptr));
  return Status::OK();
}

Status ShardReader::ReadRowGroupBrief(int group_id, int shard_id, const std::vector<std::string> &columns,
                                      std::shared_ptr<ROW_GROUP_BRIEF> *row_group_brief_ptr) {
  RETURN_UNEXPECTED_IF_NULL(row_group_brief_ptr);
  std::shared_ptr<Page> page_ptr;
  RETURN_IF_NOT_OK(shard_header_->GetPageByGroupId(group_id, shard_id, &page_ptr));
  std::string file_name = file_paths_[shard_id];
  uint64_t page_length = page_ptr->GetPageSize();
  uint64_t page_offset = page_size_ * page_ptr->GetPageID() + header_size_;
  std::vector<std::vector<uint64_t>> image_offset = GetImageOffset(page_ptr->GetPageID(), shard_id);
  auto labels_ptr = std::make_shared<std::vector<json>>();
  RETURN_IF_NOT_OK(GetLabels(page_ptr->GetPageID(), shard_id, columns, {"", ""}, &labels_ptr));
  *row_group_brief_ptr = std::make_shared<ROW_GROUP_BRIEF>(file_name, page_length, page_offset, std::move(image_offset),
                                                           std::move(*labels_ptr));
  return Status::OK();
}

Status ShardReader::ReadRowGroupCriteria(int group_id, int shard_id,
                                         const std::pair<std::string, std::string> &criteria,
                                         const std::vector<std::string> &columns,
                                         std::shared_ptr<ROW_GROUP_BRIEF> *row_group_brief_ptr) {
  RETURN_UNEXPECTED_IF_NULL(row_group_brief_ptr);
  std::shared_ptr<Page> page_ptr;
  RETURN_IF_NOT_OK(shard_header_->GetPageByGroupId(group_id, shard_id, &page_ptr));
  vector<string> criteria_list{criteria.first};
  RETURN_IF_NOT_OK(CheckColumnList(criteria_list));
  std::string file_name = file_paths_[shard_id];
  uint64_t page_length = page_ptr->GetPageSize();
  uint64_t page_offset = page_size_ * page_ptr->GetPageID() + header_size_;
  std::vector<std::vector<uint64_t>> image_offset = GetImageOffset(page_ptr->GetPageID(), shard_id, criteria);
  if (image_offset.empty()) {
    *row_group_brief_ptr = std::make_shared<ROW_GROUP_BRIEF>();
  }
  auto labels_ptr = std::make_shared<std::vector<json>>();
  RETURN_IF_NOT_OK(GetLabels(page_ptr->GetPageID(), shard_id, columns, criteria, &labels_ptr));
  *row_group_brief_ptr = std::make_shared<ROW_GROUP_BRIEF>(file_name, page_length, page_offset, std::move(image_offset),
                                                           std::move(*labels_ptr));
  return Status::OK();
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
    MS_LOG(ERROR) << "Failed to execute sql [ " << common::SafeCStr(sql) << " ], " << errmsg;
    sqlite3_free(errmsg);
    sqlite3_close(db);
    db = nullptr;
    return std::vector<std::vector<uint64_t>>();
  } else {
    MS_LOG(DEBUG) << "Succeed to get " << static_cast<int>(image_offsets.size()) << " records from index.";
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

Status ShardReader::GetPagesByCategory(int shard_id, const std::pair<std::string, std::string> &criteria,
                                       std::shared_ptr<std::vector<uint64_t>> *pages_ptr) {
  RETURN_UNEXPECTED_IF_NULL(pages_ptr);
  auto db = database_paths_[shard_id];

  std::string sql = "SELECT DISTINCT PAGE_ID_BLOB FROM INDEXES WHERE 1 = 1 ";

  if (!criteria.first.empty()) {
    auto schema = shard_header_->GetSchemas()[0]->GetSchema();
    if (kNumberFieldTypeSet.find(schema["schema"][criteria.first]["type"]) != kNumberFieldTypeSet.end()) {
      sql +=
        " AND " + criteria.first + "_" + std::to_string(column_schema_id_[criteria.first]) + " = " + criteria.second;
    } else {
      sql += " AND " + criteria.first + "_" + std::to_string(column_schema_id_[criteria.first]) + " = '" +
             criteria.second + "'";
    }
  }
  sql += ";";
  std::vector<std::vector<std::string>> page_ids;
  char *errmsg = nullptr;
  int rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &page_ids, &errmsg);
  if (rc != SQLITE_OK) {
    string ss(errmsg);
    sqlite3_free(errmsg);
    sqlite3_close(db);
    db = nullptr;
    RETURN_STATUS_UNEXPECTED(std::string("Failed to execute sql [") + common::SafeCStr(sql) + " ], " + ss);
  } else {
    MS_LOG(DEBUG) << "Succeed to get " << page_ids.size() << "pages from index.";
  }
  for (int i = 0; i < static_cast<int>(page_ids.size()); ++i) {
    (*pages_ptr)->emplace_back(std::stoull(page_ids[i][0]));
  }
  sqlite3_free(errmsg);
  return Status::OK();
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

Status ShardReader::QueryWithCriteria(sqlite3 *db, const string &sql, const string &criteria,
                                      std::shared_ptr<std::vector<std::vector<std::string>>> labels_ptr) {
  sqlite3_stmt *stmt = nullptr;
  if (sqlite3_prepare_v2(db, common::SafeCStr(sql), -1, &stmt, 0) != SQLITE_OK) {
    RETURN_STATUS_UNEXPECTED("Failed to prepare statement sql [ " + sql + " ].");
  }
  int index = sqlite3_bind_parameter_index(stmt, ":criteria");
  if (sqlite3_bind_text(stmt, index, common::SafeCStr(criteria), -1, SQLITE_STATIC) != SQLITE_OK) {
    RETURN_STATUS_UNEXPECTED("Failed to bind parameter of sql, index: " + std::to_string(index) +
                             ", field value: " + criteria);
  }
  int rc = sqlite3_step(stmt);
  while (rc != SQLITE_DONE) {
    vector<string> tmp;
    int ncols = sqlite3_column_count(stmt);
    for (int i = 0; i < ncols; i++) {
      tmp.emplace_back(reinterpret_cast<const char *>(sqlite3_column_text(stmt, i)));
    }
    labels_ptr->push_back(tmp);
    rc = sqlite3_step(stmt);
  }
  (void)sqlite3_finalize(stmt);
  return Status::OK();
}

Status ShardReader::GetLabelsFromBinaryFile(int shard_id, const std::vector<std::string> &columns,
                                            const std::vector<std::vector<std::string>> &label_offsets,
                                            std::shared_ptr<std::vector<json>> *labels_ptr) {
  RETURN_UNEXPECTED_IF_NULL(labels_ptr);
  std::string file_name = file_paths_[shard_id];
  auto realpath = FileUtils::GetRealPath(file_name.data());
  CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(), "Failed to get real path, path=" + file_name);

  std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
  fs->open(realpath.value(), std::ios::in | std::ios::binary);
  CHECK_FAIL_RETURN_UNEXPECTED(fs->good(), "Failed to open file, path: " + file_name);
  // init the return
  for (unsigned int i = 0; i < label_offsets.size(); ++i) {
    (*labels_ptr)->emplace_back(json{});
  }

  for (unsigned int i = 0; i < label_offsets.size(); ++i) {
    const auto &labelOffset = label_offsets[i];
    if (labelOffset.size() < 3) {
      fs->close();
      RETURN_STATUS_UNEXPECTED("Invalid data, labelOffset size: " + std::to_string(labelOffset.size()) +
                               " is invalid.");
    }
    uint64_t label_start = std::stoull(labelOffset[1]) + kInt64Len;
    uint64_t label_end = std::stoull(labelOffset[2]);
    int raw_page_id = std::stoi(labelOffset[0]);
    auto len = label_end - label_start;
    auto label_raw = std::vector<uint8_t>(len);
    auto &io_seekg = fs->seekg(page_size_ * raw_page_id + header_size_ + label_start, std::ios::beg);
    if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
      fs->close();
      RETURN_STATUS_UNEXPECTED("Failed to seekg file, path: " + file_name);
    }

    auto &io_read = fs->read(reinterpret_cast<char *>(&label_raw[0]), len);
    if (!io_read.good() || io_read.fail() || io_read.bad()) {
      fs->close();
      RETURN_STATUS_UNEXPECTED("Failed to read file, path: " + file_name);
    }

    json label_json = json::from_msgpack(label_raw);
    json tmp = label_json;
    for (auto &col : columns) {
      if (label_json.find(col) != label_json.end()) {
        tmp[col] = label_json[col];
      }
    }
    (*(*labels_ptr))[i] = tmp;
  }
  return Status::OK();
}
Status ShardReader::GetLabelsFromPage(int page_id, int shard_id, const std::vector<std::string> &columns,
                                      const std::pair<std::string, std::string> &criteria,
                                      std::shared_ptr<std::vector<json>> *labels_ptr) {
  RETURN_UNEXPECTED_IF_NULL(labels_ptr);
  // get page info from sqlite
  auto db = database_paths_[shard_id];
  std::string sql = "SELECT PAGE_ID_RAW, PAGE_OFFSET_RAW,PAGE_OFFSET_RAW_END FROM INDEXES WHERE PAGE_ID_BLOB = " +
                    std::to_string(page_id);
  auto label_offset_ptr = std::make_shared<std::vector<std::vector<std::string>>>();
  if (!criteria.first.empty()) {
    sql += " AND " + criteria.first + "_" + std::to_string(column_schema_id_[criteria.first]) + " = :criteria";
    RETURN_IF_NOT_OK(QueryWithCriteria(db, sql, criteria.second, label_offset_ptr));
  } else {
    sql += ";";
    char *errmsg = nullptr;
    int rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, label_offset_ptr.get(), &errmsg);
    if (rc != SQLITE_OK) {
      std::ostringstream oss;
      oss << "Failed to execute sql [ " << common::SafeCStr(sql) << " ], " << errmsg;
      sqlite3_free(errmsg);
      sqlite3_close(db);
      db = nullptr;
      RETURN_STATUS_UNEXPECTED(oss.str());
    }
    MS_LOG(DEBUG) << "Succeed to get " << label_offset_ptr->size() << " records from index.";
    sqlite3_free(errmsg);
  }
  // get labels from binary file
  return GetLabelsFromBinaryFile(shard_id, columns, *label_offset_ptr, labels_ptr);
}

Status ShardReader::GetLabels(int page_id, int shard_id, const std::vector<std::string> &columns,
                              const std::pair<std::string, std::string> &criteria,
                              std::shared_ptr<std::vector<json>> *labels_ptr) {
  RETURN_UNEXPECTED_IF_NULL(labels_ptr);
  if (all_in_index_) {
    auto db = database_paths_[shard_id];
    std::string fields;
    for (unsigned int i = 0; i < columns.size(); ++i) {
      if (i > 0) fields += ',';
      uint64_t schema_id = column_schema_id_[columns[i]];
      fields += columns[i] + "_" + std::to_string(schema_id);
    }
    if (fields.empty()) {
      fields = "*";
    }
    auto labels = std::make_shared<std::vector<std::vector<std::string>>>();
    std::string sql = "SELECT " + fields + " FROM INDEXES WHERE PAGE_ID_BLOB = " + std::to_string(page_id);
    if (!criteria.first.empty()) {
      sql += " AND " + criteria.first + "_" + std::to_string(column_schema_id_[criteria.first]) + " = " + ":criteria";
      RETURN_IF_NOT_OK(QueryWithCriteria(db, sql, criteria.second, labels));
    } else {
      sql += ";";
      char *errmsg = nullptr;
      int rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, labels.get(), &errmsg);
      if (rc != SQLITE_OK) {
        std::ostringstream oss;
        oss << "Failed to execute sql [ " << common::SafeCStr(sql) << " ], " << errmsg;
        sqlite3_free(errmsg);
        sqlite3_close(db);
        db = nullptr;
        RETURN_STATUS_UNEXPECTED(oss.str());
      } else {
        MS_LOG(DEBUG) << "Succeed to get " << static_cast<int>(labels->size()) << " records from index.";
      }
      sqlite3_free(errmsg);
    }
    for (unsigned int i = 0; i < labels->size(); ++i) {
      (*labels_ptr)->emplace_back(json{});
    }
    for (unsigned int i = 0; i < labels->size(); ++i) {
      json construct_json;
      for (unsigned int j = 0; j < columns.size(); ++j) {
        // construct json "f1": value
        auto schema = shard_header_->GetSchemas()[0]->GetSchema()["schema"];

        // convert the string to base type by schema
        if (schema[columns[j]]["type"] == "int32") {
          construct_json[columns[j]] = StringToNum<int32_t>((*labels)[i][j]);
        } else if (schema[columns[j]]["type"] == "int64") {
          construct_json[columns[j]] = StringToNum<int64_t>((*labels)[i][j]);
        } else if (schema[columns[j]]["type"] == "float32") {
          construct_json[columns[j]] = StringToNum<float>((*labels)[i][j]);
        } else if (schema[columns[j]]["type"] == "float64") {
          construct_json[columns[j]] = StringToNum<double>((*labels)[i][j]);
        } else {
          construct_json[columns[j]] = std::string((*labels)[i][j]);
        }
      }
      (*(*labels_ptr))[i] = construct_json;
    }
    return Status::OK();
  }
  return GetLabelsFromPage(page_id, shard_id, columns, criteria, labels_ptr);
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
    MS_LOG(ERROR) << "Invalid data, field " << category_field << " does not exist.";
    return -1;
  }
  std::shared_ptr<std::string> fn_ptr;
  (void)ShardIndexGenerator::GenerateFieldName(std::make_pair(map_schema_id_fields[category_field], category_field),
                                               &fn_ptr);
  std::string sql = "SELECT DISTINCT " + *fn_ptr + " FROM INDEXES";
  std::vector<std::thread> threads = std::vector<std::thread>(shard_count);
  auto category_ptr = std::make_shared<std::set<std::string>>();
  sqlite3 *db = nullptr;
  for (int x = 0; x < shard_count; x++) {
    int rc = sqlite3_open_v2(common::SafeCStr(file_paths_[x] + ".db"), &db, SQLITE_OPEN_READONLY, nullptr);
    if (SQLITE_OK != rc) {
      MS_LOG(ERROR) << "Failed to open database: " << file_paths_[x] + ".db, " << sqlite3_errmsg(db);
      return -1;
    }
    threads[x] = std::thread(&ShardReader::GetClassesInShard, this, db, x, sql, category_ptr);
  }

  for (int x = 0; x < shard_count; x++) {
    threads[x].join();
  }
  sqlite3_close(db);
  return category_ptr->size();
}

Status ShardReader::CountTotalRows(const std::vector<std::string> &file_paths, bool load_dataset,
                                   const std::shared_ptr<ShardOperator> &ops, int64_t *count, const int num_padded) {
  RETURN_IF_NOT_OK(Init(file_paths, load_dataset));
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
        root = false;
      }
    } else if (std::dynamic_pointer_cast<ShardCategory>(op)) {
      auto category_op = std::dynamic_pointer_cast<ShardCategory>(op);
      std::string category_field = category_op->GetCategoryField();
      auto num_classes = GetNumClasses(category_field);
      num_samples = category_op->GetNumSamples(num_samples, num_classes);
      if (std::dynamic_pointer_cast<ShardPkSample>(op)) {
        auto tmp = std::dynamic_pointer_cast<ShardPkSample>(op)->GetNumSamples();
        if (tmp != 0 && num_samples != -1) {
          num_samples = std::min(num_samples, tmp);
        }
        CHECK_FAIL_RETURN_UNEXPECTED(
          num_samples != -1, "Invalid input, number of samples: " + std::to_string(num_samples) +
                               " exceeds the upper limit: " + std::to_string(std::numeric_limits<int64_t>::max()));
      }
    } else if (std::dynamic_pointer_cast<ShardSample>(op)) {
      if (std::dynamic_pointer_cast<ShardDistributedSample>(op)) {
        auto sampler_op = std::dynamic_pointer_cast<ShardDistributedSample>(op);
        if (root == true) {
          sampler_op->SetNumPaddedSamples(num_padded);
          num_samples = op->GetNumSamples(num_samples, 0);
          CHECK_FAIL_RETURN_UNEXPECTED(num_samples != -1, "Invalid data, dataset size plus number of padded samples: " +
                                                            std::to_string(num_samples) +
                                                            " can not be divisible by number of shards.");
          root = false;
        }
      } else {
        num_samples = op->GetNumSamples(num_samples, 0);
      }
    } else {
      if (num_padded > 0) {
        num_samples += num_padded;
      }
    }
  }
  *count = num_samples;
  return Status::OK();
}

Status ShardReader::Open(const std::vector<std::string> &file_paths, bool load_dataset, int n_consumer,
                         const std::vector<std::string> &selected_columns,
                         const std::vector<std::shared_ptr<ShardOperator>> &operators, int num_padded, bool lazy_load) {
  lazy_load_ = lazy_load;

  // Open file and set header by ShardReader
  RETURN_IF_NOT_OK(Init(file_paths, load_dataset));
  auto thread_limit = GetMaxThreadNum();
  if (n_consumer > thread_limit) {
    n_consumer = thread_limit;
  }
  if (n_consumer < kMinConsumerCount) {
    n_consumer = kMinConsumerCount;
  }

  selected_columns_ = selected_columns;
  RETURN_IF_NOT_OK(CheckColumnList(selected_columns_));

  // Initialize argument
  shard_count_ = static_cast<int>(file_paths_.size());
  n_consumer_ = n_consumer;
  num_padded_ = num_padded;

  operators_ = operators;
  RETURN_IF_NOT_OK(Open(n_consumer));
  return Status::OK();
}

Status ShardReader::Launch(bool is_sample_read) {
  // Get all row groups' info
  auto row_group_summary = ReadRowGroupSummary();

  // Sort row group by (group_id, shard_id), prepare for parallel reading
  std::sort(row_group_summary.begin(), row_group_summary.end(), ResortRowGroups);
  if (CreateTasks(row_group_summary, operators_).IsError()) {
    interrupt_ = true;
    RETURN_STATUS_UNEXPECTED("Failed to launch read threads.");
  }
  if (is_sample_read) {
    return Status::OK();
  }
  // Start provider consumer threads
  thread_set_ = std::vector<std::thread>(n_consumer_);
  CHECK_FAIL_RETURN_UNEXPECTED(n_consumer_ > 0 && n_consumer_ <= kMaxConsumerCount,
                               "Invalid data, number of consumer: " + std::to_string(n_consumer_) +
                                 " exceeds the upper limit: " + std::to_string(kMaxConsumerCount));

  for (int x = 0; x < n_consumer_; ++x) {
    thread_set_[x] = std::thread(&ShardReader::ConsumerByRow, this, x);
  }

  MS_LOG(INFO) << "Succeed to launch read thread.";
  return Status::OK();
}

Status ShardReader::CreateTasksByCategory(const std::shared_ptr<ShardOperator> &op) {
  CheckIfColumnInIndex(selected_columns_);
  auto category_op = std::dynamic_pointer_cast<ShardCategory>(op);
  auto categories = category_op->GetCategories();
  int64_t num_elements = category_op->GetNumElements();
  int64_t num_samples = 0;
  if (std::dynamic_pointer_cast<ShardPkSample>(op)) {
    num_samples = std::dynamic_pointer_cast<ShardPkSample>(op)->GetNumSamples();
    CHECK_FAIL_RETURN_UNEXPECTED(
      num_samples >= 0,
      "Invalid input, num_samples must be greater than or equal to 0, but got " + std::to_string(num_samples));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_elements > 0, "Invalid input, num_elements must be greater than 0, but got " + std::to_string(num_elements));
  if (categories.empty() == true) {
    std::string category_field = category_op->GetCategoryField();
    int64_t num_categories = category_op->GetNumCategories();
    CHECK_FAIL_RETURN_UNEXPECTED(num_categories > 0, "Invalid input, num_categories must be greater than 0, but got " +
                                                       std::to_string(num_elements));
    auto category_ptr = std::make_shared<std::set<std::string>>();
    RETURN_IF_NOT_OK(GetAllClasses(category_field, category_ptr));
    int i = 0;
    for (auto it = category_ptr->begin(); it != category_ptr->end() && i < num_categories; ++it) {
      categories.emplace_back(category_field, *it);
      i++;
    }
  }
  // Generate a vector of task lists.  Each catogory has a list of tasks.
  std::vector<ShardTaskList> categoryTasks(categories.size());
  for (uint32_t categoryNo = 0; categoryNo < categories.size(); ++categoryNo) {
    int category_index = 0;
    for (int shard_id = 0; shard_id < shard_count_ && category_index < num_elements; ++shard_id) {
      auto pages_ptr = std::make_shared<std::vector<uint64_t>>();
      RETURN_IF_NOT_OK(GetPagesByCategory(shard_id, categories[categoryNo], &pages_ptr));
      for (const auto &page_id : *pages_ptr) {
        if (category_index >= num_elements) {
          break;
        }
        std::shared_ptr<Page> page_ptr;
        RETURN_IF_NOT_OK(shard_header_->GetPage(shard_id, page_id, &page_ptr));
        auto group_id = page_ptr->GetPageTypeID();
        std::shared_ptr<ROW_GROUP_BRIEF> row_group_brief_ptr;
        RETURN_IF_NOT_OK(
          ReadRowGroupCriteria(group_id, shard_id, categories[categoryNo], selected_columns_, &row_group_brief_ptr));
        auto offsets = std::get<3>(*row_group_brief_ptr);

        auto number_of_rows = offsets.size();
        for (uint32_t iStart = 0; iStart < number_of_rows; iStart += 1) {
          if (category_index < num_elements) {
            categoryTasks[categoryNo].InsertTask(TaskType::kCommonTask, shard_id, group_id,
                                                 std::get<3>(*row_group_brief_ptr)[iStart],
                                                 std::get<4>(*row_group_brief_ptr)[iStart]);
            category_index++;
          }
        }
        MS_LOG(INFO) << "Category #" << categoryNo << " has " << categoryTasks[categoryNo].Size() << " tasks";
      }
    }
  }
  tasks_ = ShardTaskList::Combine(categoryTasks, category_op->GetReplacement(), num_elements, num_samples);

  tasks_.InitSampleIds();
  RETURN_IF_NOT_OK((*category_op)(tasks_));
  return Status::OK();
}

Status ShardReader::CreateTasksByRow(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                                     const std::vector<std::shared_ptr<ShardOperator>> &operators) {
  CheckIfColumnInIndex(selected_columns_);
  std::shared_ptr<ROW_GROUPS> row_group_ptr;
  RETURN_IF_NOT_OK(ReadAllRowGroup(selected_columns_, &row_group_ptr));
  auto &offsets = std::get<0>(*row_group_ptr);
  auto &local_columns = std::get<1>(*row_group_ptr);
  CHECK_FAIL_RETURN_UNEXPECTED(shard_count_ <= kMaxFileCount,
                               "Invalid data, number of shards: " + std::to_string(shard_count_) +
                                 " exceeds the upper limit: " + std::to_string(kMaxFileCount));
  int sample_count = 0;
  for (int shard_id = 0; shard_id < shard_count_; shard_id++) {
    sample_count += offsets[shard_id].size();
  }
  MS_LOG(DEBUG) << "Succeed to get " << sample_count << " records from dataset.";

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
  return Status::OK();
}

Status ShardReader::CreateLazyTasksByRow(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                                         const std::vector<std::shared_ptr<ShardOperator>> &operators) {
  CheckIfColumnInIndex(selected_columns_);
  CHECK_FAIL_RETURN_UNEXPECTED(shard_count_ <= kMaxFileCount,
                               "Invalid data, number of shards: " + std::to_string(shard_count_) +
                                 " exceeds the upper limit: " + std::to_string(kMaxFileCount));
  uint32_t sample_count = shard_sample_count_[shard_sample_count_.size() - 1];
  MS_LOG(DEBUG) << "Succeed to get " << sample_count << " records from dataset.";

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
  return Status::OK();
}

Status ShardReader::CreateTasks(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
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
      RETURN_IF_NOT_OK(CreateTasksByRow(row_group_summary, operators));
    } else {
      RETURN_IF_NOT_OK(CreateLazyTasksByRow(row_group_summary, operators));
    }

    // need padded sample to the task
    if (num_padded_ > 0) {
      for (int i = 0; i < num_padded_; ++i) {
        tasks_.InsertTask(TaskType::kPaddedTask, 0, 0, {}, json());
      }
    }
  } else {
    RETURN_IF_NOT_OK(CreateTasksByCategory(operators[category_operator]));
  }
  MS_LOG(DEBUG) << "Succeed to create " << tasks_.Size() << " initial task to start with before sampling.";
  tasks_.InitSampleIds();

  for (uint32_t operator_no = 0; operator_no < operators.size(); operator_no++) {
    const auto &op = operators[operator_no];
    if (std::dynamic_pointer_cast<ShardCategory>(op)) {
      continue;
    }

    if (std::dynamic_pointer_cast<ShardDistributedSample>(op) || std::dynamic_pointer_cast<ShardShuffle>(op)) {
      op->SetShardSampleCount(shard_sample_count_);
    }
    RETURN_IF_NOT_OK((*op)(tasks_));
  }

  if (tasks_.permutation_.empty()) tasks_.MakePerm();
  num_rows_ = tasks_.Size();
  MS_LOG(INFO) << "The total number of samples is " << num_rows_
               << ", the number of samples after sampling is: " << tasks_.sample_ids_.size();

  return Status::OK();
}

Status ShardReader::ConsumerOneTask(int task_id, uint32_t consumer_id,
                                    std::shared_ptr<TASK_CONTENT> *task_content_ptr) {
  RETURN_UNEXPECTED_IF_NULL(task_content_ptr);
  // All tasks are done
  CHECK_FAIL_RETURN_UNEXPECTED(
    task_id < static_cast<int>(tasks_.Size()),
    "Invalid data, task id: " + std::to_string(task_id) + " exceeds the upper limit: " + std::to_string(tasks_.Size()));
  uint32_t shard_id = 0;
  uint32_t group_id = 0;
  uint32_t blob_start = 0;
  uint32_t blob_end = 0;
  json var_fields;
  // Pick up task from task list
  ShardTask task = tasks_.GetTaskByID(task_id);

  // check task type
  auto task_type = std::get<0>(task);
  if (task_type == TaskType::kPaddedTask) {
    *task_content_ptr =
      std::make_shared<TASK_CONTENT>(TaskType::kPaddedTask, std::vector<std::tuple<std::vector<uint8_t>, json>>());
    return Status::OK();
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
    std::shared_ptr<ROW_GROUPS> row_group_ptr;
    RETURN_IF_NOT_OK(ReadRowGroupByShardIDAndSampleID(selected_columns_, shard_id, sample_id_in_shard, &row_group_ptr));
    auto &offsets = std::get<0>(*row_group_ptr);
    auto &local_columns = std::get<1>(*row_group_ptr);

    group_id = offsets[shard_id][0][1];       // group_id
    blob_start = offsets[shard_id][0][2];     // blob start
    blob_end = offsets[shard_id][0][3];       // blob end
    var_fields = local_columns[shard_id][0];  // scalar variable field
  }

  // read the blob from data file
  std::shared_ptr<Page> page_ptr;
  RETURN_IF_NOT_OK(shard_header_->GetPageByGroupId(group_id, shard_id, &page_ptr));
  MS_LOG(DEBUG) << "Success to get page by group id: " << group_id;

  // Pack image list
  std::vector<uint8_t> images(blob_end - blob_start);
  auto file_offset = header_size_ + page_size_ * (page_ptr->GetPageID()) + blob_start;

  auto &io_seekg = file_streams_random_[consumer_id][shard_id]->seekg(file_offset, std::ios::beg);
  if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
    file_streams_random_[consumer_id][shard_id]->close();
    RETURN_STATUS_UNEXPECTED("Failed to seekg file.");
  }
  auto &io_read =
    file_streams_random_[consumer_id][shard_id]->read(reinterpret_cast<char *>(&images[0]), blob_end - blob_start);
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    file_streams_random_[consumer_id][shard_id]->close();
    RETURN_STATUS_UNEXPECTED("Failed to read file.");
  }

  // Deliver batch data to output map
  std::vector<std::tuple<std::vector<uint8_t>, json>> batch;
  batch.emplace_back(std::move(images), std::move(var_fields));

  *task_content_ptr = std::make_shared<TASK_CONTENT>(TaskType::kCommonTask, std::move(batch));
  return Status::OK();
}

void ShardReader::ConsumerByRow(int consumer_id) {
  // Set thread name
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  auto thread_id = kThreadName + std::to_string(consumer_id);
  prctl(PR_SET_NAME, common::SafeCStr(thread_id), 0, 0, 0);
#endif

  // Loop forever
  for (;;) {
    int sample_id_pos = 0;

    // Get next task ID
    sample_id_pos = sample_id_position_++;

    // All tasks are done
    if (sample_id_pos >= static_cast<int>(tasks_.sample_ids_.size())) {
      return;
    }
    auto task_content_ptr =
      std::make_shared<TASK_CONTENT>(TaskType::kCommonTask, std::vector<std::tuple<std::vector<uint8_t>, json>>());
    if (ConsumerOneTask(tasks_.sample_ids_[sample_id_pos], consumer_id, &task_content_ptr).IsError()) {
      MS_LOG(ERROR) << "Error raised in ConsumerOneTask function.";
      return;
    }
    const auto &batch = (*task_content_ptr).second;
    // Hanging if maximum map size exceeded
    //   otherwise, set batch data in map
    {
      std::unique_lock<std::mutex> lck(mtx_delivery_);
      cv_delivery_.wait(lck,
                        [sample_id_pos, this] { return interrupt_ || sample_id_pos <= deliver_id_ + kNumBatchInMap; });
      if (interrupt_) {
        return;
      }
      delivery_map_[sample_id_pos] =
        std::make_shared<std::vector<std::tuple<std::vector<uint8_t>, json>>>(std::move(batch));
    }
    cv_iterator_.notify_one();
  }
}

std::vector<std::tuple<std::vector<uint8_t>, json>> ShardReader::GetNext() {
  if (interrupt_) {
    return std::vector<std::tuple<std::vector<uint8_t>, json>>();
  }
  if (deliver_id_ >= static_cast<int>(tasks_.sample_ids_.size())) {
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

TASK_CONTENT ShardReader::GetNextById(const int64_t &task_id, const int32_t &consumer_id) {
  auto task_content_ptr =
    std::make_shared<TASK_CONTENT>(TaskType::kCommonTask, std::vector<std::tuple<std::vector<uint8_t>, json>>());
  if (interrupt_) {
    return *task_content_ptr;
  }
  (void)ConsumerOneTask(task_id, consumer_id, &task_content_ptr);
  return std::move(*task_content_ptr);
}

Status ShardReader::UnCompressBlob(const std::vector<uint8_t> &raw_blob_data,
                                   std::shared_ptr<std::vector<std::vector<uint8_t>>> *blob_data_ptr) {
  RETURN_UNEXPECTED_IF_NULL(blob_data_ptr);
  auto loaded_columns = selected_columns_.size() == 0 ? shard_column_->GetColumnName() : selected_columns_;
  auto blob_fields = GetBlobFields().second;
  for (uint32_t i_col = 0; i_col < loaded_columns.size(); ++i_col) {
    if (std::find(blob_fields.begin(), blob_fields.end(), loaded_columns[i_col]) == blob_fields.end()) continue;
    const unsigned char *data = nullptr;
    std::unique_ptr<unsigned char[]> data_ptr;
    uint64_t n_bytes = 0;
    RETURN_IF_NOT_OK(
      shard_column_->GetColumnFromBlob(loaded_columns[i_col], raw_blob_data, &data, &data_ptr, &n_bytes));
    if (data == nullptr) {
      data = reinterpret_cast<const unsigned char *>(data_ptr.get());
    }
    std::vector<uint8_t> column(data, data + (n_bytes / sizeof(unsigned char)));
    (*blob_data_ptr)->push_back(column);
  }
  return Status::OK();
}

Status ShardReader::GetTotalBlobSize(int64_t *total_blob_size) {
  *total_blob_size = total_blob_size_;
  return Status::OK();
}

void ShardReader::Reset() {
  {
    std::lock_guard<std::mutex> lck(mtx_delivery_);
    sample_id_position_ = 0;
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
      auto s = (*op)(tasks_);
      if (s.IsError()) {
        MS_LOG(WARNING) << "Failed to redo randomSampler in new epoch.";
      }
    } else if (std::dynamic_pointer_cast<ShardDistributedSample>(op)) {
      auto s = (*op)(tasks_);
      if (s.IsError()) {
        MS_LOG(WARNING) << "Failed to redo distributeSampler in new epoch.";
      }
    }
  }
  if (tasks_.permutation_.empty()) tasks_.MakePerm();
}

const std::vector<int> *ShardReader::GetSampleIds() {
  // return const reference to private sample id list.
  return &(this->tasks_.sample_ids_);
}

}  // namespace mindrecord
}  // namespace mindspore
