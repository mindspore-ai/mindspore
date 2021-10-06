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
#include "minddata/mindrecord/include/shard_index_generator.h"

#include "utils/file_utils.h"
#include "utils/ms_utils.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::DEBUG;
using mindspore::MsLogLevel::ERROR;
using mindspore::MsLogLevel::INFO;

namespace mindspore {
namespace mindrecord {
ShardIndexGenerator::ShardIndexGenerator(const std::string &file_path, bool append)
    : file_path_(file_path),
      append_(append),
      page_size_(0),
      header_size_(0),
      schema_count_(0),
      task_(0),
      write_success_(true) {}

Status ShardIndexGenerator::Build() {
  std::shared_ptr<json> header_ptr;
  RETURN_IF_NOT_OK(ShardHeader::BuildSingleHeader(file_path_, &header_ptr));
  auto ds = std::make_shared<std::vector<std::string>>();
  RETURN_IF_NOT_OK(GetDatasetFiles(file_path_, (*header_ptr)["shard_addresses"], &ds));
  ShardHeader header = ShardHeader();
  RETURN_IF_NOT_OK(header.BuildDataset(*ds));
  shard_header_ = header;
  MS_LOG(INFO) << "Initialize header from mindrecord file for index successfully.";
  return Status::OK();
}

Status ShardIndexGenerator::GetValueByField(const string &field, json input, std::shared_ptr<std::string> *value) {
  RETURN_UNEXPECTED_IF_NULL(value);
  CHECK_FAIL_RETURN_UNEXPECTED(!field.empty(), "The input field is empty.");
  CHECK_FAIL_RETURN_UNEXPECTED(!input.empty(), "The input json is empty.");

  // parameter input does not contain the field
  CHECK_FAIL_RETURN_UNEXPECTED(input.find(field) != input.end(),
                               "The field " + field + " is not found in json " + input.dump());

  // schema does not contain the field
  auto schema = shard_header_.GetSchemas()[0]->GetSchema()["schema"];
  CHECK_FAIL_RETURN_UNEXPECTED(schema.find(field) != schema.end(),
                               "The field " + field + " is not found in schema " + schema.dump());

  // field should be scalar type
  CHECK_FAIL_RETURN_UNEXPECTED(
    kScalarFieldTypeSet.find(schema[field]["type"]) != kScalarFieldTypeSet.end(),
    "The field " + field + " type is " + schema[field]["type"].dump() + " which is not retrievable.");

  if (kNumberFieldTypeSet.find(schema[field]["type"]) != kNumberFieldTypeSet.end()) {
    auto schema_field_options = schema[field];
    CHECK_FAIL_RETURN_UNEXPECTED(
      schema_field_options.find("shape") == schema_field_options.end(),
      "The field " + field + " shape is " + schema[field]["shape"].dump() + " which is not retrievable.");
    *value = std::make_shared<std::string>(input[field].dump());
  } else {
    // the field type is string in here
    *value = std::make_shared<std::string>(input[field].get<std::string>());
  }
  return Status::OK();
}

std::string ShardIndexGenerator::TakeFieldType(const string &field_path, json schema) {
  std::vector<std::string> field_name = StringSplit(field_path, kPoint);
  for (uint64_t i = 0; i < field_name.size(); ++i) {
    try {
      if (i != field_name.size() - 1) {
        // Get type information from json schema
        schema = schema.at(field_name[i]);
        schema = schema.at("properties");
      } else {
        // standard root layer exist "properties" if type is "object"
        if (schema.find("properties") != schema.end()) {
          schema = schema.at("properties");
        }
        schema = schema.at(field_name[i]);
        std::string field_type = schema.at("type").dump();
        if (field_type.length() <= 2) {
          return "";
        } else {
          return field_type.substr(1, field_type.length() - 2);
        }
      }
    } catch (...) {
      MS_LOG(WARNING) << "Exception occurred while get field type.";
      return "";
    }
  }
  return "";
}

std::string ShardIndexGenerator::ConvertJsonToSQL(const std::string &json) {
  if (kDbJsonMap.find(json) != kDbJsonMap.end()) {
    return kDbJsonMap.at(json);
  } else {
    return "TEXT";
  }
}

int ShardIndexGenerator::Callback(void *not_used, int argc, char **argv, char **az_col_name) {
  for (auto i = 0; i < argc; i++) {
    if (argv[i] != nullptr) {
      MS_LOG(INFO) << az_col_name[i] << " = " << (argv[i] ? argv[i] : "nullptr");
    }
  }
  MS_LOG(INFO) << "\n";
  return 0;
}

Status ShardIndexGenerator::ExecuteSQL(const std::string &sql, sqlite3 *db, const std::string &success_msg) {
  char *z_err_msg = nullptr;
  int rc = sqlite3_exec(db, common::SafeCStr(sql), Callback, nullptr, &z_err_msg);
  if (rc != SQLITE_OK) {
    std::ostringstream oss;
    oss << "Failed to exec sqlite3_exec, msg is: " << z_err_msg;
    MS_LOG(DEBUG) << oss.str();
    sqlite3_free(z_err_msg);
    sqlite3_close(db);
    RETURN_STATUS_UNEXPECTED(oss.str());
  } else {
    if (!success_msg.empty()) {
      MS_LOG(DEBUG) << "Suceess to exec sqlite3_exec, msg is: " << success_msg;
    }
    sqlite3_free(z_err_msg);
    return Status::OK();
  }
}

Status ShardIndexGenerator::GenerateFieldName(const std::pair<uint64_t, std::string> &field,
                                              std::shared_ptr<std::string> *fn_ptr) {
  RETURN_UNEXPECTED_IF_NULL(fn_ptr);
  // Replaces dots and dashes with underscores for SQL use
  std::string field_name = field.second;
  // white list to avoid sql injection
  std::replace_if(
    field_name.begin(), field_name.end(), [](char x) { return (x == '-' || x == '.'); }, '_');
  auto pos = std::find_if_not(field_name.begin(), field_name.end(), [](char x) {
    return (x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z') || x == '_' || (x >= '0' && x <= '9');
  });
  CHECK_FAIL_RETURN_UNEXPECTED(
    pos == field_name.end(),
    "Field name must be composed of '0-9' or 'a-z' or 'A-Z' or '_', field_name: " + field_name);
  *fn_ptr = std::make_shared<std::string>(field_name + "_" + std::to_string(field.first));
  return Status::OK();
}

Status ShardIndexGenerator::CheckDatabase(const std::string &shard_address, sqlite3 **db) {
  std::optional<std::string> dir = "";
  std::optional<std::string> local_file_name = "";
  FileUtils::SplitDirAndFileName(shard_address, &dir, &local_file_name);
  if (!dir.has_value()) {
    dir = ".";
  }

  auto realpath = FileUtils::GetRealPath(dir.value().data());
  CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(), "Get real path failed, path=" + shard_address);

  std::optional<std::string> whole_path = "";
  FileUtils::ConcatDirAndFileName(&realpath, &local_file_name, &whole_path);

  std::ifstream fin(whole_path.value());
  if (!append_ && fin.good()) {
    fin.close();
    RETURN_STATUS_UNEXPECTED("Invalid file, DB file already exist: " + shard_address);
  }
  fin.close();
  if (sqlite3_open_v2(common::SafeCStr(whole_path.value()), db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr)) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open database: " + shard_address + ", error" +
                             std::string(sqlite3_errmsg(*db)));
  }
  MS_LOG(DEBUG) << "Opened database successfully";
  return Status::OK();
}

Status ShardIndexGenerator::CreateShardNameTable(sqlite3 *db, const std::string &shard_name) {
  // create shard_name table
  std::string sql = "DROP TABLE IF EXISTS SHARD_NAME;";
  RETURN_IF_NOT_OK(ExecuteSQL(sql, db, "drop table successfully."));
  sql = "CREATE TABLE SHARD_NAME(NAME TEXT NOT NULL);";
  RETURN_IF_NOT_OK(ExecuteSQL(sql, db, "create table successfully."));
  sql = "INSERT INTO SHARD_NAME (NAME) VALUES (:SHARD_NAME);";
  sqlite3_stmt *stmt = nullptr;
  if (sqlite3_prepare_v2(db, common::SafeCStr(sql), -1, &stmt, 0) != SQLITE_OK) {
    if (stmt != nullptr) {
      (void)sqlite3_finalize(stmt);
    }
    sqlite3_close(db);
    RETURN_STATUS_UNEXPECTED("SQL error: could not prepare statement, sql: " + sql);
  }

  int index = sqlite3_bind_parameter_index(stmt, ":SHARD_NAME");
  if (sqlite3_bind_text(stmt, index, shard_name.data(), -1, SQLITE_STATIC) != SQLITE_OK) {
    (void)sqlite3_finalize(stmt);
    sqlite3_close(db);
    RETURN_STATUS_UNEXPECTED("SQL error: could not bind parameter, index: " + std::to_string(index) +
                             ", field value: " + std::string(shard_name));
  }

  if (sqlite3_step(stmt) != SQLITE_DONE) {
    (void)sqlite3_finalize(stmt);
    RETURN_STATUS_UNEXPECTED("SQL error: Could not step (execute) stmt.");
  }
  (void)sqlite3_finalize(stmt);
  return Status::OK();
}

Status ShardIndexGenerator::CreateDatabase(int shard_no, sqlite3 **db) {
  std::string shard_address = shard_header_.GetShardAddressByID(shard_no);
  CHECK_FAIL_RETURN_UNEXPECTED(!shard_address.empty(), "Shard address is empty, shard No: " + shard_no);
  std::shared_ptr<std::string> fn_ptr;
  RETURN_IF_NOT_OK(GetFileName(shard_address, &fn_ptr));
  shard_address += ".db";
  RETURN_IF_NOT_OK(CheckDatabase(shard_address, db));
  std::string sql = "DROP TABLE IF EXISTS INDEXES;";
  RETURN_IF_NOT_OK(ExecuteSQL(sql, *db, "drop table successfully."));
  sql =
    "CREATE TABLE INDEXES("
    "  ROW_ID               INT  NOT NULL, PAGE_ID_RAW          INT  NOT NULL"
    ", PAGE_OFFSET_RAW      INT  NOT NULL, PAGE_OFFSET_RAW_END  INT  NOT NULL"
    ", ROW_GROUP_ID         INT  NOT NULL, PAGE_ID_BLOB         INT  NOT NULL"
    ", PAGE_OFFSET_BLOB     INT  NOT NULL, PAGE_OFFSET_BLOB_END INT  NOT NULL";

  int field_no = 0;
  std::shared_ptr<std::string> field_ptr;
  for (const auto &field : fields_) {
    uint64_t schema_id = field.first;
    std::shared_ptr<Schema> schema_ptr;
    RETURN_IF_NOT_OK(shard_header_.GetSchemaByID(schema_id, &schema_ptr));
    json json_schema = (schema_ptr->GetSchema())["schema"];
    std::string type = ConvertJsonToSQL(TakeFieldType(field.second, json_schema));
    RETURN_IF_NOT_OK(GenerateFieldName(field, &field_ptr));
    sql += ",INC_" + std::to_string(field_no++) + " INT, " + *field_ptr + " " + type;
  }
  sql += ", PRIMARY KEY(ROW_ID";
  for (uint64_t i = 0; i < fields_.size(); ++i) {
    sql += ",INC_" + std::to_string(i);
  }
  sql += "));";
  RETURN_IF_NOT_OK(ExecuteSQL(sql, *db, "create table successfully."));
  RETURN_IF_NOT_OK(CreateShardNameTable(*db, *fn_ptr));
  return Status::OK();
}

Status ShardIndexGenerator::GetSchemaDetails(const std::vector<uint64_t> &schema_lens, std::fstream &in,
                                             std::shared_ptr<std::vector<json>> *detail_ptr) {
  RETURN_UNEXPECTED_IF_NULL(detail_ptr);
  if (schema_count_ <= kMaxSchemaCount) {
    for (int sc = 0; sc < schema_count_; ++sc) {
      std::vector<char> schema_detail(schema_lens[sc]);
      auto &io_read = in.read(&schema_detail[0], schema_lens[sc]);
      if (!io_read.good() || io_read.fail() || io_read.bad()) {
        in.close();
        RETURN_STATUS_UNEXPECTED("Failed to read file.");
      }
      auto j = json::from_msgpack(std::string(schema_detail.begin(), schema_detail.end()));
      (*detail_ptr)->emplace_back(j);
    }
  }
  return Status::OK();
}

Status ShardIndexGenerator::GenerateRawSQL(const std::vector<std::pair<uint64_t, std::string>> &fields,
                                           std::shared_ptr<std::string> *sql_ptr) {
  std::string sql =
    "INSERT INTO INDEXES (ROW_ID,ROW_GROUP_ID,PAGE_ID_RAW,PAGE_OFFSET_RAW,PAGE_OFFSET_RAW_END,"
    "PAGE_ID_BLOB,PAGE_OFFSET_BLOB,PAGE_OFFSET_BLOB_END";

  int field_no = 0;
  for (const auto &field : fields) {
    std::shared_ptr<std::string> fn_ptr;
    RETURN_IF_NOT_OK(GenerateFieldName(field, &fn_ptr));
    sql += ",INC_" + std::to_string(field_no++) + "," + *fn_ptr;
  }
  sql +=
    ") VALUES( :ROW_ID,:ROW_GROUP_ID,:PAGE_ID_RAW,:PAGE_OFFSET_RAW,:PAGE_OFFSET_RAW_END,:PAGE_ID_BLOB,"
    ":PAGE_OFFSET_BLOB,:PAGE_OFFSET_BLOB_END";
  field_no = 0;
  for (const auto &field : fields) {
    std::shared_ptr<std::string> fn_ptr;
    RETURN_IF_NOT_OK(GenerateFieldName(field, &fn_ptr));
    sql += ",:INC_" + std::to_string(field_no++) + ",:" + *fn_ptr;
  }
  sql += " )";

  *sql_ptr = std::make_shared<std::string>(sql);
  return Status::OK();
}

Status ShardIndexGenerator::BindParameterExecuteSQL(sqlite3 *db, const std::string &sql, const ROW_DATA &data) {
  sqlite3_stmt *stmt = nullptr;
  if (sqlite3_prepare_v2(db, common::SafeCStr(sql), -1, &stmt, 0) != SQLITE_OK) {
    if (stmt != nullptr) {
      (void)sqlite3_finalize(stmt);
    }
    sqlite3_close(db);
    RETURN_STATUS_UNEXPECTED("SQL error: could not prepare statement, sql: " + sql);
  }
  for (auto &row : data) {
    for (auto &field : row) {
      const auto &place_holder = std::get<0>(field);
      const auto &field_type = std::get<1>(field);
      const auto &field_value = std::get<2>(field);

      int index = sqlite3_bind_parameter_index(stmt, common::SafeCStr(place_holder));
      if (field_type == "INTEGER") {
        if (sqlite3_bind_int64(stmt, index, std::stoll(field_value)) != SQLITE_OK) {
          (void)sqlite3_finalize(stmt);
          sqlite3_close(db);
          RETURN_STATUS_UNEXPECTED("SQL error: could not bind parameter, index: " + std::to_string(index) +
                                   ", field value: " + std::string(field_value));
        }
      } else if (field_type == "NUMERIC") {
        if (sqlite3_bind_double(stmt, index, std::stold(field_value)) != SQLITE_OK) {
          (void)sqlite3_finalize(stmt);
          sqlite3_close(db);
          RETURN_STATUS_UNEXPECTED("SQL error: could not bind parameter, index: " + std::to_string(index) +
                                   ", field value: " + std::string(field_value));
        }
      } else if (field_type == "NULL") {
        if (sqlite3_bind_null(stmt, index) != SQLITE_OK) {
          (void)sqlite3_finalize(stmt);

          sqlite3_close(db);
          RETURN_STATUS_UNEXPECTED("SQL error: could not bind parameter, index: " + std::to_string(index) +
                                   ", field value: NULL");
        }
      } else {
        if (sqlite3_bind_text(stmt, index, common::SafeCStr(field_value), -1, SQLITE_STATIC) != SQLITE_OK) {
          (void)sqlite3_finalize(stmt);
          sqlite3_close(db);
          RETURN_STATUS_UNEXPECTED("SQL error: could not bind parameter, index: " + std::to_string(index) +
                                   ", field value: " + std::string(field_value));
        }
      }
    }
    if (sqlite3_step(stmt) != SQLITE_DONE) {
      (void)sqlite3_finalize(stmt);
      RETURN_STATUS_UNEXPECTED("SQL error: Could not step (execute) stmt.");
    }
    (void)sqlite3_reset(stmt);
  }
  (void)sqlite3_finalize(stmt);
  return Status::OK();
}

Status ShardIndexGenerator::AddBlobPageInfo(std::vector<std::tuple<std::string, std::string, std::string>> &row_data,
                                            const std::shared_ptr<Page> cur_blob_page, uint64_t &cur_blob_page_offset,
                                            std::fstream &in) {
  row_data.emplace_back(":PAGE_ID_BLOB", "INTEGER", std::to_string(cur_blob_page->GetPageID()));

  // blob data start
  row_data.emplace_back(":PAGE_OFFSET_BLOB", "INTEGER", std::to_string(cur_blob_page_offset));
  auto &io_seekg_blob =
    in.seekg(page_size_ * cur_blob_page->GetPageID() + header_size_ + cur_blob_page_offset, std::ios::beg);
  if (!io_seekg_blob.good() || io_seekg_blob.fail() || io_seekg_blob.bad()) {
    in.close();
    RETURN_STATUS_UNEXPECTED("Failed to seekg file.");
  }
  uint64_t image_size = 0;
  auto &io_read = in.read(reinterpret_cast<char *>(&image_size), kInt64Len);
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    MS_LOG(ERROR) << "File read failed";
    in.close();
    RETURN_STATUS_UNEXPECTED("Failed to read file.");
  }

  cur_blob_page_offset += (kInt64Len + image_size);
  row_data.emplace_back(":PAGE_OFFSET_BLOB_END", "INTEGER", std::to_string(cur_blob_page_offset));

  return Status::OK();
}

Status ShardIndexGenerator::AddIndexFieldByRawData(
  const std::vector<json> &schema_detail, std::vector<std::tuple<std::string, std::string, std::string>> &row_data) {
  auto index_fields_ptr = std::make_shared<INDEX_FIELDS>();
  RETURN_IF_NOT_OK(GenerateIndexFields(schema_detail, &index_fields_ptr));
  int index = 0;
  for (const auto &field : *index_fields_ptr) {
    // assume simple field: string , number etc.
    row_data.emplace_back(":INC_" + std::to_string(index++), "INTEGER", "0");
    row_data.emplace_back(":" + std::get<0>(field), std::get<1>(field), std::get<2>(field));
  }
  return Status::OK();
}

Status ShardIndexGenerator::GenerateRowData(int shard_no, const std::map<int, int> &blob_id_to_page_id, int raw_page_id,
                                            std::fstream &in, std::shared_ptr<ROW_DATA> *row_data_ptr) {
  RETURN_UNEXPECTED_IF_NULL(row_data_ptr);
  // current raw data page
  std::shared_ptr<Page> page_ptr;
  RETURN_IF_NOT_OK(shard_header_.GetPage(shard_no, raw_page_id, &page_ptr));
  // related blob page
  vector<pair<int, uint64_t>> row_group_list = page_ptr->GetRowGroupIds();

  // pair: row_group id, offset in raw data page
  for (pair<int, int> blob_ids : row_group_list) {
    // get blob data page according to row_group id
    auto iter = blob_id_to_page_id.find(blob_ids.first);
    CHECK_FAIL_RETURN_UNEXPECTED(iter != blob_id_to_page_id.end(), "Failed to get page id from blob id.");
    std::shared_ptr<Page> blob_page_ptr;
    RETURN_IF_NOT_OK(shard_header_.GetPage(shard_no, iter->second, &blob_page_ptr));
    // offset in current raw data page
    auto cur_raw_page_offset = static_cast<uint64_t>(blob_ids.second);
    uint64_t cur_blob_page_offset = 0;
    for (unsigned int i = blob_page_ptr->GetStartRowID(); i < blob_page_ptr->GetEndRowID(); ++i) {
      std::vector<std::tuple<std::string, std::string, std::string>> row_data;
      row_data.emplace_back(":ROW_ID", "INTEGER", std::to_string(i));
      row_data.emplace_back(":ROW_GROUP_ID", "INTEGER", std::to_string(blob_page_ptr->GetPageTypeID()));
      row_data.emplace_back(":PAGE_ID_RAW", "INTEGER", std::to_string(page_ptr->GetPageID()));

      // raw data start
      row_data.emplace_back(":PAGE_OFFSET_RAW", "INTEGER", std::to_string(cur_raw_page_offset));

      // calculate raw data end
      auto &io_seekg =
        in.seekg(page_size_ * (page_ptr->GetPageID()) + header_size_ + cur_raw_page_offset, std::ios::beg);
      if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
        in.close();
        RETURN_STATUS_UNEXPECTED("Failed to seekg file.");
      }
      std::vector<uint64_t> schema_lens;
      if (schema_count_ <= kMaxSchemaCount) {
        for (int sc = 0; sc < schema_count_; sc++) {
          uint64_t schema_size = 0;

          auto &io_read = in.read(reinterpret_cast<char *>(&schema_size), kInt64Len);
          if (!io_read.good() || io_read.fail() || io_read.bad()) {
            in.close();
            RETURN_STATUS_UNEXPECTED("Failed to read file.");
          }

          cur_raw_page_offset += (kInt64Len + schema_size);
          schema_lens.push_back(schema_size);
        }
      }
      row_data.emplace_back(":PAGE_OFFSET_RAW_END", "INTEGER", std::to_string(cur_raw_page_offset));

      // Getting schema for getting data for fields
      auto detail_ptr = std::make_shared<std::vector<json>>();
      RETURN_IF_NOT_OK(GetSchemaDetails(schema_lens, in, &detail_ptr));
      // start blob page info
      RETURN_IF_NOT_OK(AddBlobPageInfo(row_data, blob_page_ptr, cur_blob_page_offset, in));

      // start index field
      AddIndexFieldByRawData(*detail_ptr, row_data);
      (*row_data_ptr)->push_back(std::move(row_data));
    }
  }
  return Status::OK();
}

Status ShardIndexGenerator::GenerateIndexFields(const std::vector<json> &schema_detail,
                                                std::shared_ptr<INDEX_FIELDS> *index_fields_ptr) {
  RETURN_UNEXPECTED_IF_NULL(index_fields_ptr);
  // index fields
  std::vector<std::pair<uint64_t, std::string>> index_fields = shard_header_.GetFields();
  for (const auto &field : index_fields) {
    CHECK_FAIL_RETURN_UNEXPECTED(field.first < schema_detail.size(), "Index field id is out of range.");
    std::shared_ptr<std::string> field_val_ptr;
    RETURN_IF_NOT_OK(GetValueByField(field.second, schema_detail[field.first], &field_val_ptr));
    std::shared_ptr<Schema> schema_ptr;
    RETURN_IF_NOT_OK(shard_header_.GetSchemaByID(field.first, &schema_ptr));
    std::string field_type = ConvertJsonToSQL(TakeFieldType(field.second, schema_ptr->GetSchema()["schema"]));
    std::shared_ptr<std::string> fn_ptr;
    RETURN_IF_NOT_OK(GenerateFieldName(field, &fn_ptr));
    (*index_fields_ptr)->emplace_back(*fn_ptr, field_type, *field_val_ptr);
  }
  return Status::OK();
}

Status ShardIndexGenerator::ExecuteTransaction(const int &shard_no, sqlite3 *db, const std::vector<int> &raw_page_ids,
                                               const std::map<int, int> &blob_id_to_page_id) {
  // Add index data to database
  std::string shard_address = shard_header_.GetShardAddressByID(shard_no);
  CHECK_FAIL_RETURN_UNEXPECTED(!shard_address.empty(), "shard address is empty.");

  auto realpath = FileUtils::GetRealPath(shard_address.data());
  CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(), "Get real path failed, path=" + shard_address);
  std::fstream in;
  in.open(realpath.value(), std::ios::in | std::ios::binary);
  if (!in.good()) {
    in.close();
    RETURN_STATUS_UNEXPECTED("Failed to open file: " + shard_address);
  }
  (void)sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);
  for (int raw_page_id : raw_page_ids) {
    std::shared_ptr<std::string> sql_ptr;
    RELEASE_AND_RETURN_IF_NOT_OK(GenerateRawSQL(fields_, &sql_ptr), db, in);
    auto row_data_ptr = std::make_shared<ROW_DATA>();
    RELEASE_AND_RETURN_IF_NOT_OK(GenerateRowData(shard_no, blob_id_to_page_id, raw_page_id, in, &row_data_ptr), db, in);
    RELEASE_AND_RETURN_IF_NOT_OK(BindParameterExecuteSQL(db, *sql_ptr, *row_data_ptr), db, in);
    MS_LOG(INFO) << "Insert " << row_data_ptr->size() << " rows to index db.";
  }
  (void)sqlite3_exec(db, "END TRANSACTION;", nullptr, nullptr, nullptr);
  in.close();

  // Close database
  sqlite3_close(db);
  db = nullptr;
  return Status::OK();
}

Status ShardIndexGenerator::WriteToDatabase() {
  fields_ = shard_header_.GetFields();
  page_size_ = shard_header_.GetPageSize();
  header_size_ = shard_header_.GetHeaderSize();
  schema_count_ = shard_header_.GetSchemaCount();
  CHECK_FAIL_RETURN_UNEXPECTED(shard_header_.GetShardCount() <= kMaxShardCount,
                               "num shards: " + std::to_string(shard_header_.GetShardCount()) +
                                 " exceeds max count:" + std::to_string(kMaxSchemaCount));

  task_ = 0;  // set two atomic vars to initial value
  write_success_ = true;

  // spawn half the physical threads or total number of shards whichever is smaller
  const unsigned int num_workers =
    std::min(std::thread::hardware_concurrency() / 2 + 1, static_cast<unsigned int>(shard_header_.GetShardCount()));

  std::vector<std::thread> threads;
  threads.reserve(num_workers);

  for (size_t t = 0; t < threads.capacity(); t++) {
    threads.emplace_back(std::thread(&ShardIndexGenerator::DatabaseWriter, this));
  }

  for (size_t t = 0; t < threads.capacity(); t++) {
    threads[t].join();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(write_success_, "Failed to write data to db.");
  return Status::OK();
}

void ShardIndexGenerator::DatabaseWriter() {
  int shard_no = task_++;
  while (shard_no < shard_header_.GetShardCount()) {
    sqlite3 *db = nullptr;
    if (CreateDatabase(shard_no, &db).IsError()) {
      MS_LOG(ERROR) << "Failed to create Generate database.";
      write_success_ = false;
      return;
    }
    MS_LOG(INFO) << "Init index db for shard: " << shard_no << " successfully.";
    // Pre-processing page information
    auto total_pages = shard_header_.GetLastPageId(shard_no) + 1;

    std::map<int, int> blob_id_to_page_id;
    std::vector<int> raw_page_ids;
    for (uint64_t i = 0; i < total_pages; ++i) {
      std::shared_ptr<Page> page_ptr;
      if (shard_header_.GetPage(shard_no, i, &page_ptr).IsError()) {
        MS_LOG(ERROR) << "Failed to get page.";
        write_success_ = false;
        return;
      }
      if (page_ptr->GetPageType() == "RAW_DATA") {
        raw_page_ids.push_back(i);
      } else if (page_ptr->GetPageType() == "BLOB_DATA") {
        blob_id_to_page_id[page_ptr->GetPageTypeID()] = i;
      }
    }

    if (ExecuteTransaction(shard_no, db, raw_page_ids, blob_id_to_page_id).IsError()) {
      MS_LOG(ERROR) << "Failed to execute transaction.";
      write_success_ = false;
      return;
    }
    MS_LOG(INFO) << "Generate index db for shard: " << shard_no << " successfully.";
    shard_no = task_++;
  }
}
Status ShardIndexGenerator::Finalize(const std::vector<std::string> file_names) {
  CHECK_FAIL_RETURN_UNEXPECTED(!file_names.empty(), "Mindrecord files is empty.");
  ShardIndexGenerator sg{file_names[0]};
  RETURN_IF_NOT_OK(sg.Build());
  RETURN_IF_NOT_OK(sg.WriteToDatabase());
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
