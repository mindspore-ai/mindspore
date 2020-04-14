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
#include <thread>

#include "mindrecord/include/shard_index_generator.h"
#include "common/utils.h"

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

MSRStatus ShardIndexGenerator::Build() {
  ShardHeader header = ShardHeader();
  if (header.Build(file_path_) != SUCCESS) {
    MS_LOG(ERROR) << "Build shard schema failed.";
    return FAILED;
  }
  shard_header_ = header;
  MS_LOG(INFO) << "Init header from mindrecord file for index successfully.";
  return SUCCESS;
}

std::pair<MSRStatus, std::string> ShardIndexGenerator::GetValueByField(const string &field, json input) {
  if (field.empty()) {
    MS_LOG(ERROR) << "The input field is None.";
    return {FAILED, ""};
  }

  if (input.empty()) {
    MS_LOG(ERROR) << "The input json is None.";
    return {FAILED, ""};
  }

  // parameter input does not contain the field
  if (input.find(field) == input.end()) {
    MS_LOG(ERROR) << "The field " << field << " is not found in parameter " << input;
    return {FAILED, ""};
  }

  // schema does not contain the field
  auto schema = shard_header_.get_schemas()[0]->GetSchema()["schema"];
  if (schema.find(field) == schema.end()) {
    MS_LOG(ERROR) << "The field " << field << " is not found in schema " << schema;
    return {FAILED, ""};
  }

  // field should be scalar type
  if (kScalarFieldTypeSet.find(schema[field]["type"]) == kScalarFieldTypeSet.end()) {
    MS_LOG(ERROR) << "The field " << field << " type is " << schema[field]["type"] << ", it is not retrievable";
    return {FAILED, ""};
  }

  if (kNumberFieldTypeSet.find(schema[field]["type"]) != kNumberFieldTypeSet.end()) {
    auto schema_field_options = schema[field];
    if (schema_field_options.find("shape") == schema_field_options.end()) {
      return {SUCCESS, input[field].dump()};
    } else {
      // field with shape option
      MS_LOG(ERROR) << "The field " << field << " shape is " << schema[field]["shape"] << " which is not retrievable";
      return {FAILED, ""};
    }
  }

  // the field type is string in here
  return {SUCCESS, input[field].get<std::string>()};
}

std::string ShardIndexGenerator::TakeFieldType(const string &field_path, json schema) {
  std::vector<std::string> field_name = StringSplit(field_path, kPoint);
  for (uint64_t i = 0; i < field_name.size(); i++) {
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

MSRStatus ShardIndexGenerator::ExecuteSQL(const std::string &sql, sqlite3 *db, const std::string &success_msg) {
  char *z_err_msg = nullptr;
  int rc = sqlite3_exec(db, common::SafeCStr(sql), Callback, nullptr, &z_err_msg);
  if (rc != SQLITE_OK) {
    MS_LOG(ERROR) << "Sql error: " << z_err_msg;
    sqlite3_free(z_err_msg);
    return FAILED;
  } else {
    if (!success_msg.empty()) {
      MS_LOG(DEBUG) << "Sqlite3_exec exec success, msg is: " << success_msg;
    }
    sqlite3_free(z_err_msg);
    return SUCCESS;
  }
}

std::pair<MSRStatus, std::string> ShardIndexGenerator::GenerateFieldName(
  const std::pair<uint64_t, std::string> &field) {
  // Replaces dots and dashes with underscores for SQL use
  std::string field_name = field.second;
  // white list to avoid sql injection
  std::replace_if(
    field_name.begin(), field_name.end(), [](char x) { return (x == '-' || x == '.'); }, '_');
  auto pos = std::find_if_not(field_name.begin(), field_name.end(), [](char x) {
    return (x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z') || x == '_' || (x >= '0' && x <= '9');
  });
  if (pos != field_name.end()) {
    MS_LOG(ERROR) << "Field name must be composed of '0-9' or 'a-z' or 'A-Z' or '_', field_name: " << field_name;
    return {FAILED, ""};
  }
  return {SUCCESS, field_name + "_" + std::to_string(field.first)};
}

std::pair<MSRStatus, sqlite3 *> ShardIndexGenerator::CheckDatabase(const std::string &shard_address) {
  sqlite3 *db = nullptr;
  std::ifstream fin(common::SafeCStr(shard_address));
  if (!append_ && fin.good()) {
    MS_LOG(ERROR) << "DB file already exist";
    fin.close();
    return {FAILED, nullptr};
  }
  fin.close();
  int rc = sqlite3_open_v2(common::SafeCStr(shard_address), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
  if (rc) {
    MS_LOG(ERROR) << "Can't open database, error: " << sqlite3_errmsg(db);
    return {FAILED, nullptr};
  } else {
    MS_LOG(DEBUG) << "Opened database successfully";
    return {SUCCESS, db};
  }
}

MSRStatus ShardIndexGenerator::CreateShardNameTable(sqlite3 *db, const std::string &shard_name) {
  // create shard_name table
  std::string sql = "DROP TABLE IF EXISTS SHARD_NAME;";
  if (ExecuteSQL(sql, db, "drop table successfully.") != SUCCESS) {
    return FAILED;
  }
  sql = "CREATE TABLE SHARD_NAME(NAME TEXT NOT NULL);";
  if (ExecuteSQL(sql, db, "create table successfully.") != SUCCESS) {
    return FAILED;
  }
  sql = "INSERT INTO SHARD_NAME (NAME) VALUES ('" + shard_name + "');";
  if (ExecuteSQL(sql, db, "insert name successfully.") != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

std::pair<MSRStatus, sqlite3 *> ShardIndexGenerator::CreateDatabase(int shard_no) {
  std::string shard_address = shard_header_.get_shard_address_by_id(shard_no);
  if (shard_address.empty()) {
    MS_LOG(ERROR) << "Shard address is null, shard no: " << shard_no;
    return {FAILED, nullptr};
  }

  string shard_name = GetFileName(shard_address).second;
  shard_address += ".db";
  auto ret1 = CheckDatabase(shard_address);
  if (ret1.first != SUCCESS) {
    return {FAILED, nullptr};
  }
  sqlite3 *db = ret1.second;
  std::string sql = "DROP TABLE IF EXISTS INDEXES;";
  if (ExecuteSQL(sql, db, "drop table successfully.") != SUCCESS) {
    return {FAILED, nullptr};
  }
  sql =
    "CREATE TABLE INDEXES("
    "  ROW_ID               INT  NOT NULL, PAGE_ID_RAW          INT  NOT NULL"
    ", PAGE_OFFSET_RAW      INT  NOT NULL, PAGE_OFFSET_RAW_END  INT  NOT NULL"
    ", ROW_GROUP_ID         INT  NOT NULL, PAGE_ID_BLOB         INT  NOT NULL"
    ", PAGE_OFFSET_BLOB     INT  NOT NULL, PAGE_OFFSET_BLOB_END INT  NOT NULL";

  int field_no = 0;
  for (const auto &field : fields_) {
    uint64_t schema_id = field.first;
    auto result = shard_header_.GetSchemaByID(schema_id);
    if (result.second != SUCCESS) {
      return {FAILED, nullptr};
    }
    json json_schema = (result.first->GetSchema())["schema"];
    std::string type = ConvertJsonToSQL(TakeFieldType(field.second, json_schema));
    auto ret = GenerateFieldName(field);
    if (ret.first != SUCCESS) {
      return {FAILED, nullptr};
    }
    sql += ",INC_" + std::to_string(field_no++) + " INT, " + ret.second + " " + type;
  }
  sql += ", PRIMARY KEY(ROW_ID";
  for (uint64_t i = 0; i < fields_.size(); ++i) sql += ",INC_" + std::to_string(i);
  sql += "));";
  if (ExecuteSQL(sql, db, "create table successfully.") != SUCCESS) {
    return {FAILED, nullptr};
  }

  if (CreateShardNameTable(db, shard_name) != SUCCESS) {
    return {FAILED, nullptr};
  }
  return {SUCCESS, db};
}

std::pair<MSRStatus, std::vector<json>> ShardIndexGenerator::GetSchemaDetails(const std::vector<uint64_t> &schema_lens,
                                                                              std::fstream &in) {
  std::vector<json> schema_details;
  if (schema_count_ <= kMaxSchemaCount) {
    for (int sc = 0; sc < schema_count_; ++sc) {
      std::vector<char> schema_detail(schema_lens[sc]);

      auto &io_read = in.read(&schema_detail[0], schema_lens[sc]);
      if (!io_read.good() || io_read.fail() || io_read.bad()) {
        MS_LOG(ERROR) << "File read failed";
        in.close();
        return {FAILED, {}};
      }

      schema_details.emplace_back(json::from_msgpack(std::string(schema_detail.begin(), schema_detail.end())));
    }
  }

  return {SUCCESS, schema_details};
}

std::pair<MSRStatus, std::string> ShardIndexGenerator::GenerateRawSQL(
  const std::vector<std::pair<uint64_t, std::string>> &fields) {
  std::string sql =
    "INSERT INTO INDEXES (ROW_ID,ROW_GROUP_ID,PAGE_ID_RAW,PAGE_OFFSET_RAW,PAGE_OFFSET_RAW_END,"
    "PAGE_ID_BLOB,PAGE_OFFSET_BLOB,PAGE_OFFSET_BLOB_END";

  int field_no = 0;
  for (const auto &field : fields) {
    auto ret = GenerateFieldName(field);
    if (ret.first != SUCCESS) {
      return {FAILED, ""};
    }
    sql += ",INC_" + std::to_string(field_no++) + "," + ret.second;
  }
  sql +=
    ") VALUES( :ROW_ID,:ROW_GROUP_ID,:PAGE_ID_RAW,:PAGE_OFFSET_RAW,:PAGE_OFFSET_RAW_END,:PAGE_ID_BLOB,"
    ":PAGE_OFFSET_BLOB,:PAGE_OFFSET_BLOB_END";
  field_no = 0;
  for (const auto &field : fields) {
    auto ret = GenerateFieldName(field);
    if (ret.first != SUCCESS) {
      return {FAILED, ""};
    }
    sql += ",:INC_" + std::to_string(field_no++) + ",:" + ret.second;
  }
  sql += " )";
  return {SUCCESS, sql};
}

MSRStatus ShardIndexGenerator::BindParameterExecuteSQL(
  sqlite3 *db, const std::string &sql,
  const std::vector<std::vector<std::tuple<std::string, std::string, std::string>>> &data) {
  sqlite3_stmt *stmt = nullptr;
  if (sqlite3_prepare_v2(db, common::SafeCStr(sql), -1, &stmt, 0) != SQLITE_OK) {
    MS_LOG(ERROR) << "SQL error: could not prepare statement, sql: " << sql;
    return FAILED;
  }
  for (auto &row : data) {
    for (auto &field : row) {
      const auto &place_holder = std::get<0>(field);
      const auto &field_type = std::get<1>(field);
      const auto &field_value = std::get<2>(field);

      int index = sqlite3_bind_parameter_index(stmt, common::SafeCStr(place_holder));
      if (field_type == "INTEGER") {
        if (sqlite3_bind_int(stmt, index, std::stoi(field_value)) != SQLITE_OK) {
          MS_LOG(ERROR) << "SQL error: could not bind parameter, index: " << index
                        << ", field value: " << std::stoi(field_value);
          return FAILED;
        }
      } else if (field_type == "NUMERIC") {
        if (sqlite3_bind_double(stmt, index, std::stod(field_value)) != SQLITE_OK) {
          MS_LOG(ERROR) << "SQL error: could not bind parameter, index: " << index
                        << ", field value: " << std::stoi(field_value);
          return FAILED;
        }
      } else if (field_type == "NULL") {
        if (sqlite3_bind_null(stmt, index) != SQLITE_OK) {
          MS_LOG(ERROR) << "SQL error: could not bind parameter, index: " << index << ", field value: NULL";
          return FAILED;
        }
      } else {
        if (sqlite3_bind_text(stmt, index, common::SafeCStr(field_value), -1, SQLITE_STATIC) != SQLITE_OK) {
          MS_LOG(ERROR) << "SQL error: could not bind parameter, index: " << index << ", field value: " << field_value;
          return FAILED;
        }
      }
    }
    if (sqlite3_step(stmt) != SQLITE_DONE) {
      MS_LOG(ERROR) << "SQL error: Could not step (execute) stmt.";
      return FAILED;
    }
    (void)sqlite3_reset(stmt);
  }
  (void)sqlite3_finalize(stmt);
  return SUCCESS;
}

MSRStatus ShardIndexGenerator::AddBlobPageInfo(std::vector<std::tuple<std::string, std::string, std::string>> &row_data,
                                               const std::shared_ptr<Page> cur_blob_page,
                                               uint64_t &cur_blob_page_offset, std::fstream &in) {
  row_data.emplace_back(":PAGE_ID_BLOB", "INTEGER", std::to_string(cur_blob_page->get_page_id()));

  // blob data start
  row_data.emplace_back(":PAGE_OFFSET_BLOB", "INTEGER", std::to_string(cur_blob_page_offset));
  auto &io_seekg_blob =
    in.seekg(page_size_ * cur_blob_page->get_page_id() + header_size_ + cur_blob_page_offset, std::ios::beg);
  if (!io_seekg_blob.good() || io_seekg_blob.fail() || io_seekg_blob.bad()) {
    MS_LOG(ERROR) << "File seekg failed";
    in.close();
    return FAILED;
  }

  uint64_t image_size = 0;

  auto &io_read = in.read(reinterpret_cast<char *>(&image_size), kInt64Len);
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    MS_LOG(ERROR) << "File read failed";
    in.close();
    return FAILED;
  }

  cur_blob_page_offset += (kInt64Len + image_size);
  row_data.emplace_back(":PAGE_OFFSET_BLOB_END", "INTEGER", std::to_string(cur_blob_page_offset));

  return SUCCESS;
}

void ShardIndexGenerator::AddIndexFieldByRawData(
  const std::vector<json> &schema_detail, std::vector<std::tuple<std::string, std::string, std::string>> &row_data) {
  auto result = GenerateIndexFields(schema_detail);
  if (result.first == SUCCESS) {
    int index = 0;
    for (const auto &field : result.second) {
      // assume simple field: string , number etc.
      row_data.emplace_back(":INC_" + std::to_string(index++), "INTEGER", "0");
      row_data.emplace_back(":" + std::get<0>(field), std::get<1>(field), std::get<2>(field));
    }
  }
}

ROW_DATA ShardIndexGenerator::GenerateRowData(int shard_no, const std::map<int, int> &blob_id_to_page_id,
                                              int raw_page_id, std::fstream &in) {
  std::vector<std::vector<std::tuple<std::string, std::string, std::string>>> full_data;

  // current raw data page
  std::shared_ptr<Page> cur_raw_page = shard_header_.GetPage(shard_no, raw_page_id).first;

  // related blob page
  vector<pair<int, uint64_t>> row_group_list = cur_raw_page->get_row_group_ids();

  // pair: row_group id, offset in raw data page
  for (pair<int, int> blob_ids : row_group_list) {
    // get blob data page according to row_group id
    std::shared_ptr<Page> cur_blob_page = shard_header_.GetPage(shard_no, blob_id_to_page_id.at(blob_ids.first)).first;

    // offset in current raw data page
    auto cur_raw_page_offset = static_cast<uint64_t>(blob_ids.second);
    uint64_t cur_blob_page_offset = 0;
    for (unsigned int i = cur_blob_page->get_start_row_id(); i < cur_blob_page->get_end_row_id(); ++i) {
      std::vector<std::tuple<std::string, std::string, std::string>> row_data;
      row_data.emplace_back(":ROW_ID", "INTEGER", std::to_string(i));
      row_data.emplace_back(":ROW_GROUP_ID", "INTEGER", std::to_string(cur_blob_page->get_page_type_id()));
      row_data.emplace_back(":PAGE_ID_RAW", "INTEGER", std::to_string(cur_raw_page->get_page_id()));

      // raw data start
      row_data.emplace_back(":PAGE_OFFSET_RAW", "INTEGER", std::to_string(cur_raw_page_offset));

      // calculate raw data end
      auto &io_seekg =
        in.seekg(page_size_ * (cur_raw_page->get_page_id()) + header_size_ + cur_raw_page_offset, std::ios::beg);
      if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
        MS_LOG(ERROR) << "File seekg failed";
        in.close();
        return {FAILED, {}};
      }

      std::vector<uint64_t> schema_lens;
      if (schema_count_ <= kMaxSchemaCount) {
        for (int sc = 0; sc < schema_count_; sc++) {
          uint64_t schema_size = 0;

          auto &io_read = in.read(reinterpret_cast<char *>(&schema_size), kInt64Len);
          if (!io_read.good() || io_read.fail() || io_read.bad()) {
            MS_LOG(ERROR) << "File read failed";
            in.close();
            return {FAILED, {}};
          }

          cur_raw_page_offset += (kInt64Len + schema_size);
          schema_lens.push_back(schema_size);
        }
      }
      row_data.emplace_back(":PAGE_OFFSET_RAW_END", "INTEGER", std::to_string(cur_raw_page_offset));

      // Getting schema for getting data for fields
      auto st_schema_detail = GetSchemaDetails(schema_lens, in);
      if (st_schema_detail.first != SUCCESS) {
        return {FAILED, {}};
      }

      // start blob page info
      if (AddBlobPageInfo(row_data, cur_blob_page, cur_blob_page_offset, in) != SUCCESS) {
        return {FAILED, {}};
      }

      // start index field
      AddIndexFieldByRawData(st_schema_detail.second, row_data);
      full_data.push_back(std::move(row_data));
    }
  }
  return {SUCCESS, full_data};
}

INDEX_FIELDS ShardIndexGenerator::GenerateIndexFields(const std::vector<json> &schema_detail) {
  std::vector<std::tuple<std::string, std::string, std::string>> fields;
  // index fields
  std::vector<std::pair<uint64_t, std::string>> index_fields = shard_header_.get_fields();
  for (const auto &field : index_fields) {
    if (field.first >= schema_detail.size()) {
      return {FAILED, {}};
    }
    auto field_value = GetValueByField(field.second, schema_detail[field.first]);
    if (field_value.first != SUCCESS) {
      MS_LOG(ERROR) << "Get value from json by field name failed";
      return {FAILED, {}};
    }

    auto result = shard_header_.GetSchemaByID(field.first);
    if (result.second != SUCCESS) {
      return {FAILED, {}};
    }

    std::string field_type = ConvertJsonToSQL(TakeFieldType(field.second, result.first->GetSchema()["schema"]));
    auto ret = GenerateFieldName(field);
    if (ret.first != SUCCESS) {
      return {FAILED, {}};
    }

    fields.emplace_back(ret.second, field_type, field_value.second);
  }
  return {SUCCESS, std::move(fields)};
}

MSRStatus ShardIndexGenerator::ExecuteTransaction(const int &shard_no, const std::pair<MSRStatus, sqlite3 *> &db,
                                                  const std::vector<int> &raw_page_ids,
                                                  const std::map<int, int> &blob_id_to_page_id) {
  // Add index data to database
  std::string shard_address = shard_header_.get_shard_address_by_id(shard_no);
  if (shard_address.empty()) {
    MS_LOG(ERROR) << "Shard address is null";
    return FAILED;
  }

  std::fstream in;
  in.open(common::SafeCStr(shard_address), std::ios::in | std::ios::binary);
  if (!in.good()) {
    MS_LOG(ERROR) << "File could not opened";
    return FAILED;
  }
  (void)sqlite3_exec(db.second, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);
  for (int raw_page_id : raw_page_ids) {
    auto sql = GenerateRawSQL(fields_);
    if (sql.first != SUCCESS) {
      return FAILED;
    }
    auto data = GenerateRowData(shard_no, blob_id_to_page_id, raw_page_id, in);
    if (data.first != SUCCESS) {
      return FAILED;
    }
    if (BindParameterExecuteSQL(db.second, sql.second, data.second) == FAILED) {
      return FAILED;
    }
    MS_LOG(INFO) << "Insert " << data.second.size() << " rows to index db.";
  }
  (void)sqlite3_exec(db.second, "END TRANSACTION;", nullptr, nullptr, nullptr);
  in.close();

  // Close database
  if (sqlite3_close(db.second) != SQLITE_OK) {
    MS_LOG(ERROR) << "Close database failed";
    return FAILED;
  }
  return SUCCESS;
}

MSRStatus ShardIndexGenerator::WriteToDatabase() {
  fields_ = shard_header_.get_fields();
  page_size_ = shard_header_.get_page_size();
  header_size_ = shard_header_.get_header_size();
  schema_count_ = shard_header_.get_schema_count();
  if (shard_header_.get_shard_count() > kMaxShardCount) {
    MS_LOG(ERROR) << "num shards: " << shard_header_.get_shard_count() << " exceeds max count:" << kMaxSchemaCount;
    return FAILED;
  }
  task_ = 0;  // set two atomic vars to initial value
  write_success_ = true;

  // spawn half the physical threads or total number of shards whichever is smaller
  const unsigned int num_workers =
    std::min(std::thread::hardware_concurrency() / 2 + 1, static_cast<unsigned int>(shard_header_.get_shard_count()));

  std::vector<std::thread> threads;
  threads.reserve(num_workers);

  for (size_t t = 0; t < threads.capacity(); t++) {
    threads.emplace_back(std::thread(&ShardIndexGenerator::DatabaseWriter, this));
  }

  for (size_t t = 0; t < threads.capacity(); t++) {
    threads[t].join();
  }
  return write_success_ ? SUCCESS : FAILED;
}

void ShardIndexGenerator::DatabaseWriter() {
  int shard_no = task_++;
  while (shard_no < shard_header_.get_shard_count()) {
    auto db = CreateDatabase(shard_no);
    if (db.first != SUCCESS || db.second == nullptr || write_success_ == false) {
      write_success_ = false;
      return;
    }

    MS_LOG(INFO) << "Init index db for shard: " << shard_no << " successfully.";

    // Pre-processing page information
    auto total_pages = shard_header_.GetLastPageId(shard_no) + 1;

    std::map<int, int> blob_id_to_page_id;
    std::vector<int> raw_page_ids;
    for (uint64_t i = 0; i < total_pages; ++i) {
      std::shared_ptr<Page> cur_page = shard_header_.GetPage(shard_no, i).first;
      if (cur_page->get_page_type() == "RAW_DATA") {
        raw_page_ids.push_back(i);
      } else if (cur_page->get_page_type() == "BLOB_DATA") {
        blob_id_to_page_id[cur_page->get_page_type_id()] = i;
      }
    }

    if (ExecuteTransaction(shard_no, db, raw_page_ids, blob_id_to_page_id) != SUCCESS) {
      write_success_ = false;
      return;
    }
    MS_LOG(INFO) << "Generate index db for shard: " << shard_no << " successfully.";
    shard_no = task_++;
  }
}
}  // namespace mindrecord
}  // namespace mindspore
