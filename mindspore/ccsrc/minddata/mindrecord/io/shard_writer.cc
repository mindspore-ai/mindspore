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

#include "minddata/mindrecord/include/shard_writer.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "./securec.h"

namespace mindspore {
namespace mindrecord {
ShardWriter::ShardWriter()
    : shard_count_(1), header_size_(kDefaultHeaderSize), page_size_(kDefaultPageSize), row_count_(0), schema_count_(1) {
  compression_size_ = 0;
}

ShardWriter::~ShardWriter() {
  for (int i = static_cast<int>(file_streams_.size()) - 1; i >= 0; i--) {
    file_streams_[i]->close();
  }
}

Status ShardWriter::GetFullPathFromFileName(const std::vector<std::string> &paths) {
  // Get full path from file name
  for (const auto &path : paths) {
    CHECK_FAIL_RETURN_UNEXPECTED_MR(CheckIsValidUtf8(path),
                                    "Invalid file, mindrecord file name: " + path +
                                      " contains invalid uft-8 character. Please rename mindrecord file name.");
    // get realpath
    std::optional<std::string> dir = "";
    std::optional<std::string> local_file_name = "";
    FileUtils::SplitDirAndFileName(path, &dir, &local_file_name);
    if (!dir.has_value()) {
      dir = ".";
    }

    auto realpath = FileUtils::GetRealPath(dir.value().c_str());
    CHECK_FAIL_RETURN_UNEXPECTED_MR(
      realpath.has_value(),
      "Invalid dir, failed to get the realpath of mindrecord file dir. Please check path: " + dir.value());

    std::optional<std::string> whole_path = "";
    FileUtils::ConcatDirAndFileName(&realpath, &local_file_name, &whole_path);

    (void)file_paths_.emplace_back(whole_path.value());
  }
  return Status::OK();
}

Status ShardWriter::OpenDataFiles(bool append, bool overwrite) {
  // Open files
  for (const auto &file : file_paths_) {
    std::optional<std::string> dir = "";
    std::optional<std::string> local_file_name = "";
    FileUtils::SplitDirAndFileName(file, &dir, &local_file_name);
    if (!dir.has_value()) {
      dir = ".";
    }

    auto realpath = FileUtils::GetRealPath(dir.value().c_str());
    CHECK_FAIL_RETURN_UNEXPECTED_MR(
      realpath.has_value(), "Invalid file, failed to get the realpath of mindrecord files. Please check file: " + file);

    std::optional<std::string> whole_path = "";
    FileUtils::ConcatDirAndFileName(&realpath, &local_file_name, &whole_path);

    std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
    if (!append) {
      // if not append && mindrecord or db file exist
      fs->open(whole_path.value(), std::ios::in | std::ios::binary);
      std::ifstream fs_db(whole_path.value() + ".db");
      if (fs->good() || fs_db.good()) {
        fs->close();
        fs_db.close();
        if (overwrite) {
          auto res1 = std::remove(whole_path.value().c_str());
          CHECK_FAIL_RETURN_UNEXPECTED_MR(!std::ifstream(whole_path.value()) == true,
                                          "Invalid file, failed to remove the old files when trying to overwrite "
                                          "mindrecord files. Please check file path and permission: " +
                                            file);
          if (res1 == 0) {
            MS_LOG(WARNING) << "Succeed to remove the old mindrecord files, path: " << file;
          }
          auto db_file = whole_path.value() + ".db";
          auto res2 = std::remove(db_file.c_str());
          CHECK_FAIL_RETURN_UNEXPECTED_MR(!std::ifstream(whole_path.value() + ".db") == true,
                                          "Invalid file, failed to remove the old mindrecord meta files when trying to "
                                          "overwrite mindrecord files. Please check file path and permission: " +
                                            file + ".db");
          if (res2 == 0) {
            MS_LOG(WARNING) << "Succeed to remove the old mindrecord metadata files, path: " << file + ".db";
          }
        } else {
          RETURN_STATUS_UNEXPECTED_MR(
            "Invalid file, mindrecord files already exist. Please check file path: " + file +
            +".\nIf you do not want to keep the files, set the 'overwrite' parameter to True and try again.");
        }
      } else {
        fs->close();
        fs_db.close();
      }
      // open the mindrecord file to write
      fs->open(whole_path.value().data(), std::ios::out | std::ios::in | std::ios::binary | std::ios::trunc);
      if (!fs->good()) {
        RETURN_STATUS_UNEXPECTED_MR(
          "Invalid file, failed to open files for writing mindrecord files. Please check file path, permission and "
          "open file limit: " +
          file);
      }
    } else {
      // open the mindrecord file to append
      fs->open(whole_path.value().data(), std::ios::out | std::ios::in | std::ios::binary);
      if (!fs->good()) {
        fs->close();
        RETURN_STATUS_UNEXPECTED_MR(
          "Invalid file, failed to open files for appending mindrecord files. Please check file path, permission and "
          "open file limit: " +
          file);
      }
    }
    MS_LOG(INFO) << "Succeed to open mindrecord shard file, path: " << file;
    file_streams_.push_back(fs);
  }
  return Status::OK();
}

Status ShardWriter::RemoveLockFile() {
  // Remove temporary file
  int ret = std::remove(pages_file_.c_str());
  if (ret == 0) {
    MS_LOG(DEBUG) << "Succeed to remove page file, path: " << pages_file_;
  }

  ret = std::remove(lock_file_.c_str());
  if (ret == 0) {
    MS_LOG(DEBUG) << "Succeed to remove lock file, path: " << lock_file_;
  }
  return Status::OK();
}

Status ShardWriter::InitLockFile() {
  CHECK_FAIL_RETURN_UNEXPECTED_MR(file_paths_.size() != 0, "[Internal ERROR] 'file_paths_' is not initialized.");

  lock_file_ = file_paths_[0] + kLockFileSuffix;
  pages_file_ = file_paths_[0] + kPageFileSuffix;
  RETURN_IF_NOT_OK_MR(RemoveLockFile());
  return Status::OK();
}

Status ShardWriter::Open(const std::vector<std::string> &paths, bool append, bool overwrite) {
  shard_count_ = paths.size();
  CHECK_FAIL_RETURN_UNEXPECTED_MR(schema_count_ <= kMaxSchemaCount,
                                  "[Internal ERROR] 'schema_count_' should be less than or equal to " +
                                    std::to_string(kMaxSchemaCount) + ", but got: " + std::to_string(schema_count_));

  // Get full path from file name
  RETURN_IF_NOT_OK_MR(GetFullPathFromFileName(paths));
  // Open files
  RETURN_IF_NOT_OK_MR(OpenDataFiles(append, overwrite));
  // Init lock file
  RETURN_IF_NOT_OK_MR(InitLockFile());
  return Status::OK();
}

Status ShardWriter::OpenForAppend(const std::string &path) {
  RETURN_IF_NOT_OK_MR(CheckFile(path));
  std::shared_ptr<json> header_ptr;
  RETURN_IF_NOT_OK_MR(ShardHeader::BuildSingleHeader(path, &header_ptr));
  auto ds = std::make_shared<std::vector<std::string>>();
  RETURN_IF_NOT_OK_MR(GetDatasetFiles(path, (*header_ptr)["shard_addresses"], &ds));
  ShardHeader header = ShardHeader();
  RETURN_IF_NOT_OK_MR(header.BuildDataset(*ds));
  shard_header_ = std::make_shared<ShardHeader>(header);
  RETURN_IF_NOT_OK_MR(SetHeaderSize(shard_header_->GetHeaderSize()));
  RETURN_IF_NOT_OK_MR(SetPageSize(shard_header_->GetPageSize()));
  compression_size_ = shard_header_->GetCompressionSize();
  RETURN_IF_NOT_OK_MR(Open(*ds, true));
  shard_column_ = std::make_shared<ShardColumn>(shard_header_);
  return Status::OK();
}

Status ShardWriter::Commit() {
  // Read pages file
  std::ifstream page_file(pages_file_.c_str());
  if (page_file.good()) {
    page_file.close();
    RETURN_IF_NOT_OK_MR(shard_header_->FileToPages(pages_file_));
  }
  RETURN_IF_NOT_OK_MR(WriteShardHeader());
  MS_LOG(INFO) << "Succeed to write meta data.";
  // Remove lock file
  RETURN_IF_NOT_OK_MR(RemoveLockFile());

  return Status::OK();
}

Status ShardWriter::SetShardHeader(std::shared_ptr<ShardHeader> header_data) {
  CHECK_FAIL_RETURN_UNEXPECTED_MR(
    header_data->GetSchemaCount() > 0,
    "Invalid data, schema is not found in header, please use 'add_schema' to add a schema for new mindrecord files.");
  RETURN_IF_NOT_OK_MR(header_data->InitByFiles(file_paths_));
  // set fields in mindrecord when empty
  std::vector<std::pair<uint64_t, std::string>> fields = header_data->GetFields();
  if (fields.empty()) {
    MS_LOG(DEBUG) << "Index field is not set, it will be generated automatically.";
    std::vector<std::shared_ptr<Schema>> schemas = header_data->GetSchemas();
    for (const auto &schema : schemas) {
      json jsonSchema = schema->GetSchema()["schema"];
      for (const auto &el : jsonSchema.items()) {
        if (el.value()["type"] == "string" ||
            (el.value()["type"] == "int32" && el.value().find("shape") == el.value().end()) ||
            (el.value()["type"] == "int64" && el.value().find("shape") == el.value().end()) ||
            (el.value()["type"] == "float32" && el.value().find("shape") == el.value().end()) ||
            (el.value()["type"] == "float64" && el.value().find("shape") == el.value().end())) {
          fields.emplace_back(std::make_pair(schema->GetSchemaID(), el.key()));
        }
      }
    }
    // only blob data
    if (!fields.empty()) {
      RETURN_IF_NOT_OK_MR(header_data->AddIndexFields(fields));
    }
  }

  shard_header_ = header_data;
  shard_header_->SetHeaderSize(header_size_);
  shard_header_->SetPageSize(page_size_);
  shard_column_ = std::make_shared<ShardColumn>(shard_header_);
  return Status::OK();
}

Status ShardWriter::SetHeaderSize(const uint64_t &header_size) {
  // header_size [16KB, 128MB]
  CHECK_FAIL_RETURN_UNEXPECTED_MR(header_size >= kMinHeaderSize && header_size <= kMaxHeaderSize,
                                  "Invalid data, header size: " + std::to_string(header_size) +
                                    " should be in range [" + std::to_string(kMinHeaderSize) + " bytes, " +
                                    std::to_string(kMaxHeaderSize) + " bytes].");
  CHECK_FAIL_RETURN_UNEXPECTED_MR(
    header_size % 4 == 0, "Invalid data, header size " + std::to_string(header_size) + " should be divided by four.");
  header_size_ = header_size;
  return Status::OK();
}

Status ShardWriter::SetPageSize(const uint64_t &page_size) {
  // PageSize [32KB, 256MB]
  CHECK_FAIL_RETURN_UNEXPECTED_MR(page_size >= kMinPageSize && page_size <= kMaxPageSize,
                                  "Invalid data, page size: " + std::to_string(page_size) + " should be in range [" +
                                    std::to_string(kMinPageSize) + " bytes, " + std::to_string(kMaxPageSize) +
                                    " bytes].");
  CHECK_FAIL_RETURN_UNEXPECTED_MR(
    page_size % 4 == 0, "Invalid data, page size " + std::to_string(page_size) + " should be divided by four.");
  page_size_ = page_size;
  return Status::OK();
}

void ShardWriter::DeleteErrorData(std::map<uint64_t, std::vector<json>> &raw_data,
                                  std::vector<std::vector<uint8_t>> &blob_data) {
  // get wrong data location
  std::set<int, std::greater<int>> delete_set;
  for (auto &err_mg : err_mg_) {
    uint64_t id = err_mg.first;
    auto sub_err_mg = err_mg.second;
    for (auto &subMg : sub_err_mg) {
      int loc = subMg.first;
      std::string message = subMg.second;
      MS_LOG(ERROR) << "Invalid input, the " << loc + 1
                    << " th data provided by user is invalid while writing mindrecord files. Please fix the error: "
                    << message;
      (void)delete_set.insert(loc);
    }
  }

  auto it = raw_data.begin();
  if (delete_set.size() == it->second.size()) {
    raw_data.clear();
    blob_data.clear();
    return;
  }

  // delete wrong raw data
  for (auto &loc : delete_set) {
    // delete row data
    for (auto &raw : raw_data) {
      (void)raw.second.erase(raw.second.begin() + loc);
    }

    // delete blob data
    (void)blob_data.erase(blob_data.begin() + loc);
  }
}

void ShardWriter::PopulateMutexErrorData(const int &row, const std::string &message,
                                         std::map<int, std::string> &err_raw_data) {
  std::lock_guard<std::mutex> lock(check_mutex_);
  (void)err_raw_data.insert(std::make_pair(row, message));
}

Status ShardWriter::CheckDataTypeAndValue(const std::string &key, const json &value, const json &data, const int &i,
                                          std::map<int, std::string> &err_raw_data) {
  auto data_type = std::string(value["type"].get<std::string>());
  if ((data_type == "int32" && !data[key].is_number_integer()) ||
      (data_type == "int64" && !data[key].is_number_integer()) ||
      (data_type == "float32" && !data[key].is_number_float()) ||
      (data_type == "float64" && !data[key].is_number_float()) || (data_type == "string" && !data[key].is_string())) {
    std::string message = "Invalid input, for field: " + key + ", type: " + data_type +
                          " and value: " + data[key].dump() + " do not match while writing mindrecord files.";
    PopulateMutexErrorData(i, message, err_raw_data);
    RETURN_STATUS_UNEXPECTED_MR(message);
  }

  if (data_type == "int32" && data[key].is_number_integer()) {
    int64_t temp_value = data[key];
    if (static_cast<int64_t>(temp_value) < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
        static_cast<int64_t>(temp_value) > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      std::string message = "Invalid input, for field: " + key + "and its type: " + data_type +
                            ", value: " + data[key].dump() + " is out of range while writing mindrecord files.";
      PopulateMutexErrorData(i, message, err_raw_data);
      RETURN_STATUS_UNEXPECTED_MR(message);
    }
  }
  return Status::OK();
}

void ShardWriter::CheckSliceData(int start_row, int end_row, json schema, const std::vector<json> &sub_raw_data,
                                 std::map<int, std::string> &err_raw_data) {
  if (start_row < 0 || start_row > end_row || end_row > static_cast<int>(sub_raw_data.size())) {
    return;
  }
  for (int i = start_row; i < end_row; i++) {
    json data = sub_raw_data[i];

    for (auto iter = schema.begin(); iter != schema.end(); iter++) {
      std::string key = iter.key();
      json value = iter.value();
      if (data.find(key) == data.end()) {
        std::string message = "'" + key + "' object can not found in data: " + value.dump();
        PopulateMutexErrorData(i, message, err_raw_data);
        break;
      }

      if (value.size() == kInt2) {
        // Skip check since all shaped data will store as blob
        continue;
      }

      if (CheckDataTypeAndValue(key, value, data, i, err_raw_data).IsError()) {
        break;
      }
    }
  }
}

Status ShardWriter::CheckData(const std::map<uint64_t, std::vector<json>> &raw_data) {
  auto rawdata_iter = raw_data.begin();

  // make sure rawdata match schema
  for (; rawdata_iter != raw_data.end(); ++rawdata_iter) {
    // used for storing error
    std::map<int, std::string> sub_err_mg;
    int schema_id = rawdata_iter->first;
    std::shared_ptr<Schema> schema_ptr;
    RETURN_IF_NOT_OK_MR(shard_header_->GetSchemaByID(schema_id, &schema_ptr));
    json schema = schema_ptr->GetSchema()["schema"];
    for (const auto &field : schema_ptr->GetBlobFields()) {
      (void)schema.erase(field);
    }
    std::vector<json> sub_raw_data = rawdata_iter->second;

    // calculate start position and end position for each thread
    int batch_size = rawdata_iter->second.size() / shard_count_;
    int thread_num = shard_count_;
    CHECK_FAIL_RETURN_UNEXPECTED_MR(thread_num > 0, "[Internal ERROR] 'thread_num' should be positive.");
    if (thread_num > kMaxThreadCount) {
      thread_num = kMaxThreadCount;
    }
    std::vector<std::thread> thread_set(thread_num);

    // start multiple thread
    int start_row = 0, end_row = 0;
    for (int x = 0; x < thread_num; ++x) {
      if (x != thread_num - 1) {
        start_row = batch_size * x;
        end_row = batch_size * (x + 1);
      } else {
        start_row = batch_size * x;
        end_row = rawdata_iter->second.size();
      }
      thread_set[x] = std::thread(&ShardWriter::CheckSliceData, this, start_row, end_row, schema,
                                  std::ref(sub_raw_data), std::ref(sub_err_mg));
    }
    CHECK_FAIL_RETURN_UNEXPECTED_MR(
      thread_num <= kMaxThreadCount,
      "[Internal ERROR] 'thread_num' should be less than or equal to " + std::to_string(kMaxThreadCount));
    // Wait for threads done
    for (int x = 0; x < thread_num; ++x) {
      thread_set[x].join();
    }

    (void)err_mg_.insert(std::make_pair(schema_id, sub_err_mg));
  }
  return Status::OK();
}

Status ShardWriter::ValidateRawData(std::map<uint64_t, std::vector<json>> &raw_data,
                                    std::vector<std::vector<uint8_t>> &blob_data, bool sign,
                                    std::shared_ptr<std::pair<int, int>> *count_ptr) {
  RETURN_UNEXPECTED_IF_NULL_MR(count_ptr);
  auto rawdata_iter = raw_data.begin();
  schema_count_ = raw_data.size();
  CHECK_FAIL_RETURN_UNEXPECTED_MR(schema_count_ > 0, "Invalid data, the number of schema should be positive but got: " +
                                                       std::to_string(schema_count_) +
                                                       ". Please check the input schema.");

  // keep schema_id
  std::set<int64_t> schema_ids;
  row_count_ = (rawdata_iter->second).size();

  // Determine if the number of schemas is the same
  CHECK_FAIL_RETURN_UNEXPECTED_MR(shard_header_->GetSchemas().size() == schema_count_,
                                  "[Internal ERROR] 'schema_count_' and the schema count in schema: " +
                                    std::to_string(schema_count_) + " do not match.");
  // Determine raw_data size == blob_data size
  CHECK_FAIL_RETURN_UNEXPECTED_MR(raw_data[0].size() == blob_data.size(),
                                  "[Internal ERROR] raw data size: " + std::to_string(raw_data[0].size()) +
                                    " is not equal to blob data size: " + std::to_string(blob_data.size()) + ".");

  // Determine whether the number of samples corresponding to each schema is the same
  for (rawdata_iter = raw_data.begin(); rawdata_iter != raw_data.end(); ++rawdata_iter) {
    CHECK_FAIL_RETURN_UNEXPECTED_MR(row_count_ == rawdata_iter->second.size(),
                                    "[Internal ERROR] 'row_count_': " + std::to_string(rawdata_iter->second.size()) +
                                      " for each schema is not the same.");
    (void)schema_ids.insert(rawdata_iter->first);
  }
  const std::vector<std::shared_ptr<Schema>> &schemas = shard_header_->GetSchemas();
  // There is not enough data which is not matching the number of schema
  CHECK_FAIL_RETURN_UNEXPECTED_MR(!std::any_of(schemas.begin(), schemas.end(),
                                               [schema_ids](const std::shared_ptr<Schema> &schema) {
                                                 return schema_ids.find(schema->GetSchemaID()) == schema_ids.end();
                                               }),
                                  "[Internal ERROR] schema id in 'schemas' can not found in 'schema_ids'.");
  if (!sign) {
    *count_ptr = std::make_shared<std::pair<int, int>>(schema_count_, row_count_);
    return Status::OK();
  }

  // check the data according the schema
  RETURN_IF_NOT_OK_MR(CheckData(raw_data));

  // delete wrong data from raw data
  DeleteErrorData(raw_data, blob_data);

  // update raw count
  row_count_ = row_count_ - err_mg_.begin()->second.size();
  *count_ptr = std::make_shared<std::pair<int, int>>(schema_count_, row_count_);
  return Status::OK();
}

void ShardWriter::FillArray(int start, int end, std::map<uint64_t, vector<json>> &raw_data,
                            std::vector<std::vector<uint8_t>> &bin_data) {
  // Prevent excessive thread opening and cause cross-border
  if (start >= end) {
    flag_ = true;
    return;
  }
  int schema_count = static_cast<int>(raw_data.size());
  std::map<uint64_t, vector<json>>::const_iterator rawdata_iter;
  for (int x = start; x < end; ++x) {
    int cnt = 0;
    for (rawdata_iter = raw_data.begin(); rawdata_iter != raw_data.end(); ++rawdata_iter) {
      const json &line = raw_data.at(rawdata_iter->first)[x];
      std::vector<std::uint8_t> bline = json::to_msgpack(line);

      // Storage form is [Sample1-Schema1, Sample1-Schema2, Sample2-Schema1, Sample2-Schema2]
      bin_data[x * schema_count + cnt] = bline;
      cnt++;
    }
  }
}

Status ShardWriter::LockWriter(bool parallel_writer, std::unique_ptr<int> *fd_ptr) {
  if (!parallel_writer) {
    *fd_ptr = std::make_unique<int>(0);
    return Status::OK();
  }

#if defined(_WIN32) || defined(_WIN64)
  const int fd = 0;
  MS_LOG(DEBUG) << "Lock file done by Python.";

#else
  const int fd = open(lock_file_.c_str(), O_WRONLY | O_CREAT, 0666);
  if (fd >= 0) {
    flock(fd, LOCK_EX);
  } else {
    close(fd);
    RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to lock file, path: " + lock_file_);
  }
#endif

  // Open files
  file_streams_.clear();
  for (const auto &file : file_paths_) {
    auto realpath = FileUtils::GetRealPath(file.c_str());
    if (!realpath.has_value()) {
      close(fd);
      RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to get real path, path: " + file);
    }
    std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
    fs->open(realpath.value(), std::ios::in | std::ios::out | std::ios::binary);
    if (fs->fail()) {
      close(fd);
      RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to open file, path: " + file);
    }
    file_streams_.push_back(fs);
  }
  auto status = shard_header_->FileToPages(pages_file_);
  if (status.IsError()) {
    close(fd);
    RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Error raised in FileToPages function.");
  }
  *fd_ptr = std::make_unique<int>(fd);
  return Status::OK();
}

Status ShardWriter::UnlockWriter(int fd, bool parallel_writer) {
  if (!parallel_writer) {
    return Status::OK();
  }
  RETURN_IF_NOT_OK_MR(shard_header_->PagesToFile(pages_file_));
  for (int i = static_cast<int>(file_streams_.size()) - 1; i >= 0; i--) {
    file_streams_[i]->close();
  }
#if defined(_WIN32) || defined(_WIN64)
  MS_LOG(DEBUG) << "Unlock file done by Python.";

#else
  flock(fd, LOCK_UN);
  close(fd);
#endif
  return Status::OK();
}

Status ShardWriter::WriteRawDataPreCheck(std::map<uint64_t, std::vector<json>> &raw_data,
                                         std::vector<std::vector<uint8_t>> &blob_data, bool sign, int *schema_count,
                                         int *row_count) {
  // check the free disk size
  std::shared_ptr<uint64_t> size_ptr;
  RETURN_IF_NOT_OK_MR(GetDiskSize(file_paths_[0], kFreeSize, &size_ptr));
  CHECK_FAIL_RETURN_UNEXPECTED_MR(
    *size_ptr >= kMinFreeDiskSize,
    "No free disk to be used while writing mindrecord files, available free disk size: " + std::to_string(*size_ptr));
  // compress blob
  if (shard_column_->CheckCompressBlob()) {
    for (auto &blob : blob_data) {
      int64_t compression_bytes = 0;
      blob = shard_column_->CompressBlob(blob, &compression_bytes);
      compression_size_ += compression_bytes;
    }
  }

  // Add 4-bytes dummy blob data if no any blob fields
  if (blob_data.size() == 0 && raw_data.size() > 0) {
    blob_data = std::vector<std::vector<uint8_t>>(raw_data[0].size(), std::vector<uint8_t>(kUnsignedInt4, 0));
  }

  // Add dummy id if all are blob fields
  if (blob_data.size() > 0 && raw_data.size() == 0) {
    raw_data.insert(std::pair<uint64_t, std::vector<json>>(0, std::vector<json>(blob_data.size(), kDummyId)));
  }
  std::shared_ptr<std::pair<int, int>> count_ptr;
  RETURN_IF_NOT_OK_MR(ValidateRawData(raw_data, blob_data, sign, &count_ptr));
  *schema_count = (*count_ptr).first;
  *row_count = (*count_ptr).second;
  return Status::OK();
}
Status ShardWriter::MergeBlobData(const std::vector<string> &blob_fields,
                                  const std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> &row_bin_data,
                                  std::shared_ptr<std::vector<uint8_t>> *output) {
  if (blob_fields.empty()) {
    return Status::OK();
  }
  if (blob_fields.size() == 1) {
    auto &blob = row_bin_data.at(blob_fields[0]);
    auto blob_size = blob->size();
    *output = std::make_shared<std::vector<uint8_t>>(blob_size);
    std::copy(blob->begin(), blob->end(), (*output)->begin());
  } else {
    size_t output_size = 0;
    for (auto &field : blob_fields) {
      output_size += row_bin_data.at(field)->size();
    }
    output_size += blob_fields.size() * sizeof(uint64_t);
    *output = std::make_shared<std::vector<uint8_t>>(output_size);
    std::vector<uint8_t> buf(sizeof(uint64_t), 0);
    size_t idx = 0;
    for (auto &field : blob_fields) {
      auto &b = row_bin_data.at(field);
      uint64_t blob_size = b->size();
      // big edian
      for (size_t i = 0; i < buf.size(); ++i) {
        buf[buf.size() - 1 - i] = (std::numeric_limits<uint8_t>::max()) & blob_size;
        blob_size >>= 8u;
      }
      std::copy(buf.begin(), buf.end(), (*output)->begin() + idx);
      idx += buf.size();
      std::copy(b->begin(), b->end(), (*output)->begin() + idx);
      idx += b->size();
    }
  }
  return Status::OK();
}

Status ShardWriter::WriteRawData(std::map<uint64_t, std::vector<json>> &raw_data,
                                 std::vector<std::vector<uint8_t>> &blob_data, bool sign, bool parallel_writer) {
  // Lock Writer if loading data parallel
  std::unique_ptr<int> fd_ptr;
  RETURN_IF_NOT_OK_MR(LockWriter(parallel_writer, &fd_ptr));

  // Get the count of schemas and rows
  int schema_count = 0;
  int row_count = 0;

  // Serialize raw data
  RETURN_IF_NOT_OK_MR(WriteRawDataPreCheck(raw_data, blob_data, sign, &schema_count, &row_count));
  CHECK_FAIL_RETURN_UNEXPECTED_MR(row_count >= kInt0, "[Internal ERROR] the size of raw data should be positive.");
  if (row_count == kInt0) {
    return Status::OK();
  }
  std::vector<std::vector<uint8_t>> bin_raw_data(row_count * schema_count);
  // Serialize raw data
  RETURN_IF_NOT_OK_MR(SerializeRawData(raw_data, bin_raw_data, row_count));
  // Set row size of raw data
  RETURN_IF_NOT_OK_MR(SetRawDataSize(bin_raw_data));
  // Set row size of blob data
  RETURN_IF_NOT_OK_MR(SetBlobDataSize(blob_data));
  // Write data to disk with multi threads
  RETURN_IF_NOT_OK_MR(ParallelWriteData(blob_data, bin_raw_data));
  MS_LOG(INFO) << "Succeed to write " << bin_raw_data.size() << " records.";

  RETURN_IF_NOT_OK_MR(UnlockWriter(*fd_ptr, parallel_writer));

  return Status::OK();
}

Status ShardWriter::WriteRawData(std::map<uint64_t, std::vector<py::handle>> &raw_data,
                                 std::map<uint64_t, std::vector<py::handle>> &blob_data, bool sign,
                                 bool parallel_writer) {
  std::map<uint64_t, std::vector<json>> raw_data_json;
  std::map<uint64_t, std::vector<json>> blob_data_json;

  (void)std::transform(raw_data.begin(), raw_data.end(), std::inserter(raw_data_json, raw_data_json.end()),
                       [](const std::pair<uint64_t, std::vector<py::handle>> &pair) {
                         auto &py_raw_data = pair.second;
                         std::vector<json> json_raw_data;
                         (void)std::transform(py_raw_data.begin(), py_raw_data.end(), std::back_inserter(json_raw_data),
                                              [](const py::handle &obj) { return nlohmann::detail::ToJsonImpl(obj); });
                         return std::make_pair(pair.first, std::move(json_raw_data));
                       });

  (void)std::transform(blob_data.begin(), blob_data.end(), std::inserter(blob_data_json, blob_data_json.end()),
                       [](const std::pair<uint64_t, std::vector<py::handle>> &pair) {
                         auto &py_blob_data = pair.second;
                         std::vector<json> jsonBlobData;
                         (void)std::transform(py_blob_data.begin(), py_blob_data.end(),
                                              std::back_inserter(jsonBlobData),
                                              [](const py::handle &obj) { return nlohmann::detail::ToJsonImpl(obj); });
                         return std::make_pair(pair.first, std::move(jsonBlobData));
                       });

  // Serialize blob page
  auto blob_data_iter = blob_data.begin();
  auto schema_count = blob_data.size();
  auto row_count = blob_data_iter->second.size();

  std::vector<std::vector<uint8_t>> bin_blob_data(row_count * schema_count);
  // Serialize blob data
  RETURN_IF_NOT_OK_MR(SerializeRawData(blob_data_json, bin_blob_data, row_count));
  return WriteRawData(raw_data_json, bin_blob_data, sign, parallel_writer);
}

Status ShardWriter::ParallelWriteData(const std::vector<std::vector<uint8_t>> &blob_data,
                                      const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  auto shards = BreakIntoShards();
  // define the number of thread
  int thread_num = static_cast<int>(shard_count_);
  CHECK_FAIL_RETURN_UNEXPECTED_MR(thread_num > 0, "[Internal ERROR] 'thread_num' should be positive.");
  if (thread_num > kMaxThreadCount) {
    thread_num = kMaxThreadCount;
  }
  int left_thread = shard_count_;
  int current_thread = 0;
  while (left_thread) {
    if (left_thread < thread_num) {
      thread_num = left_thread;
    }
    // Start one thread for one shard
    std::vector<std::thread> thread_set(thread_num);
    if (thread_num <= kMaxThreadCount) {
      for (int x = 0; x < thread_num; ++x) {
        int start_row = shards[current_thread + x].first;
        int end_row = shards[current_thread + x].second;
        thread_set[x] = std::thread(&ShardWriter::WriteByShard, this, current_thread + x, start_row, end_row,
                                    std::ref(blob_data), std::ref(bin_raw_data));
      }
      // Wait for threads done
      for (int x = 0; x < thread_num; ++x) {
        thread_set[x].join();
      }
      left_thread -= thread_num;
      current_thread += thread_num;
    }
  }
  return Status::OK();
}

Status ShardWriter::WriteByShard(int shard_id, int start_row, int end_row,
                                 const std::vector<std::vector<uint8_t>> &blob_data,
                                 const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  MS_LOG(DEBUG) << "Shard: " << shard_id << ", start: " << start_row << ", end: " << end_row
                << ", schema size: " << schema_count_;
  if (start_row == end_row) {
    return Status::OK();
  }
  vector<std::pair<int, int>> rows_in_group;
  std::shared_ptr<Page> last_raw_page = nullptr;
  std::shared_ptr<Page> last_blob_page = nullptr;
  SetLastRawPage(shard_id, last_raw_page);
  SetLastBlobPage(shard_id, last_blob_page);

  RETURN_IF_NOT_OK_MR(CutRowGroup(start_row, end_row, blob_data, rows_in_group, last_raw_page, last_blob_page));
  RETURN_IF_NOT_OK_MR(AppendBlobPage(shard_id, blob_data, rows_in_group, last_blob_page));
  RETURN_IF_NOT_OK_MR(NewBlobPage(shard_id, blob_data, rows_in_group, last_blob_page));
  RETURN_IF_NOT_OK_MR(ShiftRawPage(shard_id, rows_in_group, last_raw_page));
  RETURN_IF_NOT_OK_MR(WriteRawPage(shard_id, rows_in_group, last_raw_page, bin_raw_data));

  return Status::OK();
}

Status ShardWriter::CutRowGroup(int start_row, int end_row, const std::vector<std::vector<uint8_t>> &blob_data,
                                std::vector<std::pair<int, int>> &rows_in_group,
                                const std::shared_ptr<Page> &last_raw_page,
                                const std::shared_ptr<Page> &last_blob_page) {
  auto n_byte_blob = last_blob_page ? last_blob_page->GetPageSize() : 0;

  auto last_raw_page_size = last_raw_page ? last_raw_page->GetPageSize() : 0;
  auto last_raw_offset = last_raw_page ? last_raw_page->GetLastRowGroupID().second : 0;
  auto n_byte_raw = last_raw_page_size - last_raw_offset;

  int page_start_row = start_row;
  CHECK_FAIL_RETURN_UNEXPECTED_MR(start_row <= end_row,
                                  "[Internal ERROR] 'start_row': " + std::to_string(start_row) +
                                    " should be less than or equal to 'end_row': " + std::to_string(end_row));

  CHECK_FAIL_RETURN_UNEXPECTED_MR(
    end_row <= static_cast<int>(blob_data_size_.size()) && end_row <= static_cast<int>(raw_data_size_.size()),
    "[Internal ERROR] 'end_row': " + std::to_string(end_row) + " should be less than 'blob_data_size': " +
      std::to_string(blob_data_size_.size()) + " and 'raw_data_size': " + std::to_string(raw_data_size_.size()) + ".");
  for (int i = start_row; i < end_row; ++i) {
    // n_byte_blob(0) indicate appendBlobPage
    if (n_byte_blob == 0 || n_byte_blob + blob_data_size_[i] > page_size_ ||
        n_byte_raw + raw_data_size_[i] > page_size_) {
      rows_in_group.emplace_back(page_start_row, i);
      page_start_row = i;
      n_byte_blob = blob_data_size_[i];
      n_byte_raw = raw_data_size_[i];
    } else {
      n_byte_blob += blob_data_size_[i];
      n_byte_raw += raw_data_size_[i];
    }
  }

  // Not forget last one
  rows_in_group.emplace_back(page_start_row, end_row);
  return Status::OK();
}

Status ShardWriter::AppendBlobPage(const int &shard_id, const std::vector<std::vector<uint8_t>> &blob_data,
                                   const std::vector<std::pair<int, int>> &rows_in_group,
                                   const std::shared_ptr<Page> &last_blob_page) {
  auto blob_row = rows_in_group[0];
  if (blob_row.first == blob_row.second) {
    return Status::OK();
  }
  // Write disk
  auto page_id = last_blob_page->GetPageID();
  auto bytes_page = last_blob_page->GetPageSize();
  auto &io_seekp = file_streams_[shard_id]->seekp(page_size_ * page_id + header_size_ + bytes_page, std::ios::beg);
  if (!io_seekp.good() || io_seekp.fail() || io_seekp.bad()) {
    file_streams_[shard_id]->close();
    RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to seekg file.");
  }

  (void)FlushBlobChunk(file_streams_[shard_id], blob_data, blob_row);

  // Update last blob page
  bytes_page += std::accumulate(blob_data_size_.begin() + blob_row.first, blob_data_size_.begin() + blob_row.second, 0);
  last_blob_page->SetPageSize(bytes_page);
  uint64_t end_row = last_blob_page->GetEndRowID() + blob_row.second - blob_row.first;
  last_blob_page->SetEndRowID(end_row);
  (void)shard_header_->SetPage(last_blob_page);
  return Status::OK();
}

Status ShardWriter::NewBlobPage(const int &shard_id, const std::vector<std::vector<uint8_t>> &blob_data,
                                const std::vector<std::pair<int, int>> &rows_in_group,
                                const std::shared_ptr<Page> &last_blob_page) {
  auto page_id = shard_header_->GetLastPageId(shard_id);
  auto page_type_id = last_blob_page ? last_blob_page->GetPageTypeID() : -1;
  auto current_row = last_blob_page ? last_blob_page->GetEndRowID() : 0;
  // index(0) indicate appendBlobPage
  for (uint32_t i = 1; i < rows_in_group.size(); ++i) {
    auto blob_row = rows_in_group[i];

    // Write 1 blob page to disk
    auto &io_seekp = file_streams_[shard_id]->seekp(page_size_ * (page_id + 1) + header_size_, std::ios::beg);
    if (!io_seekp.good() || io_seekp.fail() || io_seekp.bad()) {
      file_streams_[shard_id]->close();
      RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to seekg file.");
    }

    (void)FlushBlobChunk(file_streams_[shard_id], blob_data, blob_row);
    // Create new page info for header
    auto page_size =
      std::accumulate(blob_data_size_.begin() + blob_row.first, blob_data_size_.begin() + blob_row.second, 0);
    std::vector<std::pair<int, uint64_t>> row_group_ids;
    auto start_row = current_row;
    auto end_row = start_row + blob_row.second - blob_row.first;
    auto page = Page(++page_id, shard_id, kPageTypeBlob, ++page_type_id, start_row, end_row, row_group_ids, page_size);
    (void)shard_header_->AddPage(std::make_shared<Page>(page));
    current_row = end_row;
  }
  return Status::OK();
}

Status ShardWriter::ShiftRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                                 std::shared_ptr<Page> &last_raw_page) {
  auto blob_row = rows_in_group[0];
  if (blob_row.first == blob_row.second) {
    return Status::OK();
  }
  auto last_raw_page_size = last_raw_page ? last_raw_page->GetPageSize() : 0;
  if (std::accumulate(raw_data_size_.begin() + blob_row.first, raw_data_size_.begin() + blob_row.second, 0) +
        last_raw_page_size <=
      page_size_) {
    return Status::OK();
  }
  auto page_id = shard_header_->GetLastPageId(shard_id);
  auto last_row_group_id_offset = last_raw_page->GetLastRowGroupID().second;
  auto last_raw_page_id = last_raw_page->GetPageID();
  auto shift_size = last_raw_page_size - last_row_group_id_offset;

  std::vector<uint8_t> buf(shift_size);

  // Read last row group from previous raw data page
  CHECK_FAIL_RETURN_UNEXPECTED_MR(
    shard_id >= 0 && shard_id < file_streams_.size(),
    "[Internal ERROR] 'shard_id' should be in range [0, " + std::to_string(file_streams_.size()) + ").");

  auto &io_seekg = file_streams_[shard_id]->seekg(
    page_size_ * last_raw_page_id + header_size_ + last_row_group_id_offset, std::ios::beg);
  if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
    file_streams_[shard_id]->close();
    RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to seekg file.");
  }

  auto &io_read = file_streams_[shard_id]->read(reinterpret_cast<char *>(&buf[0]), buf.size());
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    file_streams_[shard_id]->close();
    RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to read file.");
  }

  // Merge into new row group at new raw data page
  auto &io_seekp = file_streams_[shard_id]->seekp(page_size_ * (page_id + 1) + header_size_, std::ios::beg);
  if (!io_seekp.good() || io_seekp.fail() || io_seekp.bad()) {
    file_streams_[shard_id]->close();
    RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to seekg file.");
  }

  auto &io_handle = file_streams_[shard_id]->write(reinterpret_cast<char *>(&buf[0]), buf.size());
  if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
    file_streams_[shard_id]->close();
    RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to write file.");
  }
  last_raw_page->DeleteLastGroupId();
  (void)shard_header_->SetPage(last_raw_page);

  // Refresh page info in header
  int row_group_id = last_raw_page->GetLastRowGroupID().first + 1;
  std::vector<std::pair<int, uint64_t>> row_group_ids;
  row_group_ids.emplace_back(row_group_id, 0);
  int page_type_id = last_raw_page->GetPageID();
  auto page = Page(++page_id, shard_id, kPageTypeRaw, ++page_type_id, 0, 0, row_group_ids, shift_size);
  (void)shard_header_->AddPage(std::make_shared<Page>(page));

  // Reset: last raw page
  SetLastRawPage(shard_id, last_raw_page);
  return Status::OK();
}

Status ShardWriter::WriteRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                                 std::shared_ptr<Page> &last_raw_page,
                                 const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  int last_row_group_id = last_raw_page ? last_raw_page->GetLastRowGroupID().first : -1;
  for (uint32_t i = 0; i < rows_in_group.size(); ++i) {
    const auto &blob_row = rows_in_group[i];
    if (blob_row.first == blob_row.second) {
      continue;
    }
    auto raw_size =
      std::accumulate(raw_data_size_.begin() + blob_row.first, raw_data_size_.begin() + blob_row.second, 0);
    if (!last_raw_page) {
      RETURN_IF_NOT_OK_MR(EmptyRawPage(shard_id, last_raw_page));
    } else if (last_raw_page->GetPageSize() + raw_size > page_size_) {
      RETURN_IF_NOT_OK_MR(shard_header_->SetPage(last_raw_page));
      RETURN_IF_NOT_OK_MR(EmptyRawPage(shard_id, last_raw_page));
    }
    RETURN_IF_NOT_OK_MR(AppendRawPage(shard_id, rows_in_group, i, last_row_group_id, last_raw_page, bin_raw_data));
  }
  RETURN_IF_NOT_OK_MR(shard_header_->SetPage(last_raw_page));
  return Status::OK();
}

Status ShardWriter::EmptyRawPage(const int &shard_id, std::shared_ptr<Page> &last_raw_page) {
  auto row_group_ids = std::vector<std::pair<int, uint64_t>>();
  auto page_id = shard_header_->GetLastPageId(shard_id);
  auto page_type_id = last_raw_page ? last_raw_page->GetPageID() : -1;
  auto page = Page(++page_id, shard_id, kPageTypeRaw, ++page_type_id, 0, 0, row_group_ids, 0);
  RETURN_IF_NOT_OK_MR(shard_header_->AddPage(std::make_shared<Page>(page)));
  SetLastRawPage(shard_id, last_raw_page);
  return Status::OK();
}

Status ShardWriter::AppendRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                                  const int &chunk_id, int &last_row_group_id, std::shared_ptr<Page> last_raw_page,
                                  const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  std::vector<std::pair<int, uint64_t>> row_group_ids = last_raw_page->GetRowGroupIds();
  auto last_raw_page_id = last_raw_page->GetPageID();
  auto n_bytes = last_raw_page->GetPageSize();

  //  previous raw data page
  auto &io_seekp =
    file_streams_[shard_id]->seekp(page_size_ * last_raw_page_id + header_size_ + n_bytes, std::ios::beg);
  if (!io_seekp.good() || io_seekp.fail() || io_seekp.bad()) {
    file_streams_[shard_id]->close();
    RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to seekg file.");
  }

  if (chunk_id > 0) {
    row_group_ids.emplace_back(++last_row_group_id, n_bytes);
  }
  n_bytes += std::accumulate(raw_data_size_.begin() + rows_in_group[chunk_id].first,
                             raw_data_size_.begin() + rows_in_group[chunk_id].second, 0);
  RETURN_IF_NOT_OK_MR(FlushRawChunk(file_streams_[shard_id], rows_in_group, chunk_id, bin_raw_data));

  // Update previous raw data page
  last_raw_page->SetPageSize(n_bytes);
  last_raw_page->SetRowGroupIds(row_group_ids);
  RETURN_IF_NOT_OK_MR(shard_header_->SetPage(last_raw_page));

  return Status::OK();
}

Status ShardWriter::FlushBlobChunk(const std::shared_ptr<std::fstream> &out,
                                   const std::vector<std::vector<uint8_t>> &blob_data,
                                   const std::pair<int, int> &blob_row) {
  CHECK_FAIL_RETURN_UNEXPECTED_MR(
    blob_row.first <= blob_row.second && blob_row.second <= static_cast<int>(blob_data.size()) && blob_row.first >= 0,
    "[Internal ERROR] 'blob_row': " + std::to_string(blob_row.first) + ", " + std::to_string(blob_row.second) +
      " is invalid.");
  for (int j = blob_row.first; j < blob_row.second; ++j) {
    // Write the size of blob
    uint64_t line_len = blob_data[j].size();
    auto &io_handle = out->write(reinterpret_cast<char *>(&line_len), kInt64Len);
    if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
      out->close();
      RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to write file.");
    }

    // Write the data of blob
    auto line = blob_data[j];
    auto &io_handle_data = out->write(reinterpret_cast<char *>(&line[0]), line_len);
    if (!io_handle_data.good() || io_handle_data.fail() || io_handle_data.bad()) {
      out->close();
      RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to write file.");
    }
  }
  return Status::OK();
}

Status ShardWriter::FlushRawChunk(const std::shared_ptr<std::fstream> &out,
                                  const std::vector<std::pair<int, int>> &rows_in_group, const int &chunk_id,
                                  const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  for (int i = rows_in_group[chunk_id].first; i < rows_in_group[chunk_id].second; i++) {
    // Write the size of multi schemas
    for (uint32_t j = 0; j < schema_count_; ++j) {
      uint64_t line_len = bin_raw_data[i * schema_count_ + j].size();
      auto &io_handle = out->write(reinterpret_cast<char *>(&line_len), kInt64Len);
      if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
        out->close();
        RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to write file.");
      }
    }
    // Write the data of multi schemas
    for (uint32_t j = 0; j < schema_count_; ++j) {
      auto line = bin_raw_data[i * schema_count_ + j];
      auto &io_handle = out->write(reinterpret_cast<char *>(&line[0]), line.size());
      if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
        out->close();
        RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to write file.");
      }
    }
  }
  return Status::OK();
}

// Allocate data to shards evenly
std::vector<std::pair<int, int>> ShardWriter::BreakIntoShards() {
  std::vector<std::pair<int, int>> shards;
  int row_in_shard = row_count_ / shard_count_;
  int remains = row_count_ % shard_count_;

  std::vector<int> v_list(shard_count_);
  std::iota(v_list.begin(), v_list.end(), 0);

  std::mt19937 g = GetRandomDevice();
  std::shuffle(v_list.begin(), v_list.end(), g);
  std::unordered_set<int> set(v_list.begin(), v_list.begin() + remains);

  if (shard_count_ <= kMaxShardCount) {
    int start_row = 0;
    for (int i = 0; i < shard_count_; ++i) {
      int end_row = start_row + row_in_shard;
      if (set.count(i) == 1) {
        end_row++;
      }
      shards.emplace_back(start_row, end_row);
      start_row = end_row;
    }
  }
  return shards;
}

Status ShardWriter::WriteShardHeader() {
  RETURN_UNEXPECTED_IF_NULL_MR(shard_header_);
  int64_t compression_temp = compression_size_;
  uint64_t compression_size = compression_temp > 0 ? compression_temp : 0;
  shard_header_->SetCompressionSize(compression_size);

  auto shard_header = shard_header_->SerializeHeader();
  // Write header data to multi files
  CHECK_FAIL_RETURN_UNEXPECTED_MR(
    shard_count_ <= static_cast<int>(file_streams_.size()) && shard_count_ <= static_cast<int>(shard_header.size()),
    "[Internal ERROR] 'shard_count_' should be less than or equal to 'file_stream_' size: " +
      std::to_string(file_streams_.size()) + ", and 'shard_header' size: " + std::to_string(shard_header.size()) + ".");
  if (shard_count_ <= kMaxShardCount) {
    for (int shard_id = 0; shard_id < shard_count_; ++shard_id) {
      auto &io_seekp = file_streams_[shard_id]->seekp(0, std::ios::beg);
      if (!io_seekp.good() || io_seekp.fail() || io_seekp.bad()) {
        file_streams_[shard_id]->close();
        RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to seekp file.");
      }

      std::vector<uint8_t> bin_header(shard_header[shard_id].begin(), shard_header[shard_id].end());
      uint64_t line_len = bin_header.size();
      if (line_len + kInt64Len > header_size_) {
        file_streams_[shard_id]->close();
        RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] shard header is too big.");
      }
      auto &io_handle = file_streams_[shard_id]->write(reinterpret_cast<char *>(&line_len), kInt64Len);
      if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
        file_streams_[shard_id]->close();
        RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to write file.");
      }

      auto &io_handle_header = file_streams_[shard_id]->write(reinterpret_cast<char *>(&bin_header[0]), line_len);
      if (!io_handle_header.good() || io_handle_header.fail() || io_handle_header.bad()) {
        file_streams_[shard_id]->close();
        RETURN_STATUS_UNEXPECTED_MR("[Internal ERROR] Failed to write file.");
      }
      file_streams_[shard_id]->close();
    }
  }
  return Status::OK();
}

Status ShardWriter::SerializeRawData(std::map<uint64_t, std::vector<json>> &raw_data,
                                     std::vector<std::vector<uint8_t>> &bin_data, uint32_t row_count) {
  // define the number of thread
  uint32_t thread_num = std::thread::hardware_concurrency();
  if (thread_num == 0) {
    thread_num = kThreadNumber;
  }
  // Set the number of samples processed by each thread
  int group_num = static_cast<int>(ceil(row_count * 1.0 / thread_num));
  std::vector<std::thread> thread_set(thread_num);
  int work_thread_num = 0;
  for (uint32_t x = 0; x < thread_num; ++x) {
    int start_num = x * group_num;
    int end_num = ((x + 1) * group_num > row_count) ? row_count : (x + 1) * group_num;
    if (start_num >= end_num) {
      continue;
    }
    // Define the run boundary and start the child thread
    thread_set[x] =
      std::thread(&ShardWriter::FillArray, this, start_num, end_num, std::ref(raw_data), std::ref(bin_data));
    work_thread_num++;
  }
  for (uint32_t x = 0; x < work_thread_num; ++x) {
    // Set obstacles to prevent the main thread from running
    thread_set[x].join();
  }
  CHECK_FAIL_RETURN_SYNTAX_ERROR_MR(flag_ != true, "[Internal ERROR] Error raised in FillArray function.");
  return Status::OK();
}

Status ShardWriter::SetRawDataSize(const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  raw_data_size_ = std::vector<uint64_t>(row_count_, 0);
  for (uint32_t i = 0; i < row_count_; ++i) {
    raw_data_size_[i] = std::accumulate(
      bin_raw_data.begin() + (i * schema_count_), bin_raw_data.begin() + (i * schema_count_) + schema_count_, 0,
      [](uint64_t accumulator, const std::vector<uint8_t> &row) { return accumulator + kInt64Len + row.size(); });
  }
  CHECK_FAIL_RETURN_SYNTAX_ERROR_MR(*std::max_element(raw_data_size_.begin(), raw_data_size_.end()) <= page_size_,
                                    "Invalid data, Page size: " + std::to_string(page_size_) +
                                      " is too small to save a raw row. Please try to use the mindrecord api "
                                      "'set_page_size(value)' to enable larger page size, and the value range is in [" +
                                      std::to_string(kMinPageSize) + " bytes, " + std::to_string(kMaxPageSize) +
                                      " bytes].");
  return Status::OK();
}

Status ShardWriter::SetBlobDataSize(const std::vector<std::vector<uint8_t>> &blob_data) {
  blob_data_size_ = std::vector<uint64_t>(row_count_);
  (void)std::transform(blob_data.begin(), blob_data.end(), blob_data_size_.begin(),
                       [](const std::vector<uint8_t> &row) { return kInt64Len + row.size(); });
  CHECK_FAIL_RETURN_SYNTAX_ERROR_MR(*std::max_element(blob_data_size_.begin(), blob_data_size_.end()) <= page_size_,
                                    "Invalid data, Page size: " + std::to_string(page_size_) +
                                      " is too small to save a blob row. Please try to use the mindrecord api "
                                      "'set_page_size(value)' to enable larger page size, and the value range is in [" +
                                      std::to_string(kMinPageSize) + " bytes, " + std::to_string(kMaxPageSize) +
                                      " bytes].");
  return Status::OK();
}

Status ShardWriter::SetLastRawPage(const int &shard_id, std::shared_ptr<Page> &last_raw_page) {
  // Get last raw page
  auto last_raw_page_id = shard_header_->GetLastPageIdByType(shard_id, kPageTypeRaw);
  CHECK_FAIL_RETURN_SYNTAX_ERROR_MR(last_raw_page_id >= 0, "[Internal ERROR] 'last_raw_page_id': " +
                                                             std::to_string(last_raw_page_id) + " should be positive.");
  RETURN_IF_NOT_OK_MR(shard_header_->GetPage(shard_id, last_raw_page_id, &last_raw_page));
  return Status::OK();
}

Status ShardWriter::SetLastBlobPage(const int &shard_id, std::shared_ptr<Page> &last_blob_page) {
  // Get last blob page
  auto last_blob_page_id = shard_header_->GetLastPageIdByType(shard_id, kPageTypeBlob);
  CHECK_FAIL_RETURN_SYNTAX_ERROR_MR(
    last_blob_page_id >= 0,
    "[Internal ERROR] 'last_blob_page_id': " + std::to_string(last_blob_page_id) + " should be positive.");
  RETURN_IF_NOT_OK_MR(shard_header_->GetPage(shard_id, last_blob_page_id, &last_blob_page));
  return Status::OK();
}

Status ShardWriter::Initialize(const std::unique_ptr<ShardWriter> *writer_ptr,
                               const std::vector<std::string> &file_names) {
  RETURN_UNEXPECTED_IF_NULL_MR(writer_ptr);
  RETURN_IF_NOT_OK_MR((*writer_ptr)->Open(file_names, false));
  RETURN_IF_NOT_OK_MR((*writer_ptr)->SetHeaderSize(kDefaultHeaderSize));
  RETURN_IF_NOT_OK_MR((*writer_ptr)->SetPageSize(kDefaultPageSize));
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
