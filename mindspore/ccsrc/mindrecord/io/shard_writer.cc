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

#include "mindrecord/include/shard_writer.h"
#include "common/utils.h"
#include "mindrecord/include/common/shard_utils.h"
#include "./securec.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::DEBUG;
using mindspore::MsLogLevel::ERROR;
using mindspore::MsLogLevel::INFO;

namespace mindspore {
namespace mindrecord {
ShardWriter::ShardWriter()
    : shard_count_(1),
      header_size_(kDefaultHeaderSize),
      page_size_(kDefaultPageSize),
      row_count_(0),
      schema_count_(1) {}

ShardWriter::~ShardWriter() {
  for (int i = static_cast<int>(file_streams_.size()) - 1; i >= 0; i--) {
    file_streams_[i]->close();
  }
}

MSRStatus ShardWriter::GetFullPathFromFileName(const std::vector<std::string> &paths) {
  // Get full path from file name
  for (const auto &path : paths) {
    if (!CheckIsValidUtf8(path)) {
      MS_LOG(ERROR) << "The filename contains invalid uft-8 data: " << path << ".";
      return FAILED;
    }
    char resolved_path[PATH_MAX] = {0};
    char buf[PATH_MAX] = {0};
    if (strncpy_s(buf, PATH_MAX, common::SafeCStr(path), path.length()) != EOK) {
      MS_LOG(ERROR) << "Secure func failed";
      return FAILED;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (_fullpath(resolved_path, dirname(&(buf[0])), PATH_MAX) == nullptr) {
      MS_LOG(ERROR) << "Invalid file path";
      return FAILED;
    }
    if (_fullpath(resolved_path, common::SafeCStr(path), PATH_MAX) == nullptr) {
      MS_LOG(DEBUG) << "Path " << resolved_path;
    }
#else
    if (realpath(dirname(&(buf[0])), resolved_path) == nullptr) {
      MS_LOG(ERROR) << "Invalid file path";
      return FAILED;
    }
    if (realpath(common::SafeCStr(path), resolved_path) == nullptr) {
      MS_LOG(DEBUG) << "Path " << resolved_path;
    }
#endif
    file_paths_.emplace_back(string(resolved_path));
  }
  return SUCCESS;
}

MSRStatus ShardWriter::OpenDataFiles(bool append) {
  // Open files
  for (const auto &file : file_paths_) {
    std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
    if (!append) {
      // if not append and mindrecord file exist, return FAILED
      fs->open(common::SafeCStr(file), std::ios::in | std::ios::binary);
      if (fs->good()) {
        MS_LOG(ERROR) << "MindRecord file already existed.";
        fs->close();
        return FAILED;
      }
      fs->close();

      // open the mindrecord file to write
      fs->open(common::SafeCStr(file), std::ios::out | std::ios::binary);
      if (!fs->good()) {
        MS_LOG(ERROR) << "MindRecord file could not opened.";
        return FAILED;
      }
    } else {
      // open the mindrecord file to append
      fs->open(common::SafeCStr(file), std::ios::out | std::ios::in | std::ios::binary);
      if (!fs->good()) {
        MS_LOG(ERROR) << "MindRecord file could not opened for append.";
        return FAILED;
      }
    }
    MS_LOG(INFO) << "Open shard file successfully.";
    file_streams_.push_back(fs);
  }
  return SUCCESS;
}

MSRStatus ShardWriter::RemoveLockFile() {
  // Remove temporary file
  int ret = std::remove(pages_file_.c_str());
  if (ret == 0) {
    MS_LOG(DEBUG) << "Remove page file.";
  }

  ret = std::remove(lock_file_.c_str());
  if (ret == 0) {
    MS_LOG(DEBUG) << "Remove lock file.";
  }
  return SUCCESS;
}

MSRStatus ShardWriter::InitLockFile() {
  if (file_paths_.size() == 0) {
    MS_LOG(ERROR) << "File path not initialized.";
    return FAILED;
  }

  lock_file_ = file_paths_[0] + kLockFileSuffix;
  pages_file_ = file_paths_[0] + kPageFileSuffix;

  if (RemoveLockFile() == FAILED) {
    MS_LOG(ERROR) << "Remove file failed.";
    return FAILED;
  }
  return SUCCESS;
}

MSRStatus ShardWriter::Open(const std::vector<std::string> &paths, bool append) {
  shard_count_ = paths.size();
  if (shard_count_ > kMaxShardCount || shard_count_ == 0) {
    MS_LOG(ERROR) << "The Shard Count greater than max value or equal to 0.";
    return FAILED;
  }
  if (schema_count_ > kMaxSchemaCount) {
    MS_LOG(ERROR) << "The schema Count greater than max value.";
    return FAILED;
  }

  // Get full path from file name
  if (GetFullPathFromFileName(paths) == FAILED) {
    MS_LOG(ERROR) << "Get full path from file name failed.";
    return FAILED;
  }

  // Open files
  if (OpenDataFiles(append) == FAILED) {
    MS_LOG(ERROR) << "Open data files failed.";
    return FAILED;
  }

  // Init lock file
  if (InitLockFile() == FAILED) {
    MS_LOG(ERROR) << "Init lock file failed.";
    return FAILED;
  }
  return SUCCESS;
}

MSRStatus ShardWriter::OpenForAppend(const std::string &path) {
  if (!IsLegalFile(path)) {
    return FAILED;
  }
  auto ret1 = ShardHeader::BuildSingleHeader(path);
  if (ret1.first != SUCCESS) {
    return FAILED;
  }
  auto json_header = ret1.second;
  auto ret2 = GetParentDir(path);
  if (SUCCESS != ret2.first) {
    return FAILED;
  }
  std::vector<std::string> real_addresses;
  for (const auto &path : json_header["shard_addresses"]) {
    std::string abs_path = ret2.second + string(path);
    real_addresses.emplace_back(abs_path);
  }
  ShardHeader header = ShardHeader();
  if (header.BuildDataset(real_addresses) == FAILED) {
    return FAILED;
  }
  shard_header_ = std::make_shared<ShardHeader>(header);
  MSRStatus ret = SetHeaderSize(shard_header_->GetHeaderSize());
  if (ret == FAILED) {
    return FAILED;
  }
  ret = SetPageSize(shard_header_->GetPageSize());
  if (ret == FAILED) {
    return FAILED;
  }
  ret = Open(real_addresses, true);
  if (ret == FAILED) {
    MS_LOG(ERROR) << "Open file failed";
    return FAILED;
  }
  shard_column_ = std::make_shared<ShardColumn>(shard_header_);
  return SUCCESS;
}

MSRStatus ShardWriter::Commit() {
  // Read pages file
  std::ifstream page_file(pages_file_.c_str());
  if (page_file.good()) {
    page_file.close();
    if (shard_header_->FileToPages(pages_file_) == FAILED) {
      MS_LOG(ERROR) << "Read pages from file failed";
      return FAILED;
    }
  }

  if (WriteShardHeader() == FAILED) {
    MS_LOG(ERROR) << "Write metadata failed";
    return FAILED;
  }
  MS_LOG(INFO) << "Write metadata successfully.";

  // Remove lock file
  if (RemoveLockFile() == FAILED) {
    MS_LOG(ERROR) << "Remove lock file failed.";
    return FAILED;
  }

  return SUCCESS;
}

MSRStatus ShardWriter::SetShardHeader(std::shared_ptr<ShardHeader> header_data) {
  MSRStatus ret = header_data->InitByFiles(file_paths_);
  if (ret == FAILED) {
    return FAILED;
  }

  // set fields in mindrecord when empty
  std::vector<std::pair<uint64_t, std::string>> fields = header_data->GetFields();
  if (fields.empty()) {
    MS_LOG(DEBUG) << "Missing index fields by user, auto generate index fields.";
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
      ret = header_data->AddIndexFields(fields);
      if (ret == FAILED) {
        MS_LOG(ERROR) << "Add index field failed";
        return FAILED;
      }
    }
  }

  shard_header_ = header_data;
  shard_header_->SetHeaderSize(header_size_);
  shard_header_->SetPageSize(page_size_);
  shard_column_ = std::make_shared<ShardColumn>(shard_header_);
  return SUCCESS;
}

MSRStatus ShardWriter::SetHeaderSize(const uint64_t &header_size) {
  // header_size [16KB, 128MB]
  if (header_size < kMinHeaderSize || header_size > kMaxHeaderSize) {
    MS_LOG(ERROR) << "Header size should between 16KB and 128MB.";
    return FAILED;
  }
  if (header_size % 4 != 0) {
    MS_LOG(ERROR) << "Header size should be divided by four.";
    return FAILED;
  }

  header_size_ = header_size;
  return SUCCESS;
}

MSRStatus ShardWriter::SetPageSize(const uint64_t &page_size) {
  // PageSize [32KB, 256MB]
  if (page_size < kMinPageSize || page_size > kMaxPageSize) {
    MS_LOG(ERROR) << "Page size should between 16KB and 256MB.";
    return FAILED;
  }
  if (page_size % 4 != 0) {
    MS_LOG(ERROR) << "Page size should be divided by four.";
    return FAILED;
  }
  page_size_ = page_size;
  return SUCCESS;
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
      MS_LOG(ERROR) << "For schema " << id << ", " << loc + 1 << " th data is wrong: " << message;
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

MSRStatus ShardWriter::CheckDataTypeAndValue(const std::string &key, const json &value, const json &data, const int &i,
                                             std::map<int, std::string> &err_raw_data) {
  auto data_type = std::string(value["type"].get<std::string>());

  if ((data_type == "int32" && !data[key].is_number_integer()) ||
      (data_type == "int64" && !data[key].is_number_integer()) ||
      (data_type == "float32" && !data[key].is_number_float()) ||
      (data_type == "float64" && !data[key].is_number_float()) || (data_type == "string" && !data[key].is_string())) {
    std::string message = "field: " + key + " type : " + data_type + " value: " + data[key].dump() + " is not matched";
    PopulateMutexErrorData(i, message, err_raw_data);
    return FAILED;
  }

  if (data_type == "int32" && data[key].is_number_integer()) {
    int64_t temp_value = data[key];
    if (static_cast<int64_t>(temp_value) < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
        static_cast<int64_t>(temp_value) > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      std::string message =
        "field: " + key + " type : " + data_type + " value: " + data[key].dump() + " is out of range";
      PopulateMutexErrorData(i, message, err_raw_data);
      return FAILED;
    }
  }
  return SUCCESS;
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
        std::string message = "there is not '" + key + "' object in the raw data";
        PopulateMutexErrorData(i, message, err_raw_data);
        break;
      }

      if (value.size() == kInt2) {
        // Skip check since all shaped data will store as blob
        continue;
      }

      if (CheckDataTypeAndValue(key, value, data, i, err_raw_data) != SUCCESS) {
        break;
      }
    }
  }
}

MSRStatus ShardWriter::CheckData(const std::map<uint64_t, std::vector<json>> &raw_data) {
  auto rawdata_iter = raw_data.begin();

  // make sure rawdata match schema
  for (; rawdata_iter != raw_data.end(); ++rawdata_iter) {
    // used for storing error
    std::map<int, std::string> sub_err_mg;
    int schema_id = rawdata_iter->first;
    auto result = shard_header_->GetSchemaByID(schema_id);
    if (result.second != SUCCESS) {
      return FAILED;
    }
    json schema = result.first->GetSchema()["schema"];
    for (const auto &field : result.first->GetBlobFields()) {
      (void)schema.erase(field);
    }
    std::vector<json> sub_raw_data = rawdata_iter->second;

    // calculate start position and end position for each thread
    int batch_size = rawdata_iter->second.size() / shard_count_;
    int thread_num = shard_count_;
    if (thread_num <= 0) {
      return FAILED;
    }
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
    if (thread_num > kMaxThreadCount) {
      return FAILED;
    }
    // Wait for threads done
    for (int x = 0; x < thread_num; ++x) {
      thread_set[x].join();
    }

    (void)err_mg_.insert(std::make_pair(schema_id, sub_err_mg));
  }
  return SUCCESS;
}

std::tuple<MSRStatus, int, int> ShardWriter::ValidateRawData(std::map<uint64_t, std::vector<json>> &raw_data,
                                                             std::vector<std::vector<uint8_t>> &blob_data, bool sign) {
  auto rawdata_iter = raw_data.begin();
  schema_count_ = raw_data.size();
  std::tuple<MSRStatus, int, int> failed(FAILED, 0, 0);
  if (schema_count_ == 0) {
    MS_LOG(ERROR) << "Data size is zero";
    return failed;
  }

  // keep schema_id
  std::set<int64_t> schema_ids;
  row_count_ = (rawdata_iter->second).size();
  MS_LOG(DEBUG) << "Schema count is " << schema_count_;

  // Determine if the number of schemas is the same
  if (shard_header_->GetSchemas().size() != schema_count_) {
    MS_LOG(ERROR) << "Data size is not equal with the schema size";
    return failed;
  }

  // Determine raw_data size == blob_data size
  if (raw_data[0].size() != blob_data.size()) {
    MS_LOG(ERROR) << "Raw data size is not equal blob data size";
    return failed;
  }

  // Determine whether the number of samples corresponding to each schema is the same
  for (rawdata_iter = raw_data.begin(); rawdata_iter != raw_data.end(); ++rawdata_iter) {
    if (row_count_ != rawdata_iter->second.size()) {
      MS_LOG(ERROR) << "Data size is not equal";
      return failed;
    }
    (void)schema_ids.insert(rawdata_iter->first);
  }
  const std::vector<std::shared_ptr<Schema>> &schemas = shard_header_->GetSchemas();
  if (std::any_of(schemas.begin(), schemas.end(), [schema_ids](const std::shared_ptr<Schema> &schema) {
        return schema_ids.find(schema->GetSchemaID()) == schema_ids.end();
      })) {
    // There is not enough data which is not matching the number of schema
    MS_LOG(ERROR) << "Input rawdata schema id do not match real schema id.";
    return failed;
  }

  if (!sign) {
    std::tuple<MSRStatus, int, int> success(SUCCESS, schema_count_, row_count_);
    return success;
  }

  // check the data according the schema
  if (CheckData(raw_data) != SUCCESS) {
    MS_LOG(ERROR) << "Data validate check failed";
    return std::tuple<MSRStatus, int, int>(FAILED, schema_count_, row_count_);
  }

  // delete wrong data from raw data
  DeleteErrorData(raw_data, blob_data);

  // update raw count
  row_count_ = row_count_ - err_mg_.begin()->second.size();
  std::tuple<MSRStatus, int, int> success(SUCCESS, schema_count_, row_count_);
  return success;
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

int ShardWriter::LockWriter(bool parallel_writer) {
  if (!parallel_writer) {
    return 0;
  }

#if defined(_WIN32) || defined(_WIN64)
  MS_LOG(DEBUG) << "Lock file done by python.";
  const int fd = 0;
#else
  const int fd = open(lock_file_.c_str(), O_WRONLY | O_CREAT, 0666);
  if (fd >= 0) {
    flock(fd, LOCK_EX);
  } else {
    MS_LOG(ERROR) << "Shard writer failed when locking file";
    return -1;
  }
#endif

  // Open files
  file_streams_.clear();
  for (const auto &file : file_paths_) {
    std::shared_ptr<std::fstream> fs = std::make_shared<std::fstream>();
    fs->open(common::SafeCStr(file), std::ios::in | std::ios::out | std::ios::binary);
    if (fs->fail()) {
      MS_LOG(ERROR) << "File could not opened";
      return -1;
    }
    file_streams_.push_back(fs);
  }

  if (shard_header_->FileToPages(pages_file_) == FAILED) {
    MS_LOG(ERROR) << "Read pages from file failed";
    return -1;
  }
  return fd;
}

MSRStatus ShardWriter::UnlockWriter(int fd, bool parallel_writer) {
  if (!parallel_writer) {
    return SUCCESS;
  }

  if (shard_header_->PagesToFile(pages_file_) == FAILED) {
    MS_LOG(ERROR) << "Write pages to file failed";
    return FAILED;
  }

  for (int i = static_cast<int>(file_streams_.size()) - 1; i >= 0; i--) {
    file_streams_[i]->close();
  }

#if defined(_WIN32) || defined(_WIN64)
  MS_LOG(DEBUG) << "Unlock file done by python.";
#else
  flock(fd, LOCK_UN);
  close(fd);
#endif
  return SUCCESS;
}

MSRStatus ShardWriter::WriteRawDataPreCheck(std::map<uint64_t, std::vector<json>> &raw_data,
                                            std::vector<std::vector<uint8_t>> &blob_data, bool sign, int *schema_count,
                                            int *row_count) {
  // check the free disk size
  auto st_space = GetDiskSize(file_paths_[0], kFreeSize);
  if (st_space.first != SUCCESS || st_space.second < kMinFreeDiskSize) {
    MS_LOG(ERROR) << "IO error / there is no free disk to be used";
    return FAILED;
  }

  // compress blob
  if (shard_column_->CheckCompressBlob()) {
    for (auto &blob : blob_data) {
      blob = shard_column_->CompressBlob(blob);
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

  auto v = ValidateRawData(raw_data, blob_data, sign);
  if (std::get<0>(v) == FAILED) {
    MS_LOG(ERROR) << "Validate raw data failed";
    return FAILED;
  }
  *schema_count = std::get<1>(v);
  *row_count = std::get<2>(v);
  return SUCCESS;
}

MSRStatus ShardWriter::WriteRawData(std::map<uint64_t, std::vector<json>> &raw_data,
                                    std::vector<std::vector<uint8_t>> &blob_data, bool sign, bool parallel_writer) {
  // Lock Writer if loading data parallel
  int fd = LockWriter(parallel_writer);
  if (fd < 0) {
    MS_LOG(ERROR) << "Lock writer failed";
    return FAILED;
  }

  // Get the count of schemas and rows
  int schema_count = 0;
  int row_count = 0;

  // Serialize raw data
  if (WriteRawDataPreCheck(raw_data, blob_data, sign, &schema_count, &row_count) == FAILED) {
    MS_LOG(ERROR) << "Check raw data failed";
    return FAILED;
  }

  if (row_count == kInt0) {
    MS_LOG(INFO) << "Raw data size is 0.";
    return SUCCESS;
  }

  std::vector<std::vector<uint8_t>> bin_raw_data(row_count * schema_count);

  // Serialize raw data
  if (SerializeRawData(raw_data, bin_raw_data, row_count) == FAILED) {
    MS_LOG(ERROR) << "Serialize raw data failed";
    return FAILED;
  }

  // Set row size of raw data
  if (SetRawDataSize(bin_raw_data) == FAILED) {
    MS_LOG(ERROR) << "Set raw data size failed";
    return FAILED;
  }

  // Set row size of blob data
  if (SetBlobDataSize(blob_data) == FAILED) {
    MS_LOG(ERROR) << "Set blob data size failed";
    return FAILED;
  }

  // Write data to disk with multi threads
  if (ParallelWriteData(blob_data, bin_raw_data) == FAILED) {
    MS_LOG(ERROR) << "Parallel write data failed";
    return FAILED;
  }
  MS_LOG(INFO) << "Write " << bin_raw_data.size() << " records successfully.";

  if (UnlockWriter(fd, parallel_writer) == FAILED) {
    MS_LOG(ERROR) << "Unlock writer failed";
    return FAILED;
  }

  return SUCCESS;
}

MSRStatus ShardWriter::WriteRawData(std::map<uint64_t, std::vector<py::handle>> &raw_data,
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
  if (SerializeRawData(blob_data_json, bin_blob_data, row_count) == FAILED) {
    MS_LOG(ERROR) << "Serialize raw data failed in write raw data";
    return FAILED;
  }
  return WriteRawData(raw_data_json, bin_blob_data, sign, parallel_writer);
}

MSRStatus ShardWriter::WriteRawData(std::map<uint64_t, std::vector<py::handle>> &raw_data,
                                    vector<vector<uint8_t>> &blob_data, bool sign, bool parallel_writer) {
  std::map<uint64_t, std::vector<json>> raw_data_json;
  (void)std::transform(raw_data.begin(), raw_data.end(), std::inserter(raw_data_json, raw_data_json.end()),
                       [](const std::pair<uint64_t, std::vector<py::handle>> &pair) {
                         auto &py_raw_data = pair.second;
                         std::vector<json> json_raw_data;
                         (void)std::transform(py_raw_data.begin(), py_raw_data.end(), std::back_inserter(json_raw_data),
                                              [](const py::handle &obj) { return nlohmann::detail::ToJsonImpl(obj); });
                         return std::make_pair(pair.first, std::move(json_raw_data));
                       });
  return WriteRawData(raw_data_json, blob_data, sign, parallel_writer);
}

MSRStatus ShardWriter::ParallelWriteData(const std::vector<std::vector<uint8_t>> &blob_data,
                                         const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  auto shards = BreakIntoShards();
  // define the number of thread
  int thread_num = static_cast<int>(shard_count_);
  if (thread_num < 0) {
    return FAILED;
  }
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
  return SUCCESS;
}

MSRStatus ShardWriter::WriteByShard(int shard_id, int start_row, int end_row,
                                    const std::vector<std::vector<uint8_t>> &blob_data,
                                    const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  MS_LOG(DEBUG) << "Shard: " << shard_id << ", start: " << start_row << ", end: " << end_row
                << ", schema size: " << schema_count_;
  if (start_row == end_row) {
    return SUCCESS;
  }
  vector<std::pair<int, int>> rows_in_group;
  std::shared_ptr<Page> last_raw_page = nullptr;
  std::shared_ptr<Page> last_blob_page = nullptr;
  SetLastRawPage(shard_id, last_raw_page);
  SetLastBlobPage(shard_id, last_blob_page);

  if (CutRowGroup(start_row, end_row, blob_data, rows_in_group, last_raw_page, last_blob_page) == FAILED) {
    MS_LOG(ERROR) << "Cut row group failed";
    return FAILED;
  }

  if (AppendBlobPage(shard_id, blob_data, rows_in_group, last_blob_page) == FAILED) {
    MS_LOG(ERROR) << "Append bolb page failed";
    return FAILED;
  }

  if (NewBlobPage(shard_id, blob_data, rows_in_group, last_blob_page) == FAILED) {
    MS_LOG(ERROR) << "New blob page failed";
    return FAILED;
  }

  if (ShiftRawPage(shard_id, rows_in_group, last_raw_page) == FAILED) {
    MS_LOG(ERROR) << "Shit raw page failed";
    return FAILED;
  }

  if (WriteRawPage(shard_id, rows_in_group, last_raw_page, bin_raw_data) == FAILED) {
    MS_LOG(ERROR) << "Write raw page failed";
    return FAILED;
  }

  return SUCCESS;
}

MSRStatus ShardWriter::CutRowGroup(int start_row, int end_row, const std::vector<std::vector<uint8_t>> &blob_data,
                                   std::vector<std::pair<int, int>> &rows_in_group,
                                   const std::shared_ptr<Page> &last_raw_page,
                                   const std::shared_ptr<Page> &last_blob_page) {
  auto n_byte_blob = last_blob_page ? last_blob_page->GetPageSize() : 0;

  auto last_raw_page_size = last_raw_page ? last_raw_page->GetPageSize() : 0;
  auto last_raw_offset = last_raw_page ? last_raw_page->GetLastRowGroupID().second : 0;
  auto n_byte_raw = last_raw_page_size - last_raw_offset;

  int page_start_row = start_row;
  if (start_row > end_row) {
    return FAILED;
  }
  if (end_row > static_cast<int>(blob_data_size_.size()) || end_row > static_cast<int>(raw_data_size_.size())) {
    return FAILED;
  }
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
  return SUCCESS;
}

MSRStatus ShardWriter::AppendBlobPage(const int &shard_id, const std::vector<std::vector<uint8_t>> &blob_data,
                                      const std::vector<std::pair<int, int>> &rows_in_group,
                                      const std::shared_ptr<Page> &last_blob_page) {
  auto blob_row = rows_in_group[0];
  if (blob_row.first == blob_row.second) return SUCCESS;

  // Write disk
  auto page_id = last_blob_page->GetPageID();
  auto bytes_page = last_blob_page->GetPageSize();
  auto &io_seekp = file_streams_[shard_id]->seekp(page_size_ * page_id + header_size_ + bytes_page, std::ios::beg);
  if (!io_seekp.good() || io_seekp.fail() || io_seekp.bad()) {
    MS_LOG(ERROR) << "File seekp failed";
    file_streams_[shard_id]->close();
    return FAILED;
  }

  (void)FlushBlobChunk(file_streams_[shard_id], blob_data, blob_row);

  // Update last blob page
  bytes_page += std::accumulate(blob_data_size_.begin() + blob_row.first, blob_data_size_.begin() + blob_row.second, 0);
  last_blob_page->SetPageSize(bytes_page);
  uint64_t end_row = last_blob_page->GetEndRowID() + blob_row.second - blob_row.first;
  last_blob_page->SetEndRowID(end_row);
  (void)shard_header_->SetPage(last_blob_page);
  return SUCCESS;
}

MSRStatus ShardWriter::NewBlobPage(const int &shard_id, const std::vector<std::vector<uint8_t>> &blob_data,
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
      MS_LOG(ERROR) << "File seekp failed";
      file_streams_[shard_id]->close();
      return FAILED;
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
  return SUCCESS;
}

MSRStatus ShardWriter::ShiftRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                                    std::shared_ptr<Page> &last_raw_page) {
  auto blob_row = rows_in_group[0];
  if (blob_row.first == blob_row.second) return SUCCESS;
  auto last_raw_page_size = last_raw_page ? last_raw_page->GetPageSize() : 0;
  if (std::accumulate(raw_data_size_.begin() + blob_row.first, raw_data_size_.begin() + blob_row.second, 0) +
        last_raw_page_size <=
      page_size_) {
    return SUCCESS;
  }
  auto page_id = shard_header_->GetLastPageId(shard_id);
  auto last_row_group_id_offset = last_raw_page->GetLastRowGroupID().second;
  auto last_raw_page_id = last_raw_page->GetPageID();
  auto shift_size = last_raw_page_size - last_row_group_id_offset;

  std::vector<uint8_t> buf(shift_size);

  // Read last row group from previous raw data page
  if (shard_id < 0 || shard_id >= file_streams_.size()) {
    return FAILED;
  }

  auto &io_seekg = file_streams_[shard_id]->seekg(
    page_size_ * last_raw_page_id + header_size_ + last_row_group_id_offset, std::ios::beg);
  if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
    MS_LOG(ERROR) << "File seekg failed";
    file_streams_[shard_id]->close();
    return FAILED;
  }

  auto &io_read = file_streams_[shard_id]->read(reinterpret_cast<char *>(&buf[0]), buf.size());
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    MS_LOG(ERROR) << "File read failed";
    file_streams_[shard_id]->close();
    return FAILED;
  }

  // Merge into new row group at new raw data page
  auto &io_seekp = file_streams_[shard_id]->seekp(page_size_ * (page_id + 1) + header_size_, std::ios::beg);
  if (!io_seekp.good() || io_seekp.fail() || io_seekp.bad()) {
    MS_LOG(ERROR) << "File seekp failed";
    file_streams_[shard_id]->close();
    return FAILED;
  }

  auto &io_handle = file_streams_[shard_id]->write(reinterpret_cast<char *>(&buf[0]), buf.size());
  if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
    MS_LOG(ERROR) << "File write failed";
    file_streams_[shard_id]->close();
    return FAILED;
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
  return SUCCESS;
}

MSRStatus ShardWriter::WriteRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                                    std::shared_ptr<Page> &last_raw_page,
                                    const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  int last_row_group_id = last_raw_page ? last_raw_page->GetLastRowGroupID().first : -1;
  for (uint32_t i = 0; i < rows_in_group.size(); ++i) {
    const auto &blob_row = rows_in_group[i];
    if (blob_row.first == blob_row.second) continue;
    auto raw_size =
      std::accumulate(raw_data_size_.begin() + blob_row.first, raw_data_size_.begin() + blob_row.second, 0);
    if (!last_raw_page) {
      EmptyRawPage(shard_id, last_raw_page);
    } else if (last_raw_page->GetPageSize() + raw_size > page_size_) {
      (void)shard_header_->SetPage(last_raw_page);
      EmptyRawPage(shard_id, last_raw_page);
    }
    if (AppendRawPage(shard_id, rows_in_group, i, last_row_group_id, last_raw_page, bin_raw_data) != SUCCESS) {
      return FAILED;
    }
  }
  (void)shard_header_->SetPage(last_raw_page);
  return SUCCESS;
}

void ShardWriter::EmptyRawPage(const int &shard_id, std::shared_ptr<Page> &last_raw_page) {
  auto row_group_ids = std::vector<std::pair<int, uint64_t>>();
  auto page_id = shard_header_->GetLastPageId(shard_id);
  auto page_type_id = last_raw_page ? last_raw_page->GetPageID() : -1;
  auto page = Page(++page_id, shard_id, kPageTypeRaw, ++page_type_id, 0, 0, row_group_ids, 0);
  (void)shard_header_->AddPage(std::make_shared<Page>(page));
  SetLastRawPage(shard_id, last_raw_page);
}

MSRStatus ShardWriter::AppendRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                                     const int &chunk_id, int &last_row_group_id, std::shared_ptr<Page> last_raw_page,
                                     const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  std::vector<std::pair<int, uint64_t>> row_group_ids = last_raw_page->GetRowGroupIds();
  auto last_raw_page_id = last_raw_page->GetPageID();
  auto n_bytes = last_raw_page->GetPageSize();

  //  previous raw data page
  auto &io_seekp =
    file_streams_[shard_id]->seekp(page_size_ * last_raw_page_id + header_size_ + n_bytes, std::ios::beg);
  if (!io_seekp.good() || io_seekp.fail() || io_seekp.bad()) {
    MS_LOG(ERROR) << "File seekp failed";
    file_streams_[shard_id]->close();
    return FAILED;
  }

  if (chunk_id > 0) row_group_ids.emplace_back(++last_row_group_id, n_bytes);
  n_bytes += std::accumulate(raw_data_size_.begin() + rows_in_group[chunk_id].first,
                             raw_data_size_.begin() + rows_in_group[chunk_id].second, 0);
  (void)FlushRawChunk(file_streams_[shard_id], rows_in_group, chunk_id, bin_raw_data);

  // Update previous raw data page
  last_raw_page->SetPageSize(n_bytes);
  last_raw_page->SetRowGroupIds(row_group_ids);
  (void)shard_header_->SetPage(last_raw_page);

  return SUCCESS;
}

MSRStatus ShardWriter::FlushBlobChunk(const std::shared_ptr<std::fstream> &out,
                                      const std::vector<std::vector<uint8_t>> &blob_data,
                                      const std::pair<int, int> &blob_row) {
  if (blob_row.first > blob_row.second) {
    return FAILED;
  }
  if (blob_row.second > static_cast<int>(blob_data.size()) || blob_row.first < 0) {
    return FAILED;
  }
  for (int j = blob_row.first; j < blob_row.second; ++j) {
    // Write the size of blob
    uint64_t line_len = blob_data[j].size();
    auto &io_handle = out->write(reinterpret_cast<char *>(&line_len), kInt64Len);
    if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
      MS_LOG(ERROR) << "File write failed";
      out->close();
      return FAILED;
    }

    // Write the data of blob
    auto line = blob_data[j];
    auto &io_handle_data = out->write(reinterpret_cast<char *>(&line[0]), line_len);
    if (!io_handle_data.good() || io_handle_data.fail() || io_handle_data.bad()) {
      MS_LOG(ERROR) << "File write failed";
      out->close();
      return FAILED;
    }
  }
  return SUCCESS;
}

MSRStatus ShardWriter::FlushRawChunk(const std::shared_ptr<std::fstream> &out,
                                     const std::vector<std::pair<int, int>> &rows_in_group, const int &chunk_id,
                                     const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  for (int i = rows_in_group[chunk_id].first; i < rows_in_group[chunk_id].second; i++) {
    // Write the size of multi schemas
    for (uint32_t j = 0; j < schema_count_; ++j) {
      uint64_t line_len = bin_raw_data[i * schema_count_ + j].size();
      auto &io_handle = out->write(reinterpret_cast<char *>(&line_len), kInt64Len);
      if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
        MS_LOG(ERROR) << "File write failed";
        out->close();
        return FAILED;
      }
    }
    // Write the data of multi schemas
    for (uint32_t j = 0; j < schema_count_; ++j) {
      auto line = bin_raw_data[i * schema_count_ + j];
      auto &io_handle = out->write(reinterpret_cast<char *>(&line[0]), line.size());
      if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
        MS_LOG(ERROR) << "File write failed";
        out->close();
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

// Allocate data to shards evenly
std::vector<std::pair<int, int>> ShardWriter::BreakIntoShards() {
  std::vector<std::pair<int, int>> shards;
  int row_in_shard = row_count_ / shard_count_;
  int remains = row_count_ % shard_count_;

  std::vector<int> v_list(shard_count_);
  std::iota(v_list.begin(), v_list.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(v_list.begin(), v_list.end(), g);
  std::unordered_set<int> set(v_list.begin(), v_list.begin() + remains);

  if (shard_count_ <= kMaxShardCount) {
    int start_row = 0;
    for (int i = 0; i < shard_count_; ++i) {
      int end_row = start_row + row_in_shard;
      if (set.count(i)) end_row++;
      shards.emplace_back(start_row, end_row);
      start_row = end_row;
    }
  }
  return shards;
}

MSRStatus ShardWriter::WriteShardHeader() {
  if (shard_header_ == nullptr) {
    MS_LOG(ERROR) << "Shard header is null";
    return FAILED;
  }
  auto shard_header = shard_header_->SerializeHeader();
  // Write header data to multi files
  if (shard_count_ > static_cast<int>(file_streams_.size()) || shard_count_ > static_cast<int>(shard_header.size())) {
    return FAILED;
  }
  if (shard_count_ <= kMaxShardCount) {
    for (int shard_id = 0; shard_id < shard_count_; ++shard_id) {
      auto &io_seekp = file_streams_[shard_id]->seekp(0, std::ios::beg);
      if (!io_seekp.good() || io_seekp.fail() || io_seekp.bad()) {
        MS_LOG(ERROR) << "File seekp failed";
        file_streams_[shard_id]->close();
        return FAILED;
      }

      std::vector<uint8_t> bin_header(shard_header[shard_id].begin(), shard_header[shard_id].end());
      uint64_t line_len = bin_header.size();
      if (line_len + kInt64Len > header_size_) {
        MS_LOG(ERROR) << "Shard header is too big";
        return FAILED;
      }

      auto &io_handle = file_streams_[shard_id]->write(reinterpret_cast<char *>(&line_len), kInt64Len);
      if (!io_handle.good() || io_handle.fail() || io_handle.bad()) {
        MS_LOG(ERROR) << "File write failed";
        file_streams_[shard_id]->close();
        return FAILED;
      }

      auto &io_handle_header = file_streams_[shard_id]->write(reinterpret_cast<char *>(&bin_header[0]), line_len);
      if (!io_handle_header.good() || io_handle_header.fail() || io_handle_header.bad()) {
        MS_LOG(ERROR) << "File write failed";
        file_streams_[shard_id]->close();
        return FAILED;
      }
      file_streams_[shard_id]->close();
    }
  }
  return SUCCESS;
}

MSRStatus ShardWriter::SerializeRawData(std::map<uint64_t, std::vector<json>> &raw_data,
                                        std::vector<std::vector<uint8_t>> &bin_data, uint32_t row_count) {
  // define the number of thread
  uint32_t thread_num = std::thread::hardware_concurrency();
  if (thread_num == 0) thread_num = kThreadNumber;
  // Set the number of samples processed by each thread
  int group_num = ceil(row_count * 1.0 / thread_num);
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
  return flag_ == true ? FAILED : SUCCESS;
}

MSRStatus ShardWriter::SetRawDataSize(const std::vector<std::vector<uint8_t>> &bin_raw_data) {
  raw_data_size_ = std::vector<uint64_t>(row_count_, 0);
  for (uint32_t i = 0; i < row_count_; ++i) {
    raw_data_size_[i] = std::accumulate(
      bin_raw_data.begin() + (i * schema_count_), bin_raw_data.begin() + (i * schema_count_) + schema_count_, 0,
      [](uint64_t accumulator, const std::vector<uint8_t> &row) { return accumulator + kInt64Len + row.size(); });
  }
  if (*std::max_element(raw_data_size_.begin(), raw_data_size_.end()) > page_size_) {
    MS_LOG(ERROR) << "Page size is too small to save a row!";
    return FAILED;
  }
  return SUCCESS;
}

MSRStatus ShardWriter::SetBlobDataSize(const std::vector<std::vector<uint8_t>> &blob_data) {
  blob_data_size_ = std::vector<uint64_t>(row_count_);
  (void)std::transform(blob_data.begin(), blob_data.end(), blob_data_size_.begin(),
                       [](const std::vector<uint8_t> &row) { return kInt64Len + row.size(); });
  if (*std::max_element(blob_data_size_.begin(), blob_data_size_.end()) > page_size_) {
    MS_LOG(ERROR) << "Page size is too small to save a row!";
    return FAILED;
  }
  return SUCCESS;
}

void ShardWriter::SetLastRawPage(const int &shard_id, std::shared_ptr<Page> &last_raw_page) {
  // Get last raw page
  auto last_raw_page_id = shard_header_->GetLastPageIdByType(shard_id, kPageTypeRaw);
  if (last_raw_page_id >= 0) {
    auto page = shard_header_->GetPage(shard_id, last_raw_page_id);
    last_raw_page = page.first;
  }
}

void ShardWriter::SetLastBlobPage(const int &shard_id, std::shared_ptr<Page> &last_blob_page) {
  // Get last blob page
  auto last_blob_page_id = shard_header_->GetLastPageIdByType(shard_id, kPageTypeBlob);
  if (last_blob_page_id >= 0) {
    auto page = shard_header_->GetPage(shard_id, last_blob_page_id);
    last_blob_page = page.first;
  }
}
}  // namespace mindrecord
}  // namespace mindspore
