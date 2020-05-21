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

#include "dataset/engine/datasetops/source/tf_client.h"

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <limits>
#include <algorithm>

#include "common/utils.h"
#include "proto/example.pb.h"
#include "dataset/engine/datasetops/source/storage_client.h"
#include "dataset/util/path.h"
#include "dataset/util/status.h"
#include "dataset/engine/datasetops/source/storage_op.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
// Name: Constructor
// Description: Creates the TFClient.
TFClient::TFClient(std::unique_ptr<DataSchema> schema,  // In: The schema for this storage client.
                   StorageOp *so)                       // In: The StorageOp that's using this client
    : StorageClient(std::move(schema), so),
      rows_per_buffer_(so->rows_per_buffer()),
      random_seed_generator_(so->seed()),
      random_seed_distribution_(0, std::numeric_limits<uint32_t>::max()),
      rows_per_shard_(0) {}

Status TFClient::Init() {
  // Initialize queue to hold the tf file names
  const std::string kExtensionData = ".data";
  const std::string kExtensionTF = ".tfrecord";
  bool schema_init = false;
  if (!storage_op_->dataset_files_dir().empty()) {
    MS_LOG(DEBUG) << "Reading dataset using datasetPath.";
    Path data_set_directory(storage_op_->dataset_files_dir());
    auto dirIt = Path::DirIterator::OpenDirectory(&data_set_directory);
    if (dirIt) {
      while (dirIt->hasNext()) {
        Path file = dirIt->next();
        std::string filename = file.toString();
        if ((file.Extension() == kExtensionData) || (file.Extension() == kExtensionTF)) {
          const std::vector<uint64_t> recs_lengths = ParseTfFileLines(filename);
          v_total_file_rows_.emplace_back(
            std::pair<std::string, std::vector<uint64_t>>(filename, std::move(recs_lengths)));

          // schema
          if (!schema_init) {
            RETURN_IF_NOT_OK(ParseTfFileSchema(filename));
            schema_init = true;
          }
          MS_LOG(INFO) << "found tf file: " << filename << ", num rows " << recs_lengths.size() << ".";
        }
      }
    } else {
      RETURN_STATUS_UNEXPECTED("Unable to open directory " + data_set_directory.toString());
    }
  } else {
    MS_LOG(DEBUG) << "Reading dataset using dataset files list.";
    for (auto filename : storage_op_->dataset_file_list()) {
      const std::vector<uint64_t> recs_lengths = ParseTfFileLines(filename);
      v_total_file_rows_.emplace_back(std::pair<std::string, std::vector<uint64_t>>(filename, std::move(recs_lengths)));

      // schema
      if (!schema_init) {
        RETURN_IF_NOT_OK(ParseTfFileSchema(filename));
        schema_init = true;
      }
      MS_LOG(INFO) << "Processed tf file: " << filename << ", num rows " << recs_lengths.size() << ".";
    }
  }

  RETURN_IF_NOT_OK(CalculateRowsPerDevice());
  std::sort(v_total_file_rows_.begin(), v_total_file_rows_.end());
  RETURN_IF_NOT_OK(ScatterFileRows(static_cast<uint32_t>(storage_op_->device_id()), storage_op_->shard_config(),
                                   storage_op_->seed(), storage_op_->shuffle_config()));

  CalculateNumRows();
  InitStateInfo();
  return Status::OK();
}

// Sharding will reduce the number of rows. Doing this in constructor as we only want to do this once.
void TFClient::CalculateNumRows() {
  num_rows_in_dataset_ = 0;
  for (auto rows : file_start_end_offset_) {
    num_rows_in_dataset_ += (rows.second - rows.first);
  }
}

Status TFClient::CalculateRowsPerDevice() {
  uint64_t num = std::accumulate(
    v_total_file_rows_.begin(), v_total_file_rows_.end(), 0,
    [](uint64_t value, const std::pair<std::string, std::vector<uint64_t>> &a) { return value + a.second.size(); });
  if (static_cast<uint64_t>(std::floor(num * 1.0 / storage_op_->device_num())) == 0) {
    RETURN_STATUS_UNEXPECTED("Num rows of dataset is less than device number");
  }
  rows_per_shard_ = static_cast<uint64_t>(std::ceil(num * 1.0 / storage_op_->device_num()));
  return Status::OK();
}

bool TFClient::ValidFileForShard(const uint64_t file_rows, uint64_t *start_offset, uint64_t *end_offset,
                                 const uint64_t &pre_count, uint32_t device_id) const {
  *start_offset = 0;
  *end_offset = 0;
  bool valid = false;
  uint64_t start_index = device_id * rows_per_shard_;
  uint64_t end_index = (device_id + 1) * rows_per_shard_;

  // First valid file
  if (pre_count <= start_index && pre_count + file_rows > start_index) {
    *start_offset = start_index - pre_count;
    valid = true;
    if (pre_count < end_index && pre_count + file_rows >= end_index) {
      *end_offset = end_index - pre_count;
    } else {
      *end_offset = file_rows;
    }
  }

  // Second and subsequent files
  if (pre_count > start_index && pre_count < end_index) {
    *start_offset = 0;
    valid = true;
    if (pre_count + file_rows >= end_index) {
      *end_offset = end_index - pre_count;
    } else {
      *end_offset = file_rows;
    }
  }

  return valid;
}

void TFClient::GetValidFileForShard(const std::vector<std::pair<std::string, std::vector<uint64_t>>> &v_files,
                                    uint32_t device_id) {
  uint64_t start_offset = 0;
  uint64_t end_offset = 0;
  uint64_t pre_count = 0;
  bool finish = false;
  while (!finish) {
    for (const auto &file : v_files) {
      if (ValidFileForShard(file.second.size(), &start_offset, &end_offset, pre_count, device_id)) {
        std::pair<uint32_t, uint32_t> offset(start_offset, end_offset);
        file_start_end_offset_.emplace_back(offset);
        v_file_rows_.emplace_back(file);
      }
      pre_count += file.second.size();
    }
    if (pre_count < (device_id + 1) * rows_per_shard_) {
      finish = false;
    } else {
      finish = true;
    }
  }
}

// Description: Scatter file rows to local single-P according to config info.
// There are 3 modes: ALL, UNIQUE, RANDOM. For UNIQUE and RANDOM mode, shuffleConfig controls
// whether file row vector would be shuffled or not before a new mEopch.
// For ALL mode, temporarily, we deal with epoch in python part.
Status TFClient::ScatterFileRows(uint32_t device_id, const std::string &shard_config, uint32_t seed,
                                 bool shuffle_config) {
  if (shard_config == "UNIQUE" || shard_config == "RANDOM") {
    std::vector<std::pair<std::string, std::vector<uint64_t>>> v_shuffled_total_file_rows =
      ShuffleVector(v_total_file_rows_, seed);
    GetValidFileForShard(v_shuffled_total_file_rows, device_id);
    if (shuffle_config) {
      v_total_file_rows_ = v_shuffled_total_file_rows;
    }
  } else if (shard_config == "ALL") {
    v_file_rows_.insert(v_file_rows_.end(), v_total_file_rows_.begin(), v_total_file_rows_.end());
    if (shuffle_config) {
      v_total_file_rows_ = ShuffleVector(v_total_file_rows_, seed);
    }

    for (const auto &file : v_file_rows_) {
      std::pair<uint32_t, uint32_t> offset(0, file.second.size());
      file_start_end_offset_.emplace_back(offset);
    }
  } else {
    RETURN_STATUS_UNEXPECTED("In parallel config file, wrong shuffleConfig or shardConfig provided.");
  }

  return Status::OK();
}

std::vector<std::pair<std::string, std::vector<uint64_t>>> TFClient::ShuffleVector(
  std::vector<std::pair<std::string, std::vector<uint64_t>>> v, uint32_t seed = 1) {
  std::default_random_engine randomEngine(seed);
  std::shuffle(std::begin(v), std::end(v), randomEngine);
  return v;
}

void TFClient::CalculateStartOffset(const uint64_t start_index, const uint64_t end_index,
                                    const std::vector<uint64_t> &vec_length, uint64_t *start_offset) const {
  for (size_t i = start_index; i < end_index; i++) {
    // Format of a single record:
    //  uint64    length
    //  uint32    masked crc of length
    //  byte      data[length]
    //  uint32    masked crc of data
    *start_offset += sizeof(uint64_t) + 2 * sizeof(uint32_t) + vec_length[i];
  }
}

void TFClient::InitStateInfo() {
  uint32_t start_idx = 0, record_num = 0, buffer_id = 0;
  uint64_t start_offset = 0;
  bool first_buffer = true;
  f_info_queue_.emplace_back(QFile());
  std::vector<std::pair<std::string, std::vector<uint64_t>>>::iterator itr = v_file_rows_.begin();
  uint32_t index = 0;
  while (itr != v_file_rows_.end()) {
    uint32_t file_start_index = file_start_end_offset_[index].first;
    uint32_t file_end_index = file_start_end_offset_[index].second;
    FileInfo f_info;
    f_info.fileName = itr->first;
    f_info.startRecordIdx = start_idx > file_start_index ? start_idx : file_start_index;
    if (first_buffer && f_info.startRecordIdx != 0) {
      CalculateStartOffset(0, f_info.startRecordIdx, itr->second, &start_offset);
      start_idx = static_cast<uint32_t>(f_info.startRecordIdx);
    }
    first_buffer = false;
    f_info.startOffset = start_offset;
    if (start_idx + rows_per_buffer_ - record_num < itr->second.size()) {
      uint64_t end_idx = start_idx + rows_per_buffer_ - record_num - 1;
      f_info.endRecordIdx = end_idx > (file_end_index - 1) ? (file_end_index - 1) : end_idx;
      f_info_queue_[buffer_id].push(f_info);
      CalculateStartOffset(start_idx, f_info.endRecordIdx + 1, itr->second, &start_offset);
      start_idx = start_idx + rows_per_buffer_ - record_num;
      record_num = 0;
      buffer_id++;
      f_info_queue_.emplace_back(QFile());
      if (end_idx >= file_end_index - 1) {
        start_idx = start_offset = 0;
        ++itr;
        ++index;
      }
    } else {
      f_info.endRecordIdx = itr->second.size() - 1 > file_end_index - 1 ? file_end_index - 1 : itr->second.size() - 1;
      f_info_queue_[buffer_id].push(f_info);
      if (start_idx + rows_per_buffer_ - record_num == itr->second.size()) {
        record_num = start_idx = start_offset = 0;
        buffer_id++;
        if (itr + 1 != v_file_rows_.end()) {
          f_info_queue_.emplace_back(QFile());
        }
      } else {
        record_num += static_cast<uint32_t>(itr->second.size()) - start_idx;
        start_idx = start_offset = 0;
      }
      ++itr;
      ++index;
    }
  }
}

// Name: Print()
// Description: A function that prints info about the TFClient
void TFClient::Print(std::ostream &out) const {  //  In: The output stream to print to
  out << "TF client.";
}

std::vector<uint64_t> TFClient::ParseTfFileLines(const std::string &filename) {
  std::vector<uint64_t> recs_lengths;
  std::ifstream reader;
  reader.open(filename);
  while (true) {
    if (reader.peek() == EOF) {
      reader.close();
      break;
    }

    // read length
    uint64_t record_length = 0;
    (void)reader.read(reinterpret_cast<char *>(&record_length), static_cast<std::streamsize>(sizeof(uint64_t)));
    recs_lengths.push_back(record_length);

    // ignore crc header
    (void)reader.ignore(static_cast<std::streamsize>(sizeof(uint32_t)));

    // ignore data length
    (void)reader.ignore(static_cast<std::streamsize>(record_length));

    // ignore crc footer
    (void)reader.ignore(static_cast<std::streamsize>(sizeof(uint32_t)));
  }
  return recs_lengths;
}

Status TFClient::ParseTfFileSchema(const std::string &filename) {
  std::ifstream reader;
  reader.open(filename);
  std::string serialized_example;
  // read length
  uint64_t record_length = 0;
  (void)reader.read(reinterpret_cast<char *>(&record_length), static_cast<std::streamsize>(sizeof(uint64_t)));

  // ignore crc header
  (void)reader.ignore(static_cast<std::streamsize>(sizeof(uint32_t)));

  // read serialized Example
  serialized_example.resize(record_length);
  (void)reader.read(&serialized_example[0], static_cast<std::streamsize>(record_length));

  // ignore crc footer
  (void)reader.ignore(static_cast<std::streamsize>(sizeof(uint32_t)));

  reader.close();
  dataengine::Example tf_file;
  if (!tf_file.ParseFromString(serialized_example)) {
    std::string err_msg = "parse tf_file failed, file name is " + filename;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  const dataengine::Features &example_features = tf_file.features();
  const google::protobuf::Map<std::string, dataengine::Feature> &feature_map = example_features.feature();
  for (auto it = feature_map.begin(); it != feature_map.end(); ++it) {
    col_names_.push_back(it->first);
  }
  return Status::OK();
}

// Name: Reset()
// Description: Resets any state info inside the client back to it's initialized
//              state.
Status TFClient::Reset() {
  v_file_rows_.clear();
  file_start_end_offset_.clear();

  uint32_t next_seed = random_seed_distribution_(random_seed_generator_);
  RETURN_IF_NOT_OK(ScatterFileRows(static_cast<uint32_t>(storage_op_->device_id()), storage_op_->shard_config(),
                                   next_seed, storage_op_->shuffle_config()));

  CalculateNumRows();
  uint32_t num_rows_in_file = 0;
  RETURN_IF_NOT_OK(this->numRowsFromFile(num_rows_in_file));
  if (num_rows_in_file < num_rows_in_dataset_) {
    num_rows_in_dataset_ = num_rows_in_file;
  }

  storage_op_->set_num_rows(static_cast<int32_t>(num_rows_in_dataset_));
  InitStateInfo();

  return Status::OK();
}

Status TFClient::NextFileInfo(uint32_t id, FileInfo *ptr) {
  if (f_info_queue_.empty() || id >= f_info_queue_.size() || f_info_queue_[id].empty()) {
    RETURN_STATUS_UNEXPECTED("cannot find next FileInfo in mFInfoQueue");
  }
  *ptr = f_info_queue_[id].front();
  f_info_queue_[id].pop();
  return Status::OK();
}

bool TFClient::IsMoreData(uint32_t id) { return (!f_info_queue_[id].empty()); }
}  // namespace dataset
}  // namespace mindspore
