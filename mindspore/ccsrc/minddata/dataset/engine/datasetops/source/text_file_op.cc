/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/wait_post.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
TextFileOp::TextFileOp(int32_t num_workers, int64_t total_rows, int32_t worker_connector_size,
                       std::unique_ptr<DataSchema> schema, std::vector<std::string> text_files_list,
                       int32_t op_connector_size, bool shuffle_files, int32_t num_devices, int32_t device_id)
    : NonMappableLeafOp(num_workers, worker_connector_size, total_rows, op_connector_size, shuffle_files, num_devices,
                        device_id),
      text_files_list_(std::move(text_files_list)),
      data_schema_(std::move(schema)) {}

// A print method typically used for debugging
void TextFileOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nRow count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\n"
        << DatasetName(true) << " list:\n";
    for (size_t i = 0; i < text_files_list_.size(); ++i) {
      out << " " << text_files_list_[i];
    }
    out << "\nData Schema:\n";
    out << *data_schema_ << "\n\n";
  }
}

Status TextFileOp::Init() {
  RETURN_IF_NOT_OK(filename_index_->insert(text_files_list_));

  int32_t safe_queue_size = static_cast<int32_t>(std::ceil(text_files_list_.size() / num_workers_) + 1);
  io_block_queues_.Init(num_workers_, safe_queue_size);

  jagged_rows_connector_ = std::make_unique<JaggedConnector>(num_workers_, 1, worker_connector_size_);
  return Status::OK();
}

Status TextFileOp::LoadTensor(const std::string &line, TensorRow *out_row) const {
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(line, &tensor));
  (*out_row)[0] = std::move(tensor);
  return Status::OK();
}

Status TextFileOp::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << file << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + file + " does not exist.");
  }

  std::ifstream handle(realpath.value());
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open text:" + file +
                             ", the file is damaged or permission denied.");
  }

  int64_t rows_total = 0;
  std::string line;

  while (getline(handle, line)) {
    if (line.empty()) {
      continue;
    }
    // If read to the end offset of this file, break.
    if (rows_total >= end_offset) {
      break;
    }
    // Skip line before start offset.
    if (rows_total < start_offset) {
      rows_total++;
      continue;
    }

    TensorRow tRow(1, nullptr);
    tRow.setPath({file});
    RETURN_IF_NOT_OK(LoadTensor(line, &tRow));
    RETURN_IF_NOT_OK(jagged_rows_connector_->Add(worker_id, std::move(tRow)));

    rows_total++;
  }

  return Status::OK();
}

Status TextFileOp::FillIOBlockQueue(const std::vector<int64_t> &i_keys) {
  int32_t queue_index = 0;
  int64_t pre_count = 0;
  int64_t start_offset = 0;
  int64_t end_offset = 0;
  bool finish = false;
  while (!finish) {
    std::vector<std::pair<std::string, int64_t>> file_index;
    if (!i_keys.empty()) {
      for (auto it = i_keys.begin(); it != i_keys.end(); ++it) {
        {
          if (!GetLoadIoBlockQueue()) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>((*filename_index_)[*it], *it));
      }
    } else {
      for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
        {
          if (!GetLoadIoBlockQueue()) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>(it.value(), it.key()));
      }
    }
    for (auto file_info : file_index) {
      if (NeedPushFileToBlockQueue(file_info.first, &start_offset, &end_offset, pre_count)) {
        auto ioBlock =
          std::make_unique<FilenameBlock>(file_info.second, start_offset, end_offset, IOBlock::kDeIoBlockNone);
        RETURN_IF_NOT_OK(PushIoBlockQueue(queue_index, std::move(ioBlock)));
        queue_index = (queue_index + 1) % num_workers_;
      }

      pre_count += filename_numrows_[file_info.first];
    }

    if (pre_count < (static_cast<int64_t>(device_id_) + 1) * num_rows_per_shard_) {
      finish = false;
    } else {
      finish = true;
    }
  }

  RETURN_IF_NOT_OK(PostEndOfEpoch(queue_index));
  return Status::OK();
}

int64_t TextFileOp::CountTotalRows(const std::string &file) {
  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file, " << file << " does not exist.";
    return 0;
  }

  std::ifstream handle(realpath.value());
  if (!handle.is_open()) {
    MS_LOG(ERROR) << "Invalid file, failed to open text file:" << file << ", the file is damaged or permission denied.";
    return 0;
  }

  std::string line;
  int64_t count = 0;
  while (getline(handle, line)) {
    if (!line.empty()) {
      count++;
    }
  }

  return count;
}

Status TextFileOp::CalculateNumRowsPerShard() {
  for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
    int64_t count = CountTotalRows(it.value());
    filename_numrows_[it.value()] = count;
    num_rows_ += count;
  }
  if (num_rows_ == 0) {
    std::stringstream ss;
    for (int i = 0; i < text_files_list_.size(); ++i) {
      ss << " " << text_files_list_[i];
    }
    std::string file_list = ss.str();
    RETURN_STATUS_UNEXPECTED("Invalid data, " + DatasetName(true) +
                             "Dataset API can't read the data file (interface mismatch or no data found). Check " +
                             DatasetName() + ": " + file_list);
  }

  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(num_rows_ * 1.0 / num_devices_));
  MS_LOG(DEBUG) << "Number rows per shard is " << num_rows_per_shard_;
  return Status::OK();
}

Status TextFileOp::CountAllFileRows(const std::vector<std::string> &files, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  int32_t num_workers = GlobalContext::config_manager()->num_parallel_workers();
  int32_t connector_que_size = GlobalContext::config_manager()->op_connector_size();
  int32_t worker_connector_size = GlobalContext::config_manager()->worker_connector_size();
  const int32_t shard_id = 0;
  const int32_t num_shards = 1;
  const int64_t num_samples = 0;
  bool shuffle_files = false;
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();

  // Create and initialize
  std::shared_ptr<TextFileOp> op =
    std::make_shared<TextFileOp>(num_workers, num_samples, worker_connector_size, std::move(schema), files,
                                 connector_que_size, shuffle_files, num_shards, shard_id);
  RETURN_IF_NOT_OK(op->Init());
  *count = 0;
  for (auto file : files) {
    *count += op->CountTotalRows(file);
  }
  return Status::OK();
}

Status TextFileOp::ComputeColMap() {
  // Set the column name mapping (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
