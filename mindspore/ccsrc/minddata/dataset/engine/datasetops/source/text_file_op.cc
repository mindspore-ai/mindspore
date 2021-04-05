/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <iomanip>
#include <memory>
#include <string>
#include <utility>

#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/wait_post.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
TextFileOp::Builder::Builder()
    : builder_device_id_(0), builder_num_devices_(1), builder_total_rows_(0), builder_shuffle_files_(false) {
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  builder_num_workers_ = config_manager->num_parallel_workers();
  builder_op_connector_size_ = config_manager->op_connector_size();
  builder_worker_connector_size_ = config_manager->worker_connector_size();
}

Status TextFileOp::Builder::ValidateInputs() const {
  std::string err_msg;
  err_msg += builder_num_workers_ <= 0 ? "Invalid parameter, num_parallel_workers must be greater than 0, but got " +
                                           std::to_string(builder_num_workers_) + ".\n"
                                       : "";
  err_msg += (builder_device_id_ >= builder_num_devices_ || builder_num_devices_ < 1)
               ? "Invalid parameter, num_shard must be greater than shard_id and greater than 0, got num_shard: " +
                   std::to_string(builder_num_devices_) + ", shard_id: " + std::to_string(builder_device_id_) + ".\n"
               : "";
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, err_msg);
}

Status TextFileOp::Builder::Build(std::shared_ptr<TextFileOp> *op) {
  RETURN_IF_NOT_OK(ValidateInputs());

  // Throttle the number of workers if we have more workers than files!
  if (static_cast<size_t>(builder_num_workers_) > builder_text_files_list_.size()) {
    builder_num_workers_ = builder_text_files_list_.size();
    MS_LOG(DEBUG) << "TextFileOp operator parallelism reduced to " << builder_num_workers_ << " workers.";
  }

  builder_schema_ = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(
    builder_schema_->AddColumn(ColDescriptor("text", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));

  std::shared_ptr<TextFileOp> text_file_op =
    std::make_shared<TextFileOp>(builder_num_workers_, builder_total_rows_, builder_worker_connector_size_,
                                 std::move(builder_schema_), builder_text_files_list_, builder_op_connector_size_,
                                 builder_shuffle_files_, builder_num_devices_, builder_device_id_);
  RETURN_IF_NOT_OK(text_file_op->Init());
  *op = std::move(text_file_op);

  return Status::OK();
}

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
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nText files list:\n";
    for (int i = 0; i < text_files_list_.size(); ++i) {
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

  RETURN_IF_NOT_OK(ParallelOp::CreateWorkerConnector(worker_connector_size_));

  jagged_buffer_connector_ = std::make_unique<JaggedConnector>(num_workers_, 1, worker_connector_size_);
  return Status::OK();
}

Status TextFileOp::LoadTensor(const std::string &line, TensorRow *out_row) {
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(line, &tensor));
  (*out_row)[0] = std::move(tensor);
  return Status::OK();
}

Status TextFileOp::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  std::ifstream handle(file);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + file);
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
    RETURN_IF_NOT_OK(jagged_buffer_connector_->Add(worker_id, std::move(tRow)));

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
          if (!load_io_block_queue_) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>((*filename_index_)[*it], *it));
      }
    } else {
      for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
        {
          if (!load_io_block_queue_) {
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
  std::ifstream handle(file);
  if (!handle.is_open()) {
    MS_LOG(ERROR) << "Invalid file, failed to open file: " << file;
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
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API TextFileDataset. Please check file path or dataset API.");
  }

  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(num_rows_ * 1.0 / num_devices_));
  MS_LOG(DEBUG) << "Number rows per shard is " << num_rows_per_shard_;
  return Status::OK();
}

Status TextFileOp::CountAllFileRows(const std::vector<std::string> &files, int64_t *count) {
  std::shared_ptr<TextFileOp> op;
  *count = 0;
  RETURN_IF_NOT_OK(Builder().SetTextFilesList(files).Build(&op));
  for (auto file : files) {
    *count += op->CountTotalRows(file);
  }
  return Status::OK();
}

Status TextFileOp::ComputeColMap() {
  // Set the column name mapping (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->column(i).name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
