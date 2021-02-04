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
  builder_rows_per_buffer_ = config_manager->rows_per_buffer();
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

  std::shared_ptr<TextFileOp> text_file_op = std::make_shared<TextFileOp>(
    builder_num_workers_, builder_rows_per_buffer_, builder_total_rows_, builder_worker_connector_size_,
    std::move(builder_schema_), builder_text_files_list_, builder_op_connector_size_, builder_shuffle_files_,
    builder_num_devices_, builder_device_id_);
  RETURN_IF_NOT_OK(text_file_op->Init());
  *op = std::move(text_file_op);

  return Status::OK();
}

TextFileOp::TextFileOp(int32_t num_workers, int64_t rows_per_buffer, int64_t total_rows, int32_t worker_connector_size,
                       std::unique_ptr<DataSchema> schema, std::vector<std::string> text_files_list,
                       int32_t op_connector_size, bool shuffle_files, int32_t num_device, int32_t device_id)
    : ParallelOp(num_workers, op_connector_size),
      device_id_(device_id),
      num_devices_(num_device),
      rows_per_buffer_(rows_per_buffer),
      total_rows_(total_rows),
      text_files_list_(std::move(text_files_list)),
      shuffle_files_(shuffle_files),
      data_schema_(std::move(schema)),
      all_num_rows_(0),
      num_rows_per_shard_(0),
      filename_index_(std::make_unique<StringIndex>()),
      finished_reading_dataset_(false),
      load_io_block_queue_(true),
      load_jagged_connector_(true) {
  worker_connector_size_ = worker_connector_size;
}

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
    out << "\nRows per buffer: " << rows_per_buffer_ << "\nRow count: " << total_rows_ << "\nDevice id: " << device_id_
        << "\nNumber of devices: " << num_devices_ << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no")
        << "\nText files list:\n";
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

Status TextFileOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  load_jagged_connector_ = true;
  load_io_block_queue_ = true;

  RETURN_IF_NOT_OK(ParallelOp::Reset());
  NotifyToFillIOBlockQueue();
  return Status::OK();
}

Status TextFileOp::LoadTensor(const std::string &line, std::unique_ptr<TensorQTable> *tensor_table, int64_t row) {
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(line, &tensor));
  (**tensor_table)[row][0] = std::move(tensor);
  return Status::OK();
}

Status TextFileOp::LoadFile(const std::string &file, const int64_t start_offset, const int64_t end_offset,
                            const int32_t worker_id) {
  std::ifstream handle(file);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + file);
  }

  int64_t rows_each_buffer = 0;
  int64_t rows_total = 0;
  std::string line;
  std::unique_ptr<DataBuffer> cur_buffer = std::make_unique<DataBuffer>(0, DataBuffer::BufferFlags::kDeBFlagNone);
  std::unique_ptr<TensorQTable> tensor_table = std::make_unique<TensorQTable>();

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
    tensor_table->push_back(std::move(tRow));
    RETURN_IF_NOT_OK(LoadTensor(line, &tensor_table, rows_each_buffer));
    rows_each_buffer++;
    rows_total++;
    if (rows_each_buffer == rows_per_buffer_) {
      cur_buffer->set_tensor_table(std::move(tensor_table));
      RETURN_IF_NOT_OK(jagged_buffer_connector_->Add(worker_id, std::move(cur_buffer)));

      cur_buffer = std::make_unique<DataBuffer>(0, DataBuffer::BufferFlags::kDeBFlagNone);
      tensor_table = std::make_unique<TensorQTable>();
      rows_each_buffer = 0;
    }
  }

  if (rows_each_buffer > 0) {
    cur_buffer->set_tensor_table(std::move(tensor_table));
    RETURN_IF_NOT_OK(jagged_buffer_connector_->Add(worker_id, std::move(cur_buffer)));
  }

  return Status::OK();
}

Status TextFileOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();

  std::unique_ptr<FilenameBlock> io_block;
  RETURN_IF_NOT_OK(PopIoBlockQueue(worker_id, &io_block));
  while (!io_block->eof()) {
    if (!io_block->eoe()) {
      if (load_jagged_connector_) {
        std::string filename;
        RETURN_IF_NOT_OK(io_block->GetFilename(&filename, *filename_index_));
        int64_t start_offset = io_block->GetStartOffset();
        int64_t end_offset = io_block->GetEndOffset();
        RETURN_IF_NOT_OK(LoadFile(filename, start_offset, end_offset, worker_id));
      }
    } else {
      std::unique_ptr<DataBuffer> eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
      RETURN_IF_NOT_OK(jagged_buffer_connector_->Add(worker_id, std::move(eoe_buffer)));
    }

    RETURN_IF_NOT_OK(PopIoBlockQueue(worker_id, &io_block));
  }
  return Status::OK();
}

// Pops an element from a queue in io_block_queues
Status TextFileOp::PopIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> *out_block) {
  RETURN_IF_NOT_OK(io_block_queues_[index]->PopFront(out_block));

  return Status::OK();
}

// Pushes an element to a queue in io_block_queues
Status TextFileOp::PushIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> &&io_block) {
  RETURN_IF_NOT_OK(io_block_queues_[index]->Add(std::move(io_block)));

  return Status::OK();
}

// Pushes a control indicator onto the IOBlockQueue for each worker to consume.
// When the worker pops this control indicator, it will shut itself down gracefully.
Status TextFileOp::PostEndOfData() {
  for (int i = 0; i < num_workers_; ++i) {
    std::unique_ptr<FilenameBlock> eof = std::make_unique<FilenameBlock>(IOBlock::kDeIoBlockFlagEof);
    RETURN_IF_NOT_OK(PushIoBlockQueue(i, std::move(eof)));
  }

  return Status::OK();
}

// Pushes a control indicator onto the IOBlockQueue for each worker to consume. When the worker
// pops this control indicator, it will wait until the next epoch starts and then resume execution.
Status TextFileOp::PostEndOfEpoch(int32_t queue_index) {
  for (int i = 0; i < num_workers_; ++i) {
    std::unique_ptr<FilenameBlock> eoe = std::make_unique<FilenameBlock>(IOBlock::kDeIoBlockFlagEoe);
    RETURN_IF_NOT_OK(PushIoBlockQueue((queue_index + i) % num_workers_, std::move(eoe)));
  }

  return Status::OK();
}

static void ShuffleKeys(std::vector<int64_t> *i_keys, uint32_t seed) {
  std::mt19937 rng(seed);
  std::shuffle(i_keys->begin(), i_keys->end(), rng);
}

bool TextFileOp::NeedPushFileToBlockQueue(const std::string &file_name, int64_t *start_offset, int64_t *end_offset,
                                          const int64_t &pre_count) {
  *start_offset = 0;
  *end_offset = 0;
  bool push = false;
  int64_t start_index = device_id_ * num_rows_per_shard_;
  if (device_id_ + 1 < 0) {
    MS_LOG(ERROR) << "Device id is invalid";
    return false;
  }

  int64_t end_index = (static_cast<int64_t>(device_id_) + 1) * num_rows_per_shard_;
  if (pre_count <= start_index && pre_count + filename_numrows_[file_name] > start_index) {
    *start_offset = start_index - pre_count;
    push = true;
    if (pre_count < end_index && pre_count + filename_numrows_[file_name] >= end_index) {
      *end_offset = end_index - pre_count;
    } else {
      *end_offset = filename_numrows_[file_name];
    }
  }

  if (pre_count >= start_index && pre_count < end_index) {
    *start_offset = 0;
    push = true;
    if (pre_count + filename_numrows_[file_name] >= end_index) {
      *end_offset = end_index - pre_count;
    } else {
      *end_offset = filename_numrows_[file_name];
    }
  }

  return push;
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

Status TextFileOp::WaitToFillIOBlockQueue() {
  // must be called first if called by worker spanwed by taskgroup
  TaskManager::FindMe()->Post();

  std::vector<int64_t> i_keys;
  if (shuffle_files_) {
    for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
      i_keys.push_back(it.key());
    }
  }
  uint32_t seed = 0;
  while (true) {
    RETURN_IF_NOT_OK(io_block_queue_wait_post_.Wait());
    io_block_queue_wait_post_.Clear();

    if (finished_reading_dataset_) {
      break;
    }

    if (shuffle_files_) {
      ShuffleKeys(&i_keys, num_devices_ == 1 ? GetSeed() : ++seed);
    }
    RETURN_IF_NOT_OK(FillIOBlockQueue(i_keys));
  }
  return Status::OK();
}

void TextFileOp::NotifyToFillIOBlockQueue() { io_block_queue_wait_post_.Set(); }

Status TextFileOp::operator()() {
  RETURN_IF_NOT_OK(CalculateNumRowsPerShard());

  // Move register to the front of launching thread, this will fix the problem
  // when thread exit unnormally register will failed occasionally.
  RETURN_IF_NOT_OK(io_block_queue_wait_post_.Register(tree_->AllTasks()));

  // launch one thread, responsible for filling IoBlockQueue
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(1, std::bind(&TextFileOp::WaitToFillIOBlockQueue, this), Name(), id()));

  // Read data from disk into buffers
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&TextFileOp::WorkerEntry, this, std::placeholders::_1), Name(), id()));

  // must be called after launching workers.
  TaskManager::FindMe()->Post();
  NotifyToFillIOBlockQueue();
  while (!finished_reading_dataset_) {
    int64_t buffer_id = 0;
    int32_t workers_done = 0;
    int64_t rows_read = 0;
    load_io_block_queue_ = true;

    while (workers_done < num_workers_) {
      std::unique_ptr<DataBuffer> buffer;
      RETURN_IF_NOT_OK(jagged_buffer_connector_->Pop(0, &buffer));
      if (buffer->eoe()) {
        workers_done++;
      } else if (total_rows_ == 0 || rows_read < total_rows_) {
        if ((total_rows_ > 0) && (rows_read + buffer->NumRows() > total_rows_)) {
          int64_t rowsToRemove = buffer->NumRows() - (total_rows_ - rows_read);
          RETURN_IF_NOT_OK(buffer->SliceOff(rowsToRemove));
        }
        rows_read += buffer->NumRows();
        buffer->set_id(buffer_id++);
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(buffer)));
      } else {
        // end of epoch
        load_jagged_connector_ = false;
        load_io_block_queue_ = false;
      }
    }

    std::unique_ptr<DataBuffer> eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
    RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eoe_buffer)));

    if (IsLastIteration()) {
      finished_reading_dataset_ = true;
      NotifyToFillIOBlockQueue();
    } else {
      jagged_buffer_connector_->DoReset();
      buffer_id = 0;
      // Self-reset to start a new iteration
      RETURN_IF_NOT_OK(Reset());
    }
    UpdateRepeatAndEpochCounter();
  }

  std::unique_ptr<DataBuffer> eof_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF);
  RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eof_buffer)));

  RETURN_IF_NOT_OK(PostEndOfData());

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
    all_num_rows_ += count;
  }
  if (all_num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API TextFileDataset. Please check file path or dataset API.");
  }

  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(all_num_rows_ * 1.0 / num_devices_));
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
