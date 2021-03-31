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
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"

#include <algorithm>
#include <fstream>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "proto/example.pb.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/wait_post.h"
#include "utils/system/crc32c.h"

namespace mindspore {
namespace dataset {
TFReaderOp::Builder::Builder()
    : builder_device_id_(0), builder_num_devices_(1), builder_total_rows_(0), builder_equal_rows_per_shard_(false) {
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  builder_num_workers_ = config_manager->num_parallel_workers();
  builder_worker_connector_size_ = config_manager->worker_connector_size();
  builder_op_connector_size_ = config_manager->op_connector_size();
  builder_rows_per_buffer_ = config_manager->rows_per_buffer();
  builder_shuffle_files_ = false;
  builder_data_schema_ = std::make_unique<DataSchema>();
}

bool TFReaderOp::ValidateFirstRowCrc(const std::string &filename) {
  std::ifstream reader;
  reader.open(filename);
  if (!reader) {
    return false;
  }

  // read data
  int64_t record_length = 0;
  (void)reader.read(reinterpret_cast<char *>(&record_length), static_cast<std::streamsize>(sizeof(int64_t)));

  // read crc from file
  uint32_t masked_crc = 0;
  (void)reader.read(reinterpret_cast<char *>(&masked_crc), static_cast<std::streamsize>(sizeof(uint32_t)));

  // generate crc from data
  uint32_t generated_crc =
    system::Crc32c::GetMaskCrc32cValue(reinterpret_cast<char *>(&record_length), sizeof(int64_t));

  return masked_crc == generated_crc;
}

Status TFReaderOp::Builder::ValidateInputs() const {
  std::string err_msg;

  if (builder_num_workers_ <= 0) {
    err_msg += "Invalid parameter, num_parallel_workers must be greater than 0, but got " +
               std::to_string(builder_num_workers_) + ".\n";
  }

  if (builder_device_id_ >= builder_num_devices_ || builder_num_devices_ < 1) {
    err_msg += "Invalid parameter, num_shard must be greater than shard_id and greater than 0, got num_shard: " +
               std::to_string(builder_num_devices_) + ", shard_id: " + std::to_string(builder_device_id_) + ".\n";
  }

  std::vector<std::string> invalid_files(builder_dataset_files_list_.size());
  auto it = std::copy_if(builder_dataset_files_list_.begin(), builder_dataset_files_list_.end(), invalid_files.begin(),
                         [](const std::string &filename) { return !ValidateFirstRowCrc(filename); });
  invalid_files.resize(std::distance(invalid_files.begin(), it));

  if (!invalid_files.empty()) {
    err_msg += "Invalid file, the following files either cannot be opened, or are not valid tfrecord files:\n";

    std::string accumulated_filenames = std::accumulate(
      invalid_files.begin(), invalid_files.end(), std::string(""),
      [](const std::string &accumulated, const std::string &next) { return accumulated + "    " + next + "\n"; });
    err_msg += accumulated_filenames;
  }

  return err_msg.empty() ? Status::OK() : Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, err_msg);
}

Status TFReaderOp::Builder::Build(std::shared_ptr<TFReaderOp> *out_tf_reader_op) {
  RETURN_IF_NOT_OK(ValidateInputs());

  // Throttle the number of workers if we have more workers than files!
  if (static_cast<size_t>(builder_num_workers_) > builder_dataset_files_list_.size()) {
    builder_num_workers_ = builder_dataset_files_list_.size();
    MS_LOG(WARNING) << "TFReader operator parallelism reduced to " << builder_num_workers_ << " workers.";
  }

  std::shared_ptr<TFReaderOp> new_tf_reader_op = std::make_shared<TFReaderOp>(
    builder_num_workers_, builder_worker_connector_size_, builder_rows_per_buffer_, builder_total_rows_,
    builder_dataset_files_list_, std::move(builder_data_schema_), builder_op_connector_size_, builder_columns_to_load_,
    builder_shuffle_files_, builder_num_devices_, builder_device_id_, builder_equal_rows_per_shard_);

  RETURN_IF_NOT_OK(new_tf_reader_op->Init());
  *out_tf_reader_op = std::move(new_tf_reader_op);
  return Status::OK();
}

TFReaderOp::TFReaderOp(int32_t num_workers, int32_t worker_connector_size, int64_t rows_per_buffer,
                       int64_t total_num_rows, std::vector<std::string> dataset_files_list,
                       std::unique_ptr<DataSchema> data_schema, int32_t op_connector_size,
                       std::vector<std::string> columns_to_load, bool shuffle_files, int32_t num_device,
                       int32_t device_id, bool equal_rows_per_shard)
    : ParallelOp(num_workers, op_connector_size),
      device_id_(device_id),
      num_devices_(num_device),
      rows_per_buffer_(rows_per_buffer),
      total_rows_(total_num_rows),
      dataset_files_list_(std::move(dataset_files_list)),
      columns_to_load_(std::move(columns_to_load)),
      finished_reading_dataset_(false),
      shuffle_files_(shuffle_files),
      data_schema_(std::move(data_schema)),
      filename_index_(std::make_unique<StringIndex>()),
      load_io_block_queue_(true),
      load_jagged_connector_(true),
      num_rows_(0),
      num_rows_per_shard_(0),
      equal_rows_per_shard_(equal_rows_per_shard) {
  worker_connector_size_ = worker_connector_size;
}

// A print method typically used for debugging
void TFReaderOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nRows per buffer: " << rows_per_buffer_ << "\nTotal rows: " << total_rows_ << "\nDevice id: " << device_id_
        << "\nNumber of devices: " << num_devices_ << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no")
        << "\nDataset files list: Size: " << dataset_files_list_.size() << "\n";
    for (int i = 0; i < dataset_files_list_.size(); ++i) {
      out << " " << dataset_files_list_[i];
    }
    if (!columns_to_load_.empty()) {
      out << "\nColumns to load:\n";
      for (int i = 0; i < columns_to_load_.size(); ++i) {
        out << " " << columns_to_load_[i];
      }
    }
    out << "\nData Schema:\n";
    out << *data_schema_ << "\n\n";
  }
}

Status TFReaderOp::Init() {
  if (data_schema_->Empty()) {
    RETURN_IF_NOT_OK(CreateSchema(dataset_files_list_[0], columns_to_load_));
  }

  if (total_rows_ == 0) {
    total_rows_ = data_schema_->num_rows();
  }
  if (total_rows_ < 0) {
    RETURN_STATUS_UNEXPECTED("Invalid parameter, num_sample or num_row for TFRecordDataset must be greater than 0.");
  }

  // Build the index with our files such that each file corresponds to a key id.
  RETURN_IF_NOT_OK(filename_index_->insert(dataset_files_list_));

  // The creation of the internal connector has been delayed until now, since we may have adjusted the
  // number of workers.  Now that the worker count is established, create the connector now in the
  // parallel op base.
  RETURN_IF_NOT_OK(ParallelOp::CreateWorkerConnector(worker_connector_size_));

  jagged_buffer_connector_ = std::make_unique<JaggedConnector>(num_workers_, 1, worker_connector_size_);

  // temporary: make size large enough to hold all files + EOE to avoid hangs
  int32_t safe_queue_size = static_cast<int32_t>(std::ceil(dataset_files_list_.size() / num_workers_)) + 1;
  io_block_queues_.Init(num_workers_, safe_queue_size);

  return Status::OK();
}

Status TFReaderOp::CalculateNumRowsPerShard() {
  if (!equal_rows_per_shard_) {
    return Status::OK();
  }

  for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
    std::vector<std::string> file(1, it.value());
    int64_t num = CountTotalRowsSectioned(file, 0, 1);
    filename_numrows_[it.value()] = num;
    num_rows_ += num;
  }
  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(num_rows_ * 1.0 / num_devices_));
  if (num_rows_per_shard_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API TFRecordDataset. Please check file path or dataset API.");
  }
  return Status::OK();
}
// Class functor operator () override.
// All dataset operators operate by launching a thread (see ExecutionTree). This class functor will
// provide the master loop that drives the logic for performing the work
Status TFReaderOp::operator()() {
  RETURN_IF_NOT_OK(CalculateNumRowsPerShard());

  // Put here to avoid register failed when Worker_Entry thread exits unexpected
  RETURN_IF_NOT_OK(io_block_queue_wait_post_.Register(tree_->AllTasks()));

  // launch one thread, responsible for filling mIOBlockQueue
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(1, std::bind(&TFReaderOp::WaitToFillIOBlockQueue, this), "", id()));

  // launch num_workers_ worker threads, responsible for pulling from the IOBlockQueue and reading
  // data from disk into buffers
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&TFReaderOp::WorkerEntry, this, std::placeholders::_1), "", id()));

  // must be called after launching workers. workers can't be spawned after this post,
  // so workers have to be kept alive until the end of the program
  TaskManager::FindMe()->Post();

  NotifyToFillIOBlockQueue();
  while (!finished_reading_dataset_) {
    int64_t buffer_id = 0;
    int32_t workers_done = 0;
    int64_t rows_read = 0;
    {
      std::unique_lock<std::mutex> lock(load_io_block_queue_mutex_);
      load_io_block_queue_ = true;
    }

    while (workers_done < num_workers_) {
      std::unique_ptr<DataBuffer> fetched_buffer;
      RETURN_IF_NOT_OK(jagged_buffer_connector_->Pop(0, &fetched_buffer));
      if (fetched_buffer->eoe()) {
        workers_done++;
      } else if (total_rows_ == 0 || rows_read < total_rows_) {
        // we need to push a buffer
        if (total_rows_ > 0 && rows_read + fetched_buffer->NumRows() > total_rows_) {
          // this is last buffer we need, and we only need a part of it
          int64_t rowsToRemove = fetched_buffer->NumRows() - (total_rows_ - rows_read);
          RETURN_IF_NOT_OK(fetched_buffer->SliceOff(rowsToRemove));
        }

        rows_read += fetched_buffer->NumRows();
        fetched_buffer->set_id(buffer_id);
        buffer_id++;
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(fetched_buffer)));
      } else {
        // user specified number of rows they want, and we read enough rows
        //
        // IOBlockQueue thread needs to:
        // -stop pushing stuff to IOBlockQueue
        // -call PostEndOfEpoch (will send EOE)
        // -wait for reset
        //
        // Worker threads need to:
        // -stop reading the file they are currently reading and throw it away
        // -keep pulling, but dont read other files (eventually skips all IOBlocks and will get EOE)
        //
        // Master thread needs to:
        // -tell IOBlockQueue thread to stop pushing
        // -tell worker threads to stop reading the file tey are currently reading
        // -keep pulling until EOE

        // don't think we need a lock for now
        load_jagged_connector_ = false;

        std::unique_lock<std::mutex> lock(load_io_block_queue_mutex_);
        load_io_block_queue_ = false;
      }
    }

    // all workers finished reading for this epoch, and we have read all the data from all workers
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

// static local-only helper function
static void shuffleKeys(std::vector<int64_t> *i_keys, uint32_t seed) {
  std::mt19937 rng(seed);
  std::shuffle(i_keys->begin(), i_keys->end(), rng);
}

// The entry point for when workers are launched.
Status TFReaderOp::WorkerEntry(int32_t worker_id) {
  // must be called first if called by worker spawned by taskgroup
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
        MS_LOG(DEBUG) << "TFReader operator worker " << worker_id << " loaded file " << filename << ".";
      }
    } else {
      std::unique_ptr<DataBuffer> eoe_buffer = std::make_unique<DataBuffer>(1, DataBuffer::kDeBFlagEOE);
      RETURN_IF_NOT_OK(jagged_buffer_connector_->Add(worker_id, std::move(eoe_buffer)));
    }

    RETURN_IF_NOT_OK(PopIoBlockQueue(worker_id, &io_block));
  }

  return Status::OK();
}

// Pushes a control indicator onto the IOBlockQueue for each worker to consume.
// When the worker pops this control indicator, it will shut itself down gracefully.
Status TFReaderOp::PostEndOfData() {
  for (int i = 0; i < num_workers_; ++i) {
    std::unique_ptr<FilenameBlock> eof = std::make_unique<FilenameBlock>(IOBlock::kDeIoBlockFlagEof);
    RETURN_IF_NOT_OK(PushIoBlockQueue(i, std::move(eof)));
  }

  return Status::OK();
}

// Pushes a control indicator onto the IOBlockQueue for each worker to consume. When the worker
// pops this control indicator, it will wait until the next epoch starts and then resume execution.
Status TFReaderOp::PostEndOfEpoch(int32_t queue_index) {
  for (int i = 0; i < num_workers_; ++i) {
    std::unique_ptr<FilenameBlock> eoe = std::make_unique<FilenameBlock>(IOBlock::kDeIoBlockFlagEoe);
    RETURN_IF_NOT_OK(PushIoBlockQueue((queue_index + i) % num_workers_, std::move(eoe)));
  }

  return Status::OK();
}

bool TFReaderOp::NeedPushFileToBlockQueue(const std::string &file_name, int64_t *start_offset, int64_t *end_offset,
                                          const int64_t &pre_count) {
  *start_offset = 0;
  *end_offset = 0;
  bool push = false;
  int64_t start_index = device_id_ * num_rows_per_shard_;
  if (device_id_ + 1 < 0) {
    MS_LOG(ERROR) << "Device id is invalid.";
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

Status TFReaderOp::FillIOBlockShuffle(const std::vector<int64_t> &i_keys) {
  int32_t queue_index = 0;
  int32_t key_index = 0;
  int64_t pre_count = 0;
  int64_t start_offset = 0;
  int64_t end_offset = 0;
  bool finish = false;
  bool end_of_epoch = false;
  while (!finish) {
    for (auto it = i_keys.begin(); it != i_keys.end(); ++it) {
      {
        std::unique_lock<std::mutex> lock(load_io_block_queue_mutex_);
        if (load_io_block_queue_ == false) {
          end_of_epoch = true;
          break;
        }
      }
      if (!equal_rows_per_shard_) {
        if (key_index++ % num_devices_ == device_id_) {
          auto ioBlock = std::make_unique<FilenameBlock>(*it, kInvalidOffset, kInvalidOffset, IOBlock::kDeIoBlockNone);
          RETURN_IF_NOT_OK(PushIoBlockQueue(queue_index, std::move(ioBlock)));
          queue_index = (queue_index + 1) % num_workers_;
        }
      } else {
        // Do an index lookup using that key to get the filename.
        std::string file_name = (*filename_index_)[*it];
        if (NeedPushFileToBlockQueue(file_name, &start_offset, &end_offset, pre_count)) {
          auto ioBlock = std::make_unique<FilenameBlock>(*it, start_offset, end_offset, IOBlock::kDeIoBlockNone);
          RETURN_IF_NOT_OK(PushIoBlockQueue(queue_index, std::move(ioBlock)));
          MS_LOG(DEBUG) << "File name " << *it << " start offset " << start_offset << " end_offset " << end_offset;
          queue_index = (queue_index + 1) % num_workers_;
        }

        pre_count += filename_numrows_[file_name];
      }
    }
    if (equal_rows_per_shard_ && pre_count < (static_cast<int64_t>(device_id_) + 1) * num_rows_per_shard_ &&
        !end_of_epoch) {
      finish = false;
    } else {
      finish = true;
    }
  }
  RETURN_IF_NOT_OK(PostEndOfEpoch(queue_index));
  return Status::OK();
}

Status TFReaderOp::FillIOBlockNoShuffle() {
  int32_t queue_index = 0;
  int32_t key_index = 0;
  int64_t pre_count = 0;
  int64_t start_offset = 0;
  int64_t end_offset = 0;
  bool finish = false;
  bool end_of_epoch = false;
  while (!finish) {
    // Iterate over all the keys and add one key to each block.
    for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
      {
        std::unique_lock<std::mutex> lock(load_io_block_queue_mutex_);
        if (load_io_block_queue_ == false) {
          end_of_epoch = true;
          break;
        }
      }
      if (!equal_rows_per_shard_) {
        if (key_index++ % num_devices_ == device_id_) {
          auto ioBlock =
            std::make_unique<FilenameBlock>(it.key(), kInvalidOffset, kInvalidOffset, IOBlock::kDeIoBlockNone);
          RETURN_IF_NOT_OK(PushIoBlockQueue(queue_index, std::move(ioBlock)));
          queue_index = (queue_index + 1) % num_workers_;
        }
      } else {
        std::string file_name = it.value();
        if (NeedPushFileToBlockQueue(file_name, &start_offset, &end_offset, pre_count)) {
          auto ioBlock = std::make_unique<FilenameBlock>(it.key(), start_offset, end_offset, IOBlock::kDeIoBlockNone);
          RETURN_IF_NOT_OK(PushIoBlockQueue(queue_index, std::move(ioBlock)));
          queue_index = (queue_index + 1) % num_workers_;
        }

        pre_count += filename_numrows_[file_name];
      }
    }
    if (equal_rows_per_shard_ && pre_count < (static_cast<int64_t>(device_id_) + 1) * num_rows_per_shard_ &&
        !end_of_epoch) {
      finish = false;
    } else {
      finish = true;
    }
  }

  RETURN_IF_NOT_OK(PostEndOfEpoch(queue_index));
  return Status::OK();
}

// Called asynchronously by another thread. Will wait until notified to fill the IOBlockQueue.
Status TFReaderOp::WaitToFillIOBlockQueue() {
  // must be called first if called by worker spawned by taskgroup
  TaskManager::FindMe()->Post();

  std::vector<int64_t> i_keys;
  // Generate a vector of keys that we can shuffle
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
      shuffleKeys(&i_keys, num_devices_ == 1 ? GetSeed() : ++seed);
      RETURN_IF_NOT_OK(FillIOBlockShuffle(i_keys));
    } else {  // shuffle_files_ == false
      RETURN_IF_NOT_OK(FillIOBlockNoShuffle());
    }
  }

  return Status::OK();
}

// Notifies the thread which called WaitToFillIOBlockQueue to resume execution.
void TFReaderOp::NotifyToFillIOBlockQueue() { io_block_queue_wait_post_.Set(); }

// Pops an element from a queue in io_block_queues
Status TFReaderOp::PopIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> *out_block) {
  RETURN_IF_NOT_OK(io_block_queues_[index]->PopFront(out_block));

  return Status::OK();
}

// Pushes an element to a queue in io_block_queues
Status TFReaderOp::PushIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> &&io_block) {
  RETURN_IF_NOT_OK(io_block_queues_[index]->Add(std::move(io_block)));

  return Status::OK();
}

// Reads a tf_file file and loads the data into multiple buffers.
Status TFReaderOp::LoadFile(const std::string &filename, const int64_t start_offset, const int64_t end_offset,
                            const int32_t &worker_id) {
  std::ifstream reader;
  reader.open(filename);
  if (!reader) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + filename);
  }

  int64_t rows_read = 0;
  int64_t rows_total = 0;
  std::unique_ptr<DataBuffer> current_buffer = std::make_unique<DataBuffer>(0, DataBuffer::BufferFlags::kDeBFlagNone);
  std::unique_ptr<TensorQTable> new_tensor_table = std::make_unique<TensorQTable>();

  while (reader.peek() != EOF) {
    if (!load_jagged_connector_) {
      break;
    }
    RETURN_IF_INTERRUPTED();

    // read length
    int64_t record_length = 0;
    (void)reader.read(reinterpret_cast<char *>(&record_length), static_cast<std::streamsize>(sizeof(int64_t)));

    // ignore crc header
    (void)reader.ignore(static_cast<std::streamsize>(sizeof(int32_t)));

    // read serialized Example
    std::string serialized_example;
    serialized_example.resize(record_length);
    (void)reader.read(&serialized_example[0], static_cast<std::streamsize>(record_length));
    if (start_offset == kInvalidOffset || (rows_total >= start_offset && rows_total < end_offset)) {
      dataengine::Example tf_file;
      if (!tf_file.ParseFromString(serialized_example)) {
        std::string errMsg = "Invalid file, failed to parse tfrecord file : " + filename;
        MS_LOG(DEBUG) << errMsg + ", details of string: " << serialized_example;
        RETURN_STATUS_UNEXPECTED(errMsg);
      }
      int32_t num_columns = data_schema_->NumColumns();
      TensorRow newRow(num_columns, nullptr);
      std::vector<std::string> file_path(num_columns, filename);
      newRow.setPath(file_path);
      new_tensor_table->push_back(std::move(newRow));
      RETURN_IF_NOT_OK(LoadExample(&tf_file, &new_tensor_table, rows_read));
      rows_read++;
    }

    // ignore crc footer
    (void)reader.ignore(static_cast<std::streamsize>(sizeof(int32_t)));
    rows_total++;

    if (rows_read == rows_per_buffer_) {
      current_buffer->set_tensor_table(std::move(new_tensor_table));
      RETURN_IF_NOT_OK(jagged_buffer_connector_->Add(worker_id, std::move(current_buffer)));

      current_buffer = std::make_unique<DataBuffer>(0, DataBuffer::BufferFlags::kDeBFlagNone);
      new_tensor_table = std::make_unique<TensorQTable>();
      rows_read = 0;
    }
  }

  if (rows_read > 0) {
    current_buffer->set_tensor_table(std::move(new_tensor_table));
    RETURN_IF_NOT_OK(jagged_buffer_connector_->Add(worker_id, std::move(current_buffer)));
  }

  return Status::OK();
}

// Parses a single row and puts the data into a tensor table.
Status TFReaderOp::LoadExample(const dataengine::Example *tf_file, std::unique_ptr<TensorQTable> *tensor_table,
                               int64_t row) {
  int32_t num_columns = data_schema_->NumColumns();
  for (int32_t col = 0; col < num_columns; ++col) {
    const ColDescriptor current_col = data_schema_->column(col);
    const dataengine::Features &example_features = tf_file->features();
    const google::protobuf::Map<std::string, dataengine::Feature> &feature_map = example_features.feature();
    auto iter_column = feature_map.find(current_col.name());
    if (iter_column == feature_map.end()) {
      RETURN_STATUS_UNEXPECTED("Invalid parameter, column name: " + current_col.name() + " does not exist.");
    }
    const dataengine::Feature &column_values_list = iter_column->second;
    RETURN_IF_NOT_OK(LoadFeature(tensor_table, column_values_list, current_col, row, col));
  }

  return Status::OK();
}

// Parses a single cell and puts the data into a tensor table.
Status TFReaderOp::LoadFeature(const std::unique_ptr<TensorQTable> *tensor_table,
                               const dataengine::Feature &column_values_list, const ColDescriptor &current_col,
                               int64_t row, int32_t col) {
  const dataengine::Feature::KindCase column_list_type = column_values_list.kind_case();
  std::unique_ptr<float[]> float_array;     // For staging data from protobuf deserialization
  const unsigned char *data_ptr = nullptr;  // Generic pointer used for populating the Tensor

  // This variable will point into the above staging variables.
  // Also used for creating shape attributes.
  int32_t num_elements = 0;

  // we build a tensor first a read directly into it if we need to cast
  std::shared_ptr<Tensor> ts;

  // Depending on the type of data from the tf_file, we want to extract 2 things:
  // 1) A pointer to the data as a const unsigned char *
  // 2) The number of elements of the data
  // After those are determined, we can then build the tensor to represent this data.
  switch (column_list_type) {
    case dataengine::Feature::KindCase::kBytesList: {
      RETURN_IF_NOT_OK(LoadBytesList(current_col, column_values_list, &num_elements, &ts));

      break;
    }
    case dataengine::Feature::KindCase::kFloatList: {
      RETURN_IF_NOT_OK(LoadFloatList(current_col, column_values_list, &num_elements, &float_array));

      data_ptr = reinterpret_cast<const unsigned char *>(float_array.get());

      // only floatList needs to create the tensor here, other two lists read directly
      // into the tensor
      TensorShape current_shape = TensorShape::CreateUnknownRankShape();
      RETURN_IF_NOT_OK(current_col.MaterializeTensorShape(num_elements, &current_shape));
      RETURN_IF_NOT_OK(Tensor::CreateFromMemory(current_shape, current_col.type(), data_ptr, &ts));
      break;
    }
    case dataengine::Feature::KindCase::kInt64List: {
      RETURN_IF_NOT_OK(LoadIntListSwitch(current_col, column_values_list, &num_elements, &ts));
      break;
    }
    case dataengine::Feature::KindCase::KIND_NOT_SET: {
      std::string err_msg = "Invalid data, tf_file column type must be uint8, int64 or float32.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    default: {
      std::string err_msg = "Invalid data, tf_file column type must be uint8, int64 or float32.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }

  (**tensor_table)[row][col] = std::move(ts);

  return Status::OK();
}

// Overrides base class reset method. Cleans up any state info from it's previous execution and
// reinitializes itself so that it can be executed again, as if it was just created.
Status TFReaderOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  // start workers first, otherwise IOBlocks will fall through if workers see it before this is set to true
  load_jagged_connector_ = true;

  {
    std::unique_lock<std::mutex> lock(load_io_block_queue_mutex_);
    load_io_block_queue_ = true;
  }

  RETURN_IF_NOT_OK(ParallelOp::Reset());
  NotifyToFillIOBlockQueue();

  return Status::OK();
}

Status TFReaderOp::LoadBytesList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                                 int32_t *num_elements, std::shared_ptr<Tensor> *tensor) {
  // kBytesList can map to the following DE types ONLY!
  // DE_UINT8, DE_INT8
  // Must be single byte type for each element!
  if (current_col.type() != DataType::DE_UINT8 && current_col.type() != DataType::DE_INT8 &&
      current_col.type() != DataType::DE_STRING) {
    std::string err_msg = "Invalid data, invalid data type for Tensor at column: " + current_col.name() +
                          ", data type should be int8, uint8 or string, but got " + current_col.type().ToString();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  const dataengine::BytesList &bytes_list = column_values_list.bytes_list();

  *num_elements = bytes_list.value_size();

  if (current_col.type() == DataType::DE_STRING) {
    TensorShape shape = TensorShape::CreateScalar();
    RETURN_IF_NOT_OK(current_col.MaterializeTensorShape(*num_elements, &shape));
    RETURN_IF_NOT_OK(Tensor::CreateFromByteList(bytes_list, shape, tensor));
    return Status::OK();
  }

  uint64_t max_size = 0;
  for (uint32_t i = 0; i < bytes_list.value_size(); ++i) {
#if defined(__APPLE__)
    max_size = fmax(max_size, bytes_list.value(i).size());
#else
    max_size = std::max(max_size, bytes_list.value(i).size());
#endif
  }

  int64_t pad_size = max_size;

  // if user provides a shape in the form of [-1, d1, 2d, ... , dn], we need to pad to d1 * d2 * ... * dn
  if (current_col.hasShape()) {
    TensorShape cur_shape = current_col.shape();
    if (cur_shape.Size() >= 2 && cur_shape[0] == TensorShape::kDimUnknown) {
      int64_t new_pad_size = 1;
      for (int i = 1; i < cur_shape.Size(); ++i) {
        if (cur_shape[i] == TensorShape::kDimUnknown) {
          std::string err_msg =
            "Invalid data, more than one unknown dimension in the shape of column: " + current_col.name();
          RETURN_STATUS_UNEXPECTED(err_msg);
        }
        new_pad_size *= cur_shape[i];
      }
      pad_size = new_pad_size;
    } else {
      if (cur_shape.known() && cur_shape.NumOfElements() != max_size) {
        std::string err_msg = "Shape in schema's column '" + current_col.name() + "' is incorrect." +
                              "\nshape received: " + cur_shape.ToString() +
                              "\ntotal elements in shape received: " + std::to_string(cur_shape.NumOfElements()) +
                              "\nexpected total elements in shape: " + std::to_string(max_size);
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
    }
  }

  // know how many elements there are and the total bytes, create tensor here:
  TensorShape current_shape = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(current_col.MaterializeTensorShape((*num_elements) * pad_size, &current_shape));
  RETURN_IF_NOT_OK(Tensor::CreateFromByteList(bytes_list, current_shape, current_col.type(), pad_size, tensor));

  return Status::OK();
}

Status TFReaderOp::LoadFloatList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                                 int32_t *num_elements, std::unique_ptr<float[]> *float_array) {
  // KFloatList can only map to DE types:
  // DE_FLOAT32
  if (current_col.type() != DataType::DE_FLOAT32) {
    std::string err_msg = "Invalid data, invalid data type for Tensor at column: " + current_col.name() +
                          ", data type should be string, but got " + current_col.type().ToString();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  const dataengine::FloatList &float_list = column_values_list.float_list();

  // Identify how many values we have and then create a local array of these
  // to deserialize into
  *num_elements = float_list.value_size();
  *float_array = std::make_unique<float[]>(*num_elements);
  for (int i = 0; i < float_list.value_size(); ++i) {
    (*float_array)[i] = float_list.value(i);
  }

  return Status::OK();
}

// Determines which template type to use and calls LoadIntList
Status TFReaderOp::LoadIntListSwitch(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                                     int32_t *num_elements, std::shared_ptr<Tensor> *tensor) {
  if (current_col.type() == DataType::DE_UINT64) {
    RETURN_IF_NOT_OK(LoadIntList<uint64_t>(current_col, column_values_list, num_elements, tensor));
  } else if (current_col.type() == DataType::DE_INT64) {
    RETURN_IF_NOT_OK(LoadIntList<int64_t>(current_col, column_values_list, num_elements, tensor));
  } else if (current_col.type() == DataType::DE_UINT32) {
    RETURN_IF_NOT_OK(LoadIntList<uint32_t>(current_col, column_values_list, num_elements, tensor));
  } else if (current_col.type() == DataType::DE_INT32) {
    RETURN_IF_NOT_OK(LoadIntList<int32_t>(current_col, column_values_list, num_elements, tensor));
  } else if (current_col.type() == DataType::DE_UINT16) {
    RETURN_IF_NOT_OK(LoadIntList<uint16_t>(current_col, column_values_list, num_elements, tensor));
  } else if (current_col.type() == DataType::DE_INT16) {
    RETURN_IF_NOT_OK(LoadIntList<int16_t>(current_col, column_values_list, num_elements, tensor));
  } else if (current_col.type() == DataType::DE_UINT8) {
    RETURN_IF_NOT_OK(LoadIntList<uint8_t>(current_col, column_values_list, num_elements, tensor));
  } else if (current_col.type() == DataType::DE_INT8) {
    RETURN_IF_NOT_OK(LoadIntList<int8_t>(current_col, column_values_list, num_elements, tensor));
  } else {
    std::string err_msg = "Invalid data, invalid datatype for Tensor at column: " + current_col.name() +
                          ", data type should be uint64, int64, uint32, int32, uint16, int16, uint8 or int8" +
                          ", but got " + current_col.type().ToString();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}

// Reads values from a bytes list and casts the value to type T, must be an integral type
// compatible with int64_t
template <typename T>
Status TFReaderOp::LoadIntList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                               int32_t *num_elements, std::shared_ptr<Tensor> *tensor) {
  if (!(current_col.type().IsInt())) {
    std::string err_msg = "Invalid data, invalid data type for Tensor at column: " + current_col.name() +
                          ", data type should be int, but got " + current_col.type().ToString();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  const dataengine::Int64List &int64_list = column_values_list.int64_list();

  // Identify how many values we have and then create a local array of these
  // to deserialize into
  *num_elements = int64_list.value_size();

  // know how many elements there are, create tensor here:
  TensorShape current_shape = TensorShape::CreateUnknownRankShape();
  RETURN_IF_NOT_OK(current_col.MaterializeTensorShape(*num_elements, &current_shape));
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(current_shape, current_col.type(), tensor));

  int64_t i = 0;
  auto it = (*tensor)->begin<T>();
  for (; it != (*tensor)->end<T>(); i++, ++it) {
    T element = static_cast<T>(int64_list.value(i));
    *it = element;
  }

  return Status::OK();
}

Status TFReaderOp::CreateSchema(const std::string tf_file, std::vector<std::string> columns_to_load) {
  std::ifstream reader;
  reader.open(tf_file);

  // read length
  int64_t record_length = 0;
  (void)reader.read(reinterpret_cast<char *>(&record_length), static_cast<std::streamsize>(sizeof(int64_t)));

  // ignore crc header
  (void)reader.ignore(static_cast<std::streamsize>(sizeof(int32_t)));

  // read serialized Example
  std::string serialized_example;
  serialized_example.resize(record_length);
  (void)reader.read(&serialized_example[0], static_cast<std::streamsize>(record_length));

  dataengine::Example example;
  if (!example.ParseFromString(serialized_example)) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to parse tfrecord file: " + serialized_example);
  }

  const dataengine::Features &example_features = example.features();
  const google::protobuf::Map<std::string, dataengine::Feature> &feature_map = example_features.feature();

  if (columns_to_load.empty()) {
    (void)std::transform(feature_map.begin(), feature_map.end(), std::back_inserter(columns_to_load),
                         [](const auto &it) -> std::string { return it.first; });
    std::sort(columns_to_load.begin(), columns_to_load.end());
  }

  for (const auto &curr_col_name : columns_to_load) {
    auto it = feature_map.find(curr_col_name);
    if (it == feature_map.end()) {
      RETURN_STATUS_UNEXPECTED("Invalid data, failed to find column name: " + curr_col_name);
    }
    std::string column_name = it->first;

    std::string column_type;

    const dataengine::Feature &feature = it->second;
    const dataengine::Feature::KindCase kind_case = feature.kind_case();
    switch (kind_case) {
      case dataengine::Feature::KindCase::kBytesList:
        column_type = "uint8";
        break;

      case dataengine::Feature::KindCase::kFloatList:
        column_type = "float32";
        break;

      case dataengine::Feature::KindCase::kInt64List:
        column_type = "int64";
        break;

      case dataengine::Feature::KindCase::KIND_NOT_SET:
        RETURN_STATUS_UNEXPECTED("Invalid data, tf_file column type must be uint8, int64 or float32.");

      default:
        RETURN_STATUS_UNEXPECTED("Invalid data, tf_file column type must be uint8, int64 or float32.");
    }

    RETURN_IF_NOT_OK(
      data_schema_->AddColumn(ColDescriptor(column_name, DataType(column_type), TensorImpl::kFlexible, 1)));
  }

  return Status::OK();
}

Status TFReaderOp::CountTotalRows(int64_t *out_total_rows, const std::vector<std::string> &filenames, int64_t threads,
                                  bool estimate) {
  try {
    if (threads > filenames.size()) {
      threads = filenames.size();
    }

    std::vector<std::future<int64_t>> async_results;

    int64_t chunk_size = filenames.size() / threads;
    int64_t remainder = filenames.size() % threads;

    int64_t begin = 0;
    int64_t end = begin;
    for (int i = 0; i < threads; i++) {
      end += chunk_size;
      if (remainder > 0) {
        end++;
        remainder--;
      }

      if (estimate) {
        // Parse a single file for each chunk with estimate mode on
        async_results.push_back(std::async(std::launch::async, &CountTotalRowsSectioned, filenames, begin, begin + 1));
      } else {
        // Parse the whole chunk with estimate mode off
        async_results.push_back(std::async(std::launch::async, &CountTotalRowsSectioned, filenames, begin, end));
      }

      begin = end;
    }

    int64_t total_rows = 0;
    for (int i = 0; i < async_results.size(); i++) {
      total_rows += async_results[i].get();
    }

    if (estimate) {
      // Each thread only scans 1 file
      // Estimated total rows = Average rows * total number of files
      total_rows = total_rows / threads * filenames.size();
    }

    *out_total_rows = total_rows;
  } catch (const std::exception &e) {
    std::string err_msg = "Unexpected error occurred: ";
    err_msg += e.what();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}

int64_t TFReaderOp::CountTotalRowsSectioned(const std::vector<std::string> &filenames, int64_t begin, int64_t end) {
  int64_t rows_read = 0;
  for (int i = begin; i < end; i++) {
    std::ifstream reader;
    reader.open(filenames[i]);
    if (!reader) {
      MS_LOG(DEBUG) << "TFReader operator failed to open file " << filenames[i] << ".";
    }

    while (reader.peek() != EOF) {
      // read length
      int64_t record_length = 0;
      (void)reader.read(reinterpret_cast<char *>(&record_length), static_cast<std::streamsize>(sizeof(int64_t)));

      // ignore crc header
      (void)reader.ignore(static_cast<std::streamsize>(sizeof(int32_t)));

      // ignore tf_file contents
      (void)reader.ignore(static_cast<std::streamsize>(record_length));

      // ignore crc footer
      (void)reader.ignore(static_cast<std::streamsize>(sizeof(int32_t)));

      rows_read++;
    }
  }

  return rows_read;
}

Status TFReaderOp::ComputeColMap() {
  // Construct the column name map for this operator (base class field)
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
