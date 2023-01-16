/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/nonmappable_leaf_op.h"

#include <utility>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
NonMappableLeafOp::NonMappableLeafOp(int32_t num_workers, int32_t worker_connector_size, int64_t total_num_rows,
                                     int32_t op_connector_size, bool shuffle_files, int32_t num_devices,
                                     int32_t device_id, const CompressionType &compression_type)
    : ParallelOp(num_workers, op_connector_size),
      device_id_(device_id),
      num_devices_(num_devices),
      load_jagged_connector_(true),
      filename_index_(std::make_unique<StringIndex>()),
      finished_reading_dataset_(false),
      total_rows_(total_num_rows),
      load_io_block_queue_(true),
      shuffle_files_(shuffle_files),
      num_rows_per_shard_(0),
      compression_type_(compression_type),
      num_rows_(0),
      shuffled_keys_({}),
      seed_(0) {
  worker_connector_size_ = worker_connector_size;
}

// Class functor operator () override.
// All dataset operators operate by launching a thread (see ExecutionTree). This class functor will
// provide the master loop that drives the logic for performing the work
Status NonMappableLeafOp::operator()() {
  RETURN_IF_NOT_OK(CalculateNumRowsPerShard());

  // Put here to avoid register failed when Worker_Entry thread exits unexpected
  RETURN_IF_NOT_OK(io_block_queue_wait_post_.Register(tree_->AllTasks()));

  // launch one thread, responsible for filling mIOBlockQueue
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(1, std::bind(&NonMappableLeafOp::WaitToFillIOBlockQueue, this), "", id()));

  // launch num_workers_ worker threads, responsible for pulling from the IOBlockQueue and reading
  // data from disk into TensorRows
  RETURN_IF_NOT_OK(RegisterAndLaunchThreads());

  // must be called after launching workers. workers can't be spawned after this post,
  // so workers have to be kept alive until the end of the program
  TaskManager::FindMe()->Post();

  NotifyToFillIOBlockQueue();
  while (!finished_reading_dataset_) {
    int32_t workers_done = 0;
    int64_t rows_read = 0;
    {
      std::unique_lock<std::mutex> lock(load_io_block_queue_mutex_);
      load_io_block_queue_ = true;
    }

    while (workers_done < num_workers_) {
      TensorRow fetched_row;
      RETURN_IF_NOT_OK(jagged_rows_connector_->Pop(0, &fetched_row));
      if (fetched_row.eoe()) {
        workers_done++;
      } else if ((compression_type_ == CompressionType::NONE || compression_type_ == CompressionType::GZIP_WITH_COUNT ||
                  compression_type_ == CompressionType::ZLIB_WITH_COUNT) &&
                 (total_rows_ == 0 || rows_read < total_rows_)) {
        // we need to push a row
        RETURN_IF_NOT_OK(out_connector_->Add(std::move(fetched_row)));
        rows_read++;
      } else if ((compression_type_ == CompressionType::GZIP || compression_type_ == CompressionType::ZLIB) &&
                 (rows_read < total_rows_ * num_devices_)) {
        // for compressed version, total_rows_ is total rows that will be read per shard
        // we need to push a row
        RETURN_IF_NOT_OK(out_connector_->Add(std::move(fetched_row)));
        rows_read++;
      } else {
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
        {
          std::unique_lock<std::mutex> lock(load_jagged_connector_mutex_);
          load_jagged_connector_ = false;
        }
        {
          std::unique_lock<std::mutex> lock(load_io_block_queue_mutex_);
          load_io_block_queue_ = false;
        }
      }
    }

    // all workers finished reading for this epoch, and we have read all the data from all workers
    RETURN_IF_NOT_OK(out_connector_->SendEOE());

    if (IsLastIteration()) {
      finished_reading_dataset_ = true;
      NotifyToFillIOBlockQueue();
    } else {
      jagged_rows_connector_->DoReset();
      // Self-reset to start a new iteration
      RETURN_IF_NOT_OK(Reset());
    }
    UpdateRepeatAndEpochCounter();
  }

  RETURN_IF_NOT_OK(out_connector_->SendEOF());

  RETURN_IF_NOT_OK(PostEndOfData());

  return Status::OK();
}

// The entry point for when workers are launched.
Status NonMappableLeafOp::WorkerEntry(int32_t worker_id) {
  // must be called first if called by worker spawned by taskgroup
  TaskManager::FindMe()->Post();

  std::unique_ptr<FilenameBlock> io_block;
  RETURN_IF_NOT_OK(PopIoBlockQueue(worker_id, &io_block));

  while (!io_block->eof()) {
    if (!io_block->eoe()) {
      if (GetLoadJaggedConnector()) {
        std::string filename;
        RETURN_IF_NOT_OK(io_block->GetFilename(&filename, *filename_index_));
        int64_t start_offset = io_block->GetStartOffset();
        int64_t end_offset = io_block->GetEndOffset();
        RETURN_IF_NOT_OK(LoadFile(filename, start_offset, end_offset, worker_id));
        MS_LOG(DEBUG) << Name() << " operator worker " << worker_id << " loaded file " << filename << ".";
      }
    } else {
      TensorRow eoe = TensorRow(TensorRow::kFlagEOE);
      RETURN_IF_NOT_OK(jagged_rows_connector_->Add(worker_id, std::move(eoe)));
    }

    RETURN_IF_NOT_OK(PopIoBlockQueue(worker_id, &io_block));
  }

  return Status::OK();
}

// Pushes a control indicator onto the IOBlockQueue for each worker to consume.
// When the worker pops this control indicator, it will shut itself down gracefully.
Status NonMappableLeafOp::PostEndOfData() {
  for (int i = 0; i < num_workers_; ++i) {
    std::unique_ptr<FilenameBlock> eof = std::make_unique<FilenameBlock>(IOBlock::kDeIoBlockFlagEof);
    RETURN_IF_NOT_OK(PushIoBlockQueue(i, std::move(eof)));
  }

  return Status::OK();
}

// Pushes a control indicator onto the IOBlockQueue for each worker to consume. When the worker
// pops this control indicator, it will wait until the next epoch starts and then resume execution.
Status NonMappableLeafOp::PostEndOfEpoch(int32_t queue_index) {
  for (int i = 0; i < num_workers_; ++i) {
    std::unique_ptr<FilenameBlock> eoe = std::make_unique<FilenameBlock>(IOBlock::kDeIoBlockFlagEoe);
    RETURN_IF_NOT_OK(PushIoBlockQueue((queue_index + i) % num_workers_, std::move(eoe)));
  }

  return Status::OK();
}

// Notifies the thread which called WaitToFillIOBlockQueue to resume execution.
void NonMappableLeafOp::NotifyToFillIOBlockQueue() { io_block_queue_wait_post_.Set(); }

// Pops an element from a queue in io_block_queues
Status NonMappableLeafOp::PopIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> *out_block) {
  RETURN_IF_NOT_OK(io_block_queues_[index]->PopFront(out_block));
  return Status::OK();
}

// Pushes an element to a queue in io_block_queues
Status NonMappableLeafOp::PushIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> &&io_block) {
  RETURN_IF_NOT_OK(io_block_queues_[index]->Add(std::move(io_block)));
  return Status::OK();
}

// Overrides base class reset method. Cleans up any state info from it's previous execution and
// reinitializes itself so that it can be executed again, as if it was just created.
Status NonMappableLeafOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  // start workers first, otherwise IOBlocks will fall through if workers see it before this is set to true
  {
    std::unique_lock<std::mutex> lock(load_jagged_connector_mutex_);
    load_jagged_connector_ = true;
  }

  {
    std::unique_lock<std::mutex> lock(load_io_block_queue_mutex_);
    load_io_block_queue_ = true;
  }

  NotifyToFillIOBlockQueue();

  return Status::OK();
}

bool NonMappableLeafOp::NeedPushFileToBlockQueue(const std::string &file_name, int64_t *start_offset,
                                                 int64_t *end_offset, const int64_t &pre_count) {
  *start_offset = 0;
  *end_offset = 0;
  bool push = false;
  int64_t start_index = device_id_ * num_rows_per_shard_;
  if (device_id_ + 1 < 0) {
    MS_LOG(ERROR) << "Invalid device id, device id should be greater than or equal 0, but got "
                  << std::to_string(device_id_);
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

void NonMappableLeafOp::ShuffleKeys() {
  std::mt19937 rng(num_devices_ == 1 ? GetSeed() : ++seed_);
  std::shuffle(shuffled_keys_.begin(), shuffled_keys_.end(), rng);
}

Status NonMappableLeafOp::WaitToFillIOBlockQueue() {
  // must be called first if called by worker spanwed by taskgroup
  TaskManager::FindMe()->Post();

  while (true) {
    RETURN_IF_NOT_OK(io_block_queue_wait_post_.Wait());
    io_block_queue_wait_post_.Clear();

    if (finished_reading_dataset_) {
      break;
    }

    if (shuffle_files_) {
      ShuffleKeys();
    }
    RETURN_IF_NOT_OK(FillIOBlockQueue(shuffled_keys_));
  }
  return Status::OK();
}

Status NonMappableLeafOp::PrepareOperator() {
  // Run any common code from super class first before adding our own
  RETURN_IF_NOT_OK(DatasetOp::PrepareOperator());

  if (shuffle_files_) {
    for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
      shuffled_keys_.push_back(it.key());
    }
    // in reset mode, shuffled_keys needs to be ordered in the rsetting epoch
    if (GlobalContext::config_manager()->fast_recovery() && op_current_repeats_ > 0) {
      for (auto i = 0; i < op_current_repeats_; i++) {
        ShuffleKeys();
      }
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
