/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/cache_base_op.h"
#include <iomanip>
#include <iostream>
#include <utility>
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
// A print method typically used for debugging
void CacheBase::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nCache client:\n" << *cache_client_ << "\n\n";
  }
}
// Overrides base class reset method.  When an operator does a reset, it cleans up any state
// info from it's previous execution and then initializes itself so that it can be executed
// again.
Status CacheBase::Reset() {
  if (sampler_ != nullptr) {
    RETURN_IF_NOT_OK(sampler_->ResetSampler());
  }
  // Wake up the workers to get them going again in a new epoch
  MS_LOG(DEBUG) << Name() << " resetting.";
  epoch_sync_.Set();
  return Status::OK();
}
CacheBase::CacheBase(int32_t num_workers, int32_t op_connector_size, int32_t rows_per_buf,
                     std::shared_ptr<CacheClient> cache_client, std::shared_ptr<Sampler> sampler)
    : ParallelOp(num_workers, op_connector_size, std::move(sampler)),
      row_cnt_(0),
      num_cache_miss_(0),
      cache_client_(std::move(cache_client)),
      rows_per_buffer_(rows_per_buf),
      // We can cause deadlock if this internal Connector size is too small.
      keys_miss_(num_workers_, 1, connector_capacity_),
      prefetch_size_(cache_client_->getPrefetchSize()) {
  io_block_queues_.Init(num_workers, op_connector_size);
  prefetch_queues_.Init(num_workers, op_connector_size);
  sampler_queue_ = std::make_unique<Queue<std::shared_ptr<Tensor>>>(op_connector_size);
}
// Common function to fetch samples from the sampler and send them using the io_block_queues to
// the parallel workers
Status CacheBase::FetchSamplesToWorkers() {
  int64_t buf_cnt = 0;
  int64_t wait_cnt = 0;
  // Kick off several threads which will prefetch prefetch_size_ rows in advance. The rows_per_buffers_
  // is too small (1 by default) and won't help performance.
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask("Dispatcher", std::bind(&CacheBase::Dispatcher, this)));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_, std::bind(&CacheBase::Prefetcher, this, std::placeholders::_1)));
  // Instead of sending sampler id to WorkerEntry, we send them to the Prefetcher which will redirect them
  // to the WorkerEntry.
  do {
    epoch_sync_.Clear();
    if (AllowCacheMiss() && wait_cnt > 0) {
      MS_LOG(WARNING) << "Epoch: " << wait_cnt << " Cache Miss : " << num_cache_miss_
                      << " Total number of rows : " << row_cnt_;
    }
    num_cache_miss_ = 0;
    row_cnt_ = 0;
    ++wait_cnt;
    std::vector<row_id_type> keys;
    keys.reserve(rows_per_buffer_);
    std::unique_ptr<DataBuffer> sampler_buffer;
    RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    while (!sampler_buffer->eoe()) {
      TensorRow sample_row;
      RETURN_IF_NOT_OK(sampler_buffer->PopRow(&sample_row));
      std::shared_ptr<Tensor> sample_ids = sample_row[0];
      // Send the sampler tensor to other thread for prefetching. We are using shared pointer so it
      // won't go out scope until it is really not in use.
      RETURN_IF_NOT_OK(sampler_queue_->Add(sample_ids));
      for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); itr++) {
        keys.push_back(*itr);
        ++row_cnt_;
        if (row_cnt_ % rows_per_buffer_ == 0) {
          auto blk = std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone));
          RETURN_IF_NOT_OK(io_block_queues_[buf_cnt++ % num_workers_]->Add(std::move(blk)));
          keys.clear();
        }
      }
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    }
    if (!keys.empty()) {
      auto blk = std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone));
      RETURN_IF_NOT_OK(io_block_queues_[buf_cnt++ % num_workers_]->Add(std::move(blk)));
    }
    // send the eoe
    RETURN_IF_NOT_OK(
      io_block_queues_[(buf_cnt++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
    // If repeat but the not last repeat, wait for reset.
    if (!IsLastIteration()) {
      MS_LOG(DEBUG) << Name() << " Waiting for reset. Count " << wait_cnt << " Buffer sent " << buf_cnt;
      RETURN_IF_NOT_OK(epoch_sync_.Wait());
    } else {
      // We can break out from the loop.
      break;
    }
    UpdateRepeatAndEpochCounter();
  } while (true);
  // Flow the eof before exit
  RETURN_IF_NOT_OK(
    io_block_queues_[(buf_cnt++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof)));
  // Shutdown threads
  std::shared_ptr<Tensor> empty;
  RETURN_IF_NOT_OK(sampler_queue_->Add(std::move(empty)));
  for (int32_t i = 0; i < num_workers_; i++) {
    RETURN_IF_NOT_OK(
      io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
  }
  // Dump the last epoch result (approximately) without waiting for the worker threads to come back.
  if (AllowCacheMiss()) {
    MS_LOG(WARNING) << "Epoch: " << wait_cnt << " Cache Miss : " << num_cache_miss_
                    << " Total number of rows : " << row_cnt_;
  }
  return Status::OK();
}

Status CacheBase::FetchFromCache(int32_t worker_id) {
  int64_t buffer_id = worker_id;
  std::unique_ptr<IOBlock> blk;
  do {
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&blk));
    if (blk->eof()) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF)));
    } else if (blk->eoe()) {
      if (AllowCacheMiss()) {
        // This code path is for CacheLookupOp acting as a sampler. If we get a eoe from
        // a sampler, send a eoe to physical leaf op as well.
        std::vector<row_id_type> eoe;
        eoe.push_back(eoe_row_id);
        RETURN_IF_NOT_OK(keys_miss_.Push(worker_id, eoe));
      }
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE)));
    } else {
      std::vector<int64_t> keys;
      RETURN_IF_NOT_OK(blk->GetKeys(&keys));
      if (keys.empty()) {
        // empty key is a quit signal for workers
        break;
      }
      std::unique_ptr<DataBuffer> db = std::make_unique<DataBuffer>(buffer_id, DataBuffer::kDeBFlagNone);
      std::unique_ptr<TensorQTable> que = std::make_unique<TensorQTable>();
      std::vector<row_id_type> cache_miss;
      cache_miss.reserve(keys.size());
      for (auto row_id : keys) {
        TensorRow row;
        // Block until the row shows up in the pool.
        RETURN_IF_NOT_OK(prefetch_.PopFront(row_id, &row));
        if (row.empty()) {
          cache_miss.push_back(row_id);
        }
        que->push_back(std::move(row));
      }
      db->set_tensor_table(std::move(que));
      if (AllowCacheMiss()) {
        // Because of the way connector works, we push unconditionally even cache_miss can be empty.
        RETURN_IF_NOT_OK(keys_miss_.Push(worker_id, cache_miss));
      }
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::move(db)));
      buffer_id += num_workers_;
    }
  } while (true);
  return Status::OK();
}

Status CacheBase::RegisterResources() {
  RETURN_IF_NOT_OK(epoch_sync_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(prefetch_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(sampler_queue_->Register(tree_->AllTasks()));
  return Status::OK();
}

CacheBase::~CacheBase() = default;

Status CacheBase::UpdateColumnMapFromCache() {
  Status rc;
  // Get the schema from the server. It may not be there yet. So tolerate the error.
  if (column_name_id_map_.empty()) {
    rc = cache_client_->FetchSchema(&column_name_id_map_);
    if (rc == Status(StatusCode::kFileNotExist)) {
      MS_LOG(DEBUG) << "Schema not in the server yet.";
      rc = Status::OK();
    }
  }
  return rc;
}

Status CacheBase::Dispatcher() {
  TaskManager::FindMe()->Post();
  int64_t buf_cnt = 0;
  int64_t num_row = 0;
  std::vector<row_id_type> keys;
  keys.reserve(prefetch_size_);
  do {
    keys.clear();
    std::shared_ptr<Tensor> sample_ids;
    RETURN_IF_NOT_OK(sampler_queue_->PopFront(&sample_ids));
    if (sample_ids == nullptr) {
      // A null shared pointer signal times to quit.
      // Also signal all prefetchers to quit.
      for (int32_t i = 0; i < num_workers_; i++) {
        RETURN_IF_NOT_OK(
          prefetch_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
      }
      break;
    }
    // Now we distribute the sampler ids to each prefetcher according to the prefetch size.
    for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); itr++) {
      keys.push_back(*itr);
      ++num_row;
      if (num_row % prefetch_size_ == 0) {
        auto blk = std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone));
        RETURN_IF_NOT_OK(prefetch_queues_[buf_cnt++ % num_workers_]->Add(std::move(blk)));
        keys.clear();
      }
    }
    // Send the remaining sample id
    if (!keys.empty()) {
      auto blk = std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone));
      RETURN_IF_NOT_OK(prefetch_queues_[buf_cnt++ % num_workers_]->Add(std::move(blk)));
    }
  } while (true);
  return Status::OK();
}

Status CacheBase::Prefetcher(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::vector<row_id_type> prefetch_keys;
  prefetch_keys.reserve(prefetch_size_);
  do {
    prefetch_keys.clear();
    std::unique_ptr<IOBlock> blk;
    RETURN_IF_NOT_OK(prefetch_queues_[worker_id]->PopFront(&blk));
    RETURN_IF_NOT_OK(blk->GetKeys(&prefetch_keys));
    if (prefetch_keys.empty()) {
      // Empty keys mean time to quit.
      break;
    }
    TensorTable ttbl;
    RETURN_IF_NOT_OK(cache_client_->GetRows(prefetch_keys, &ttbl));
    auto row_it = ttbl.begin();
    for (auto row_id : prefetch_keys) {
      auto &row = *row_it;
      if (row.empty()) {
        if (AllowCacheMiss()) {
          ++num_cache_miss_;
        } else {
          std::string errMsg = "Row id " + std::to_string(row_id) + " not found.";
          RETURN_STATUS_UNEXPECTED(errMsg);
        }
      }
      // Put the prefetch row into the pool and wake up any WorkerEntry to wait for the row
      RETURN_IF_NOT_OK(prefetch_.Add(row_id, std::move(row)));
      ++row_it;
    }
  } while (true);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
