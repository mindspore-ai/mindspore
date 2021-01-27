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
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  return Status::OK();
}
CacheBase::CacheBase(int32_t num_workers, int32_t op_connector_size, int32_t rows_per_buf,
                     std::shared_ptr<CacheClient> cache_client, std::shared_ptr<SamplerRT> sampler)
    : ParallelOp(num_workers, op_connector_size, std::move(sampler)),
      row_cnt_(0),
      num_cache_miss_(0),
      cache_client_(std::move(cache_client)),
      rows_per_buffer_(rows_per_buf),
      prefetch_size_(rows_per_buffer_),
      num_prefetchers_(num_workers_) {
  // Adjust the prefetch size based on the number of workers.
  auto prefetch_sz_per_thread = cache_client_->GetPrefetchSize() / num_prefetchers_;
  if (prefetch_size_ < prefetch_sz_per_thread) {
    prefetch_size_ = prefetch_sz_per_thread;
    MS_LOG(DEBUG) << "Per worker prefetch size : " << prefetch_size_;
  }
  io_block_queues_.Init(num_workers, op_connector_size);
  prefetch_queues_.Init(num_prefetchers_, op_connector_size);
  // We can cause deadlock if this internal Connector size is too small.
  keys_miss_ = std::make_unique<Connector<std::vector<row_id_type>>>(num_prefetchers_, 1, connector_capacity_);
}
// Common function to fetch samples from the sampler and send them using the io_block_queues to
// the parallel workers
Status CacheBase::FetchSamplesToWorkers() {
  int64_t buf_cnt = 0;
  int64_t wait_cnt = 0;
  int64_t prefetch_cnt = 0;
  // Kick off several threads which will prefetch prefetch_size_ rows in advance. The rows_per_buffers_
  // is too small (1 by default) and won't help performance.
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_prefetchers_, std::bind(&CacheBase::Prefetcher, this, std::placeholders::_1), Name()));
  auto send_to_que = [](QueueList<std::unique_ptr<IOBlock>> &qList, int32_t worker_id,
                        std::vector<row_id_type> &keys) -> Status {
    auto blk = std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone));
    RETURN_IF_NOT_OK(qList[worker_id]->Add(std::move(blk)));
    return Status::OK();
  };
  // Instead of sending sampler id to WorkerEntry, we send them to the Prefetcher which will redirect them
  // to the WorkerEntry.
  do {
    if (AllowCacheMiss() && wait_cnt > 0 && wait_cnt % op_num_repeats_per_epoch() == 0) {
      MS_LOG(INFO) << "Epoch: " << op_current_epochs_ << " Cache Miss : " << num_cache_miss_
                   << " Total number of rows : " << row_cnt_;
    }
    num_cache_miss_ = 0;
    row_cnt_ = 0;
    ++wait_cnt;
    std::vector<row_id_type> keys;
    keys.reserve(rows_per_buffer_);
    std::vector<row_id_type> prefetch_keys;
    prefetch_keys.reserve(prefetch_size_);
    std::unique_ptr<DataBuffer> sampler_buffer;
    RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    while (!sampler_buffer->eoe()) {
      TensorRow sample_row;
      RETURN_IF_NOT_OK(sampler_buffer->PopRow(&sample_row));
      std::shared_ptr<Tensor> sample_ids = sample_row[0];
      for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); itr++) {
        ++row_cnt_;
        prefetch_keys.push_back(*itr);
        // Batch enough rows for performance reason.
        if (row_cnt_ % prefetch_size_ == 0) {
          RETURN_IF_NOT_OK(send_to_que(prefetch_queues_, prefetch_cnt++ % num_prefetchers_, prefetch_keys));
          // Now we tell the WorkerEntry to wait for them to come back. If prefetch_size_ is a multiple
          // of rows_per_buffer_, the keys vector will always be empty. But it can be partially filled.
          // The only requirement we set up is rows_per_buffer_ is less than or equal to prefetch_size_.
          for (auto row_id : prefetch_keys) {
            keys.push_back(row_id);
            if (keys.size() == rows_per_buffer_) {
              RETURN_IF_NOT_OK(send_to_que(io_block_queues_, buf_cnt++ % num_workers_, keys));
              keys.clear();
            }
          }
          prefetch_keys.clear();
        }
      }
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    }
    // Deal with any partial keys left.
    if (!prefetch_keys.empty()) {
      RETURN_IF_NOT_OK(send_to_que(prefetch_queues_, prefetch_cnt++ % num_prefetchers_, prefetch_keys));
      for (auto row_id : prefetch_keys) {
        keys.push_back(row_id);
        if (keys.size() == rows_per_buffer_) {
          RETURN_IF_NOT_OK(send_to_que(io_block_queues_, buf_cnt++ % num_workers_, keys));
          keys.clear();
        }
      }
    }
    if (!keys.empty()) {
      RETURN_IF_NOT_OK(send_to_que(io_block_queues_, buf_cnt++ % num_workers_, keys));
    }
    // send the eoe
    RETURN_IF_NOT_OK(
      io_block_queues_[(buf_cnt++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
    RETURN_IF_NOT_OK(prefetch_queues_[(prefetch_cnt++) % num_prefetchers_]->Add(
      std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
    // If repeat but the not last repeat, wait for reset.
    if (!IsLastIteration()) {
      MS_LOG(DEBUG) << Name() << " Waiting for reset. Count " << wait_cnt << " Buffer sent " << buf_cnt;
    } else {
      // We can break out from the loop.
      break;
    }
    if (epoch_sync_flag_) {
      // If epoch_sync_flag_ is set, then master thread sleeps until all the worker threads have finished their job for
      // the current epoch.
      RETURN_IF_NOT_OK(WaitForWorkers());
    }
    // If not the last repeat, self-reset and go to loop again.
    if (!IsLastIteration()) RETURN_IF_NOT_OK(Reset());
    UpdateRepeatAndEpochCounter();
  } while (true);
  // Flow the eof before exit
  RETURN_IF_NOT_OK(
    io_block_queues_[(buf_cnt++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof)));
  // Shutdown threads
  for (int32_t i = 0; i < num_workers_; i++) {
    RETURN_IF_NOT_OK(
      io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
  }
  // Dump the last epoch result (approximately) without waiting for the worker threads to come back.
  if (AllowCacheMiss()) {
    MS_LOG(INFO) << "Epoch: " << wait_cnt / op_num_repeats_per_epoch() << " Cache Miss : " << num_cache_miss_
                 << " Total number of rows : " << row_cnt_;
  }
  return Status::OK();
}

Status CacheBase::FetchFromCache(int32_t worker_id) {
  int64_t buffer_id = worker_id;
  std::unique_ptr<IOBlock> blk;
  do {
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&blk));
    if (blk->wait()) {
      // Sync io_block is a signal that master thread wants us to pause and sync with other workers.
      // The last guy who comes to this sync point should reset the counter and wake up the master thread.
      if (++num_workers_paused_ == num_workers_) {
        wait_for_workers_post_.Set();
      }
    } else if (blk->eof()) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF)));
    } else if (blk->eoe()) {
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
      for (auto row_id : keys) {
        TensorRow row;
        // Block until the row shows up in the pool.
        RETURN_IF_NOT_OK(GetPrefetchRow(row_id, &row));
        if (row.empty()) {
          if (AllowCacheMiss()) {
            ++num_cache_miss_;
          } else {
            std::string errMsg = "Row id " + std::to_string(row_id) + " not found.";
            RETURN_STATUS_UNEXPECTED(errMsg);
          }
        }
        que->push_back(std::move(row));
      }
      db->set_tensor_table(std::move(que));
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::move(db)));
      buffer_id += num_workers_;
    }
  } while (true);
  return Status::OK();
}

Status CacheBase::RegisterResources() {
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(prefetch_queues_.Register(tree_->AllTasks()));
  return Status::OK();
}

CacheBase::~CacheBase() = default;

Status CacheBase::UpdateColumnMapFromCache() {
  Status rc;
  // Get the schema from the server. It may not be there yet. So tolerate the error.
  if (column_name_id_map_.empty()) {
    rc = cache_client_->FetchSchema(&column_name_id_map_);
    if (rc == Status(StatusCode::kMDFileNotExist)) {
      MS_LOG(DEBUG) << "Schema not in the server yet.";
      rc = Status::OK();
    }
  }
  return rc;
}

Status CacheBase::GetPrefetchRow(row_id_type row_id, TensorRow *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  CHECK_FAIL_RETURN_UNEXPECTED(row_id >= 0, "Expect positive row id");
  RETURN_IF_NOT_OK(prefetch_.PopFront(row_id, out));
  return Status::OK();
}

Status CacheBase::PrefetchRows(const std::vector<row_id_type> &keys, std::vector<row_id_type> *cache_miss) {
  RETURN_UNEXPECTED_IF_NULL(cache_miss);
  std::vector<row_id_type> prefetch_keys;
  prefetch_keys.reserve(keys.size());

  // Filter out all those keys that unlikely we will find at the server
  for (auto row_id : keys) {
    if (cache_client_->KeyIsCacheMiss(row_id)) {
      // Just put an empty row in the cache.
      TensorRow row;
      row.setId(row_id);
      RETURN_IF_NOT_OK(prefetch_.Add(row_id, std::move(row)));
      cache_miss->push_back(row_id);
    } else {
      prefetch_keys.push_back(row_id);
    }
  }
  // Early exit if nothing to fetch
  if (prefetch_keys.empty()) {
    return Status::OK();
  }
  // Get the rows from the server
  TensorTable ttbl;
  RETURN_IF_NOT_OK(cache_client_->GetRows(prefetch_keys, &ttbl));
  auto row_it = ttbl.begin();
  for (auto row_id : prefetch_keys) {
    auto &row = *row_it;
    if (row.empty()) {
      cache_miss->push_back(row_id);
    }
    // Put the prefetch row into the pool and wake up any WorkerEntry to wait for the row
    RETURN_IF_NOT_OK(prefetch_.Add(row_id, std::move(row)));
    ++row_it;
  }
  return Status::OK();
}

Status CacheBase::Prefetcher(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::vector<row_id_type> prefetch_keys;
  prefetch_keys.reserve(prefetch_size_);
  std::vector<row_id_type> cache_miss;
  cache_miss.reserve(prefetch_size_);
  do {
    prefetch_keys.clear();
    cache_miss.clear();
    std::unique_ptr<IOBlock> blk;
    RETURN_IF_NOT_OK(prefetch_queues_[worker_id]->PopFront(&blk));
    CHECK_FAIL_RETURN_UNEXPECTED(!blk->eof(), "Expect eoe or a regular io block");
    if (!blk->eoe()) {
      RETURN_IF_NOT_OK(blk->GetKeys(&prefetch_keys));
      Status rc;
      const int32_t max_retries = 5;
      int32_t retry_count = 0;
      do {
        rc = PrefetchRows(prefetch_keys, &cache_miss);
        if (rc == StatusCode::kMDNetWorkError && retry_count < max_retries) {
          // If we get some network error, we will attempt some retries
          retry_count++;
        } else if (rc.IsError() && rc.StatusCode() != StatusCode::kMDInterrupted) {
          MS_LOG(WARNING) << rc.ToString();
          return rc;
        }
      } while (rc == StatusCode::kMDNetWorkError);
      // In case any thread is waiting for the rows to come back and blocked on a semaphore,
      // we will put an empty row in the local cache.
      if (rc.IsError() && AllowCacheMiss()) {
        for (auto row_id : prefetch_keys) {
          TensorRow row;
          row.setId(row_id);
          RETURN_IF_NOT_OK(prefetch_.Add(row_id, std::move(row)));
          cache_miss.push_back(row_id);
        }
      }
    } else {
      if (AllowCacheMiss()) {
        // This code path is for CacheLookupOp acting as a sampler. If we get a eoe from
        // a sampler, send a eoe to physical leaf op as well.
        cache_miss.push_back(eoe_row_id);
      }
    }
    if (AllowCacheMiss()) {
      // Because of the way connector works, we push unconditionally even cache_miss can be empty.
      RETURN_IF_NOT_OK(keys_miss_->Push(worker_id, cache_miss));
    }
  } while (true);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
