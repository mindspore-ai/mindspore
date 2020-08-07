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
    : ParallelOp(num_workers, op_connector_size, sampler),
      cache_client_(cache_client),
      rows_per_buffer_(rows_per_buf),
      // We can cause deadlock if this internal Connector size is too small.
      keys_miss_(num_workers_, 1, connector_capacity_) {
  io_block_queues_.Init(num_workers, op_connector_size);
}
// Common function to fetch samples from the sampler and send them using the io_block_queues to
// the parallel workers
Status CacheBase::FetchSamplesToWorkers() {
  int64_t buf_cnt = 0;
  int64_t wait_cnt = 0;
  do {
    epoch_sync_.Clear();
    std::vector<row_id_type> keys;
    int64_t row_cnt = 0;
    keys.reserve(rows_per_buffer_);
    std::unique_ptr<DataBuffer> sampler_buffer;
    RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    while (!sampler_buffer->eoe()) {
      TensorRow sample_row;
      RETURN_IF_NOT_OK(sampler_buffer->PopRow(&sample_row));
      std::shared_ptr<Tensor> sample_ids = sample_row[0];
      for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); itr++) {
        keys.push_back(*itr);
        ++row_cnt;
        if (row_cnt % rows_per_buffer_ == 0) {
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
      MS_LOG(DEBUG) << Name() << " Waiting for reset. Count " << ++wait_cnt << " Buffer sent " << buf_cnt;
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
  // Ask all the workers to quit.
  for (int32_t i = 0; i < num_workers_; i++) {
    RETURN_IF_NOT_OK(
      io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
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
      TensorTable ttbl;
      RETURN_IF_NOT_OK(cache_client_->GetRows(keys, &ttbl));
      auto row_it = ttbl.begin();
      std::vector<row_id_type> cache_miss;
      cache_miss.reserve(keys.size());
      for (auto row_id : keys) {
        auto &row = *row_it;
        if (row.empty()) {
          if (AllowCacheMiss()) {
            cache_miss.push_back(row_id);
          } else {
            std::string errMsg = "Row id " + std::to_string(row_id) + " not found.";
            RETURN_STATUS_UNEXPECTED(errMsg);
          }
        }
        que->push_back(std::move(row));
        ++row_it;
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
  return Status::OK();
}
CacheBase::~CacheBase() {}
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
}  // namespace dataset
}  // namespace mindspore
