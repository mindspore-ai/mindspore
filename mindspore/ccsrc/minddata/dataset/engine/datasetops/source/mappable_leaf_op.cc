/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
MappableLeafOp::MappableLeafOp(int32_t num_wkrs, int32_t queue_size, std::shared_ptr<SamplerRT> sampler)
    : ParallelOp(num_wkrs, queue_size, std::move(sampler)) {}

// Main logic, Register Queue with TaskGroup, launch all threads and do the functor's work
Status MappableLeafOp::operator()() {
  RETURN_IF_NOT_OK(LaunchThreadsAndInitOp());
  TensorRow sample_row;
  RETURN_IF_NOT_OK(sampler_->GetNextSample(&sample_row));
  int64_t row_cnt = 0;
  while (true) {  // each iteration is 1 epoch, breaks when IsLastIteration() is true
    while (sample_row.eoe() == false) {
      std::shared_ptr<Tensor> sample_ids = sample_row[0];
      for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); ++itr) {
        if ((*itr) >= num_rows_) {
          MS_LOG(WARNING) << "Skipping sample with ID: " << *itr << " since it is out of bound: " << num_rows_;
          continue;  // index out of bound, skipping
        }
        RETURN_IF_NOT_OK(
          io_block_queues_[row_cnt++ % num_workers_]->Add(std::make_unique<IOBlock>(*itr, IOBlock::kDeIoBlockNone)));
      }
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sample_row));
    }
    if (IsLastIteration()) {
      std::unique_ptr<IOBlock> eoe_block = std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe);
      std::unique_ptr<IOBlock> eof_block = std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof);
      RETURN_IF_NOT_OK(io_block_queues_[(row_cnt++) % num_workers_]->Add(std::move(eoe_block)));
      RETURN_IF_NOT_OK(io_block_queues_[(row_cnt++) % num_workers_]->Add(std::move(eof_block)));
      for (int32_t i = 0; i < num_workers_; ++i) {
        RETURN_IF_NOT_OK(
          io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
      }
      return Status::OK();
    } else {  // not the last repeat.
      RETURN_IF_NOT_OK(
        io_block_queues_[(row_cnt++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
    }

    if (epoch_sync_flag_) {
      // If epoch_sync_flag_ is set, then master thread sleeps until all the worker threads have finished their job for
      // the current epoch.
      RETURN_IF_NOT_OK(WaitForWorkers());
    }
    // If not the last repeat, self-reset and go to loop again.
    if (!IsLastIteration()) {
      RETURN_IF_NOT_OK(Reset());
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sample_row));
    }
    UpdateRepeatAndEpochCounter();
  }
}

// Reset Sampler and wakeup Master thread (functor)
Status MappableLeafOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  RETURN_IF_NOT_OK(sampler_->ResetSampler());
  return Status::OK();
}

// hand shake with Sampler, allow Sampler to call RandomAccessOp's functions to get NumRows
Status MappableLeafOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}

// contains the main logic of pulling a IOBlock from IOBlockQueue, load a row and push the row to out_connector_
// IMPORTANT: 1 IOBlock produces 1 row
Status MappableLeafOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::unique_ptr<IOBlock> io_block;
  RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  while (io_block != nullptr) {
    if (io_block->wait()) {
      // Sync io_block is a signal that master thread wants us to pause and sync with other workers.
      // The last guy who comes to this sync point should reset the counter and wake up the master thread.
      if (++num_workers_paused_ == num_workers_) {
        wait_for_workers_post_.Set();
      }
    } else if (io_block->eoe()) {
      RETURN_IF_NOT_OK(out_connector_->SendEOE(worker_id));
    } else if (io_block->eof()) {
      RETURN_IF_NOT_OK(out_connector_->SendEOF(worker_id));
    } else {
      std::vector<int64_t> keys;
      RETURN_IF_NOT_OK(io_block->GetKeys(&keys));
      if (keys.empty()) return Status::OK();  // empty key is a quit signal for workers
      TensorRow trow;
      RETURN_IF_NOT_OK(this->LoadTensorRow(keys[0], &trow));
      RETURN_IF_NOT_OK(out_connector_->Add(std::move(trow), worker_id));
    }
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  }
  RETURN_STATUS_UNEXPECTED("[Internal ERROR] Unexpected nullptr received in worker.");
}
}  // namespace dataset
}  // namespace mindspore
