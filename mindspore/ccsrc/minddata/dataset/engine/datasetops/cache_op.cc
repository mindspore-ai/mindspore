/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/cache_op.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
// Constructor of CacheOp
CacheOp::CacheOp(int32_t num_workers, int32_t op_connector_size, std::shared_ptr<CacheClient> cache_client,
                 std::shared_ptr<SamplerRT> sampler)
    : CacheBase(num_workers, op_connector_size, std::move(cache_client), std::move(sampler)),
      num_guys_in_(0),
      phase_(Phase::kBuildPhase) {}

// Destructor
CacheOp::~CacheOp() = default;

// This class functor will provide the master loop that drives the logic for performing the work
Status CacheOp::operator()() {
  RETURN_UNEXPECTED_IF_NULL(tree_);
  if (!sampler_) {
    RETURN_STATUS_UNEXPECTED("Invalid sampler, CacheOp requires a sampler before it can be executed, but got nullptr.");
  }
  RETURN_IF_NOT_OK(RegisterResources());

  // required task group sync after launching workers
  TaskManager::FindMe()->Post();
  // Wait for the workers to finish caching the rows.
  RETURN_IF_NOT_OK(WaitForCachingAllRows());
  // Current repeats and current epochs may have increased when caching all rows with DatasetOp::GetNextInput.
  // But they shouldn't be increased because now cache op is starting to act as a leaf and its epoch hasn't started.
  op_current_repeats_ = 0;
  op_current_epochs_ = 0;
  RETURN_IF_NOT_OK(FetchSamplesToWorkers());
  return Status::OK();
}

Status CacheOp::CacheAllRows(int32_t worker_id) {
  // If the current phase is to fill the cache, do it then.
  if (phase_ == Phase::kBuildPhase) {
    MS_LOG(INFO) << "CacheOp first epoch SAVE mode started. Worker: " << worker_id;
    // SAVE mode loop
    TensorRow row;
    RETURN_IF_NOT_OK(cache_workers_in_queue_[worker_id]->PopFront(&row));
    while (!row.eof()) {
      if (!row.eoe()) {
        Status rc;
        // Do the Async write if we attach to the shared memory.
        rc = cache_client_->AsyncWriteRow(row);
        if (rc.StatusCode() == StatusCode::kMDNotImplementedYet) {
          RETURN_IF_NOT_OK(cache_client_->WriteRow(row));
        } else if (rc.IsError()) {
          return rc;
        }
      } else {
        // In a repeat-over-cache scenario, any of the "real" leaf operators below us have been set up
        // as non-repeating leaf ops.  As such, they only do one epoch and then quit.  Since we got the
        // the eoe to indicate the end of the epoch, we should next expect to get the eof.
        // Drain this eof so that we don't leave it sitting there on a connector that we'll never fetch
        // from again.
        RETURN_IF_NOT_OK(cache_workers_in_queue_[worker_id]->PopFront(&row));
        if (!row.eof()) {
          RETURN_STATUS_UNEXPECTED("[Internal ERROR] Cache op expects to get an eof after eoe from child.");
        }
        break;
      }
      RETURN_IF_NOT_OK(cache_workers_in_queue_[worker_id]->PopFront(&row));
    }
  }
  // Let the main guy know we are done.
  auto last_guy_in = num_guys_in_.fetch_add(1);
  if ((last_guy_in + 1) == num_workers_) {
    rows_cache_done_.Set();
  } else {
    // Let's do a sync up here.
    RETURN_IF_NOT_OK(rows_cache_done_.Wait());
  }
  return Status::OK();
}
Status CacheOp::WaitForCachingAllRows() {
  // Fetch all rows and wait for workers to cache them
  if (phase_ == Phase::kBuildPhase) {
    // We will take the chance to cache the schema at the server.
    RETURN_IF_NOT_OK(cache_client_->CacheSchema(column_name_id_map()));
    // SAVE mode loop
    TensorRow new_row;
    auto child_iterator = std::make_unique<ChildIterator>(this, 0, 0);
    int64_t ctr = 0;
    do {
      RETURN_IF_NOT_OK(child_iterator->FetchNextTensorRow(&new_row));
      RETURN_IF_NOT_OK(cache_workers_in_queue_[ctr++ % num_workers_]->EmplaceBack(std::move(new_row)));
    } while (!new_row.eof());

    for (int32_t i = 1; i < num_workers_; i++) {
      RETURN_IF_NOT_OK(cache_workers_in_queue_[ctr++ % num_workers_]->EmplaceBack(TensorRow(TensorRow::kFlagEOF)));
    }
  }

  // Wait for the workers to finish caching the rows.
  RETURN_IF_NOT_OK(rows_cache_done_.Wait());
  // Move from build phase to fetch phase if we are the one to fill the cache
  if (phase_ == Phase::kBuildPhase) {
    RETURN_IF_NOT_OK(cache_client_->FlushAsyncWriteBuffer());  // One more flush
    RETURN_IF_NOT_OK(cache_client_->BuildPhaseDone());
    // Move to the next phase
    phase_ = Phase::kFetchPhase;
  }
  // If we are not the one to create the cache,
  // wait until the state changed from build phase to fetch base.
  bool BuildPhaseDone = true;
  do {
    int8_t out;
    RETURN_IF_NOT_OK(cache_client_->GetState(&out));
    auto state = static_cast<CacheServiceState>(out);
    switch (state) {
      case CacheServiceState::kBuildPhase:
        // Do nothing. Continue to wait.
        BuildPhaseDone = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(kPhaseCheckIntervalInMilliSec));
        break;
      case CacheServiceState::kFetchPhase:
        BuildPhaseDone = true;
        break;
      case CacheServiceState::kOutOfMemory:
        RETURN_STATUS_OOM("Out of memory.");
      case CacheServiceState::kNoSpace:
        RETURN_STATUS_ERROR(StatusCode::kMDNoSpace,
                            "Cache server is running of out spill storage, check memory usage.");
      case CacheServiceState::kNone:
      case CacheServiceState::kError:
      default:
        RETURN_STATUS_UNEXPECTED("Unexpected Cache server state: " + std::to_string(out));
    }
  } while (!BuildPhaseDone);
  // Get statistics from the server, and if we are not the one to create the cache,
  // wait until the state changed from build phase to fetch base.
  CacheServiceStat stat{};
  RETURN_IF_NOT_OK(cache_client_->GetStat(&stat));
  const row_id_type min_key = stat.min_row_id;
  const row_id_type max_key = stat.max_row_id;
  num_rows_ = max_key - min_key + 1;
  MS_LOG(INFO) << "Number of rows cached: " << num_rows_;
  MS_LOG(INFO) << "Number of rows cached in memory : " << stat.num_mem_cached;
  MS_LOG(INFO) << "Number of rows spilled to disk : " << stat.num_disk_cached;
  MS_LOG(INFO) << "Average cache size : " << stat.avg_cache_sz;
  // Now all rows are cached and we have done a sync point check up. Next phase is
  // is pick up fetch input from sampler and pass up to the caller.
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}
Status CacheOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(CacheAllRows(worker_id));
  RETURN_IF_NOT_OK(FetchFromCache(worker_id));
  return Status::OK();
}
Status CacheOp::RegisterResources() {
  RETURN_UNEXPECTED_IF_NULL(tree_);
  cache_workers_in_queue_.Init(num_workers_, oc_queue_size_);
  RETURN_IF_NOT_OK(cache_workers_in_queue_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(CacheBase::RegisterResources());
  RETURN_IF_NOT_OK(rows_cache_done_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(keys_miss_->Register(tree_->AllTasks()));
  return Status::OK();
}

// Base-class override for special eoe handler.
// CacheOp must override this because it shall not perform default handling of eoe. Instead
// the CacheOp manages actions related to the end of the epoch.
Status CacheOp::EoeReceived(int32_t worker_id) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}
// Base-class override for handling cases when an eof is received.
Status CacheOp::EofReceived(int32_t worker_id) {
  // eofReceived is overloaded because we want to manually handle this eof.
  // Specifically, the default behavior is to pack it and flow it up to the next connection.
  // In this case, we want a no-op behavior so that we can perform correct action.
  return Status::OK();
}

Status CacheOp::PrepareOperator() {
  // Run any common code from super class first before adding our own
  RETURN_IF_NOT_OK(DatasetOp::PrepareOperator());
  // Get the computed check sum from all ops in our cache path below us and ask the cache op to create it's cache
  uint32_t cache_crc = DatasetOp::GenerateCRC(shared_from_this());
  // This is a non-mappable cache op so the id's need to be generated.
  // Construct the cache
  const bool generate_ids = true;
  Status rc = cache_client_->CreateCache(cache_crc, generate_ids);
  if (rc.StatusCode() == StatusCode::kMDDuplicateKey) {
    // We are told the cache has been created already. So we skip the build phase.
    phase_ = Phase::kFetchPhase;
    rc = Status::OK();
  }
  RETURN_IF_NOT_OK(rc);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
