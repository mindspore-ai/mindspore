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
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <utility>
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/system_pool.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
CacheMergeOp::~CacheMergeOp() = default;
void CacheMergeOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\n\n";
  }
}

CacheMergeOp::CacheMergeOp(int32_t numWorkers, int32_t opConnectorSize, int32_t numCleaners,
                           std::shared_ptr<CacheClient> cache_client, const std::shared_ptr<SamplerRT> &sampler)
    : ParallelOp(numWorkers, opConnectorSize, sampler),
      num_cleaners_(numCleaners),
      cache_client_(std::move(cache_client)),
      cache_missing_rows_(true) {}

Status CacheMergeOp::operator()() {
  // A queue of row id to let cleaner send cache miss rows to the cache server
  // We don't want a small queue as this will block the parallel op workers.
  // A row id is 8 byte integer. So bigger size doesn't consume a lot of memory.
  static const int32_t queue_sz = 512;
  io_que_ = std::make_unique<Queue<row_id_type>>(queue_sz);
  RETURN_IF_NOT_OK(io_que_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(
    num_workers_, std::bind(&CacheMergeOp::WorkerEntry, this, std::placeholders::_1), Name() + "::WorkerEntry", id()));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_,
                                        std::bind(&CacheMergeOp::CacheMissWorkerEntry, this, std::placeholders::_1),
                                        Name() + "::CacheMissWorkerEntry", id()));
  // One dedicated thread to move TensorRow from the pool to the cache server
  for (auto i = 0; i < num_cleaners_; ++i) {
    RETURN_IF_NOT_OK(
      tree_->AllTasks()->CreateAsyncTask("Cleaner", std::bind(&CacheMergeOp::Cleaner, this), nullptr, id()));
  }
  TaskManager::FindMe()->Post();
  return Status::OK();
}

// Each parallel worker will pop from the CacheHit stream. If there is a missing TensorRow, we will wait
// until it shows up in the pool.
Status CacheMergeOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::shared_ptr<DatasetOp> cache_hit_stream = child_[kCacheHitChildIdx];
  std::unique_ptr<DataBuffer> db_ptr;
  RETURN_IF_NOT_OK(cache_hit_stream->GetNextBuffer(&db_ptr, worker_id));
  while (!db_ptr->eof()) {
    if (db_ptr->eoe()) {
      RETURN_IF_NOT_OK(EoeReceived(worker_id));
      db_ptr.reset();
      RETURN_IF_NOT_OK(cache_hit_stream->GetNextBuffer(&db_ptr, worker_id));
    } else {
      // See if there is any missing row
      auto tbl = std::make_unique<TensorQTable>();
      while (db_ptr->NumRows() > 0) {
        TensorRow row;
        RETURN_IF_NOT_OK(db_ptr->PopRow(&row));
        if (row.empty()) {
          auto row_id = row.getId();
          // Block until the row shows up in the pool.
          RETURN_IF_NOT_OK(cache_miss_.PopFront(row_id, &row));
        }
        tbl->push_back(std::move(row));
      }
      db_ptr->set_tensor_table(std::move(tbl));
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::move(db_ptr)));
      RETURN_IF_NOT_OK(cache_hit_stream->GetNextBuffer(&db_ptr, worker_id));
    }
  }
  RETURN_IF_NOT_OK(EofReceived(worker_id));
  return Status::OK();
}

Status CacheMergeOp::CacheMissWorkerEntry(int32_t workerId) {
  TaskManager::FindMe()->Post();
  // We will simply pop TensorRow from the stream and insert them into the pool and
  // wake up any worker that is awaiting on the missing TensorRow.
  // If we see an eoe, ignore it. For eof, we exit.
  std::shared_ptr<DatasetOp> cache_missing_stream = child_[kCacheMissChildIdx];
  // Before we start, cache the schema at the server. Pick one of the workers
  // do it. The schema should have been done at prepare time.
  if (workerId == 0) {
    RETURN_IF_NOT_OK(cache_client_->CacheSchema(column_name_id_map()));
  }
  std::unique_ptr<DataBuffer> db_ptr;
  RETURN_IF_NOT_OK(cache_missing_stream->GetNextBuffer(&db_ptr, workerId));
  while (!db_ptr->eof()) {
    if (db_ptr->eoe()) {
      // Ignore it.
      MS_LOG(DEBUG) << "Ignore eoe";
      // However we need to flush any left over from the async write buffer. But any error
      // we are getting will just to stop caching but the pipeline will continue
      Status rc;
      if ((rc = cache_client_->FlushAsyncWriteBuffer()).IsError()) {
        cache_missing_rows_ = false;
        if (rc == StatusCode::kMDOutOfMemory || rc == kMDNoSpace) {
          cache_client_->ServerRunningOutOfResources();
        } else {
          MS_LOG(INFO) << "Async row flushing not successful: " << rc.ToString();
        }
      }
    } else {
      while (db_ptr->NumRows() > 0) {
        TensorRow row;
        RETURN_IF_NOT_OK(db_ptr->PopRow(&row));
        row_id_type row_id = row.getId();
        if (row_id < 0) {
          std::string errMsg = "Expect positive row id: " + std::to_string(row_id);
          RETURN_STATUS_UNEXPECTED(errMsg);
        }
        if (cache_missing_rows_) {
          // Technically number of this row shows up in the cache miss stream is equal to the number
          // of P() call. However the cleaner wants it too. So we need an extra copy.
          TensorRowCacheRequest *rq;
          RETURN_IF_NOT_OK(GetRq(row_id, &rq));
          if (rq->GetState() == TensorRowCacheRequest::State::kEmpty) {
            // We will send the request async. But any error we most
            // likely ignore and continue.
            Status rc;
            rc = rq->AsyncSendCacheRequest(cache_client_, row);
            if (rc.IsOk()) {
              RETURN_IF_NOT_OK(io_que_->EmplaceBack(row_id));
            } else if (rc == StatusCode::kMDOutOfMemory || rc == kMDNoSpace) {
              cache_missing_rows_ = false;
              cache_client_->ServerRunningOutOfResources();
            }
          }
        }
        RETURN_IF_NOT_OK(cache_miss_.Add(row_id, std::move(row)));
      }
    }
    RETURN_IF_NOT_OK(cache_missing_stream->GetNextBuffer(&db_ptr, workerId));
  }
  return Status::OK();
}

Status CacheMergeOp::Cleaner() {
  TaskManager::FindMe()->Post();
  while (true) {
    row_id_type row_id;
    RETURN_IF_NOT_OK(io_que_->PopFront(&row_id));
    if (row_id < 0) {
      break;
    }
    // Locate the cache request
    TensorRowCacheRequest *rq;
    RETURN_IF_NOT_OK(GetRq(row_id, &rq));
    // If already flushed, move on to the next one.
    if (rq->GetState() == TensorRowCacheRequest::State::kClean) {
      continue;
    }
    Status rc = rq->CheckCacheResult();
    if (rc.IsError()) {
      // If interrupt, time to quit.
      if (rc == StatusCode::kMDInterrupted) {
        return Status::OK();
      } else if (rc == StatusCode::kMDOutOfMemory || rc == kMDNoSpace) {
        // The server is hitting some limit and we will turn off caching from now on.
        cache_missing_rows_ = false;
        cache_client_->ServerRunningOutOfResources();
      } else {
        MS_LOG(INFO) << "Cache row not successful: " << rc.ToString();
        // Bad rc should not bring down the pipeline. We will simply continue and
        // change the state back to empty. We don't need a CAS from CLEAN back to EMPTY.
        rq->SetState(TensorRowCacheRequest::State::kEmpty);
      }
    }
  }
  return Status::OK();
}

Status CacheMergeOp::PrepareOperator() {  // Run any common code from super class first before adding our own
                                          // specific logic
  CHECK_FAIL_RETURN_UNEXPECTED(child_.size() == 2, "Incorrect number of children");
  RETURN_IF_NOT_OK(DatasetOp::PrepareOperator());
  // Get the computed check sum from all ops in the cache miss class
  uint32_t cache_crc = DatasetOp::GenerateCRC(child_[kCacheMissChildIdx]);
  // This is a mappable cache op so the id's need to be generated.
  // Construct the cache
  const bool generate_ids = false;
  Status rc = cache_client_->CreateCache(cache_crc, generate_ids);
  if (rc.StatusCode() == StatusCode::kMDDuplicateKey) {
    // We are told the cache has been created already.
    MS_LOG(INFO) << "Cache created already";
    rc = Status::OK();
  }
  RETURN_IF_NOT_OK(rc);
  return Status::OK();
}

Status CacheMergeOp::ComputeColMap() {
  CHECK_FAIL_RETURN_UNEXPECTED(child_[kCacheMissChildIdx] != nullptr, "Invalid data, cache miss stream empty.");
  if (column_name_id_map().empty()) {
    column_name_id_map_ = child_[kCacheMissChildIdx]->column_name_id_map();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!column_name_id_map().empty(), "Invalid data, column_name_id_map is empty.");
  return Status::OK();
}

// Builder constructor. Creates the builder object.
CacheMergeOp::Builder::Builder() : build_cache_client_(nullptr), build_sampler_(nullptr) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  build_num_workers_ = cfg->num_parallel_workers();
  build_op_connector_size_ = cfg->op_connector_size();
  build_num_cleaners_ = cfg->num_parallel_workers();
}

// Check if the required parameters are set by the builder.
Status CacheMergeOp::Builder::SanityCheck() const {
  if (build_cache_client_ == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "Invalid parameter, CacheMergeOp requires a CacheClient, but got nullptr.");
  }
  // Make sure the cache client has a valid session
  if (!build_cache_client_->session_id()) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "Invalid parameter, cache client for CacheMergeOp requires a session id which is not equal to 0.");
  }
  return Status::OK();
}

// The builder "build" method creates the final object and does some init on it
Status CacheMergeOp::Builder::Build(std::shared_ptr<CacheMergeOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<CacheMergeOp>(build_num_workers_, build_op_connector_size_, build_num_cleaners_,
                                        build_cache_client_, build_sampler_);
  return Status::OK();
}

Status CacheMergeOp::EoeReceived(int32_t worker_id) {
  // Send the eoe up.
  MS_LOG(DEBUG) << "Cache merge sending eoe";
  return DatasetOp::EoeReceived(worker_id);
}

// Base-class override for handling cases when an eof is received.
Status CacheMergeOp::EofReceived(int32_t worker_id) {
  // Send the eof up.
  MS_LOG(DEBUG) << "Cache merge sending eof";
  return DatasetOp::EofReceived(worker_id);
}

Status CacheMergeOp::GetRq(row_id_type row_id, CacheMergeOp::TensorRowCacheRequest **out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  std::unique_lock<std::mutex> lock(mux_);
  auto it = io_request_.find(row_id);
  if (it != io_request_.end()) {
    *out = it->second.GetMutablePointer();
  } else {
    // We will create a new one.
    auto alloc = SystemPool::GetAllocator<TensorRowCacheRequest>();
    auto r = io_request_.emplace(row_id, MemGuard<TensorRowCacheRequest, Allocator<TensorRowCacheRequest>>(alloc));
    if (r.second) {
      auto &mem = r.first->second;
      RETURN_IF_NOT_OK(mem.allocate(1));
      *out = mem.GetMutablePointer();
    } else {
      RETURN_STATUS_UNEXPECTED("Invalid data, map insert fail.");
    }
  }
  return Status::OK();
}

Status CacheMergeOp::TensorRowCacheRequest::AsyncSendCacheRequest(const std::shared_ptr<CacheClient> &cc,
                                                                  const TensorRow &row) {
  auto expected = State::kEmpty;
  if (st_.compare_exchange_strong(expected, State::kDirty)) {
    // We will do a deep copy but write directly into CacheRequest protobuf or shared memory
    Status rc;
    rc = cc->AsyncWriteRow(row);
    if (rc.StatusCode() == StatusCode::kMDNotImplementedYet) {
      cleaner_copy_ = std::make_shared<CacheRowRequest>(cc.get());
      rc = cleaner_copy_->SerializeCacheRowRequest(cc.get(), row);
      if (rc.IsOk()) {
        // Send the request async. The cleaner will check the return code.
        rc = cc->PushRequest(cleaner_copy_);
      }
    } else if (rc.IsOk()) {
      // Set the state to clean even though it still sits in the cache client async buffer.
      // The cleaner will then ignore it once the state is clean.
      st_ = State::kClean;
    }
    if (rc.IsError()) {
      // Clean up the shared pointer and reset the state back to empty
      cleaner_copy_.reset();
      st_ = State::kEmpty;
    }
    return rc;
  }
  return Status::OK();
}

Status CacheMergeOp::TensorRowCacheRequest::CheckCacheResult() {
  auto expected = State::kDirty;
  if (st_.compare_exchange_strong(expected, State::kClean)) {
    // Success or not, we will release the memory.
    // We simply move it out of the structure and let it go out of scope.
    auto cache_request = std::move(cleaner_copy_);
    RETURN_IF_NOT_OK(cache_request->Wait());
    return Status::OK();
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
