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
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/engine/execution_tree.h"
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
                           std::shared_ptr<CacheClient> cache_client, const std::shared_ptr<Sampler> &sampler)
    : ParallelOp(numWorkers, opConnectorSize, sampler), num_cleaners_(numCleaners), cache_client_(cache_client) {}
Status CacheMergeOp::operator()() {
  // A queue of row id to let cleaner send cache miss rows to the cache server
  // We don't want a small queue as this will block the parallel op workers.
  // A row id is 8 byte integer. So bigger size doesn't consume a lot of memory.
  static const int32_t queue_sz = 512;
  io_que_ = std::make_unique<Queue<row_id_type>>(queue_sz);
  RETURN_IF_NOT_OK(io_que_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&CacheMergeOp::WorkerEntry, this, std::placeholders::_1)));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&CacheMergeOp::CacheMissWorkerEntry, this, std::placeholders::_1)));
  // One dedicated thread to move TensorRow from the pool to the cache server
  for (auto i = 0; i < num_cleaners_; ++i) {
    RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask("Cleaner", std::bind(&CacheMergeOp::Cleaner, this)));
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
          TensorRowRequest *rq = nullptr;
          RETURN_IF_NOT_OK(GetRq(row_id, &rq));
          // Block until the row shows up in the pool.
          RETURN_IF_NOT_OK(rq->Wait(&row));
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
    } else {
      while (db_ptr->NumRows() > 0) {
        TensorRow row;
        RETURN_IF_NOT_OK(db_ptr->PopRow(&row));
        row_id_type row_id = row.getId();
        if (row_id < 0) {
          std::string errMsg = "Expect positive row id: " + std::to_string(row_id);
          RETURN_STATUS_UNEXPECTED(errMsg);
        }
        TensorRowRequest *rq = nullptr;
        RETURN_IF_NOT_OK(GetRq(row_id, &rq));
        rq->WakeUpAny(std::move(row));
        // Let the cleaner to flush out this row (async) to the cache server.
        RETURN_IF_NOT_OK(io_que_->EmplaceBack(row_id));
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
    TensorRowRequest *rq = nullptr;
    RETURN_IF_NOT_OK(GetRq(row_id, &rq));
    if (rq->GetState() == TensorRowRequest::State::kClean) {
      // If already flushed, move on to the next one.
      continue;
    }
    TensorRow row;
    RETURN_IF_NOT_OK(rq->Release(&row));
    CHECK_FAIL_RETURN_UNEXPECTED(!row.empty(), "Programming error.");
    Status rc = cache_client_->WriteRow(row);
    // Bad rc should not bring down the pipeline
    if (rc.IsError()) {
      MS_LOG(WARNING) << "Cache not successful." << rc.ToString();
    }
    rq->SetState(TensorRowRequest::State::kClean);
  }
  return Status::OK();
}

Status CacheMergeOp::GetRq(row_id_type row_id, CacheMergeOp::TensorRowRequest **out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  std::unique_lock<std::mutex> lck(mux_);
  auto it = cache_miss_map_.find(row_id);
  if (it != cache_miss_map_.end()) {
    *out = it->second.GetMutablePointer();
  } else {
    // We will create a new one.
    auto alloc = Services::GetAllocator<TensorRowRequest>();
    auto r = cache_miss_map_.emplace(row_id, MemGuard<TensorRowRequest, Allocator<TensorRowRequest>>(alloc));
    if (r.second) {
      auto &mem = r.first->second;
      RETURN_IF_NOT_OK(mem.allocate(1, row_id));
      *out = mem.GetMutablePointer();
    } else {
      RETURN_STATUS_UNEXPECTED("Map insert fail.");
    }
  }
  return Status::OK();
}
Status CacheMergeOp::PrepareNodePostAction() {  // Run any common code from super class first before adding our own
                                                // specific logic
  CHECK_FAIL_RETURN_UNEXPECTED(child_.size() == 2, "Incorrect number of children");
  RETURN_IF_NOT_OK(ParallelOp::PrepareNodePostAction());
  // Get the computed check sum from all ops in the cache miss class
  uint32_t cache_crc = DatasetOp::GenerateCRC(child_[kCacheMissChildIdx]);
  // This is a mappable cache op so the id's need to be generated.
  // Construct the cache
  const bool generate_ids = false;
  Status rc = cache_client_->CreateCache(cache_crc, generate_ids);
  if (rc.get_code() == StatusCode::kDuplicateKey) {
    // We are told the cache has been created already.
    MS_LOG(INFO) << "Cache created already";
    rc = Status::OK();
  }
  RETURN_IF_NOT_OK(rc);
  return Status::OK();
}
Status CacheMergeOp::ComputeColMap() {
  CHECK_FAIL_RETURN_UNEXPECTED(child_[kCacheMissChildIdx] != nullptr, "Cache miss stream empty");
  if (column_name_id_map().empty()) {
    column_name_id_map_ = child_[kCacheMissChildIdx]->column_name_id_map();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!column_name_id_map().empty(), "No column map detected");
  return Status::OK();
}
Status CacheMergeOp::TensorRowRequest::Wait(TensorRow *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  // Block until the missing row is in the pool.
  RETURN_IF_NOT_OK(use_count_.P());
  std::unique_lock<std::mutex> lck(dq_mux_);
  CHECK_FAIL_RETURN_UNEXPECTED(!row_.empty(), "Programming error");
  *out = std::move(row_.front());
  row_.pop_front();
  return Status::OK();
}
void CacheMergeOp::TensorRowRequest::WakeUpAny(TensorRow &&row) {
  std::unique_lock<std::mutex> lck(dq_mux_);
  // Technically number of this row shows up in the cache miss stream is equal to the number
  // of P() call. However the cleaner wants it too. So we need an extra copy.
  if (GetState() == State::kEmpty) {
    // We will do a deep copy
    for (auto &ts : row) {
      std::shared_ptr<Tensor> out_ts;
      Tensor::CreateFromTensor(ts, &out_ts);
      cleaner_copy_.push_back(out_ts);
    }
    cleaner_copy_.setId(row.getId());
    // Change the state to dirty
    SetState(State::kDirty);
  }
  row_.push_back(std::move(row));
  // Bump up the use count by 1. This wake up any parallel worker which is waiting
  // for this row.
  use_count_.V();
}
Status CacheMergeOp::TensorRowRequest::Release(TensorRow *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  // We are not holding any mutex here because the cleaner isn't really touching the deque row_.
  // In case we have multiple cleaners and they all see the copy, only one of them will
  // get it.
  auto expected = State::kDirty;
  if (st_.compare_exchange_strong(expected, State::kClean)) {
    *out = std::move(cleaner_copy_);
  }
  return Status::OK();
}
// Builder constructor. Creates the builder object.
CacheMergeOp::Builder::Builder() : build_cache_client_(nullptr), build_sampler_(nullptr) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  build_num_workers_ = cfg->num_parallel_workers();
  build_op_connector_size_ = cfg->op_connector_size();
  build_num_cleaners_ = 1;
}

// Check if the required parameters are set by the builder.
Status CacheMergeOp::Builder::SanityCheck() const {
  if (build_cache_client_ == nullptr) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "CacheMergeOp requires a CacheClient");
  }
  // Make sure the cache client has a valid session
  if (!build_cache_client_->session_id()) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "Cache client for CacheMergeOp is missing session id");
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

// Pre-Visitor accept method for NodePass
Status CacheMergeOp::PreAccept(NodePass *p, bool *modified) {
  // Downcast shared pointer then call the pre-visitation
  return p->PreRunOnNode(shared_from_base<CacheMergeOp>(), modified);
}

// Visitor accept method for NodePass
Status CacheMergeOp::Accept(NodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->RunOnNode(shared_from_base<CacheMergeOp>(), modified);
}

Status CacheMergeOp::EoeReceived(int32_t worker_id) {
  // If we are in a repeat path, send the eoe up.
  // Otherwise ignore it.
  if (op_total_repeats_ > 1) {
    return DatasetOp::EoeReceived(worker_id);
  }
  return Status::OK();
}

// Base-class override for handling cases when an eof is received.
Status CacheMergeOp::EofReceived(int32_t worker_id) {
  // If we are not in a repeated path, then the merge op gets a eof by itself, without first
  // getting an eoe.  However, the logic demands that all epochs close with an eoe first before eof.
  // Thus, generate an eoe first, before flowing up the eof in the non-repeated case. Base class
  // provides that for us.
  if (op_total_repeats_ == 1) {
    MS_LOG(DEBUG) << "Cache merge sending eoe";
    RETURN_IF_NOT_OK(DatasetOp::EoeReceived(worker_id));
  }
  MS_LOG(DEBUG) << "Cache merge sending eof";
  return DatasetOp::EofReceived(worker_id);
}
}  // namespace dataset
}  // namespace mindspore
