/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/engine/cache/cache_server.h"
#include "minddata/dataset/engine/cache/cache_service.h"
#include "minddata/dataset/engine/cache/cache_request.h"
#include "minddata/dataset/util/bit.h"

namespace mindspore {
namespace dataset {
Status CacheServer::DoServiceStart() {
  if (!top_.empty()) {
    Path spill(top_);
    RETURN_IF_NOT_OK(spill.CreateDirectories());
    MS_LOG(INFO) << "CacheServer will use disk folder: " << top_;
  }
  RETURN_IF_NOT_OK(vg_.ServiceStart());
  cache_q_ = std::make_shared<Queue<BaseRequest *>>(1024);
  RETURN_IF_NOT_OK(cache_q_->Register(&vg_));
  auto f = std::bind(&CacheServer::ServerRequest, this);
  // Spawn a a few threads to serve the request.
  for (auto i = 0; i < num_workers_; ++i) {
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("Cache server", f));
  }
  return Status::OK();
}

Status CacheServer::DoServiceStop() {
  Status rc;
  Status rc2;
  // First stop all the threads.
  RETURN_IF_NOT_OK(vg_.ServiceStop());
  // Clean up all the caches if any.
  UniqueLock lck(&rwLock_);
  auto it = all_caches_.begin();
  while (it != all_caches_.end()) {
    auto cs = std::move(it->second);
    rc2 = cs->ServiceStop();
    if (rc2.IsError()) {
      rc = rc2;
    }
    ++it;
  }
  return rc;
}

CacheService *CacheServer::GetService(connection_id_type id) const {
  SharedLock lck(&rwLock_);
  auto it = all_caches_.find(id);
  if (it != all_caches_.end()) {
    return it->second.get();
  }
  return nullptr;
}

Status CacheServer::CreateService(connection_id_type connection_id, uint64_t cache_mem_sz,
                                  BaseRequest::CreateCacheFlag flag, std::string *out_cookie) {
  // We can't do spilling unless this server is setup with a spill path in the first place
  bool spill = (flag & BaseRequest::CreateCacheFlag::kSpillToDisk) == BaseRequest::CreateCacheFlag::kSpillToDisk;
  bool generate_id =
    (flag & BaseRequest::CreateCacheFlag::kGenerateRowId) == BaseRequest::CreateCacheFlag::kGenerateRowId;
  if (spill && top_.empty()) {
    RETURN_STATUS_UNEXPECTED("Server is not set up with spill support.");
  }
  RETURN_UNEXPECTED_IF_NULL(out_cookie);
  *out_cookie = "";
  // Before creating the cache, first check if this is a request for a shared usage of an existing cache
  // If two CreateService come in with identical connection_id, we need to serialize the create.
  // The first create will be successful and be given a special cookie.
  UniqueLock lck(&rwLock_);
  auto end = all_caches_.end();
  auto it = all_caches_.find(connection_id);
  if (it == end) {
    std::unique_ptr<CacheService> cs;
    try {
      cs = std::make_unique<CacheService>(cache_mem_sz, spill ? top_ : "", generate_id);
      RETURN_IF_NOT_OK(cs->ServiceStart());
      *out_cookie = cs->cookie();
      all_caches_.emplace(connection_id, std::move(cs));
    } catch (const std::bad_alloc &e) {
      return Status(StatusCode::kOutOfMemory);
    }
  } else {
    MS_LOG(INFO) << "Duplicate request for " + std::to_string(connection_id) + " to create cache service";
    // We can return OK but we will return a duplicate key so user can act accordingly to either ignore it
    // treat it as OK.
    return Status(StatusCode::kDuplicateKey);
  }
  return Status::OK();
}

/// This is the main loop the cache server thread(s) are running.
/// Each thread will pop a request and save the result in the same request.
/// The sender will wait on the wait post in the request. Once the request
/// is fulfilled, the server thread will do a post signalling the request is
/// is processed.
/// \return
Status CacheServer::ServerRequest() {
  TaskManager::FindMe()->Post();
  // Loop forever until we are interrupted.
  while (true) {
    BaseRequest *base_rq = nullptr;
    RETURN_IF_NOT_OK(cache_q_->PopFront(&base_rq));
    auto cs = GetService(base_rq->connection_id_);
    // Except for creating a new session, we expect cs is not null.
    switch (base_rq->type_) {
      case BaseRequest::RequestType::kCacheRow: {
        if (cs == nullptr) {
          std::string errMsg = "Cache id " + std::to_string(base_rq->connection_id_) + " not found";
          base_rq->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
        } else {
          auto *rq = reinterpret_cast<CacheRowRequest *>(base_rq);
          // Only if the cookie matches, we can accept insert into this cache that has a build phase
          if (!cs->HasBuildPhase() || rq->cookie_ == cs->cookie()) {
            rq->rc_ = cs->CacheRow(rq->buffers_, &rq->row_id_from_server_);
          } else {
            return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Cookie mismatch");
          }
        }
        break;
      }
      case BaseRequest::RequestType::kBatchFetchRows: {
        if (cs == nullptr) {
          std::string errMsg = "Cache id " + std::to_string(base_rq->connection_id_) + " not found";
          base_rq->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
        } else {
          auto *rq = reinterpret_cast<BatchFetchRequest *>(base_rq);
          rq->rc_ = cs->BatchFetch(rq->row_id_, &rq->mem_);
        }
        break;
      }
      case BaseRequest::RequestType::kCreateCache: {
        // If the cache is already created we still need to run the creation so that we do sanity checks on the
        // client id and return the cache id back to the user.
        auto *rq = reinterpret_cast<CreationCacheRequest *>(base_rq);
        rq->rc_ = CreateService(rq->connection_id_, rq->cache_mem_sz, rq->flag_, &rq->cookie_);
        break;
      }
      case BaseRequest::RequestType::kPurgeCache: {
        if (cs != nullptr) {
          base_rq->rc_ = cs->Purge();
        } else {
          // it is already purged. Ignore it.
          base_rq->rc_ = Status::OK();
        }
        break;
      }
      case BaseRequest::RequestType::kDestroyCache: {
        if (cs != nullptr) {
          // We need a strong lock to protect the map.
          connection_id_type id = base_rq->connection_id_;
          UniqueLock lck(&rwLock_);
          // std::map will invoke the constructor of CacheService. So we don't need to do anything here.
          auto n = all_caches_.erase(id);
          if (n == 0) {
            // It has been destroyed by another duplicate request.
            MS_LOG(INFO) << "Duplicate request for " + std::to_string(id) + " to create cache service";
          }
          base_rq->rc_ = Status::OK();
        } else {
          // it is already destroyed. Ignore it.
          base_rq->rc_ = Status::OK();
        }
        break;
      }
      case BaseRequest::RequestType::kGetStat: {
        if (cs == nullptr) {
          std::string errMsg = "Session " + std::to_string(base_rq->connection_id_) + " not found";
          base_rq->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
        } else {
          auto *rq = reinterpret_cast<GetStatRequest *>(base_rq);
          CacheService::ServiceStat svc_stat;
          rq->rc_ = cs->GetStat(&svc_stat);
          if (rq->rc_.IsOk()) {
            flatbuffers::FlatBufferBuilder fbb;
            ServiceStatMsgBuilder bld(fbb);
            bld.add_num_disk_cached(svc_stat.stat_.num_disk_cached);
            bld.add_num_mem_cached(svc_stat.stat_.num_mem_cached);
            bld.add_max_row_id(svc_stat.max_);
            bld.add_min_row_id(svc_stat.min_);
            bld.add_state(svc_stat.state_);
            auto offset = bld.Finish();
            fbb.Finish(offset);
            rq->rc_ = rq->mem_.allocate(fbb.GetSize());
            if (rq->rc_.IsOk()) {
              WritableSlice dest(rq->mem_.GetMutablePointer(), fbb.GetSize());
              ReadableSlice src(fbb.GetBufferPointer(), fbb.GetSize());
              RETURN_IF_NOT_OK(WritableSlice::Copy(&dest, src));
            }
          }
        }
        break;
      }
      case BaseRequest::RequestType::kCacheSchema: {
        if (cs == nullptr) {
          std::string errMsg = "Session " + std::to_string(base_rq->connection_id_) + " not found";
          base_rq->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
        } else {
          auto *rq = reinterpret_cast<CacheSchemaRequest *>(base_rq);
          rq->rc_ = cs->CacheSchema(rq->buf_, rq->len_of_buf_);
        }
        break;
      }
      case BaseRequest::RequestType::kFetchSchema: {
        if (cs == nullptr) {
          std::string errMsg = "Session " + std::to_string(base_rq->connection_id_) + " not found";
          base_rq->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
        } else {
          auto *rq = reinterpret_cast<FetchSchemaRequest *>(base_rq);
          rq->rc_ = cs->FetchSchema(&rq->mem_);
        }
        break;
      }
      case BaseRequest::RequestType::kBuildPhaseDone: {
        if (cs == nullptr) {
          std::string errMsg = "Session " + std::to_string(base_rq->connection_id_) + " not found";
          base_rq->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
        } else {
          auto *rq = reinterpret_cast<BuildPhaseDoneRequest *>(base_rq);
          // We can only allow to switch phase is the cookie match.
          if (rq->cookie_ == cs->cookie()) {
            rq->rc_ = cs->BuildPhaseDone();
          } else {
            return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Cookie mismatch");
          }
        }
        break;
      }
      default:
        base_rq->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Unknown request type");
    }
    // Notify it is done, and move on to the next request.
    base_rq->wp_.Set();
  }
  return Status::OK();
}
CacheServer::CacheServer(const std::string &spill_path, int32_t num_workers)
    : top_(spill_path), num_workers_(num_workers) {}
}  // namespace dataset
}  // namespace mindspore
