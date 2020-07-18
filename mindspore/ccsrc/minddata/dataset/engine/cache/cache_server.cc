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
#include <algorithm>
#include <functional>
#include <limits>
#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/engine/cache/cache_service.h"
#include "minddata/dataset/engine/cache/cache_request.h"
#include "minddata/dataset/util/bit.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/random.h"
#ifdef CACHE_LOCAL_CLIENT
#include "minddata/dataset/util/sig_handler.h"
#endif

namespace mindspore {
namespace dataset {
CacheServer *CacheServer::instance_ = nullptr;
std::once_flag CacheServer::init_instance_flag_;
Status CacheServer::DoServiceStart() {
#ifdef CACHE_LOCAL_CLIENT
  // We need to destroy the shared memory if user hits Control-C
  RegisterHandlers();
#endif
  if (!top_.empty()) {
    Path spill(top_);
    RETURN_IF_NOT_OK(spill.CreateDirectories());
    MS_LOG(INFO) << "CacheServer will use disk folder: " << top_;
  }
  RETURN_IF_NOT_OK(vg_.ServiceStart());
  // There will be num_workers_ threads working on the grpc queue and
  // the same number of threads working on the CacheServerRequest queue.
  // Like a connector object we will set up the same number of queues but
  // we do not need to preserve any order. We will set the capacity of
  // each queue to be 128 since we are just pushing memory pointers which
  // is only 8 byte each.
  const int32_t que_capacity = 128;
  // This is the request queue from the client
  cache_q_ = std::make_shared<QueueList<CacheServerRequest *>>();
  cache_q_->Init(num_workers_, que_capacity);
  // For the grpc completion queue to work, we need to allocate some
  // tags which in our case are instances of CacheServerQuest.
  // They got recycled and we will allocate them in advance and push
  // them into some free list. We need more (two or three times) the
  // size of the cache_q. While each worker is working on a CacheSerRequest,
  // we need some extra running injecting in the the qrpc completion queue.
  const int32_t multiplier = 3;
  const int32_t free_list_capacity = multiplier * (que_capacity + 1);
  free_list_ = std::make_shared<QueueList<CacheServerRequest *>>();
  free_list_->Init(num_workers_, free_list_capacity);
  // We need to have a reference to the services memory pool in case
  // the Services goes out of scope earlier than us since it is a singleton
  mp_ = Services::GetInstance().GetServiceMemPool();
  Allocator<CacheServerRequest> alloc(mp_);
  tag_.reserve(num_workers_);
  // Now we populate all free list.
  for (auto m = 0; m < num_workers_; ++m) {
    // Ideally we allocate all the free list in one malloc. But it turns out it exceeds the
    // Arena size. So we will we will allocate one segment at a time.
    auto my_tag = std::make_unique<MemGuard<CacheServerRequest, Allocator<CacheServerRequest>>>(alloc);
    // Allocate the tag and assign it the current queue
    RETURN_IF_NOT_OK(my_tag->allocate(free_list_capacity, m));
    for (int i = 0; i < free_list_capacity; ++i) {
      RETURN_IF_NOT_OK(free_list_->operator[](m)->Add((*my_tag)[i]));
    }
    tag_.push_back(std::move(my_tag));
  }
  RETURN_IF_NOT_OK(cache_q_->Register(&vg_));
  RETURN_IF_NOT_OK(free_list_->Register(&vg_));
  // Spawn a few threads to serve the real request.
  auto f = std::bind(&CacheServer::ServerRequest, this, std::placeholders::_1);
  for (auto i = 0; i < num_workers_; ++i) {
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("Cache service worker", std::bind(f, i)));
  }
  // Start the comm layer
  try {
    comm_layer_ = std::make_shared<CacheServerGreeterImpl>(port_, shared_memory_sz_in_gb_);
    RETURN_IF_NOT_OK(comm_layer_->Run());
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  // Finally loop forever to handle the request.
  auto r = std::bind(&CacheServer::RpcRequest, this, std::placeholders::_1);
  for (auto i = 0; i < num_workers_; ++i) {
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("rpc worker", std::bind(r, i)));
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

Status CacheServer::CreateService(CacheRequest *rq, CacheReply *reply) {
  CHECK_FAIL_RETURN_UNEXPECTED(rq->has_connection_info(), "Missing connection info");
  std::string cookie;
  auto session_id = rq->connection_info().session_id();
  auto crc = rq->connection_info().crc();
  // We concat both numbers to form the internal connection id.
  auto connection_id = GetConnectionID(session_id, crc);
  CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing info to create cache");
  auto &create_cache_buf = rq->buf_data(0);
  auto p = flatbuffers::GetRoot<CreateCacheRequestMsg>(create_cache_buf.data());
  auto flag = static_cast<CreateCacheRequest::CreateCacheFlag>(p->flag());
  auto cache_mem_sz = p->cache_mem_sz();
  // We can't do spilling unless this server is setup with a spill path in the first place
  bool spill =
    (flag & CreateCacheRequest::CreateCacheFlag::kSpillToDisk) == CreateCacheRequest::CreateCacheFlag::kSpillToDisk;
  bool generate_id =
    (flag & CreateCacheRequest::CreateCacheFlag::kGenerateRowId) == CreateCacheRequest::CreateCacheFlag::kGenerateRowId;
  if (spill && top_.empty()) {
    RETURN_STATUS_UNEXPECTED("Server is not set up with spill support.");
  }
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<flatbuffers::String> off_cookie;
  // Before creating the cache, first check if this is a request for a shared usage of an existing cache
  // If two CreateService come in with identical connection_id, we need to serialize the create.
  // The first create will be successful and be given a special cookie.
  UniqueLock lck(&rwLock_);
  // Early exit if we are doing global shutdown
  if (global_shutdown_) {
    return Status::OK();
  }
  auto end = all_caches_.end();
  auto it = all_caches_.find(connection_id);
  bool duplicate = false;
  if (it == end) {
    std::unique_ptr<CacheService> cs;
    try {
      cs = std::make_unique<CacheService>(cache_mem_sz, spill ? top_ : "", generate_id);
      RETURN_IF_NOT_OK(cs->ServiceStart());
      cookie = cs->cookie();
      all_caches_.emplace(connection_id, std::move(cs));
    } catch (const std::bad_alloc &e) {
      return Status(StatusCode::kOutOfMemory);
    }
  } else {
    duplicate = true;
    MS_LOG(INFO) << "Duplicate request for " + std::to_string(connection_id) + " to create cache service";
  }
  off_cookie = fbb.CreateString(cookie);
  CreateCacheReplyMsgBuilder bld(fbb);
  bld.add_connection_id(connection_id);
  bld.add_cookie(off_cookie);
  auto off = bld.Finish();
  fbb.Finish(off);
  reply->set_result(fbb.GetBufferPointer(), fbb.GetSize());
  // Track the history of all the sessions that we have created so far.
  history_sessions_.insert(session_id);
  // We can return OK but we will return a duplicate key so user can act accordingly to either ignore it
  // treat it as OK.
  return duplicate ? Status(StatusCode::kDuplicateKey) : Status::OK();
}

Status CacheServer::DestroyCache(CacheService *cs, CacheRequest *rq) {
  // We need a strong lock to protect the map.
  UniqueLock lck(&rwLock_);
  // it is already destroyed. Ignore it.
  if (cs != nullptr) {
    auto id = rq->connection_id();
    MS_LOG(WARNING) << "Dropping cache with connection id " << std::to_string(id);
    // std::map will invoke the destructor of CacheService. So we don't need to do anything here.
    auto n = all_caches_.erase(id);
    if (n == 0) {
      // It has been destroyed by another duplicate request.
      MS_LOG(INFO) << "Duplicate request for " + std::to_string(id) + " to create cache service";
    }
  }
  return Status::OK();
}

inline Status CacheRow(CacheService *cs, CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  if (cs == nullptr) {
    std::string errMsg = "Cache id " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    auto sz = rq->buf_data_size();
    std::vector<const void *> buffers;
    buffers.reserve(sz);
    // First piece of data is the cookie and is required
    CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing cookie");
    auto &cookie = rq->buf_data(0);
    // Only if the cookie matches, we can accept insert into this cache that has a build phase
    if (!cs->HasBuildPhase() || cookie == cs->cookie()) {
      // Push the address of each buffer (in the form of std::string coming in from protobuf) into
      // a vector of buffer
      for (auto i = 1; i < sz; ++i) {
        buffers.push_back(rq->buf_data(i).data());
      }
      row_id_type id = -1;
      RETURN_IF_NOT_OK(cs->CacheRow(buffers, &id));
      reply->set_result(std::to_string(id));
    } else {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Cookie mismatch");
    }
  }
  return Status::OK();
}

Status CacheServer::FastCacheRow(CacheService *cs, CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  auto shared_pool = comm_layer_->GetSharedMemoryPool();
  auto *base = shared_pool->SharedMemoryBaseAddr();
  // Ensure we got 3 pieces of data coming in
  CHECK_FAIL_RETURN_UNEXPECTED(rq->buf_data_size() == 3, "Incomplete data");
  // First piece of data is the cookie and is required
  auto &cookie = rq->buf_data(0);
  // Second piece of data is the address where we can find the serialized data
  auto addr = strtoll(rq->buf_data(1).data(), nullptr, 10);
  auto p = reinterpret_cast<void *>(reinterpret_cast<int64_t>(base) + addr);
  // Third piece of data is the size of the serialized data that we need to transfer
  auto sz = strtoll(rq->buf_data(2).data(), nullptr, 10);
  // Successful or not, we need to free the memory on exit.
  Status rc;
  if (cs == nullptr) {
    std::string errMsg = "Cache id " + std::to_string(connection_id) + " not found";
    rc = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    // Only if the cookie matches, we can accept insert into this cache that has a build phase
    if (!cs->HasBuildPhase() || cookie == cs->cookie()) {
      row_id_type id = -1;
      ReadableSlice src(p, sz);
      rc = cs->FastCacheRow(src, &id);
      reply->set_result(std::to_string(id));
    } else {
      rc = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Cookie mismatch");
    }
  }
  // Return the block to the shared memory.
  shared_pool->Deallocate(p);
  return rc;
}

Status CacheServer::BatchFetchRows(CacheService *cs, CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  if (cs == nullptr) {
    std::string errMsg = "Cache id " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing row id");
    auto &row_id_buf = rq->buf_data(0);
    auto p = flatbuffers::GetRoot<TensorRowIds>(row_id_buf.data());
    std::vector<row_id_type> row_id;
    auto sz = p->row_id()->size();
    row_id.reserve(sz);
    for (auto i = 0; i < sz; ++i) {
      row_id.push_back(p->row_id()->Get(i));
    }
    int64_t mem_sz = 0;
    std::vector<key_size_pair> v;
    RETURN_IF_NOT_OK(cs->PreBatchFetch(row_id, &v, &mem_sz));
    auto client_flag = rq->flag();
    bool local_client = BitTest(client_flag, kLocalClientSupport);
    // For large amount data to be sent back, we will use shared memory provided it is a local
    // client that has local bypass support
    bool local_bypass = local_client ? (mem_sz >= kLocalByPassThreshold) : false;
    reply->set_flag(local_bypass ? kDataIsInSharedMemory : 0);
    if (local_bypass) {
      // We will use shared memory
      auto shared_pool = comm_layer_->GetSharedMemoryPool();
      auto *base = shared_pool->SharedMemoryBaseAddr();
      void *q = nullptr;
      RETURN_IF_NOT_OK(shared_pool->Allocate(mem_sz, &q));
      WritableSlice dest(q, mem_sz);
      RETURN_IF_NOT_OK(cs->BatchFetch(row_id, v, &dest));
      // We can't return the absolute address which makes no sense to the client.
      // Instead we return the difference.
      auto difference = reinterpret_cast<int64_t>(q) - reinterpret_cast<int64_t>(base);
      reply->set_result(std::to_string(difference));
    } else {
      // We are going to use std::string to allocate and hold the result which will be eventually
      // 'moved' to the protobuf message (which underneath is also a std::string) for the purpose
      // to minimize memory copy.
      std::string mem;
      try {
        mem.resize(mem_sz);
        CHECK_FAIL_RETURN_UNEXPECTED(mem.capacity() >= mem_sz, "Programming error");
      } catch (const std::bad_alloc &e) {
        return Status(StatusCode::kOutOfMemory);
      }
      WritableSlice dest(mem.data(), mem_sz);
      RETURN_IF_NOT_OK(cs->BatchFetch(row_id, v, &dest));
      reply->set_result(std::move(mem));
    }
  }
  return Status::OK();
}

inline Status GetStat(CacheService *cs, CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    CacheService::ServiceStat svc_stat;
    RETURN_IF_NOT_OK(cs->GetStat(&svc_stat));
    flatbuffers::FlatBufferBuilder fbb;
    ServiceStatMsgBuilder bld(fbb);
    bld.add_num_disk_cached(svc_stat.stat_.num_disk_cached);
    bld.add_num_mem_cached(svc_stat.stat_.num_mem_cached);
    bld.add_avg_cache_sz(svc_stat.stat_.average_cache_sz);
    bld.add_max_row_id(svc_stat.max_);
    bld.add_min_row_id(svc_stat.min_);
    bld.add_state(svc_stat.state_);
    auto offset = bld.Finish();
    fbb.Finish(offset);
    reply->set_result(fbb.GetBufferPointer(), fbb.GetSize());
  }
  return Status::OK();
}

inline Status CacheSchema(CacheService *cs, CacheRequest *rq) {
  auto connection_id = rq->connection_id();
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing schema information");
    auto &create_schema_buf = rq->buf_data(0);
    RETURN_IF_NOT_OK(cs->CacheSchema(create_schema_buf.data(), create_schema_buf.size()));
  }
  return Status::OK();
}

inline Status FetchSchema(CacheService *cs, CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    // We are going to use std::string to allocate and hold the result which will be eventually
    // 'moved' to the protobuf message (which underneath is also a std::string) for the purpose
    // to minimize memory copy.
    std::string mem;
    RETURN_IF_NOT_OK(cs->FetchSchema(&mem));
    reply->set_result(std::move(mem));
  }
  return Status::OK();
}

inline Status BuildPhaseDone(CacheService *cs, CacheRequest *rq) {
  auto connection_id = rq->connection_id();
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    // First piece of data is the cookie
    CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing cookie");
    auto &cookie = rq->buf_data(0);
    // We can only allow to switch phase is the cookie match.
    if (cookie == cs->cookie()) {
      RETURN_IF_NOT_OK(cs->BuildPhaseDone());
    } else {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Cookie mismatch");
    }
  }
  return Status::OK();
}

Status CacheServer::PurgeCache(CacheService *cs) {
  SharedLock lck(&rwLock_);
  // If shutdown in progress, ignore the command.
  if (global_shutdown_) {
    return Status::OK();
  }
  // it is already purged. Ignore it.
  if (cs != nullptr) {
    RETURN_IF_NOT_OK(cs->Purge());
  }
  return Status::OK();
}

inline Status GenerateClientSessionID(session_id_type session_id, CacheReply *reply) {
  reply->set_result(std::to_string(session_id));
  return Status::OK();
}

/// \brief This is the main loop the cache server thread(s) are running.
/// Each thread will pop a request and send the result back to the client using grpc
/// \return
Status CacheServer::ServerRequest(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  auto &my_que = cache_q_->operator[](worker_id);
  // Loop forever until we are interrupted or shutdown.
  while (!global_shutdown_) {
    CacheServerRequest *cache_req = nullptr;
    RETURN_IF_NOT_OK(my_que->PopFront(&cache_req));
    auto &rq = cache_req->rq_;
    auto &reply = cache_req->reply_;
    CacheService *cs = nullptr;
    // Request comes in roughly two sets. One set is at the cache level with a connection id.
    // The other set is working at a high level and without a connection id
    if (!rq.has_connection_info()) {
      cs = GetService(rq.connection_id());
    }
    // Except for creating a new session, we expect cs is not null.
    switch (cache_req->type_) {
      case BaseRequest::RequestType::kCacheRow: {
        // Look into the flag to see where we can find the data and
        // call the appropriate method.
        auto flag = rq.flag();
        if (BitTest(flag, kDataIsInSharedMemory)) {
          cache_req->rc_ = FastCacheRow(cs, &rq, &reply);
        } else {
          cache_req->rc_ = CacheRow(cs, &rq, &reply);
        }
        break;
      }
      case BaseRequest::RequestType::kBatchFetchRows: {
        cache_req->rc_ = BatchFetchRows(cs, &rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kCreateCache: {
        cache_req->rc_ = CreateService(&rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kPurgeCache: {
        cache_req->rc_ = PurgeCache(cs);
        break;
      }
      case BaseRequest::RequestType::kDestroyCache: {
        cache_req->rc_ = DestroyCache(cs, &rq);
        break;
      }
      case BaseRequest::RequestType::kGetStat: {
        cache_req->rc_ = GetStat(cs, &rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kCacheSchema: {
        cache_req->rc_ = CacheSchema(cs, &rq);
        break;
      }
      case BaseRequest::RequestType::kFetchSchema: {
        cache_req->rc_ = FetchSchema(cs, &rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kBuildPhaseDone: {
        cache_req->rc_ = BuildPhaseDone(cs, &rq);
        break;
      }
      case BaseRequest::RequestType::kDropSession: {
        cache_req->rc_ = DestroySession(&rq);
        break;
      }
      case BaseRequest::RequestType::kGenerateSessionId: {
        cache_req->rc_ = GenerateClientSessionID(GenerateSessionID(), &reply);
        break;
      }
      case BaseRequest::RequestType::kAllocateSharedBlock: {
        cache_req->rc_ = AllocateSharedMemory(&rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kFreeSharedBlock: {
        cache_req->rc_ = FreeSharedMemory(&rq);
        break;
      }
      case BaseRequest::RequestType::kStopService: {
        // This command shutdowns everything.
        cache_req->rc_ = GlobalShutdown();
        break;
      }
      default:
        std::string errMsg("Unknown request type : ");
        errMsg += std::to_string(static_cast<uint16_t>(cache_req->type_));
        cache_req->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
    }
    // Notify it is done, and move on to the next request.
    Status2CacheReply(cache_req->rc_, &reply);
    cache_req->st_ = CacheServerRequest::STATE::FINISH;
    // We will re-tag the request back to the grpc queue. Once it comes back from the client,
    // the CacheServerRequest, i.e. the pointer cache_req, will be free
    if (!global_shutdown_) {
      cache_req->responder_.Finish(reply, grpc::Status::OK, cache_req);
    }
  }
  return Status::OK();
}

connection_id_type CacheServer::GetConnectionID(session_id_type session_id, uint32_t crc) const {
  connection_id_type connection_id =
    (static_cast<connection_id_type>(session_id) << 32u) | static_cast<connection_id_type>(crc);
  return connection_id;
}

session_id_type CacheServer::GetSessionID(connection_id_type connection_id) const {
  return static_cast<session_id_type>(connection_id >> 32u);
}

CacheServer::CacheServer(const std::string &spill_path, int32_t num_workers, int32_t port,
                         int32_t shared_meory_sz_in_gb)
    : top_(spill_path),
      num_workers_(num_workers),
      port_(port),
      shared_memory_sz_in_gb_(shared_meory_sz_in_gb),
      global_shutdown_(false) {}

Status CacheServer::Run() {
  RETURN_IF_NOT_OK(ServiceStart());
  // This is called by the main function and we shouldn't exit. Otherwise the main thread
  // will just shutdown. So we will call some function that never return unless error.
  // One good case will be simply to wait for all threads to return.
  RETURN_IF_NOT_OK(vg_.join_all(Task::WaitFlag::kBlocking));
  return Status::OK();
}

Status CacheServer::GetFreeRequestTag(int32_t queue_id, CacheServerRequest **q) {
  RETURN_UNEXPECTED_IF_NULL(q);
  CacheServer &cs = CacheServer::GetInstance();
  CacheServerRequest *p;
  RETURN_IF_NOT_OK(cs.free_list_->operator[](queue_id)->PopFront(&p));
  *q = p;
  return Status::OK();
}

Status CacheServer::ReturnRequestTag(CacheServerRequest *p) {
  RETURN_UNEXPECTED_IF_NULL(p);
  int32_t myQID = p->getQid();
  // Free any memory from the protobufs
  p->~CacheServerRequest();
  // Re-initialize the memory
  new (p) CacheServerRequest(myQID);
  // Now we return it back to free list.
  CacheServer &cs = CacheServer::GetInstance();
  RETURN_IF_NOT_OK(cs.free_list_->operator[](myQID)->Add(p));
  return Status::OK();
}

Status CacheServer::DestroySession(CacheRequest *rq) {
  CHECK_FAIL_RETURN_UNEXPECTED(rq->has_connection_info(), "Missing session id");
  auto drop_session_id = rq->connection_info().session_id();
  UniqueLock lck(&rwLock_);
  for (auto &cs : all_caches_) {
    auto connection_id = cs.first;
    auto session_id = GetSessionID(connection_id);
    // We can just call DestroyCache() but we are holding a lock already. Doing so will cause deadlock.
    // So we will just manually do it.
    if (session_id == drop_session_id) {
      // std::map will invoke the destructor of CacheService. So we don't need to do anything here.
      auto n = all_caches_.erase(connection_id);
      MS_LOG(INFO) << "Destroy " << n << " copies of cache with id " << connection_id;
    }
  }
  return Status::OK();
}

session_id_type CacheServer::GenerateSessionID() const {
  SharedLock lock(&rwLock_);
  auto mt = GetRandomDevice();
  std::uniform_int_distribution<session_id_type> distribution(0, std::numeric_limits<session_id_type>::max());
  session_id_type session_id;
  bool duplicate = false;
  do {
    session_id = distribution(mt);
    auto it = history_sessions_.find(session_id);
    duplicate = (it != history_sessions_.end());
  } while (duplicate);
  return session_id;
}

Status CacheServer::AllocateSharedMemory(CacheRequest *rq, CacheReply *reply) {
  auto requestedSz = strtoll(rq->buf_data(0).data(), nullptr, 10);
  auto shared_pool = comm_layer_->GetSharedMemoryPool();
  auto *base = shared_pool->SharedMemoryBaseAddr();
  void *p = nullptr;
  RETURN_IF_NOT_OK(shared_pool->Allocate(requestedSz, &p));
  // We can't return the absolute address which makes no sense to the client.
  // Instead we return the difference.
  auto difference = reinterpret_cast<int64_t>(p) - reinterpret_cast<int64_t>(base);
  reply->set_result(std::to_string(difference));
  return Status::OK();
}

Status CacheServer::FreeSharedMemory(CacheRequest *rq) {
  auto shared_pool = comm_layer_->GetSharedMemoryPool();
  auto *base = shared_pool->SharedMemoryBaseAddr();
  auto addr = strtoll(rq->buf_data(0).data(), nullptr, 10);
  auto p = reinterpret_cast<void *>(reinterpret_cast<int64_t>(base) + addr);
  shared_pool->Deallocate(p);
  return Status::OK();
}

Status CacheServer::RpcRequest(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(comm_layer_->HandleRequest(worker_id));
  return Status::OK();
}

Status CacheServer::GlobalShutdown() {
  // Let's shutdown in proper order.
  bool expected = false;
  if (global_shutdown_.compare_exchange_strong(expected, true)) {
    MS_LOG(WARNING) << "Shutting down server.";
    // Shutdown the grpc queue. No longer accept any new comer.
    // The threads we spawn to work on the grpc queue will exit themselves once
    // they notice the queue has been shutdown.
    comm_layer_->Shutdown();
    // Now we interrupt any threads that are waiting on cache_q_
    vg_.interrupt_all();
    // The next thing to do drop all the caches.
    UniqueLock lck(&rwLock_);
    for (auto &it : all_caches_) {
      auto id = it.first;
      MS_LOG(WARNING) << "Dropping cache with connection id " << std::to_string(id);
      // Wait for all outstanding work to be finished.
      auto &cs = it.second;
      UniqueLock cs_lock(&cs->rw_lock_);
      // std::map will invoke the destructor of CacheService. So we don't need to do anything here.
      (void)all_caches_.erase(id);
    }
  }
  return Status::OK();
}

Status CacheServer::Builder::SanityCheck() {
  if (shared_memory_sz_in_gb_ <= 0) {
    RETURN_STATUS_UNEXPECTED("Shared memory size (in GB unit) must be positive");
  }
  if (num_workers_ <= 0) {
    RETURN_STATUS_UNEXPECTED("Number of parallel workers must be positive");
  }
  if (!top_.empty()) {
    auto p = top_.data();
    if (p[0] != '/') {
      RETURN_STATUS_UNEXPECTED("Spilling directory must be an absolute path");
    }
    // Check if the spill directory is writable
    Path spill(top_);
    auto t = spill / Services::GetUniqueID();
    Status rc = t.CreateDirectory();
    if (rc.IsOk()) {
      rc = t.Remove();
    }
    if (rc.IsError()) {
      RETURN_STATUS_UNEXPECTED("Spilling directory is not writable\n" + rc.ToString());
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
