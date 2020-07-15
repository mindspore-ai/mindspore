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

#include <iomanip>
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/cache/cache_request.h"
#include "minddata/dataset/util/bit.h"

namespace mindspore {
namespace dataset {

// Constructor
CacheClient::CacheClient(uint32_t session_id, uint64_t cache_mem_sz, bool spill)
    : server_connection_id_(0), session_id_(session_id), cache_crc_(0), cache_mem_sz_(cache_mem_sz), spill_(spill) {}

// print method for display cache details
void CacheClient::Print(std::ostream &out) const {
  out << "  Session id: " << session_id_ << "\n  Cache crc: " << cache_crc_
      << "\n  Server cache id: " << server_connection_id_ << "\n  Cache mem size: " << cache_mem_sz_
      << "\n  Spilling: " << std::boolalpha << spill_;
}

Status CacheClient::WriteRow(const TensorRow &row, row_id_type *row_id_from_server) const {
  CacheRowRequest rq(server_connection_id_, cookie());
  RETURN_IF_NOT_OK(rq.SerializeCacheRowRequest(row));
  RETURN_IF_NOT_OK(CacheServer::GetInstance().PushRequest(&rq));
  RETURN_IF_NOT_OK(rq.Wait());
  if (row_id_from_server != nullptr) {
    *row_id_from_server = rq.GetRowIdAfterCache();
  }
  return Status::OK();
}

Status CacheClient::WriteBuffer(std::unique_ptr<DataBuffer> &&in) const {
  std::unique_ptr<DataBuffer> db_ptr = std::move(in);
  auto num_rows = db_ptr->NumRows();
  std::vector<TensorRow> all_rows;
  if (num_rows > 0) {
    all_rows.reserve(num_rows);
    // Break down the DataBuffer into TensorRow. We will send the requests async
    // and then do a final wait.
    MemGuard<CacheRowRequest> rq_arr;
    RETURN_IF_NOT_OK(rq_arr.allocate(num_rows, server_connection_id_, cookie()));
    CacheServer &cs = CacheServer::GetInstance();
    for (auto i = 0; i < num_rows; ++i) {
      TensorRow row;
      auto rq = rq_arr[i];
      RETURN_IF_NOT_OK(db_ptr->PopRow(&row));
      RETURN_IF_NOT_OK(rq->SerializeCacheRowRequest(row));
      RETURN_IF_NOT_OK(cs.PushRequest(rq));
      // We can't let row go out of scope. Otherwise it will free all the tensor memory.
      // So park it in the vector. When this function go out of scope, its memory
      // will be freed.
      all_rows.push_back(std::move(row));
    }
    // Now we wait for the requests to be done.
    for (auto i = 0; i < num_rows; ++i) {
      auto rq = rq_arr[i];
      RETURN_IF_NOT_OK(rq->Wait());
    }
  }
  return Status::OK();
}

Status CacheClient::GetRows(const std::vector<row_id_type> &row_id, TensorTable *out) const {
  RETURN_UNEXPECTED_IF_NULL(out);
  BatchFetchRequest rq(server_connection_id_, row_id);
  RETURN_IF_NOT_OK(CacheServer::GetInstance().PushRequest(&rq));
  RETURN_IF_NOT_OK(rq.Wait());
  RETURN_IF_NOT_OK(rq.RestoreRows(out));
  return Status::OK();
}

Status CacheClient::CreateCache(uint32_t tree_crc, bool generate_id) {
  UniqueLock lck(&mux_);
  // To create a cache, we identify ourself at the client by:
  // - the shared session id
  // - a crc for the tree nodes from the cache downward
  // Pack these 2 into a single 64 bit request id
  //
  // Consider this example:
  // tree1: tfreader --> map(decode) --> cache (session id = 1, crc = 123) --> batch
  // tree2: cifar10 --> map(rotate) --> cache (session id = 1, crc = 456) --> batch
  // These are different trees in a single session, but the user wants to share the cache.
  // This is not allowed because the data of these caches are different.
  //
  // Consider this example:
  // tree1: tfreader --> map(decode) --> cache (session id = 1, crc = 123) --> batch
  // tree2: tfreader --> map(decode) --> cache (session id = 1, crc = 123) --> map(rotate) --> batch
  // These are different trees in the same session, but the cached data is the same, so it is okay
  // to allow the sharing of this cache between these pipelines.

  // The CRC is computed by the tree prepare phase and passed to this function when creating the cache.
  // If we already have a server_connection_id_, then it means this same cache client has already been used
  // to create a cache and some other tree is trying to use the same cache.
  // That is allowed, however the crc better match!
  if (server_connection_id_) {
    if (cache_crc_ != tree_crc) {
      RETURN_STATUS_UNEXPECTED("Attempt to re-use a cache for a different tree!");
    }
    // Check the state of the server. For non-mappable case where there is a build phase and a fetch phase, we should
    // skip the build phase.
    lck.Unlock();  // GetStat will grab the mutex again. So unlock it to prevent deadlock.
    CacheClient::ServiceStat stat{};
    RETURN_IF_NOT_OK(GetStat(&stat));
    if (stat.cache_service_state == static_cast<uint8_t>(CacheService::State::kFetchPhase)) {
      return Status(StatusCode::kDuplicateKey, __LINE__, __FILE__, "Not an error and we should bypass the build phase");
    }
  } else {
    cache_crc_ = tree_crc;  // It's really a new cache we're creating so save our crc in the client
    // Combine the session and crc.  This will form our client cache identifier.
    connection_id_type connection_identification = (static_cast<uint64_t>(session_id_) << 32) | cache_crc_;
    // Now execute the cache create request using this identifier and other configs
    BaseRequest::CreateCacheFlag createFlag = BaseRequest::CreateCacheFlag::kNone;
    if (spill_) {
      createFlag |= BaseRequest::CreateCacheFlag::kSpillToDisk;
    }
    if (generate_id) {
      createFlag |= BaseRequest::CreateCacheFlag::kGenerateRowId;
    }
    CreationCacheRequest rq(connection_identification, cache_mem_sz_, createFlag);
    RETURN_IF_NOT_OK(CacheServer::GetInstance().PushRequest(&rq));
    Status rc = rq.Wait();
    if (rc.IsOk() || rc.get_code() == StatusCode::kDuplicateKey) {
      server_connection_id_ = rq.GetServerConnectionId();
      if (rc.IsOk()) {
        // The 1st guy creating the cache will get a cookie back.
        // But this object may be shared among pipelines and we don't want
        // overwrite it.
        cookie_ = rq.cookie();
      }
    }
    // We are not resetting the Duplicate key return code. We are passing it back to the CacheOp. This will tell the
    // CacheOp to bypass the build phase.
    return rc;
  }
  return Status::OK();
}

Status CacheClient::PurgeCache() {
  UniqueLock lck(&mux_);
  PurgeCacheRequest rq(server_connection_id_);
  RETURN_IF_NOT_OK(CacheServer::GetInstance().PushRequest(&rq));
  return rq.Wait();
}

Status CacheClient::DestroyCache() {
  UniqueLock lck(&mux_);
  DestroyCacheRequest rq(server_connection_id_);
  RETURN_IF_NOT_OK(CacheServer::GetInstance().PushRequest(&rq));
  return rq.Wait();
}

Status CacheClient::GetStat(ServiceStat *stat) {
  SharedLock lck(&mux_);
  RETURN_UNEXPECTED_IF_NULL(stat);
  GetStatRequest rq(server_connection_id_);
  RETURN_IF_NOT_OK(CacheServer::GetInstance().PushRequest(&rq));
  RETURN_IF_NOT_OK(rq.Wait());
  stat->num_disk_cached = rq.GetNumDiskCached();
  stat->num_mem_cached = rq.GetNumMemCached();
  stat->min_row_id = rq.GetMinRowId();
  stat->max_row_id = rq.GetMaxRowId();
  stat->cache_service_state = rq.GetState();
  return Status::OK();
}

Status CacheClient::CacheSchema(const std::unordered_map<std::string, int32_t> &map) {
  SharedLock lck(&mux_);
  CacheSchemaRequest rq(server_connection_id_);
  RETURN_IF_NOT_OK(rq.SerializeCacheSchemaRequest(map));
  RETURN_IF_NOT_OK(CacheServer::GetInstance().PushRequest(&rq));
  RETURN_IF_NOT_OK(rq.Wait());
  return Status::OK();
}

Status CacheClient::FetchSchema(std::unordered_map<std::string, int32_t> *map) {
  SharedLock lck(&mux_);
  RETURN_UNEXPECTED_IF_NULL(map);
  FetchSchemaRequest rq(server_connection_id_);
  RETURN_IF_NOT_OK(CacheServer::GetInstance().PushRequest(&rq));
  RETURN_IF_NOT_OK(rq.Wait());
  *map = rq.GetColumnMap();
  return Status::OK();
}

Status CacheClient::BuildPhaseDone() const {
  SharedLock lck(&mux_);
  BuildPhaseDoneRequest rq(server_connection_id_, cookie());
  RETURN_IF_NOT_OK(CacheServer::GetInstance().PushRequest(&rq));
  RETURN_IF_NOT_OK(rq.Wait());
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
