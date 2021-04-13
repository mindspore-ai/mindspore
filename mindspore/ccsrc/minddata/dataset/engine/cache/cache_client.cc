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

#include <unistd.h>
#include <iomanip>
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/cache/cache_request.h"
#include "minddata/dataset/engine/cache/cache_fbb.h"
#include "minddata/dataset/util/bit.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
CacheClient::Builder::Builder()
    : session_id_(0), cache_mem_sz_(0), spill_(false), hostname_(""), port_(0), num_connections_(0), prefetch_size_(0) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  hostname_ = cfg->cache_host();
  port_ = cfg->cache_port();
  num_connections_ = cfg->num_connections();  // number of async tcp/ip connections
  prefetch_size_ = cfg->prefetch_size();      // prefetch size
}

Status CacheClient::Builder::Build(std::shared_ptr<CacheClient> *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  RETURN_IF_NOT_OK(SanityCheck());
  *out = std::make_shared<CacheClient>(session_id_, cache_mem_sz_, spill_, hostname_, port_, num_connections_,
                                       prefetch_size_);
  return Status::OK();
}

Status CacheClient::Builder::SanityCheck() {
  CHECK_FAIL_RETURN_SYNTAX_ERROR(session_id_ > 0, "session id must be positive");
  CHECK_FAIL_RETURN_SYNTAX_ERROR(cache_mem_sz_ >= 0, "cache memory size must not be negative. (0 implies unlimited");
  CHECK_FAIL_RETURN_SYNTAX_ERROR(num_connections_ > 0, "rpc connections must be positive");
  CHECK_FAIL_RETURN_SYNTAX_ERROR(prefetch_size_ > 0, "prefetch size must be positive");
  CHECK_FAIL_RETURN_SYNTAX_ERROR(!hostname_.empty(), "hostname must not be empty");
  CHECK_FAIL_RETURN_SYNTAX_ERROR(port_ > 1024, "Port must be in range (1025..65535)");
  CHECK_FAIL_RETURN_SYNTAX_ERROR(port_ <= 65535, "Port must be in range (1025..65535)");
  CHECK_FAIL_RETURN_SYNTAX_ERROR(hostname_ == "127.0.0.1",
                                 "now cache client has to be on the same host with cache server");
  return Status::OK();
}

// Constructor
CacheClient::CacheClient(session_id_type session_id, uint64_t cache_mem_sz, bool spill, std::string hostname,
                         int32_t port, int32_t num_connections, int32_t prefetch_size)
    : server_connection_id_(0),
      cache_mem_sz_(cache_mem_sz),
      spill_(spill),
      client_id_(-1),
      local_bypass_(false),
      num_connections_(num_connections),
      prefetch_size_(prefetch_size),
      fetch_all_keys_(true) {
  cinfo_.set_session_id(session_id);
  comm_ = std::make_shared<CacheClientGreeter>(hostname, port, num_connections_);
}

CacheClient::~CacheClient() {
  cache_miss_keys_wp_.Set();
  // Manually release the async buffer because we need the comm layer.
  if (async_buffer_stream_) {
    async_buffer_stream_->ReleaseBuffer();
  }
  if (client_id_ != -1) {
    try {
      // Send a message to the server, saying I am done.
      auto rq = std::make_shared<ConnectResetRequest>(server_connection_id_, client_id_);
      Status rc = PushRequest(rq);
      if (rc.IsOk()) {
        rc = rq->Wait();
        if (rc.IsOk()) {
          MS_LOG(INFO) << "Disconnect from server successful";
        }
      }
    } catch (const std::exception &e) {
      // Can't do anything in destructor. So just log the error.
      MS_LOG(ERROR) << e.what();
    }
  }
  (void)comm_->ServiceStop();
}

// print method for display cache details
void CacheClient::Print(std::ostream &out) const {
  out << "  Session id: " << session_id() << "\n  Cache crc: " << cinfo_.crc()
      << "\n  Server cache id: " << server_connection_id_ << "\n  Cache mem size: " << GetCacheMemSz()
      << "\n  Spilling: " << std::boolalpha << isSpill() << "\n  Number of rpc workers: " << GetNumConnections()
      << "\n  Prefetch size: " << GetPrefetchSize() << "\n  Local client support: " << std::boolalpha
      << SupportLocalClient();
}

Status CacheClient::WriteRow(const TensorRow &row, row_id_type *row_id_from_server) const {
  auto rq = std::make_shared<CacheRowRequest>(this);
  RETURN_IF_NOT_OK(rq->SerializeCacheRowRequest(this, row));
  RETURN_IF_NOT_OK(PushRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  if (row_id_from_server != nullptr) {
    *row_id_from_server = rq->GetRowIdAfterCache();
  }
  return Status::OK();
}

Status CacheClient::WriteBuffer(std::unique_ptr<DataBuffer> &&in) const {
  std::unique_ptr<DataBuffer> db_ptr = std::move(in);
  auto num_rows = db_ptr->NumRows();
  // We will send the requests async first on all rows and do a final wait.
  if (num_rows > 0) {
    auto arr = std::make_unique<std::shared_ptr<CacheRowRequest>[]>(num_rows);
    for (auto i = 0; i < num_rows; ++i) {
      TensorRow row;
      RETURN_IF_NOT_OK(db_ptr->PopRow(&row));
      arr[i] = std::make_shared<CacheRowRequest>(this);
      RETURN_IF_NOT_OK(arr[i]->SerializeCacheRowRequest(this, row));
      RETURN_IF_NOT_OK(PushRequest(arr[i]));
    }
    // Now we wait for them to come back
    for (auto i = 0; i < num_rows; ++i) {
      RETURN_IF_NOT_OK(arr[i]->Wait());
    }
  }
  return Status::OK();
}

Status CacheClient::AsyncWriteRow(const TensorRow &row) {
  if (async_buffer_stream_ == nullptr) {
    return Status(StatusCode::kMDNotImplementedYet);
  }
  RETURN_IF_NOT_OK(async_buffer_stream_->AsyncWrite(row));
  return Status::OK();
}

Status CacheClient::AsyncWriteBuffer(std::unique_ptr<DataBuffer> &&in) {
  if (async_buffer_stream_ == nullptr) {
    return Status(StatusCode::kMDNotImplementedYet);
  } else {
    Status rc;
    std::unique_ptr<TensorQTable> tensor_table = std::make_unique<TensorQTable>();
    auto num_rows = in->NumRows();
    if (num_rows > 0) {
      for (auto i = 0; i < num_rows; ++i) {
        TensorRow row;
        RETURN_IF_NOT_OK(in->PopRow(&row));
        rc = AsyncWriteRow(row);
        if (rc.StatusCode() == StatusCode::kMDNotImplementedYet) {
          tensor_table->push_back(row);
        } else if (rc.IsError()) {
          return rc;
        }
      }
    }
    // If not all of them can be sent async, return what's left back to the caller.
    if (!tensor_table->empty()) {
      in->set_tensor_table(std::move(tensor_table));
      return Status(StatusCode::kMDNotImplementedYet);
    }
  }
  return Status::OK();
}

Status CacheClient::GetRows(const std::vector<row_id_type> &row_id, TensorTable *out) const {
  RETURN_UNEXPECTED_IF_NULL(out);
  auto rq = std::make_shared<BatchFetchRequest>(this, row_id);
  RETURN_IF_NOT_OK(PushRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  int64_t mem_addr;
  Status rc = rq->RestoreRows(out, comm_->SharedMemoryBaseAddr(), &mem_addr);
  // Free the memory by sending a request back to the server.
  if (mem_addr != -1) {
    auto mfree_req = std::make_shared<FreeSharedBlockRequest>(server_connection_id_, client_id_, mem_addr);
    Status rc2 = PushRequest(mfree_req);
    // But we won't wait for the result for the sake of performance.
    if (rc.IsOk() && rc2.IsError()) {
      rc = rc2;
    }
  }
  return rc;
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
    if (cinfo_.crc() != tree_crc) {
      RETURN_STATUS_UNEXPECTED("Cannot re-use a cache for a different tree!");
    }
    // Check the state of the server. For non-mappable case where there is a build phase and a fetch phase, we should
    // skip the build phase.
    lck.Unlock();  // GetStat will grab the mutex again. So unlock it to prevent deadlock.
    int8_t out;
    RETURN_IF_NOT_OK(GetState(&out));
    auto cache_state = static_cast<CacheServiceState>(out);
    if (cache_state == CacheServiceState::kFetchPhase ||
        (cache_state == CacheServiceState::kBuildPhase && cookie_.empty())) {
      return Status(StatusCode::kMDDuplicateKey, __LINE__, __FILE__,
                    "Not an error and we should bypass the build phase");
    }
    if (async_buffer_stream_) {
      // Reset the async buffer stream to its initial state. Any stale status and data would get cleaned up.
      RETURN_IF_NOT_OK(async_buffer_stream_->Reset());
    }
  } else {
    cinfo_.set_crc(tree_crc);  // It's really a new cache we're creating so save our crc in the client
    // Now execute the cache create request using this identifier and other configs
    CreateCacheRequest::CreateCacheFlag createFlag = CreateCacheRequest::CreateCacheFlag::kNone;
    if (spill_) {
      createFlag |= CreateCacheRequest::CreateCacheFlag::kSpillToDisk;
    }
    if (generate_id) {
      createFlag |= CreateCacheRequest::CreateCacheFlag::kGenerateRowId;
    }
    // Start the comm layer to receive reply
    RETURN_IF_NOT_OK(comm_->ServiceStart());
    // Initiate connection
    auto rq = std::make_shared<CreateCacheRequest>(this, cinfo_, cache_mem_sz_, createFlag);
    RETURN_IF_NOT_OK(PushRequest(rq));
    Status rc = rq->Wait();
    bool success = (rc.IsOk() || rc.StatusCode() == StatusCode::kMDDuplicateKey);
    // If we get kDuplicateKey, it just means we aren't the first one to create the cache,
    // and we will continue to parse the result.
    if (rc.StatusCode() == StatusCode::kMDDuplicateKey) {
      RETURN_IF_NOT_OK(rq->PostReply());
    }
    if (success) {
      // Attach to shared memory for local client
      RETURN_IF_NOT_OK(comm_->AttachToSharedMemory(&local_bypass_));
      if (local_bypass_) {
        async_buffer_stream_ = std::make_shared<AsyncBufferStream>();
        RETURN_IF_NOT_OK(async_buffer_stream_->Init(this));
      }
    }
    // We are not resetting the Duplicate key return code. We are passing it back to the CacheOp. This will tell the
    // CacheOp to bypass the build phase.
    return rc;
  }
  return Status::OK();
}

Status CacheClient::DestroyCache() {
  UniqueLock lck(&mux_);
  auto rq = std::make_shared<DestroyCacheRequest>(server_connection_id_);
  RETURN_IF_NOT_OK(PushRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  return Status::OK();
}

Status CacheClient::GetStat(CacheServiceStat *stat) {
  SharedLock lck(&mux_);
  RETURN_UNEXPECTED_IF_NULL(stat);
  // GetStat has an external interface, so we have to make sure we have a valid connection id first
  CHECK_FAIL_RETURN_UNEXPECTED(server_connection_id_ != 0, "GetStat called but the cache is not in use yet.");

  auto rq = std::make_shared<GetStatRequest>(server_connection_id_);
  RETURN_IF_NOT_OK(PushRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  rq->GetStat(stat);
  return Status::OK();
}

Status CacheClient::GetState(int8_t *out) {
  SharedLock lck(&mux_);
  RETURN_UNEXPECTED_IF_NULL(out);
  CHECK_FAIL_RETURN_UNEXPECTED(server_connection_id_ != 0, "GetState called but the cache is not in use yet.");
  auto rq = std::make_shared<GetCacheStateRequest>(server_connection_id_);
  RETURN_IF_NOT_OK(PushRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  *out = rq->GetState();
  return Status::OK();
}

Status CacheClient::CacheSchema(const std::unordered_map<std::string, int32_t> &map) {
  SharedLock lck(&mux_);
  auto rq = std::make_shared<CacheSchemaRequest>(server_connection_id_);
  RETURN_IF_NOT_OK(rq->SerializeCacheSchemaRequest(map));
  RETURN_IF_NOT_OK(PushRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  return Status::OK();
}

Status CacheClient::FetchSchema(std::unordered_map<std::string, int32_t> *map) {
  SharedLock lck(&mux_);
  RETURN_UNEXPECTED_IF_NULL(map);
  auto rq = std::make_shared<FetchSchemaRequest>(server_connection_id_);
  RETURN_IF_NOT_OK(PushRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  *map = rq->GetColumnMap();
  return Status::OK();
}

Status CacheClient::BuildPhaseDone() const {
  SharedLock lck(&mux_);
  auto rq = std::make_shared<BuildPhaseDoneRequest>(server_connection_id_, cookie());
  RETURN_IF_NOT_OK(PushRequest(rq));
  RETURN_IF_NOT_OK(rq->Wait());
  return Status::OK();
}

Status CacheClient::PushRequest(std::shared_ptr<BaseRequest> rq) const { return comm_->HandleRequest(std::move(rq)); }

void CacheClient::ServerRunningOutOfResources() {
  bool expected = true;
  if (fetch_all_keys_.compare_exchange_strong(expected, false)) {
    Status rc;
    // Server runs out of memory or disk space to cache any more rows.
    // First of all, we will turn off the locking.
    auto toggle_write_mode_rq = std::make_shared<ToggleWriteModeRequest>(server_connection_id_, false);
    rc = PushRequest(toggle_write_mode_rq);
    if (rc.IsError()) {
      return;
    }
    // Wait until we can toggle the state of the server to non-locking
    rc = toggle_write_mode_rq->Wait();
    if (rc.IsError()) {
      return;
    }
    // Now we get a list of all the keys not cached at the server so
    // we can filter out at the prefetch level.
    auto cache_miss_rq = std::make_shared<GetCacheMissKeysRequest>(server_connection_id_);
    rc = PushRequest(cache_miss_rq);
    if (rc.IsError()) {
      return;
    }
    rc = cache_miss_rq->Wait();
    if (rc.IsError()) {
      return;
    }
    // We will get back a vector of row id between [min,max] that are absent in the server.
    auto &row_id_buf = cache_miss_rq->reply_.result();
    auto p = flatbuffers::GetRoot<TensorRowIds>(row_id_buf.data());
    std::vector<row_id_type> row_ids;
    auto sz = p->row_id()->size();
    row_ids.reserve(sz);
    for (auto i = 0; i < sz; ++i) {
      row_ids.push_back(p->row_id()->Get(i));
    }
    cache_miss_keys_ = std::make_unique<CacheMissKeys>(row_ids);
    // We are all set.
    cache_miss_keys_wp_.Set();
  }
}

CacheClient::CacheMissKeys::CacheMissKeys(const std::vector<row_id_type> &v) {
  auto it = v.begin();
  min_ = *it;
  ++it;
  max_ = *it;
  ++it;
  while (it != v.end()) {
    gap_.insert(*it);
    ++it;
  }
  MS_LOG(INFO) << "# of cache miss keys between min(" << min_ << ") and max(" << max_ << ") is " << gap_.size();
}

bool CacheClient::CacheMissKeys::KeyIsCacheMiss(row_id_type key) {
  if (key > max_ || key < min_) {
    return true;
  } else if (key == min_ || key == max_) {
    return false;
  } else {
    auto it = gap_.find(key);
    return it != gap_.end();
  }
}

CacheClient::AsyncBufferStream::AsyncBufferStream() : cc_(nullptr), offset_addr_(-1), cur_(0) {}

CacheClient::AsyncBufferStream::~AsyncBufferStream() {
  (void)vg_.ServiceStop();
  (void)ReleaseBuffer();
}

Status CacheClient::AsyncBufferStream::ReleaseBuffer() {
  if (offset_addr_ != -1) {
    auto mfree_req =
      std::make_shared<FreeSharedBlockRequest>(cc_->server_connection_id_, cc_->GetClientId(), offset_addr_);
    offset_addr_ = -1;
    RETURN_IF_NOT_OK(cc_->PushRequest(mfree_req));
    RETURN_IF_NOT_OK(mfree_req->Wait());
  }
  return Status::OK();
}

Status CacheClient::AsyncBufferStream::Init(CacheClient *cc) {
  cc_ = cc;
  // Allocate shared memory from the server
  auto mem_rq = std::make_shared<AllocateSharedBlockRequest>(cc_->server_connection_id_, cc_->GetClientId(),
                                                             kAsyncBufferSize * kNumAsyncBuffer);
  RETURN_IF_NOT_OK(cc->PushRequest(mem_rq));
  RETURN_IF_NOT_OK(mem_rq->Wait());
  offset_addr_ = mem_rq->GetAddr();
  // Now we need to add that to the base address of where we attach.
  auto base = cc->SharedMemoryBaseAddr();
  auto start = reinterpret_cast<int64_t>(base) + offset_addr_;
  for (auto i = 0; i < kNumAsyncBuffer; ++i) {
    // We only need to set the pointer during init. Other fields will be set dynamically.
    buf_arr_[i].buffer_ = reinterpret_cast<void *>(start + i * kAsyncBufferSize);
  }
  buf_arr_[0].bytes_avail_ = kAsyncBufferSize;
  buf_arr_[0].num_ele_ = 0;
  RETURN_IF_NOT_OK(vg_.ServiceStart());
  return Status::OK();
}

Status CacheClient::AsyncBufferStream::AsyncWrite(const TensorRow &row) {
  std::vector<ReadableSlice> v;
  v.reserve(row.size() + 1);
  std::shared_ptr<flatbuffers::FlatBufferBuilder> fbb;
  RETURN_IF_NOT_OK(::mindspore::dataset::SerializeTensorRowHeader(row, &fbb));
  int64_t sz = fbb->GetSize();
  v.emplace_back(fbb->GetBufferPointer(), sz);
  for (const auto &ts : row) {
    sz += ts->SizeInBytes();
    v.emplace_back(ts->GetBuffer(), ts->SizeInBytes());
  }
  // If the size is too big, tell the user to send it directly.
  if (sz > kAsyncBufferSize) {
    return Status(StatusCode::kMDNotImplementedYet);
  }
  std::unique_lock<std::mutex> lock(mux_);
  // Check error from the server side while we have the lock;
  RETURN_IF_NOT_OK(flush_rc_);
  AsyncWriter *asyncWriter = &buf_arr_[cur_];
  if (asyncWriter->bytes_avail_ < sz) {
    // Flush and switch to a new buffer while we have the lock.
    RETURN_IF_NOT_OK(SyncFlush(AsyncFlushFlag::kCallerHasXLock));
    // Refresh the pointer after we switch
    asyncWriter = &buf_arr_[cur_];
  }
  RETURN_IF_NOT_OK(asyncWriter->Write(sz, v));
  return Status::OK();
}

Status CacheClient::AsyncBufferStream::SyncFlush(AsyncFlushFlag flag) {
  std::unique_lock lock(mux_, std::defer_lock_t());
  bool callerHasXLock = (flag & AsyncFlushFlag::kCallerHasXLock) == AsyncFlushFlag::kCallerHasXLock;
  if (!callerHasXLock) {
    lock.lock();
  }
  auto *asyncWriter = &buf_arr_[cur_];
  if (asyncWriter->num_ele_) {
    asyncWriter->rq.reset(
      new BatchCacheRowsRequest(cc_, offset_addr_ + cur_ * kAsyncBufferSize, asyncWriter->num_ele_));
    flush_rc_ = cc_->PushRequest(asyncWriter->rq);
    if (flush_rc_.IsOk()) {
      // If we are asked to wait, say this is the final flush, just wait for its completion.
      bool blocking = (flag & AsyncFlushFlag::kFlushBlocking) == AsyncFlushFlag::kFlushBlocking;
      if (blocking) {
        // Make sure we are done with all the buffers
        for (auto i = 0; i < kNumAsyncBuffer; ++i) {
          if (buf_arr_[i].rq) {
            Status rc = buf_arr_[i].rq->Wait();
            if (rc.IsError()) {
              flush_rc_ = rc;
            }
            buf_arr_[i].rq.reset();
          }
        }
      }
      // Prepare for the next buffer.
      cur_ = (cur_ + 1) % kNumAsyncBuffer;
      asyncWriter = &buf_arr_[cur_];
      // Update the cur_ while we have the lock.
      // Before we do anything, make sure the cache server has done with this buffer, or we will corrupt its content
      // Also we can also pick up any error from previous flush.
      if (asyncWriter->rq) {
        // Save the result into a common area, so worker can see it and quit.
        flush_rc_ = asyncWriter->rq->Wait();
        asyncWriter->rq.reset();
      }
      asyncWriter->bytes_avail_ = kAsyncBufferSize;
      asyncWriter->num_ele_ = 0;
    }
  }
  return flush_rc_;
}

Status CacheClient::AsyncBufferStream::AsyncWriter::Write(int64_t sz, const std::vector<ReadableSlice> &v) {
  CHECK_FAIL_RETURN_UNEXPECTED(sz <= bytes_avail_, "Programming error");
  for (auto &p : v) {
    auto write_sz = p.GetSize();
    WritableSlice dest(reinterpret_cast<char *>(buffer_) + kAsyncBufferSize - bytes_avail_, write_sz);
    RETURN_IF_NOT_OK(WritableSlice::Copy(&dest, p));
    bytes_avail_ -= write_sz;
  }
  ++num_ele_;
  return Status::OK();
}

Status CacheClient::AsyncBufferStream::Reset() {
  // Clean up previous running state to be prepared for a new run.
  cur_ = 0;
  flush_rc_ = Status::OK();
  for (auto i = 0; i < kNumAsyncBuffer; ++i) {
    buf_arr_[i].bytes_avail_ = kAsyncBufferSize;
    buf_arr_[i].num_ele_ = 0;
    buf_arr_[i].rq.reset();
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
