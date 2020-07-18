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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_REQ_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_REQ_H_

#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef ENABLE_CACHE
#include "proto/cache_grpc.grpc.pb.h"
#endif
#include "proto/cache_grpc.pb.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/engine/cache/de_tensor_generated.h"
#include "minddata/dataset/util/slice.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
class CacheClient;
/// \brief Statistic structure for GetStat request
struct CacheServiceStat {
  int64_t num_mem_cached;
  int64_t num_disk_cached;
  int64_t avg_cache_sz;
  row_id_type min_row_id;
  row_id_type max_row_id;
  int8_t cache_service_state;
};

/// \brief CacheClient communicates with CacheServer using Requests.
class BaseRequest {
 public:
  // Request types
  enum class RequestType : int16_t {
    kCacheRow = 0,
    kBatchFetchRows = 1,
    kCreateCache = 2,
    kPurgeCache = 3,
    kDestroyCache = 4,
    kGetStat = 5,
    kCacheSchema = 6,
    kFetchSchema = 7,
    kBuildPhaseDone = 8,
    kDropSession = 9,
    kGenerateSessionId = 10,
    kAllocateSharedBlock = 11,
    kFreeSharedBlock = 12,
    kStopService = 13,
    // Add new request before it.
    kRequestUnknown = 32767
  };

  friend class CacheServer;
  friend class CacheServerRequest;
  friend class CacheClientGreeter;
  friend class CacheClientRequestTag;

  /// \brief Base class of a cache server request
  /// \param type Type of the request
  explicit BaseRequest(RequestType type) : type_(type) { rq_.set_type(static_cast<google::int32>(type_)); }
  virtual ~BaseRequest() = default;

  /// \brief A print method for debugging
  /// \param out The output stream to write output to
  virtual void Print(std::ostream &out) const { out << "Request type: " << static_cast<int16_t>(type_); }

  /// \brief << Stream output operator overload
  /// \param out reference to the output stream
  /// \param rq reference to the BaseRequest
  /// \return the output stream
  friend std::ostream &operator<<(std::ostream &out, const BaseRequest &rq) {
    rq.Print(out);
    return out;
  }

  /// \brief Derived class can implement extra work to be done before the request is sent to the server
  virtual Status Prepare() { return Status::OK(); }

  /// \brief Derived class can implement extra work to be done after the server sends the request
  virtual Status PostReply() { return Status::OK(); }

  /// \brief A method for the client to wait for the availability of the result back from the server.
  /// \return Status object
  Status Wait();

 protected:
  CacheRequest rq_;   // This is what we send to the server
  CacheReply reply_;  // This is what the server send back

 private:
  RequestType type_;
  WaitPost wp_;  // A sync area used by the client side.
};

class FreeSharedBlockRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit FreeSharedBlockRequest(connection_id_type connection_id, int64_t addr)
      : BaseRequest(RequestType::kFreeSharedBlock) {
    rq_.set_connection_id(connection_id);
    rq_.add_buf_data(std::to_string(addr));
  }
  ~FreeSharedBlockRequest() = default;
};

/// \brief Request to cache a single TensorRow
class CacheRowRequest : public BaseRequest {
 public:
  friend class CacheServer;
  friend class CacheClient;
  explicit CacheRowRequest(connection_id_type connection_id, const std::string &cookie, bool local_bypass)
      : BaseRequest(RequestType::kCacheRow),
        support_local_bypass_(local_bypass),
        addr_(-1),
        sz_(0),
        row_id_from_server_(-1) {
    rq_.set_connection_id(connection_id);
    rq_.add_buf_data(cookie);
  }
  ~CacheRowRequest() = default;

  /// \brief Serialize a TensorRow for streaming to the cache server
  /// \param row TensorRow
  /// \return Status object
  Status SerializeCacheRowRequest(const CacheClient *cc, const TensorRow &row);

  /// \brief Sanity check before we send the row.
  /// \return Status object
  Status Prepare() override;

  /// \brief Override the base function get the row id returned from the server
  /// \return Status object
  Status PostReply() override;

  /// \brief Return the row id assigned to this row for non-mappable dataset
  /// \return row id of the cached row
  row_id_type GetRowIdAfterCache() { return row_id_from_server_; }

  /// \brief If we are doing local bypass, fill in extra request information of where the data is located.
  void AddDataLocation() {
    if (support_local_bypass_) {
      rq_.add_buf_data(std::to_string(addr_));
      rq_.add_buf_data(std::to_string(sz_));
    }
  }

  /// \brief If we fail to send the data to the server using shared memory method, we should release
  /// the shared memory by sending another request. The following function will generate a suitable
  /// request for the CacheClient to send.
  std::shared_ptr<FreeSharedBlockRequest> GenerateFreeBlockRequest() {
    return std::make_shared<FreeSharedBlockRequest>(rq_.connection_id(), addr_);
  }

 private:
  bool support_local_bypass_;
  int64_t addr_;
  int64_t sz_;
  row_id_type row_id_from_server_;
};

/// \brief Request to fetch rows in batch
class BatchFetchRequest : public BaseRequest {
 public:
  friend class CacheServer;
  friend class CacheService;
  BatchFetchRequest(connection_id_type connection_id, const std::vector<row_id_type> &row_id, bool local_bypass);
  ~BatchFetchRequest() = default;
  Status RestoreRows(TensorTable *out, const void *baseAddr, int64_t *out_addr);

 private:
  bool support_local_bypass_;
  std::vector<row_id_type> row_id_;
};

/// \brief Request to create a cache for the current connection
class CreateCacheRequest : public BaseRequest {
 public:
  friend class CacheServer;
  enum class CreateCacheFlag : uint32_t { kNone = 0, kSpillToDisk = 1, kGenerateRowId = 1u << 1L };

  /// \brief Constructor
  /// \param connection_id
  /// \param cache_mem_sz Maximum memory assigned for this connection. 0 means unlimited
  /// \param flag Attributes of the cache.
  explicit CreateCacheRequest(const CacheClientInfo &cinfo, uint64_t cache_mem_sz,
                              CreateCacheFlag flag = CreateCacheFlag::kNone);
  ~CreateCacheRequest() = default;
  void ParseResult(connection_id_type *id, std::string *out) {
    auto p = flatbuffers::GetRoot<CreateCacheReplyMsg>(reply_.result().data());
    *id = p->connection_id();
    *out = p->cookie()->str();
  }

  /// Overload the base class Prepare
  Status Prepare() override;

 private:
  uint64_t cache_mem_sz_;
  CreateCacheFlag flag_;
};

/// \brief Request to purge a cache.
class PurgeCacheRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit PurgeCacheRequest(connection_id_type connection_id) : BaseRequest(RequestType::kPurgeCache) {
    rq_.set_connection_id(connection_id);
  }
  ~PurgeCacheRequest() = default;
};

/// \brief Request to destroy a cache
class DestroyCacheRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit DestroyCacheRequest(connection_id_type connection_id) : BaseRequest(RequestType::kDestroyCache) {
    rq_.set_connection_id(connection_id);
  }
  ~DestroyCacheRequest() = default;
};

/// \brief Obtain the statistics of the current connection
class GetStatRequest : public BaseRequest {
 public:
  friend class CacheServer;
  friend class CacheService;
  explicit GetStatRequest(connection_id_type connection_id) : BaseRequest(RequestType::kGetStat) {
    rq_.set_connection_id(connection_id);
  }

  ~GetStatRequest() = default;

  /// \brief Override base function to process the result.
  Status PostReply() override;

  void GetStat(CacheServiceStat *stat) {
    if (stat != nullptr) {
      (*stat) = stat_;
    }
  }

 private:
  CacheServiceStat stat_{};
};

/// \brief Request to cache a schema
class CacheSchemaRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit CacheSchemaRequest(connection_id_type connection_id) : BaseRequest(RequestType::kCacheSchema) {
    rq_.set_connection_id(connection_id);
  }
  ~CacheSchemaRequest() = default;

  Status SerializeCacheSchemaRequest(const std::unordered_map<std::string, int32_t> &map);
};

/// \brief Request to fetch a schema
class FetchSchemaRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit FetchSchemaRequest(connection_id_type connection_id) : BaseRequest(RequestType::kFetchSchema) {
    rq_.set_connection_id(connection_id);
  }
  ~FetchSchemaRequest() = default;

  Status PostReply() override;

  std::unordered_map<std::string, int32_t> GetColumnMap();

 private:
  std::unordered_map<std::string, int32_t> column_name_id_map_;
};

/// \brief Request to change a cache from build phase to read phase. Applies to non-mappable cache only.
class BuildPhaseDoneRequest : public BaseRequest {
 public:
  friend class CacheServer;
  BuildPhaseDoneRequest(connection_id_type connection_id, const std::string &cookie)
      : BaseRequest(RequestType::kBuildPhaseDone), cookie_(cookie) {
    rq_.set_connection_id(connection_id);
    rq_.add_buf_data(cookie_);
  }
  ~BuildPhaseDoneRequest() = default;

 private:
  std::string cookie_;
};

/// \brief Request to drop all the caches in the current session
class DropSessionRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit DropSessionRequest(const CacheClientInfo &cinfo) : BaseRequest(RequestType::kDropSession) {
    rq_.mutable_connection_info()->operator=(cinfo);
  }
  ~DropSessionRequest() = default;
};

class GenerateSessionIdRequest : public BaseRequest {
 public:
  friend class CacheServer;
  GenerateSessionIdRequest() : BaseRequest(RequestType::kGenerateSessionId) {
    // We don't have anything client info nor connection id to send. But we will manually
    // set the connection id to 0.
    rq_.set_connection_id(0);
  }

  ~GenerateSessionIdRequest() = default;

  session_id_type GetSessionId() { return atoi(reply_.result().data()); }
};

class AllocateSharedBlockRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit AllocateSharedBlockRequest(connection_id_type connection_id, size_t requestedSz)
      : BaseRequest(RequestType::kAllocateSharedBlock) {
    rq_.set_connection_id(connection_id);
    rq_.add_buf_data(std::to_string(requestedSz));
  }
  ~AllocateSharedBlockRequest() = default;

  /// \brief On return from the server, we get the (relative) address where
  /// the free block is located.
  /// \return
  int64_t GetAddr() {
    auto addr = strtoll(reply_.result().data(), nullptr, 10);
    return addr;
  }
};

class ShutdownRequest : public BaseRequest {
 public:
  friend class CacheServer;
  ShutdownRequest() : BaseRequest(RequestType::kStopService) {}
  ~ShutdownRequest() = default;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_SERVICE_H_
