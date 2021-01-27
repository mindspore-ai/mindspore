/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd

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
  int64_t num_numa_hit;
  row_id_type min_row_id;
  row_id_type max_row_id;
  int8_t cache_service_state;
};

struct CacheServerCfgInfo {
  int32_t num_workers;
  int8_t log_level;
  std::string spill_dir;
};

/// \brief Info structure ListSessionsRequest
struct SessionCacheInfo {
  session_id_type session_id;
  connection_id_type connection_id;
  CacheServiceStat stats;
};

/// \brief CacheClient communicates with CacheServer using Requests.
class BaseRequest {
 public:
  // Request types
  enum class RequestType : int16_t {
    kCacheRow = 0,
    kBatchFetchRows = 1,
    kCreateCache = 2,
    kGetCacheMissKeys = 3,
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
    kHeartBeat = 14,
    kToggleWriteMode = 15,
    kListSessions = 16,
    kConnectReset = 17,
    kInternalFetchRow = 18,
    kBatchCacheRows = 19,
    kInternalCacheRow = 20,
    kGetCacheState = 21,
    // Add new request before it.
    kRequestUnknown = 32767
  };

  friend class CacheServer;
  friend class CacheServerRequest;
  friend class CacheClientGreeter;
  friend class CacheClientRequestTag;
  friend class CacheClient;
  friend class CacheService;
  friend class CacheServerGreeterImpl;

  /// \brief Base class of a cache server request
  /// \param type Type of the request
  explicit BaseRequest(RequestType type) : type_(type) {
    rq_.set_type(static_cast<int16_t>(type_));
    rq_.set_client_id(-1);
    rq_.set_flag(0);
  }
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

  /// \brief Return if the request is of row request type
  /// \return True if the request is row-related request
  bool IsRowRequest() const {
    return type_ == RequestType::kBatchCacheRows || type_ == RequestType::kBatchFetchRows ||
           type_ == RequestType::kInternalCacheRow || type_ == RequestType::kInternalFetchRow ||
           type_ == RequestType::kCacheRow;
  }

  /// \brief Return if the request is of admin request type
  /// \return True if the request is admin-related request
  bool IsAdminRequest() const {
    return type_ == RequestType::kCreateCache || type_ == RequestType::kDestroyCache ||
           type_ == RequestType::kGetStat || type_ == RequestType::kGetCacheState ||
           type_ == RequestType::kAllocateSharedBlock || type_ == RequestType::kFreeSharedBlock ||
           type_ == RequestType::kCacheSchema || type_ == RequestType::kFetchSchema ||
           type_ == RequestType::kBuildPhaseDone || type_ == RequestType::kToggleWriteMode ||
           type_ == RequestType::kConnectReset || type_ == RequestType::kStopService ||
           type_ == RequestType::kHeartBeat || type_ == RequestType::kGetCacheMissKeys;
  }

  /// \brief Return if the request is of session request type
  /// \return True if the request is session-related request
  bool IsSessionRequest() const {
    return type_ == RequestType::kGenerateSessionId || type_ == RequestType::kDropSession ||
           type_ == RequestType::kListSessions;
  }

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
  explicit FreeSharedBlockRequest(connection_id_type connection_id, int32_t client_id, int64_t addr)
      : BaseRequest(RequestType::kFreeSharedBlock) {
    rq_.set_connection_id(connection_id);
    rq_.add_buf_data(std::to_string(addr));
    rq_.set_client_id(client_id);
  }
  ~FreeSharedBlockRequest() override = default;
};

/// \brief Request to cache a single TensorRow
class CacheRowRequest : public BaseRequest {
 public:
  friend class CacheServer;
  friend class CacheClient;
  explicit CacheRowRequest(const CacheClient *cc);
  ~CacheRowRequest() override = default;

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
    return std::make_shared<FreeSharedBlockRequest>(rq_.connection_id(), rq_.client_id(), addr_);
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
  BatchFetchRequest(const CacheClient *cc, const std::vector<row_id_type> &row_id);
  ~BatchFetchRequest() override = default;
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
  explicit CreateCacheRequest(CacheClient *cc, const CacheClientInfo &cinfo, uint64_t cache_mem_sz,
                              CreateCacheFlag flag = CreateCacheFlag::kNone);
  ~CreateCacheRequest() override = default;

  /// Overload the base class Prepare/PostReply
  Status Prepare() override;
  Status PostReply() override;

 private:
  uint64_t cache_mem_sz_;
  CreateCacheFlag flag_;
  CacheClient *cc_;
};

/// \brief Request to get all the keys not present at the server.
/// \note Only applicable to mappable case
class GetCacheMissKeysRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit GetCacheMissKeysRequest(connection_id_type connection_id) : BaseRequest(RequestType::kGetCacheMissKeys) {
    rq_.set_connection_id(connection_id);
  }
  ~GetCacheMissKeysRequest() override = default;
};

/// \brief Request to destroy a cache
class DestroyCacheRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit DestroyCacheRequest(connection_id_type connection_id) : BaseRequest(RequestType::kDestroyCache) {
    rq_.set_connection_id(connection_id);
  }
  ~DestroyCacheRequest() override = default;
};

/// \brief Obtain the statistics of the current connection
class GetStatRequest : public BaseRequest {
 public:
  friend class CacheServer;
  friend class CacheService;
  explicit GetStatRequest(connection_id_type connection_id) : BaseRequest(RequestType::kGetStat) {
    rq_.set_connection_id(connection_id);
  }

  ~GetStatRequest() override = default;

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

/// \brief Get the state of a cache service
class GetCacheStateRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit GetCacheStateRequest(connection_id_type connection_id)
      : BaseRequest(RequestType::kGetCacheState), cache_service_state_(0) {
    rq_.set_connection_id(connection_id);
  }
  ~GetCacheStateRequest() override = default;

  Status PostReply() override;

  auto GetState() const { return cache_service_state_; }

 private:
  int8_t cache_service_state_;
};

/// \brief Request to cache a schema
class CacheSchemaRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit CacheSchemaRequest(connection_id_type connection_id) : BaseRequest(RequestType::kCacheSchema) {
    rq_.set_connection_id(connection_id);
  }
  ~CacheSchemaRequest() override = default;

  Status SerializeCacheSchemaRequest(const std::unordered_map<std::string, int32_t> &map);
};

/// \brief Request to fetch a schema
class FetchSchemaRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit FetchSchemaRequest(connection_id_type connection_id) : BaseRequest(RequestType::kFetchSchema) {
    rq_.set_connection_id(connection_id);
  }
  ~FetchSchemaRequest() override = default;

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
  ~BuildPhaseDoneRequest() override = default;

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
  ~DropSessionRequest() override = default;
};

class GenerateSessionIdRequest : public BaseRequest {
 public:
  friend class CacheServer;
  GenerateSessionIdRequest() : BaseRequest(RequestType::kGenerateSessionId) {
    // We don't have anything client info nor connection id to send. But we will manually
    // set the connection id to 0.
    rq_.set_connection_id(0);
  }

  ~GenerateSessionIdRequest() override = default;

  session_id_type GetSessionId() { return atoi(reply_.result().data()); }
};

class ListSessionsRequest : public BaseRequest {
 public:
  friend class CacheServer;
  ListSessionsRequest() : BaseRequest(RequestType::kListSessions) {
    // This request is not specific to any cache or session
    rq_.set_connection_id(0);
  }

  ~ListSessionsRequest() override = default;

  /// \brief Override base function to process the result.
  Status PostReply() override;

  void GetSessionCacheInfo(std::vector<SessionCacheInfo> *info) {
    if (info != nullptr) {
      (*info) = session_info_list_;
    }
  }

  std::vector<SessionCacheInfo> GetSessionCacheInfo() { return session_info_list_; }

  std::vector<session_id_type> GetSessionIds() {
    std::vector<session_id_type> session_ids;
    for (auto session_info : session_info_list_) {
      session_ids.push_back(session_info.session_id);
    }
    return session_ids;
  }

  CacheServerCfgInfo GetServerStat() { return server_cfg_; }

 private:
  std::vector<SessionCacheInfo> session_info_list_;
  CacheServerCfgInfo server_cfg_{};
};

class AllocateSharedBlockRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit AllocateSharedBlockRequest(connection_id_type connection_id, int32_t client_id, size_t requestedSz)
      : BaseRequest(RequestType::kAllocateSharedBlock) {
    rq_.set_connection_id(connection_id);
    rq_.add_buf_data(std::to_string(requestedSz));
    rq_.set_client_id(client_id);
  }
  ~AllocateSharedBlockRequest() override = default;

  /// \brief On return from the server, we get the (relative) address where
  /// the free block is located.
  /// \return
  int64_t GetAddr() {
    auto addr = strtoll(reply_.result().data(), nullptr, 10);
    return addr;
  }
};

class ToggleWriteModeRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit ToggleWriteModeRequest(connection_id_type connection_id, bool on_off)
      : BaseRequest(RequestType::kToggleWriteMode) {
    rq_.set_connection_id(connection_id);
    rq_.add_buf_data(on_off ? "on" : "off");
  }
  ~ToggleWriteModeRequest() override = default;
};

class ServerStopRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit ServerStopRequest(int32_t qID) : BaseRequest(RequestType::kStopService) {
    rq_.add_buf_data(std::to_string(qID));
  }
  ~ServerStopRequest() = default;
  Status PostReply() override;
};

class ConnectResetRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit ConnectResetRequest(connection_id_type connection_id, int32_t client_id)
      : BaseRequest(RequestType::kConnectReset) {
    rq_.set_connection_id(connection_id);
    rq_.set_client_id(client_id);
  }
  ~ConnectResetRequest() override = default;

  /// Override the base class function
  Status Prepare() override {
    CHECK_FAIL_RETURN_UNEXPECTED(rq_.client_id() != -1, "Invalid client id");
    return Status::OK();
  }
};

class BatchCacheRowsRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit BatchCacheRowsRequest(const CacheClient *cc, int64_t addr, int32_t num_ele);
  ~BatchCacheRowsRequest() override = default;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_SERVICE_H_
