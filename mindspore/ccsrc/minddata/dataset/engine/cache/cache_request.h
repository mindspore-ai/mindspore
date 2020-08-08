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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/engine/cache/de_tensor_generated.h"
#include "minddata/dataset/util/slice.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
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
    // Add new request before it.
    kRequestUnknown = 32767
  };
  // For kCreateCache
  enum class CreateCacheFlag : uint32_t { kNone = 0, kSpillToDisk = 1, kGenerateRowId = 1u << 1L };
  friend class CacheServer;
  /// \brief Base class of a cache server request
  /// \param connection_id A combination of session id and crc that uniquely identifies a connection.
  /// \param type Type of the request
  explicit BaseRequest(connection_id_type connection_id, RequestType type)
      : type_(type), connection_id_(connection_id) {}
  virtual ~BaseRequest() = default;
  /// \brief Wait for the completion of a request
  /// \return Status returned from the cache server
  Status Wait() {
    RETURN_IF_NOT_OK(wp_.Wait());
    return rc_;
  }

  /// \brief Getter function of the current connection id
  /// \return Connection id
  connection_id_type GetServerConnectionId() const { return connection_id_; }

 private:
  RequestType type_;
  connection_id_type connection_id_;
  Status rc_;
  WaitPost wp_;
};
/// \brief Request to cache a single TensorRow
class CacheRowRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit CacheRowRequest(connection_id_type connection_id, const std::string &cookie)
      : BaseRequest(connection_id, RequestType::kCacheRow), row_id_from_server_(-1), cookie_(cookie) {}
  ~CacheRowRequest() = default;

  /// \brief Serialize a TensorRow for streaming to the cache server
  /// \param row TensorRow
  /// \return Status object
  Status SerializeCacheRowRequest(const TensorRow &row);
  /// \brief Return the row id assigned to this row for non-mappable dataset
  /// \return row id of the cached row
  row_id_type GetRowIdAfterCache() { return row_id_from_server_; }

 private:
  std::shared_ptr<flatbuffers::FlatBufferBuilder> fbb_;
  row_id_type row_id_from_server_;
  std::vector<const void *> buffers_;
  std::string cookie_;

  /// \brief Private function to serialize one TensorRow
  /// \param row TensorRow
  /// \return Status object
  Status SerializeTensorRowHeader(const TensorRow &row);
  /// \brief Private function to serialize one Tensor
  /// \param ts_ptr Tensor
  /// \return Status object
  Status SerializeOneTensorMeta(const std::shared_ptr<Tensor> &ts_ptr, flatbuffers::Offset<TensorMetaMsg> *out_off);
};
/// \brief Request to fetch rows in batch
class BatchFetchRequest : public BaseRequest {
 public:
  friend class CacheServer;
  friend class CacheService;
  BatchFetchRequest(connection_id_type connection_id, const std::vector<row_id_type> &row_id)
      : BaseRequest(connection_id, RequestType::kBatchFetchRows), row_id_(row_id) {}
  ~BatchFetchRequest() = default;
  Status RestoreRows(TensorTable *out);

 private:
  std::vector<row_id_type> row_id_;
  MemGuard<uint8_t> mem_;
  Status RestoreOneTensor(const TensorMetaMsg *col_ts, const ReadableSlice &data, std::shared_ptr<Tensor> *out);
};
/// \brief Request to create a cache for the current connection
class CreationCacheRequest : public BaseRequest {
 public:
  friend class CacheServer;
  /// \brief Constructor
  /// \param connection_id
  /// \param cache_mem_sz Maximum memory assigned for this connection. 0 means unlimited
  /// \param flag Attributes of the cache.
  explicit CreationCacheRequest(connection_id_type connection_id, uint64_t cache_mem_sz,
                                CreateCacheFlag flag = CreateCacheFlag::kNone)
      : BaseRequest(connection_id, RequestType::kCreateCache), cache_mem_sz(cache_mem_sz), flag_(flag) {}

  ~CreationCacheRequest() = default;

  std::string cookie() const { return cookie_; }

 private:
  uint64_t cache_mem_sz;
  CreateCacheFlag flag_;
  std::string cookie_;
};
/// \brief Request to purge a cache.
class PurgeCacheRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit PurgeCacheRequest(connection_id_type connection_id) : BaseRequest(connection_id, RequestType::kPurgeCache) {}

  ~PurgeCacheRequest() = default;
};
/// \brief Request to destroy a cache
class DestroyCacheRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit DestroyCacheRequest(connection_id_type connection_id)
      : BaseRequest(connection_id, RequestType::kDestroyCache) {}

  /// \brief Destructor
  ~DestroyCacheRequest() = default;
};
/// \brief Obtain the statistics of the current connection
class GetStatRequest : public BaseRequest {
 public:
  friend class CacheServer;
  friend class CacheService;
  explicit GetStatRequest(connection_id_type connection_id) : BaseRequest(connection_id, RequestType::kGetStat) {}

  ~GetStatRequest() = default;

  row_id_type GetMinRowId() const {
    auto *msg = flatbuffers::GetRoot<ServiceStatMsg>(mem_.GetPointer());
    return msg->min_row_id();
  }
  row_id_type GetMaxRowId() const {
    auto *msg = flatbuffers::GetRoot<ServiceStatMsg>(mem_.GetPointer());
    return msg->max_row_id();
  }
  int64_t GetNumMemCached() const {
    auto *msg = flatbuffers::GetRoot<ServiceStatMsg>(mem_.GetPointer());
    return msg->num_mem_cached();
  }
  int64_t GetNumDiskCached() const {
    auto *msg = flatbuffers::GetRoot<ServiceStatMsg>(mem_.GetPointer());
    return msg->num_disk_cached();
  }
  uint8_t GetState() const {
    auto *msg = flatbuffers::GetRoot<ServiceStatMsg>(mem_.GetPointer());
    return msg->state();
  }

 private:
  MemGuard<uint8_t> mem_;
};
/// \brief Request to cache a schema
class CacheSchemaRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit CacheSchemaRequest(connection_id_type connection_id)
      : BaseRequest(connection_id, RequestType::kCacheSchema), buf_(nullptr), len_of_buf_(0) {}
  ~CacheSchemaRequest() = default;

  Status SerializeCacheSchemaRequest(const std::unordered_map<std::string, int32_t> &map);
  const void *GetBuffer() const { return buf_; }

 private:
  std::shared_ptr<flatbuffers::FlatBufferBuilder> fbb_;
  const void *buf_;
  int64_t len_of_buf_;
};
/// \brief Request to fetch a schema
class FetchSchemaRequest : public BaseRequest {
 public:
  friend class CacheServer;
  explicit FetchSchemaRequest(connection_id_type connection_id)
      : BaseRequest(connection_id, RequestType::kFetchSchema) {}
  ~FetchSchemaRequest() = default;

  std::unordered_map<std::string, int32_t> GetColumnMap();

 private:
  MemGuard<uint8_t> mem_;
  std::unordered_map<std::string, int32_t> column_name_id_map_;
};
/// \brief Request to change a cache from build phase to read phase. Applies to non-mappable cache only.
class BuildPhaseDoneRequest : public BaseRequest {
 public:
  friend class CacheServer;
  BuildPhaseDoneRequest(connection_id_type connection_id, const std::string &cookie)
      : BaseRequest(connection_id, RequestType::kBuildPhaseDone), cookie_(cookie) {}

  ~BuildPhaseDoneRequest() = default;

 private:
  std::string cookie_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_SERVICE_H_
