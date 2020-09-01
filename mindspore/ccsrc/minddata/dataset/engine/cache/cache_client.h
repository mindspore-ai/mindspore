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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_CLIENT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_CLIENT_H_

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/core/config_manager.h"
#ifdef ENABLE_CACHE
#include "minddata/dataset/engine/cache/cache_grpc_client.h"
#else
#include "minddata/dataset/engine/cache/stub/cache_grpc_client.h"
#endif
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/util/lock.h"

namespace mindspore {
namespace dataset {
/// \brief A CacheClient is a bridge between a DatasetOp and a CacheServer. All communications are through
/// a CacheClient. Typical tasks including like creating a cache service, cache a data buffer, restore a previously
/// rows, etc.
class CacheClient {
 public:
  friend class CacheMergeOp;

  /// \brief A builder to help creating a CacheClient object
  class Builder {
   public:
    Builder();

    ~Builder() = default;

    /// Setter function to set the session id
    /// \param session_id
    /// \return Builder object itself.
    Builder &SetSessionId(session_id_type session_id) {
      session_id_ = session_id;
      return *this;
    }

    /// Setter function to set the cache memory size
    /// \param cache_mem_sz
    /// \return Builder object itself
    Builder &SetCacheMemSz(uint64_t cache_mem_sz) {
      cache_mem_sz_ = cache_mem_sz;
      return *this;
    }

    /// Setter function to spill attribute
    /// \param spill
    /// Builder object itself
    Builder &SetSpill(bool spill) {
      spill_ = spill;
      return *this;
    }

    /// Setter function to set rpc hostname
    /// \param host
    /// \return Builder object itself
    Builder &SetHostname(std::string host) {
      hostname_ = std::move(host);
      return *this;
    }

    /// Setter function to set tcpip port
    /// \param port
    /// \return Builder object itself.
    Builder &SetPort(int32_t port) {
      port_ = port;
      return *this;
    }

    /// Setter function to set number of async rpc workers
    /// \param num_workers
    /// \return Builder object itself
    Builder &SetNumWorkers(int32_t num_workers) {
      num_workers_ = num_workers;
      return *this;
    }

    /// Setter function to set prefetch amount for fetching rows from cache server
    /// \param prefetch_sz
    /// \return Builder object itself
    Builder &SetPrefetchSize(int32_t prefetch_sz) {
      prefetch_size_ = prefetch_sz;
      return *this;
    }

    /// Getter functions
    session_id_type getSessionId() const { return session_id_; }
    uint64_t getCacheMemSz() const { return cache_mem_sz_; }
    bool isSpill() const { return spill_; }
    const std::string &getHostname() const { return hostname_; }
    int32_t getPort() const { return port_; }
    int32_t getNumWorkers() const { return num_workers_; }
    int32_t getPrefetchSize() const { return prefetch_size_; }

    Status SanityCheck();

    Status Build(std::shared_ptr<CacheClient> *out);

   private:
    session_id_type session_id_;
    uint64_t cache_mem_sz_;
    bool spill_;
    std::string hostname_;
    int32_t port_;
    int32_t num_workers_;
    int32_t prefetch_size_;
  };

  /// \brief Constructor
  /// \param session_id A user assigned session id for the current pipeline
  /// \param cache_mem_sz Size of the memory set aside for the row caching. 0 for unlimited
  /// \param spill Spill to disk if out of memory
  CacheClient(session_id_type session_id, uint64_t cache_mem_sz, bool spill, std::string hostname, int32_t port,
              int32_t num_workers, int32_t prefetch_size);

  /// \brief Destructor
  ~CacheClient() { (void)comm_->ServiceStop(); }

  /// \brief Send a TensorRow to the cache server
  /// \param[in] row
  /// \param[out] row_id_from_server Optional. The row id assigned by the server for non-mappable dataset
  /// \return return code
  Status WriteRow(const TensorRow &row, row_id_type *row_id_from_server = nullptr) const;

  /// \brief Send a DataBuffer to the cache server
  /// \param in Unique pointer of the DataBuffer to be cached
  /// \return return code
  Status WriteBuffer(std::unique_ptr<DataBuffer> &&in) const;

  /// \brief Fetch a list of rows from the cache server. An empty TensorRow will be returned if there is
  /// any cache miss
  /// \param row_id A vector of row id's
  /// \param out A TensorTable of TensorRows.
  /// \return return code
  Status GetRows(const std::vector<row_id_type> &row_id, TensorTable *out) const;

  /// \brief Create a cache.
  /// \param tree_crc  A crc that was generated during tree prepare phase
  /// \param generate_id Let the cache service generate row id
  /// \return Status object
  Status CreateCache(uint32_t tree_crc, bool generate_id);

  /// \brief Purge a cache. Cache can be reused after reset.
  /// \return Status object
  Status PurgeCache();

  /// \brief Destroy a cache. Like Purge but the cache is deleted and can't be reused.
  /// \return Status object
  Status DestroyCache();

  /// \brief Get the statistics from a cache.
  /// \param[in/out] Pointer to a pre-allocated ServiceStat object
  /// \return Status object
  Status GetStat(CacheServiceStat *);

  /// \brief Cache the schema at the cache server
  /// \param map The unordered map of the schema
  /// \return Status object
  Status CacheSchema(const std::unordered_map<std::string, int32_t> &map);

  /// \brief Fetch the schema from the cache server
  /// \param map Pointer to pre-allocated map object
  /// \return Status object.
  Status FetchSchema(std::unordered_map<std::string, int32_t> *map);

  /// \brief Change the state from build phase to read phase. Applicable to non-mappable dataset only. Only the cache
  /// client that holds cookie can be allowed to make this request
  /// \return Status object
  Status BuildPhaseDone() const;

  /// \brief A print method typically used for debugging
  /// \param out The output stream to write output to
  void Print(std::ostream &out) const;

  /// \brief Stream output operator overload
  /// \return the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const CacheClient &cc) {
    cc.Print(out);
    return out;
  }

  /// \brief Every cache server has a cookie which uniquely identifies the CacheClient that creates it.
  /// \return Cookie
  std::string cookie() const { return cookie_; }

  /// \brief Send a request async to the server
  /// \param rq BaseRequest
  /// \return Status object
  Status PushRequest(std::shared_ptr<BaseRequest> rq) const;

  /// \brief If the remote server supports local bypass using shared memory
  /// \return boolean value
  bool SupportLocalClient() const { return local_bypass_; }

  /// \brief Return the base memory address if we attach to any shared memory.
  auto SharedMemoryBaseAddr() const { return comm_->SharedMemoryBaseAddr(); }

  /// Getter functions
  session_id_type session_id() const { return cinfo_.session_id(); }
  uint64_t getCacheMemSz() const { return cache_mem_sz_; }
  bool isSpill() const { return spill_; }
  const std::string &getHostname() const { return hostname_; }
  int32_t getPort() const { return port_; }
  int32_t getNumWorkers() const { return num_workers_; }
  int32_t getPrefetchSize() const { return prefetch_size_; }

 private:
  mutable RWLock mux_;
  uint64_t cache_mem_sz_;
  bool spill_;
  // The session_id_ and cache_crc_ work together to uniquely identify this particular cache and allow
  // sharing of the cache.
  CacheClientInfo cinfo_;
  // The server_connection_id_ is the actual id we use for operations after the cache is built
  connection_id_type server_connection_id_;
  // Some magic cookie returned from the cache server.
  std::string cookie_;
  // Comm layer
  bool local_bypass_;
  std::string hostname_;
  int32_t port_;
  int32_t num_workers_;
  int32_t prefetch_size_;
  mutable std::shared_ptr<CacheClientGreeter> comm_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_CLIENT_H_
