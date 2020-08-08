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

#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/cache/cache_server.h"
#include "minddata/dataset/engine/cache/de_tensor_generated.h"
#include "minddata/dataset/util/lock.h"

namespace mindspore {
namespace dataset {
/// \brief A CacheClient is a bridge between a DatasetOp and a CacheServer. All communications are through
/// a CacheClient. Typical tasks including like creating a cache service, cache a data buffer, restore a previously
/// rows, etc.
class CacheClient {
 public:
  /// \brief Constructor
  /// \param session_id A user assigned session id for the current pipeline
  /// \param cache_mem_sz Size of the memory set aside for the row caching. 0 for unlimited
  /// \param spill Spill to disk if out of memory
  CacheClient(uint32_t session_id, uint64_t cache_mem_sz, bool spill);

  /// \brief Destructor
  ~CacheClient() = default;

  /// \brief Getter function for returning the current session id
  /// \return session id
  uint64_t session_id() const { return session_id_; }

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
  struct ServiceStat {
    int64_t num_mem_cached;
    int64_t num_disk_cached;
    row_id_type min_row_id;
    row_id_type max_row_id;
    int8_t cache_service_state;
  };
  Status GetStat(ServiceStat *);

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

 private:
  mutable RWLock mux_;
  uint64_t cache_mem_sz_;
  bool spill_;
  // The session_id_ and cache_crc_ work together to uniquely identify this particular cache and allow
  // sharing of the cache.
  uint32_t session_id_;
  uint32_t cache_crc_;
  // The server_connection_id_ is the actual id we use for operations after the cache is built
  connection_id_type server_connection_id_;
  // Some magic cookie returned from the cache server.
  std::string cookie_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_CLIENT_H_
