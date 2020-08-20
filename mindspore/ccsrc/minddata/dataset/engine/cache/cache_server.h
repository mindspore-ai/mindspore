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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_SERVER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_SERVER_H_

#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include "minddata/dataset/engine/cache/cache_service.h"
#include "minddata/dataset/engine/cache/cache_grpc_server.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/arena.h"
#include "minddata/dataset/util/cache_pool.h"
#include "minddata/dataset/util/lock.h"
#include "minddata/dataset/util/service.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/system_pool.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
/// \brief A server which provides CacheService services.
class CacheServer : public Service {
 public:
  friend class Services;
  using cache_index = std::map<connection_id_type, std::unique_ptr<CacheService>>;
  class Builder {
   public:
    Builder() : top_("/tmp"), num_workers_(32), port_(50052), shared_memory_sz_in_gb_(4), memory_cap_ratio_(0.8) {}

    ~Builder() = default;

    /// \brief Getter functions
    const std::string &GetTop() const { return top_; }
    int32_t GetNumWorkers() const { return num_workers_; }
    int32_t GetPort() const { return port_; }
    int32_t GetSharedMemorySzInGb() const { return shared_memory_sz_in_gb_; }
    float GetMemoryCapRatio() const { return memory_cap_ratio_; }

    Builder &SetRootDirectory(std::string root) {
      top_ = std::move(root);
      return *this;
    }
    Builder &SetNumWorkers(int32_t n) {
      num_workers_ = n;
      return *this;
    }
    Builder &SetPort(int32_t p) {
      port_ = p;
      return *this;
    }
    Builder &SetSharedMemorySizeInGB(int32_t sz) {
      shared_memory_sz_in_gb_ = sz;
      return *this;
    }
    Builder &SetMemoryCapRatio(float ratio) {
      memory_cap_ratio_ = ratio;
      return *this;
    }

    Status SanityCheck();

    void Print(std::ostream &out) const {
      out << "Summary of the cache server configuration\n"
          << "Spill directory: " << GetTop() << "\n"
          << "Number of parallel workers: " << GetNumWorkers() << "\n"
          << "Tcp/ip port: " << GetPort() << "\n"
          << "Shared memory size (in GB): " << GetSharedMemorySzInGb() << "\n"
          << "Memory cap ratio: " << GetMemoryCapRatio();
    }

    friend std::ostream &operator<<(std::ostream &out, const Builder &bld) {
      bld.Print(out);
      return out;
    }

    Status Build() {
      RETURN_IF_NOT_OK(SanityCheck());
      // We need to bring up the Task Manager by bringing up the Services singleton.
      RETURN_IF_NOT_OK(Services::CreateInstance());
      RETURN_IF_NOT_OK(
        CacheServer::CreateInstance(top_, num_workers_, port_, shared_memory_sz_in_gb_, memory_cap_ratio_));
      return Status::OK();
    }

   private:
    std::string top_;
    int32_t num_workers_;
    int32_t port_;
    int32_t shared_memory_sz_in_gb_;
    float memory_cap_ratio_;

    /// \brief Sanity checks on the shared memory.
    /// \return Status object
    Status IpcResourceCleanup();
  };

  CacheServer(const CacheServer &) = delete;
  CacheServer &operator=(const CacheServer &) = delete;
  CacheServer(CacheServer &&) = delete;
  CacheServer &operator=(CacheServer &) = delete;
  Status DoServiceStart() override;
  Status DoServiceStop() override;
  ~CacheServer() override { (void)ServiceStop(); }

  static Status CreateInstance(const std::string &spill_path, int32_t num_workers, int32_t port,
                               int32_t shared_memory_sz, float memory_cap_ratio) {
    std::call_once(init_instance_flag_, [&]() -> Status {
      auto &SvcManager = Services::GetInstance();
      RETURN_IF_NOT_OK(
        SvcManager.AddHook(&instance_, spill_path, num_workers, port, shared_memory_sz, memory_cap_ratio));
      return Status::OK();
    });
    return Status::OK();
  }

  static CacheServer &GetInstance() { return *instance_; }

  /// \brief For the current demonstration, a cache client contacts cache server using a Queue.
  /// \param rq
  /// \return Status object
  Status PushRequest(int32_t queue_id, CacheServerRequest *rq) {
    RETURN_UNEXPECTED_IF_NULL(rq);
    RETURN_IF_NOT_OK(cache_q_->operator[](queue_id)->Add(rq));
    return Status::OK();
  }

  /// \\brief Kick off server threads. Never return unless error out.
  Status Run(SharedMessage::queue_id_t msg_qid);

  /// \brief Get a free tag
  /// \param q[in] pointer to a pointer to a CacheServerRequest
  /// \return Status object
  static Status GetFreeRequestTag(int32_t queue_id, CacheServerRequest **q);

  /// \brief Return a tag to the free list
  /// \param p[in] pointer to already finished CacheServerRequest tag
  /// \return Status object
  static Status ReturnRequestTag(CacheServerRequest *p);

  /// \brief This returns the size (in bytes) of the physical RAM on the machine.
  /// \return the size (in bytes) of the physical RAM on the machine.
  static int64_t GetTotalSystemMemory();

  /// \brief Internally this is how much we will try to use without exceeding the limit
  /// \return Internal cap maximum
  int64_t GetAvailableSystemMemory() { return memory_cap_; }

  /// \brief Find out the current memory usage
  int64_t GetMemoryUsage() { return cur_mem_usage_; }

  /// \brief This updates our current memory usage.
  enum MemUsageOp : int8_t { kAllocate = 1, kFree = 2 };
  void UpdateMemoryUsage(int64_t sz, MemUsageOp op) {
    if (op == MemUsageOp::kAllocate) {
      cur_mem_usage_ += sz;
    } else {
      cur_mem_usage_ -= sz;
    }
  }

 private:
  static std::once_flag init_instance_flag_;
  static CacheServer *instance_;
  mutable RWLock rwLock_;
  mutable RWLock sessions_lock_;
  std::string top_;
  cache_index all_caches_;
  std::map<session_id_type, std::set<connection_id_type>> active_sessions_;
  std::shared_ptr<QueueList<CacheServerRequest *>> cache_q_;
  std::shared_ptr<QueueList<CacheServerRequest *>> free_list_;
  std::vector<std::unique_ptr<MemGuard<CacheServerRequest, Allocator<CacheServerRequest>>>> tag_;
  std::shared_ptr<CacheServerGreeterImpl> comm_layer_;
  std::shared_ptr<MemoryPool> mp_;
  TaskGroup vg_;
  int32_t num_workers_;
  int32_t port_;
  int32_t shared_memory_sz_in_gb_;
  std::atomic<bool> global_shutdown_;
  float memory_cap_ratio_;
  int64_t memory_cap_;
  std::atomic<int64_t> cur_mem_usage_;

  /// \brief Constructor
  /// \param spill_path Top directory for spilling buffers to.
  /// \param num_workers Number of threads for handling requests.
  explicit CacheServer(const std::string &spill_path, int32_t num_workers, int32_t port, int32_t share_memory_sz_in_gb,
                       float memory_cap_ratio);

  /// \brief Locate a cache service from connection id.
  /// \return Pointer to cache service. Null if not found
  CacheService *GetService(connection_id_type id) const;

  /// \brief Create a cache service. We allow multiple clients to create the same cache service.
  /// Subsequent duplicate requests are ignored. The first cache client to create the service will be given
  /// a special unique cookie.
  /// \return Status object
  Status CreateService(CacheRequest *rq, CacheReply *reply);

  /// \brief Destroy a cache service
  /// \param rq
  /// \return Status object
  Status DestroyCache(CacheRequest *rq);

  /// \brief Entry point for all internal server threads.
  Status ServerRequest(int32_t worker_id);

  /// \brief Entry point for all grpc threads.
  /// \return
  Status RpcRequest(int32_t worker_id);

  Status DestroySession(CacheRequest *rq);

  /// \brief Create a connection id from a session id and a crc
  /// \param session_id
  /// \param crc
  /// \return connection id
  connection_id_type GetConnectionID(session_id_type session_id, uint32_t crc) const;

  /// \brief Extract the session id from a connection id
  /// \param connection_id
  /// \return session id
  session_id_type GetSessionID(connection_id_type connection_id) const;

  /// \brief Generate a session ID for the client
  /// \return Session ID
  session_id_type GenerateSessionID();

  /// \brief Handle kAllocateSharedBlock request
  /// \param rq CacheRequest
  /// \param reply CacheReply
  /// \return Status object
  Status AllocateSharedMemory(CacheRequest *rq, CacheReply *reply);

  /// \brief Handle kFreeSharedBlock request
  /// \param rq
  /// \return Status object
  Status FreeSharedMemory(CacheRequest *rq);

  /// \brief Handle CacheRow request
  /// \note There are two different implementation depends if shared memory is used for transportation.
  /// \return Status object
  Status FastCacheRow(CacheRequest *rq, CacheReply *reply);
  Status CacheRow(CacheRequest *rq, CacheReply *reply);

  /// \brief Internal function to do row batch fetch
  /// \param rq Request
  /// \param reply Reply
  /// \return Status object
  Status BatchFetchRows(CacheRequest *rq, CacheReply *reply);

  /// \brief Internal function to get statistics
  /// \param rq
  /// \param reply
  /// \return Status object
  Status GetStat(CacheRequest *rq, CacheReply *reply);

  /// \brief Cache a schema request
  /// \param rq
  /// \return Status object
  Status CacheSchema(CacheRequest *rq);

  /// \brief Fetch a schema request
  /// \param rq
  /// \param reply
  /// \return Status object
  Status FetchSchema(CacheRequest *rq, CacheReply *reply);

  /// \brief Mark Build phase done (for non-mappable case)
  /// \param rq
  /// \return Status object
  Status BuildPhaseDone(CacheRequest *rq);

  /// \brief A proper shutdown of the server
  /// \return Status object
  Status GlobalShutdown();

  /// \brief Find keys that will be cache miss
  /// \return Status object
  Status GetCacheMissKeys(CacheRequest *rq, CacheReply *reply);

  /// \brief Toggle write mode for a service
  Status ToggleWriteMode(CacheRequest *rq);

  /// \brief List the sessions and their caches
  /// \param reply
  /// \return Status object
  Status ListSessions(CacheReply *reply);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CACHE_TENSOR_H_
