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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_SERVER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_SERVER_H_

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <thread>
#include "minddata/dataset/engine/cache/cache_arena.h"
#include "minddata/dataset/engine/cache/cache_hw.h"
#include "minddata/dataset/engine/cache/cache_numa.h"
#include "minddata/dataset/engine/cache/cache_service.h"
#include "minddata/dataset/engine/cache/cache_grpc_server.h"
#include "minddata/dataset/engine/cache/cache_pool.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/arena.h"
#include "minddata/dataset/util/lock.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/semaphore.h"
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
    Builder();

    ~Builder() = default;

    /// \brief Getter functions
    const std::string &GetTop() const { return top_; }
    int32_t GetNumWorkers() const { return num_workers_; }
    int32_t GetPort() const { return port_; }
    int32_t GetSharedMemorySzInGb() const { return shared_memory_sz_in_gb_; }
    float GetMemoryCapRatio() const { return memory_cap_ratio_; }
    int8_t GetLogLevel() const { return log_level_; }

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
    Builder &SetLogLevel(int8_t log_level) {
      log_level_ = log_level;
      return *this;
    }

    Status SanityCheck();

    void Print(std::ostream &out) const {
      out << "Summary of the cache server configuration\n"
          << "Spill directory: " << GetTop() << "\n"
          << "Number of parallel workers: " << GetNumWorkers() << "\n"
          << "Tcp/ip port: " << GetPort() << "\n"
          << "Shared memory size (in GB): " << GetSharedMemorySzInGb() << "\n"
          << "Memory cap ratio: " << GetMemoryCapRatio() << "\n"
          << "Log level: " << std::to_string(GetLogLevel());
    }

    friend std::ostream &operator<<(std::ostream &out, const Builder &bld) {
      bld.Print(out);
      return out;
    }

    Status Build() {
      // Get information of numa architecture and adjust num_workers_ based on numa count
      hw_info_ = std::make_shared<CacheServerHW>();
      RETURN_IF_NOT_OK(hw_info_->GetNumaNodeInfo());
      std::string warning_string;
      if (num_workers_ == -1) {
        // if the user did not provide a value for num_workers, set it to half of num_cpu as default and adjust it if
        // the default is not the optimal.
        int32_t dft_num_workers = std::thread::hardware_concurrency() > 2 ? std::thread::hardware_concurrency() / 2 : 1;
        num_workers_ = AdjustNumWorkers(dft_num_workers);
      } else {
        // if the users have given their own value, adjust it and provide a warning if it got changed.
        int32_t num_workers_new = AdjustNumWorkers(num_workers_);
        if (num_workers_ != num_workers_new) {
          warning_string =
            "The configuration of workers on the cache server is dependent on the NUMA architecture of the server. "
            "The current setting is not the optimal for the NUMA architecture. Re-adjusting the number of workers "
            "to optimal setting of " +
            std::to_string(num_workers_new) + ".\n";
          MS_LOG(INFO) << warning_string;
        }
        num_workers_ = num_workers_new;
      }
      RETURN_IF_NOT_OK(SanityCheck());
      // We need to bring up the Task Manager by bringing up the Services singleton.
      RETURN_IF_NOT_OK(Services::CreateInstance());
      RETURN_IF_NOT_OK(CacheServer::CreateInstance(top_, num_workers_, port_, shared_memory_sz_in_gb_,
                                                   memory_cap_ratio_, log_level_, std::move(hw_info_)));
      return Status(StatusCode::kSuccess, warning_string);
    }

   private:
    std::string top_;
    int32_t num_workers_;
    int32_t port_;
    int32_t shared_memory_sz_in_gb_;
    float memory_cap_ratio_;
    int8_t log_level_;
    std::shared_ptr<CacheServerHW> hw_info_;

    /// \brief Sanity checks on the shared memory.
    /// \return Status object
    Status IpcResourceCleanup();

    /// \brief Adjust the value of num_workers if it's not the optimal to NUMA architecture.
    int32_t AdjustNumWorkers(int32_t num_workers);
  };

  CacheServer(const CacheServer &) = delete;
  CacheServer &operator=(const CacheServer &) = delete;
  CacheServer(CacheServer &&) = delete;
  CacheServer &operator=(CacheServer &) = delete;
  Status DoServiceStart() override;
  Status DoServiceStop() override;
  ~CacheServer() override { (void)ServiceStop(); }

  static Status CreateInstance(const std::string &spill_path, int32_t num_workers, int32_t port,
                               int32_t shared_memory_sz, float memory_cap_ratio, int8_t log_level,
                               std::shared_ptr<CacheServerHW> hw_info) {
    std::call_once(init_instance_flag_, [&]() -> Status {
      auto &SvcManager = Services::GetInstance();
      RETURN_IF_NOT_OK(SvcManager.AddHook(&instance_, spill_path, num_workers, port, shared_memory_sz, memory_cap_ratio,
                                          log_level, hw_info));
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
  static Status GetFreeRequestTag(CacheServerRequest **q);

  /// \brief Return a tag to the free list
  /// \param p[in] pointer to already finished CacheServerRequest tag
  /// \return Status object
  static Status ReturnRequestTag(CacheServerRequest *p);

  /// Return an instance of the numa control
  std::shared_ptr<CacheServerHW> GetHWControl() { return hw_info_; }

  /// \brief Set CPU affinity
  Status SetAffinity(const Task &tk, numa_id_t numa_node) { return hw_info_->SetAffinity(tk, numa_node); }

  /// \brief return number of workers
  auto GetNumWorkers() const { return num_workers_; }

  /// \brief return number of grpc workers
  auto GetNumGrpcWorkers() const { return num_grpc_workers_; }

  /// \brief return number of numa nodes
  auto GetNumaNodeCount() const { return hw_info_->GetNumaNodeCount(); }

  /// \brief Assign a worker by a numa id
  /// \return worker id
  worker_id_t GetWorkerByNumaId(numa_id_t node_id) const;

  /// \brief Randomly pick a worker
  /// \return worker id
  worker_id_t GetRandomWorker() const;

  /// \brief Check if we bind threads to numa cores
  bool IsNumaAffinityOn() const { return numa_affinity_; }

  /// \brief Return the memory cap ratio
  float GetMemoryCapRatio() const { return memory_cap_ratio_; }

  /// \brief Function to handle a row request
  /// \param[in] cache_req A row request to handle
  /// \param[out] internal_request Indicator if the request is an internal request
  /// \return Status object
  Status ProcessRowRequest(CacheServerRequest *cache_req, bool *internal_request);

  /// \brief Function to handle an admin request
  /// \param[in] cache_req An admin request to handle
  /// \return Status object
  Status ProcessAdminRequest(CacheServerRequest *cache_req);

  /// \brief Function to handle a session request
  /// \param[in] cache_req A session request to handle
  /// \return Status object
  Status ProcessSessionRequest(CacheServerRequest *cache_req);

  /// \brief How a request is handled.
  /// \note that it can be process immediately by a grpc thread or routed to a server thread
  /// which is pinned to some numa node core.
  Status ProcessRequest(CacheServerRequest *cache_req);

  void GlobalShutdown();

  /// \brief This returns where we attach to the shared memory.
  /// Some gRPC requests will ask for a shared memory block, and
  /// we can't return the absolute address as this makes no sense
  /// in the client. So instead we will return an address relative
  /// to the base address of the shared memory where we attach to.
  /// \return Base address of the shared memory.
  const void *SharedMemoryBaseAddr() const { return shm_->SharedMemoryBaseAddr(); }

  /// \brief Return the public key of the shared memory.
  int32_t GetKey() const { return shm_->GetKey(); }

  Status AllocateSharedMemory(int32_t client_id, size_t sz, void **p);

  void DeallocateSharedMemory(int32_t client_id, void *p);

 private:
  static std::once_flag init_instance_flag_;
  static CacheServer *instance_;
  mutable RWLock rwLock_;
  mutable RWLock sessions_lock_;
  std::string top_;
  cache_index all_caches_;
  std::set<session_id_type> active_sessions_;
  std::shared_ptr<QueueList<CacheServerRequest *>> cache_q_;
  std::shared_ptr<CacheServerGreeterImpl> comm_layer_;
  TaskGroup vg_;
  int32_t num_workers_;
  int32_t num_grpc_workers_;
  int32_t port_;
  int32_t shared_memory_sz_in_gb_;
  int8_t log_level_;  // log_level is saved here for informational purpose only. It's not a functional field.
  std::atomic<bool> global_shutdown_;
  float memory_cap_ratio_;
  std::shared_ptr<CacheServerHW> hw_info_;
  std::map<worker_id_t, Task *> numa_tasks_;
  bool numa_affinity_;
  std::vector<int32_t> shutdown_qIDs_;
  std::unique_ptr<CachedSharedMemory> shm_;

  /// \brief Constructor
  /// \param spill_path Top directory for spilling buffers to.
  /// \param num_workers Number of threads for handling requests.
  explicit CacheServer(const std::string &spill_path, int32_t num_workers, int32_t port, int32_t share_memory_sz_in_gb,
                       float memory_cap_ratio, int8_t log_level, std::shared_ptr<CacheServerHW> hw_info);

  /// \brief Locate a cache service from connection id.
  /// \return Pointer to cache service. Null if not found
  CacheService *GetService(connection_id_type id) const;

  /// \brief Going over existing cache service and calculate how much we have consumed so far, a new cache service
  /// can only be created if there is still enough avail memory left
  /// \param cache_mem_sz Requested memory for a new cache service
  /// \return Status object
  Status GlobalMemoryCheck(uint64_t cache_mem_sz);

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
  Status ServerRequest(worker_id_t worker_id);

  /// \brief Entry point for all grpc threads.
  /// \return
  Status RpcRequest(worker_id_t worker_id);

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

  /// \brief Internal function to get statistics
  /// \param rq
  /// \param reply
  /// \return Status object
  Status GetStat(CacheRequest *rq, CacheReply *reply);

  /// \brief Internal function to get cache state
  /// \param rq
  /// \param reply
  /// \return Status object
  Status GetCacheState(CacheRequest *rq, CacheReply *reply);

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
  Status AcknowledgeShutdown(CacheServerRequest *cache_req);

  /// \brief Find keys that will be cache miss
  /// \return Status object
  Status GetCacheMissKeys(CacheRequest *rq, CacheReply *reply);

  /// \brief Toggle write mode for a service
  Status ToggleWriteMode(CacheRequest *rq);

  /// \brief List the sessions and their caches
  /// \param reply
  /// \return Status object
  Status ListSessions(CacheReply *reply);

  /// \brief Connect request by a pipeline
  Status ConnectReset(CacheRequest *rq);

  /// \brief This is an internal structure used by Batch processing.
  /// This is how it works internally. For batch fetch/cache, the grpc thread
  /// will intercept the request and breaks it down into multiple internal requests
  /// and spread over all the server workers. But each internal request consumes
  /// one free tag and we may run out of free tags if they don't return promptly.
  /// So we will let the server thread return the free tag immediately but the put
  /// the return code in this following structure. GRPC thread must wait until all
  /// the rc come back.
  class BatchWait : public std::enable_shared_from_this<BatchWait> {
   public:
    explicit BatchWait(int n) : expected_(n), num_back_(0) {
      expected_ = n;
      rc_lists_.reserve(expected_);
    }
    ~BatchWait() = default;

    std::weak_ptr<BatchWait> GetBatchWait() { return weak_from_this(); }

    Status Set(Status rc) {
      CHECK_FAIL_RETURN_UNEXPECTED(expected_ > num_back_, "Programming error");
      std::unique_lock<std::mutex> lck(mux_);
      rc_lists_.push_back(std::move(rc));
      ++num_back_;
      if (num_back_ == expected_) {
        wp_.Set();
      }
      return Status::OK();
    }

    Status Wait() { return wp_.Wait(); }

    Status GetRc() {
      Status rc;
      for (auto &cache_rc : rc_lists_) {
        if (cache_rc.IsError() && cache_rc != StatusCode::kMDInterrupted && rc.IsOk()) {
          rc = cache_rc;
        }
      }
      return rc;
    }

   private:
    std::mutex mux_;
    WaitPost wp_;
    int64_t expected_;
    int64_t num_back_;
    std::vector<Status> rc_lists_;
  };

  /// \brief Internal function to do row batch fetch
  /// \param rq Request
  /// \param reply Reply
  /// \return Status object
  Status BatchFetchRows(CacheRequest *rq, CacheReply *reply);

  /// \brief Main function to fetch rows in batch. The output is a contiguous memory which will be decoded
  /// by the CacheClient. Cache miss is not an error, and will be coded in the output to mark an empty row.
  /// \param[in] v A vector of row id.
  /// \param[out] out A contiguous memory buffer that holds the requested rows.
  /// \return Status object
  Status BatchFetch(const std::shared_ptr<flatbuffers::FlatBufferBuilder> &fbb, WritableSlice *out);
  Status BatchCacheRows(CacheRequest *rq);

  Status InternalFetchRow(CacheRequest *rq);
  Status InternalCacheRow(CacheRequest *rq, CacheReply *reply);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CACHE_TENSOR_H_
