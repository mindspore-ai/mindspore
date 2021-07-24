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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_GRPC_CLIENT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_GRPC_CLIENT_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/engine/cache/cache_ipc.h"
#include "minddata/dataset/util/service.h"
#include "minddata/dataset/util/task_manager.h"
namespace mindspore {
namespace dataset {
/// \brief A client view of gRPC request
/// Like the class CacheServerRequest, this is used as a tag to inject into the gRPC
/// completion queue. The thread that makes the rpc request will wait on a wait post
/// area for the reply to come back. Since this tag will be deleted from memory and
/// we thus we need to work on a shared pointer of the BaseRequest such that its
/// use count is at least two. Otherwise either thread will be referencing stale memory.
/// \see CacheServerRequest
class CacheClientRequestTag {
 public:
  friend class CacheClientGreeter;
  explicit CacheClientRequestTag(std::shared_ptr<BaseRequest> rq, int64_t seqNo)
      : base_rq_(std::move(rq)), seq_no_(seqNo) {}
  ~CacheClientRequestTag() = default;

  /// \brief Notify the client that a result has come back from the server
  void Notify() { base_rq_->wp_.Set(); }

 private:
  std::shared_ptr<BaseRequest> base_rq_;
  grpc::Status rc_;
  grpc::ClientContext ctx_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<CacheReply>> rpc_;
  int64_t seq_no_;
};

/// \brief A GRPC layer to convert BaseRequest into protobuf and send to the cache server using gRPC
/// \see BaseRequest
class CacheClientGreeter : public Service {
  friend class CacheClient;

 public:
  constexpr static int32_t kRequestTimeoutDeadlineInSec = 60;
  constexpr static int32_t kWaitForNewEventDeadlineInSec = 1;
  explicit CacheClientGreeter(const std::string &hostname, int32_t port, int32_t num_connections);
  ~CacheClientGreeter();

  /// Override base Service class
  Status DoServiceStart() override;
  Status DoServiceStop() override;

  /// \brief Send the request to the server
  /// \return Status object
  Status HandleRequest(std::shared_ptr<BaseRequest> rq);

  /// \brief A handful of threads will be handling async reply from the server
  /// \return
  Status WorkerEntry();

  /// \brief Kick off threads to receive reply from the server
  Status DispatchWorkers(int32_t num_workers);

  /// \brief Attach to shared memory for local client
  /// \note Called after we have established a connection.
  /// \return Status object.
  Status AttachToSharedMemory(bool *local_bypass);

  /// \brief This returns where we attach to the shared memory.
  /// \return Base address of the shared memory.
  const void *SharedMemoryBaseAddr() const { return mem_.SharedMemoryBaseAddr(); }

  std::string GetHostname() const { return hostname_; }
  int32_t GetPort() const { return port_; }

 private:
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<CacheServerGreeter::Stub> stub_;
  grpc::CompletionQueue cq_;
  TaskGroup vg_;
  int32_t num_connections_;
  std::atomic<int64_t> request_cnt_;
  mutable std::mutex mux_;
  std::map<int64_t, std::unique_ptr<CacheClientRequestTag>> req_;
  SharedMemory mem_;
  std::string hostname_;
  int32_t port_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_GRPC_CLIENT_H_
