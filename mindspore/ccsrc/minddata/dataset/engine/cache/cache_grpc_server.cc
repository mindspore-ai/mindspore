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
#include <chrono>
#include <limits>
#include "minddata/dataset/engine/cache/cache_grpc_server.h"
#include "minddata/dataset/engine/cache/cache_server.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
CacheServerGreeterImpl::CacheServerGreeterImpl(int32_t port, int32_t shared_memory_sz_in_gb)
    : port_(port), shm_pool_sz_in_gb_(shared_memory_sz_in_gb), shm_key_(-1) {
  // Setup a path for unix socket.
  unix_socket_ = PortToUnixSocketPath(port);
  // We can't generate the ftok key yet until the unix_socket_ is created
}

void CacheServerGreeterImpl::Shutdown() {
  if (server_) {
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
    server_->Shutdown(deadline);
  }
  // Always shutdown the completion queue after the server.
  if (cq_) {
    cq_->Shutdown();
    // We need to drain the queue. All the tag is coming from
    // the Services pool which will be shutdown as well. So we
    // ignore the tag.
    void *tag;
    bool success;
    while (cq_->Next(&tag, &success)) {
    }
  }
}

CacheServerGreeterImpl::~CacheServerGreeterImpl() { Shutdown(); }

Status CacheServerGreeterImpl::Run() {
  // To listen on all interfaces, use 0.0.0.0
  // Future, allow the user to choose listening interface.  For now, default to localhost
  std::string host("127.0.0.1");
  std::string server_address = host + ":" + std::to_string(port_);
  grpc::ServerBuilder builder;
  // Default message size for gRPC is 4MB. Increase it to 2g-1
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int32_t>::max());
  int port_tcpip = 0;
#if CACHE_LOCAL_CLIENT
  int port_local = 0;
  // We also optimize on local clients on the same machine using unix socket
  builder.AddListeningPort("unix://" + unix_socket_, grpc::InsecureServerCredentials(), &port_local);
#endif
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(), &port_tcpip);
  builder.RegisterService(&svc_);
  cq_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();
  if (server_) {
    MS_LOG(INFO) << "Server listening on " << server_address;
#if CACHE_LOCAL_CLIENT
    RETURN_IF_NOT_OK(CachedSharedMemoryArena::CreateArena(&shm_pool_, port_, shm_pool_sz_in_gb_));
    shm_key_ = shm_pool_->GetKey();
    MS_LOG(INFO) << "Creation of local socket and shared memory successful. Shared memory key " << shm_key_;
    auto cs = CacheServer::GetInstance().GetHWControl();
    // This shared memory is a hot memory and we will interleave among all the numa nodes.
    cs->InterleaveMemory(const_cast<void *>(shm_pool_->SharedMemoryBaseAddr()), shm_pool_sz_in_gb_ * 1073741824L);
#endif
  } else {
    std::string errMsg = "Fail to start server. ";
    if (port_tcpip != port_) {
      errMsg += "Unable to bind to tcpip port " + std::to_string(port_) + ".";
    }
#if CACHE_LOCAL_CLIENT
    if (port_local == 0) {
      errMsg += " Unable to create unix socket " + unix_socket_ + ".";
    }
#endif
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  return Status::OK();
}

Status CacheServerGreeterImpl::HandleRequest(int32_t worker_id) {
  bool success;
  void *tag;
  // We loop through the grpc queue. Each connection if successful
  // will come back with our own tag which is an instance of CacheServerRequest
  // and we simply call its functor. But first we need to create these instances
  // and inject them into the grpc queue.
  CacheServerRequest *p;
  // Get a free tag from my free list.
  RETURN_IF_NOT_OK(CacheServer::GetFreeRequestTag(worker_id, &p));
  RETURN_IF_NOT_OK((*p)(&svc_, cq_.get()));
  do {
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
    // Set a timeout for one second. Check for interrupt if we need to do early exit.
    auto r = cq_->AsyncNext(&tag, &success, deadline);
    if (r == grpc_impl::CompletionQueue::NextStatus::GOT_EVENT) {
      if (success) {
        auto rq = static_cast<CacheServerRequest *>(tag);
        RETURN_IF_NOT_OK((*rq)(&svc_, cq_.get()));
      }
    } else if (r == grpc_impl::CompletionQueue::NextStatus::TIMEOUT) {
      // If we are interrupted, exit. Otherwise wait again.
      RETURN_IF_INTERRUPTED();
    } else {
      // Queue is drained.
      break;
    }
  } while (true);
  return Status::OK();
}

Status CacheServerRequest::operator()(CacheServerGreeter::AsyncService *svc, grpc::ServerCompletionQueue *cq) {
  auto myQID = getQid();
  if (st_ == STATE::CREATE) {
    st_ = STATE::PROCESS;
    svc->RequestCacheServerRequest(&ctx_, &rq_, &responder_, cq, cq, this);
  } else if (st_ == STATE::PROCESS) {
    auto &cs = CacheServer::GetInstance();
    // Get a new tag and handle the next request before we serve the current request.
    // The tag will be recycled when its state is changed to FINISH.
    // The number of free list queues is the same as the number of grpc threads.
    // Where we get the free list it doesn't matter (as long we return it back to the right queue).
    // We can round robin, use the qid or even use the worker id. We will use the free list queue
    // where the current request comes from.
    CacheServerRequest *next_rq;
    RETURN_IF_NOT_OK(CacheServer::GetFreeRequestTag(myQID, &next_rq));
    RETURN_IF_NOT_OK((*next_rq)(svc, cq));
    // Now we continue with the current request.
    // First thing we need to extract the type from the incoming request.
    // When this object was first created (i.e. STATE::CREATE), we set the type to UNKNOWN.
    type_ = static_cast<RequestType>(rq_.type());
    // Now we pass the address of this instance to CacheServer's main loop.
    MS_LOG(DEBUG) << "Handle request " << *this;
    // We will distribute the request evenly (or randomly) over all the numa nodes.
    // The exception is BatchFetch which we need to pre-process here.
    if (type_ == BaseRequest::RequestType::kBatchFetchRows) {
      rc_ = cs.BatchFetchRows(&rq_, &reply_);
      if (!rc_.IsInterrupted()) {
        Status2CacheReply(rc_, &reply_);
        st_ = CacheServerRequest::STATE::FINISH;
        responder_.Finish(reply_, grpc::Status::OK, this);
      } else {
        return rc_;
      }
    } else {
      // When the number of grpc workers is the same as the server workers, we will use this queue id
      // and push to the corresponding queue.
      bool random = cs.GetNumWorkers() != cs.GetNumGrpcWorkers();
      worker_id_t worker_id = random ? cs.GetRandomWorker() : myQID;
      RETURN_IF_NOT_OK(cs.PushRequest(worker_id, this));
    }
  } else if (st_ == STATE::FINISH) {
    MS_LOG(DEBUG) << *this << " Finished.";
    // Return back to the free list.
    RETURN_IF_NOT_OK(CacheServer::ReturnRequestTag(this));
  }
  return Status::OK();
}

void CacheServerRequest::Print(std::ostream &out) const {
  if (rq_.has_connection_info()) {
    out << "Session Id: " << rq_.connection_info().session_id() << " CRC: " << rq_.connection_info().crc();
  } else {
    out << "Connection Id: " << rq_.connection_id();
  }
  out << " ";
  BaseRequest::Print(out);
}

Status CacheServerGreeterImpl::MonitorUnixSocket() {
  TaskManager::FindMe()->Post();
#if CACHE_LOCAL_CLIENT
  Path p(unix_socket_);
  do {
    RETURN_IF_INTERRUPTED();
    // If the unix socket is recreated for whatever reason, this server instance will be stale and
    // no other process and communicate with us. In this case we need to shutdown ourselves.
    if (p.Exists()) {
      SharedMemory::shm_key_t key;
      RETURN_IF_NOT_OK(PortToFtok(port_, &key));
      if (key != shm_key_) {
        std::string errMsg = "Detecting unix socket has changed. Previous key " + std::to_string(shm_key_) +
                             ". New key " + std::to_string(key) + ". Shutting down server";
        MS_LOG(ERROR) << errMsg;
        RETURN_STATUS_UNEXPECTED(errMsg);
      }
    } else {
      MS_LOG(WARNING) << "Unix socket is removed.";
      TaskManager::WakeUpWatchDog();
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
  } while (true);
#endif
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
