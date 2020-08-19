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
#include "minddata/dataset/engine/cache/cache_grpc_server.h"
#include <limits>
#include "minddata/dataset/engine/cache/cache_server.h"
#include "minddata/dataset/util/path.h"
#include "utils/log_adapter.h"
namespace mindspore {
namespace dataset {
CacheServerGreeterImpl::CacheServerGreeterImpl(int32_t port, int32_t shared_memory_sz_in_gb)
    : port_(port), shm_pool_sz_in_gb_(shared_memory_sz_in_gb) {
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

Status CacheServerGreeterImpl::IpcResourceCleanup() {
#if CACHE_LOCAL_CLIENT
  int err;
  auto shm_key = PortToFtok(port_, &err);
  // We are expecting the unix path doesn't exist.
  if (shm_key == (key_t)-1) {
    return Status::OK();
  }
  // Attach to the shared memory
  auto shm_id = shmget(shm_key, 0, 0);
  if (shm_id == -1) {
    return Status::OK();
  }
  struct shmid_ds ds {};
  auto inx = shmctl(shm_id, IPC_STAT, &ds);
  if (inx == -1) {
    std::string errMsg = "Unable to query shared memory with id " + std::to_string(shm_id);
    errMsg += "\nPlesae remove it manually using ipcrm -m command";
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  if (ds.shm_nattch == 0) {
    // Stale shared memory from last time.
    // Remove both the memory and the socket path
    inx = shmctl(shm_id, IPC_RMID, nullptr);
    if (inx == -1) {
      std::string errMsg = "Unable to remove shared memory with id " + std::to_string(shm_id);
      errMsg += ". Errno :" + std::to_string(errno);
      errMsg += "\nPlesae remove it manually using ipcrm -m command";
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
    Path p(unix_socket_);
    (void)p.Remove();
  } else {
    // Server is already up.
    MS_LOG(ERROR) << "Cache server is already up and running";
    // We return a duplicate error. The main() will intercept
    // and output a proper message
    return Status(StatusCode::kDuplicateKey);
  }
#endif
  return Status::OK();
}

Status CacheServerGreeterImpl::Run() {
  // To listen on all interfaces, use 0.0.0.0
  // Use 127.0.0.1 if just locally on the same machine.
  std::string host("0.0.0.0");  // listen on all interfaces.
  std::string server_address = host + ":" + std::to_string(port_);
  grpc::ServerBuilder builder;
  // Default message size for gRPC is 4MB. Increase it to 2g-1
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int32_t>::max());
  int port_tcpip = 0;
#if CACHE_LOCAL_CLIENT
  int port_local = 0;
  // Check if we need to do clean up on the shared memory if the server
  // came down unexpectedly like SEGV
  RETURN_IF_NOT_OK(IpcResourceCleanup());
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
    MS_LOG(INFO) << "Creation of local socket and shared memory successful";
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
    // Get a new tag and handle the next request before we serve the current request.
    // The tag will be recycled when its state is changed to FINISH
    CacheServerRequest *next_rq;
    RETURN_IF_NOT_OK(CacheServer::GetFreeRequestTag(myQID, &next_rq));
    RETURN_IF_NOT_OK((*next_rq)(svc, cq));
    // Now we continue with the current request.
    // First thing we need to extract the type from the incoming request.
    // When this object was first created (i.e. STATE::CREATE), we set the type to UNKNOWN.
    type_ = static_cast<RequestType>(rq_.type());
    // Now we pass the address of this instance to CacheServer's main loop.
    MS_LOG(DEBUG) << "Handle request " << *this;
    auto &cs = CacheServer::GetInstance();
    RETURN_IF_NOT_OK(cs.PushRequest(myQID, this));
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
}  // namespace dataset
}  // namespace mindspore
