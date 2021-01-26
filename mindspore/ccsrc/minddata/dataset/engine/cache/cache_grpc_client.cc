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
#include "minddata/dataset/engine/cache/cache_grpc_client.h"
#include <chrono>
namespace mindspore {
namespace dataset {
CacheClientGreeter::~CacheClientGreeter() { (void)ServiceStop(); }

CacheClientGreeter::CacheClientGreeter(const std::string &hostname, int32_t port, int32_t num_connections)
    : num_connections_(num_connections), request_cnt_(0), hostname_(std::move(hostname)), port_(port) {
  grpc::ChannelArguments args;
  // We need to bump up the message size to unlimited. The default receiving
  // message limit is 4MB which is not big enough.
  args.SetMaxReceiveMessageSize(-1);
  MS_LOG(INFO) << "Hostname: " << hostname_ << ", port: " << std::to_string(port_);
#if CACHE_LOCAL_CLIENT
  // Try connect locally to the unix_socket first as the first preference
  // Need to resolve hostname to ip address rather than to do a string compare
  if (hostname == "127.0.0.1") {
    std::string target = "unix://" + PortToUnixSocketPath(port);
    channel_ = grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args);
  } else {
#endif
    std::string target = hostname + ":" + std::to_string(port);
    channel_ = grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args);
#if CACHE_LOCAL_CLIENT
  }
#endif
  stub_ = CacheServerGreeter::NewStub(channel_);
}

Status CacheClientGreeter::AttachToSharedMemory(bool *local_bypass) {
  *local_bypass = false;
#if CACHE_LOCAL_CLIENT
  SharedMemory::shm_key_t shm_key;
  RETURN_IF_NOT_OK(PortToFtok(port_, &shm_key));
  // Attach to the shared memory
  mem_.SetPublicKey(shm_key);
  RETURN_IF_NOT_OK(mem_.Attach());
  *local_bypass = true;
#endif
  return Status::OK();
}

Status CacheClientGreeter::DoServiceStart() {
  RETURN_IF_NOT_OK(vg_.ServiceStart());
  RETURN_IF_NOT_OK(DispatchWorkers(num_connections_));
  return Status::OK();
}

Status CacheClientGreeter::DoServiceStop() {
  // Shutdown the queue. We don't accept any more new incomers.
  cq_.Shutdown();
  // Shutdown the TaskGroup.
  vg_.interrupt_all();
  vg_.join_all(Task::WaitFlag::kNonBlocking);
  // Drain the queue. We know how many requests we send out
  while (!req_.empty()) {
    bool success;
    void *tag;
    while (cq_.Next(&tag, &success)) {
      auto r = reinterpret_cast<CacheClientRequestTag *>(tag);
      req_.erase(r->seqNo_);
    }
  }
  return Status::OK();
}

Status CacheClientGreeter::HandleRequest(std::shared_ptr<BaseRequest> rq) {
  // If there is anything extra we need to do before we send.
  RETURN_IF_NOT_OK(rq->Prepare());
  auto seqNo = request_cnt_.fetch_add(1);
  auto tag = std::make_unique<CacheClientRequestTag>(std::move(rq), seqNo);
  // One minute timeout
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(60);
  tag->ctx_.set_deadline(deadline);
  tag->rpc_ = stub_->PrepareAsyncCacheServerRequest(&tag->ctx_, tag->base_rq_->rq_, &cq_);
  tag->rpc_->StartCall();
  auto ccReqTag = tag.get();
  // Insert it into the map.
  {
    std::unique_lock<std::mutex> lck(mux_);
    auto r = req_.emplace(seqNo, std::move(tag));
    if (!r.second) {
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__);
    }
  }
  // Last step is to tag the request.
  ccReqTag->rpc_->Finish(&ccReqTag->base_rq_->reply_, &ccReqTag->rc_, ccReqTag);
  return Status::OK();
}

Status CacheClientGreeter::WorkerEntry() {
  TaskManager::FindMe()->Post();
  do {
    bool success;
    void *tag;
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
    // Set a timeout for one second. Check for interrupt if we need to do early exit.
    auto r = cq_.AsyncNext(&tag, &success, deadline);
    if (r == grpc_impl::CompletionQueue::NextStatus::GOT_EVENT) {
      auto rq = reinterpret_cast<CacheClientRequestTag *>(tag);
      if (success) {
        auto &rc = rq->rc_;
        if (!rc.ok()) {
          auto error_code = rq->rc_.error_code();
          std::string err_msg;
          if (error_code == grpc::StatusCode::UNAVAILABLE) {
            err_msg = "Cache server with port " + std::to_string(port_) +
                      " is unreachable. Make sure the server is running. GRPC Code " + std::to_string(error_code);
          } else {
            err_msg = rq->rc_.error_message() + ". GRPC Code " + std::to_string(error_code);
          }
          Status remote_rc = Status(StatusCode::kMDNetWorkError, __LINE__, __FILE__, err_msg);
          Status2CacheReply(remote_rc, &rq->base_rq_->reply_);
        }
        // Notify the waiting thread.
        rq->Notify();
      }
      {
        // We can now free the memory
        std::unique_lock<std::mutex> lck(mux_);
        auto seqNo = rq->seqNo_;
        auto n = req_.erase(seqNo);
        CHECK_FAIL_RETURN_UNEXPECTED(n == 1, "Sequence " + std::to_string(seqNo) + " not found");
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

Status CacheClientGreeter::DispatchWorkers(int32_t num_workers) {
  auto f = std::bind(&CacheClientGreeter::WorkerEntry, this);
  for (auto i = 0; i < num_workers; ++i) {
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("Async reply", f));
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
