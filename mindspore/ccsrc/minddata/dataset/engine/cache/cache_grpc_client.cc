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
Status CacheClientRequestTag::MakeCall(CacheServerGreeter::Stub *stub, grpc::CompletionQueue *cq,
                                       std::unique_ptr<CacheClientRequestTag> &&tag) {
  // If there is anything extra we need to do before we send.
  RETURN_IF_NOT_OK(tag->base_rq_->Prepare());
  // One minute timeout
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(60);
  tag->ctx_.set_deadline(deadline);
  tag->rpc_ = stub->PrepareAsyncCacheServerRequest(&tag->ctx_, tag->base_rq_->rq_, cq);
  tag->rpc_->StartCall();
  // Last step is we release the ownership and transfer it to the completion queue.
  // The memory will be released by WorkerEntry or by the destructor when we drain the queue
  auto ccReqTag = tag.release();
  ccReqTag->rpc_->Finish(&ccReqTag->base_rq_->reply_, &ccReqTag->rc_,
                         ccReqTag);  // inject this object into the completion queue
  return Status::OK();
}

CacheClientGreeter::~CacheClientGreeter() {
  (void)ServiceStop();
  // Detach from shared memory if any
  if (shmat_addr_ != nullptr) {
    shmdt(shmat_addr_);
    shmat_addr_ = nullptr;
  }
}

CacheClientGreeter::CacheClientGreeter(const std::string &hostname, int32_t port, int32_t num_workers)
    : num_workers_(num_workers), shm_key_(-1), shm_id_(-1), shmat_addr_(nullptr) {
  grpc::ChannelArguments args;
  // We need to bump up the message size to unlimited. The default receiving
  // message limit is 4MB which is not big enough.
  args.SetMaxReceiveMessageSize(-1);
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

Status CacheClientGreeter::AttachToSharedMemory(int32_t port, bool *local_bypass) {
  *local_bypass = false;
#if CACHE_LOCAL_CLIENT
  int err;
  shm_key_ = PortToFtok(port, &err);
  if (shm_key_ == (key_t)-1) {
    std::string errMsg = "Ftok failed with errno " + std::to_string(err);
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  // Attach to the shared memory
  shm_id_ = shmget(shm_key_, 0, 0);
  if (shm_id_ == -1) {
    RETURN_STATUS_UNEXPECTED("Shmget failed. Errno " + std::to_string(errno));
  }
  shmat_addr_ = shmat(shm_id_, nullptr, 0);
  if (shmat_addr_ == reinterpret_cast<void *>(-1)) {
    RETURN_STATUS_UNEXPECTED("Shared memory attach failed. Errno " + std::to_string(errno));
  }
  *local_bypass = true;
#endif
  return Status::OK();
}

Status CacheClientGreeter::DoServiceStart() {
  RETURN_IF_NOT_OK(vg_.ServiceStart());
  RETURN_IF_NOT_OK(DispatchWorkers(num_workers_));
  return Status::OK();
}

Status CacheClientGreeter::DoServiceStop() {
  // Shutdown the queue. We don't accept any more new incomers.
  cq_.Shutdown();
  // Shutdown the TaskGroup.
  vg_.interrupt_all();
  vg_.join_all(Task::WaitFlag::kNonBlocking);
  // Drain the queue
  bool success;
  void *tag;
  while (cq_.Next(&tag, &success)) {
    auto r = reinterpret_cast<CacheClientRequestTag *>(tag);
    delete r;
  }
  return Status::OK();
}

Status CacheClientGreeter::HandleRequest(std::shared_ptr<BaseRequest> rq) {
  auto tag = std::make_unique<CacheClientRequestTag>(std::move(rq));
  return tag->MakeCall(stub_.get(), &cq_, std::move(tag));
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
          std::string errMsg = rq->rc_.error_message() + ". GRPC Code " + std::to_string(error_code);
          Status remote_rc = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
          Status2CacheReply(remote_rc, &rq->base_rq_->reply_);
        }
        // Notify the waiting thread.
        rq->Notify();
      }
      // We can now free the memory
      delete rq;
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
