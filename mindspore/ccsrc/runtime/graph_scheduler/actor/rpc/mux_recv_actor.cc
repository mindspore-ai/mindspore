/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/rpc/mux_recv_actor.h"
#include "distributed/constants.h"

namespace mindspore {
namespace runtime {
using distributed::kFinalizeMuxRecvActor;

void MuxRecvActor::SetMessageHandler() {
  MS_EXCEPTION_IF_NULL(server_);
  server_->SetMessageHandler(std::bind(&MuxRecvActor::HandleMessage, this, std::placeholders::_1));
}

MessageBase *MuxRecvActor::HandleMessage(MessageBase *const msg) {
  // Block the message handler if the context is invalid.
  std::unique_lock<std::mutex> lock(context_mtx_);
  context_cv_.wait(lock, [this] { return is_context_valid_; });
  lock.unlock();

  if (finalized_) {
    return distributed::rpc::NULL_MSG;
  }

  // If use void* data, the cv has already been notified in AllocateMessage.
  if (common::GetEnv("use_void").empty()) {
    // The mux recv actor receives requests for the service process. Currently, the requests are processed serially.
    std::unique_lock<std::mutex> is_ready_lock(is_ready_mtx_);
    is_ready_cv_.wait(is_ready_lock, [this] { return is_ready_.load(); });
    is_ready_ = false;
  }

  if (msg == nullptr || op_context_ == nullptr) {
    return distributed::rpc::NULL_MSG;
  }

  // Save from actor url.
  from_actor_aid_ = msg->From();

  ActorDispatcher::Send(GetAID(), &MuxRecvActor::RunOpInterProcessData, msg, op_context_);
  return distributed::rpc::NULL_MSG;
}

void MuxRecvActor::ParseFinalizeReqData(size_t data_len, const MessageBase *const msg, bool *need_finalize) {
  MS_EXCEPTION_IF_NULL(msg);
  MS_EXCEPTION_IF_NULL(need_finalize);

  size_t req_data_size = 0;
  RpcDataPtr finaliz_req_data;
  if (common::GetEnv("use_void").empty()) {
    req_data_size = msg->body.size();
    finaliz_req_data = const_cast<RpcDataPtr>(msg->body.c_str());
  } else {
    MS_EXCEPTION_IF_NULL(msg->data);
    req_data_size = msg->size;
    finaliz_req_data = static_cast<RpcDataPtr>(msg->data);
  }
  if (data_len == req_data_size) {
    return;
  }

  size_t remainder_len = req_data_size - data_len;
  size_t finalize_header_size = strlen(kFinalizeMuxRecvActor);
  if (remainder_len <= finalize_header_size) {
    MS_LOG(EXCEPTION) << "Not found msg header[" << kFinalizeMuxRecvActor << "] in received message";
  }

  if (remainder_len - finalize_header_size != 1) {
    MS_LOG(EXCEPTION) << "Invalid finalize request message";
  }

  const void *need_finalize_actor_data = finaliz_req_data + data_len + finalize_header_size;
  MS_EXCEPTION_IF_NULL(need_finalize_actor_data);
  bool finalize_in_msg = *(static_cast<const bool *>(need_finalize_actor_data));
  MS_LOG(INFO) << "Received a message which contains finalize command: " << finalize_in_msg;
  if (!finalize_in_msg) {
    return;
  }

  recv_finalize_msg_cnt_++;
  if (recv_finalize_msg_cnt_ == ClusterContext::instance()->node_num(distributed::kEnvRoleOfWorker)) {
    *need_finalize = true;
    // Finalize loop of runtime.
    MS_EXCEPTION_IF_NULL(op_context_);
    SET_OPCONTEXT_SUCCESS_RET((*op_context_));
  }
}

void *MuxRecvActor::AllocateMessage(size_t size) {
  // Block the message handler if the context is invalid.
  std::unique_lock<std::mutex> lock(context_mtx_);
  context_cv_.wait(lock, [this] { return is_context_valid_; });
  lock.unlock();

  // The mux recv actor receives requests for the service process. Currently, the requests are processed serially.
  if (!common::GetEnv("use_void").empty()) {
    std::unique_lock<std::mutex> is_ready_lock(is_ready_mtx_);
    is_ready_cv_.wait(is_ready_lock, [this] { return is_ready_.load(); });
    is_ready_ = false;
  }

  return AllocateMemByDeviceRes(size);
}

void MuxRecvActor::UpdateStatus() {
  std::unique_lock<std::mutex> is_ready_lock(is_ready_mtx_);
  is_ready_ = true;
  is_ready_cv_.notify_one();
}

void MuxRecvActor::Finalize() {
  std::unique_lock<std::mutex> lock(context_mtx_);
  finalized_ = true;
  is_ready_ = true;
  is_context_valid_ = true;

  op_context_ = nullptr;
  context_cv_.notify_all();
  is_ready_cv_.notify_all();
}
}  // namespace runtime
}  // namespace mindspore
