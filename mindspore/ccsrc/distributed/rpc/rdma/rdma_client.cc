/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/backend/distributed/rpc/rdma/rdma_client.h"

#include <string>

namespace mindspore {
namespace distributed {
namespace rpc {
bool RDMAClient::Initialize() {
  // The initialization of URPC RDMA is implemented in 'Connect' function.
  return true;
}

void RDMAClient::Finalize() {
  if (urpc_session_ != nullptr) {
    urpc_close_func(urpc_session_);
  }
  if (kURPCInited) {
    urpc_uninit_func();
    kURPCInited = false;
  }
  kConnectedSession.clear();
}

bool RDMAClient::Connect(const std::string &dst_url, size_t retry_count, const MemFreeCallback &) {
  if (!ParseURL(dst_url, &ip_addr_, &port_)) {
    MS_LOG(EXCEPTION) << "Failed to parse url " << dst_url;
  }
  if (!InitializeURPC(dev_name_, ip_addr_, port_)) {
    return false;
  }

  // If this process has already connected to this server, there's no need to call urpc_connect.
  if (kConnectedSession.count(dst_url) != 0) {
    MS_LOG(INFO) << "This process has already connected to server " << dst_url;
    urpc_session_ = kConnectedSession[dst_url];
    return true;
  }

  for (size_t i = 0; i < retry_count; i++) {
    urpc_session_ = urpc_connect_func(const_cast<char *>(ip_addr_.c_str()), port_, nullptr);
    if (urpc_session_ == nullptr) {
      MS_LOG(WARNING) << "Failed to call urpc_connect to " << ip_addr_ << ":" << port_ << ". Retry to reconnect("
                      << (i + 1) << "/" << retry_count << ")...";
      sleep(kRetryConnectInterval);
    } else {
      kConnectedSession[dst_url] = urpc_session_;
      MS_LOG(INFO) << "Successfully connect to server " << ip_addr_ << ":" << port_;
      return true;
    }
  }
  MS_LOG(EXCEPTION) << "Failed to call urpc_connect to " << ip_addr_ << ":" << port_ << " after " << retry_count
                    << " times. Please check Mindspore info log or URPC log directory: /var/log/umdk/urpc.";
}

bool RDMAClient::IsConnected(const std::string &dst_url) { return false; }

bool RDMAClient::Disconnect(const std::string &dst_url, size_t timeout_in_sec) {
  if (urpc_session_ != nullptr) {
    urpc_close_func(urpc_session_);
    urpc_session_ = nullptr;
  }
  return true;
}

bool RDMAClient::SendSync(std::unique_ptr<MessageBase> &&msg, size_t *const send_bytes) {
  MS_EXCEPTION_IF_NULL(msg);
  size_t msg_size = msg->size;
  void *msg_buf = msg->data;
  MS_EXCEPTION_IF_NULL(msg_buf);

  void *urpc_msg_buf = urpc_allocator_->alloc(msg_size);
  MS_EXCEPTION_IF_NULL(urpc_msg_buf);
  if (memcpy_s(urpc_msg_buf, msg_size, msg_buf, msg_size) != EOK) {
    MS_LOG(EXCEPTION) << "Failed to memcpy_s data to urpc_msg_buf with size " << msg_size;
  }
  struct urpc_sgl sgl;
  sgl.sge[0].addr = reinterpret_cast<uintptr_t>(urpc_msg_buf);
  sgl.sge[0].length = msg_size;
  sgl.sge[0].flag = URPC_SGE_FLAG_RENDEZVOUS | URPC_SGE_FLAG_RESERVE_BUF;
  sgl.sge_num = 1;

  struct urpc_send_wr send_wr = {};
  send_wr.func_id = msg->func_id_;
  send_wr.send_mode = URPC_SEND_MODE_SYNC;
  send_wr.req = &sgl;
  struct urpc_sgl rsp_sgl = {0};
  send_wr.sync.rsp = &rsp_sgl;

  MS_LOG(DEBUG) << "Start sending message to server with func_id: " << send_wr.func_id;
  if (urpc_send_request_func(urpc_session_, &send_wr, nullptr) < 0) {
    MS_LOG(ERROR) << "Failed to send request for function call: " << send_wr.func_id;
    return false;
  }
  auto rsp_data = reinterpret_cast<char *>(rsp_sgl.sge[0].addr);
  MS_LOG(DEBUG) << "Sending success. Server response message is " << rsp_data;

  // Release URPC memory.
  urpc_allocator_->free(rsp_data);
  urpc_allocator_->free(urpc_msg_buf);
  return true;
}

void RDMAClient::SendAsync(std::unique_ptr<MessageBase> &&msg) {
  MS_EXCEPTION_IF_NULL(msg);
  size_t msg_size = msg->size;
  void *msg_buf = msg->data;
  MS_EXCEPTION_IF_NULL(msg_buf);

  void *urpc_msg_buf = urpc_allocator_->alloc(msg_size);
  MS_EXCEPTION_IF_NULL(urpc_msg_buf);
  if (memcpy_s(urpc_msg_buf, msg_size, msg_buf, msg_size) != EOK) {
    MS_LOG(EXCEPTION) << "Failed to memcpy_s data to urpc_msg_buf with size " << msg_size;
  }
  struct urpc_sgl sgl;
  sgl.sge[0].addr = reinterpret_cast<uintptr_t>(urpc_msg_buf);
  sgl.sge[0].length = msg_size;
  sgl.sge[0].flag = URPC_SGE_FLAG_RENDEZVOUS | URPC_SGE_FLAG_RESERVE_BUF;
  sgl.sge_num = 1;

  std::unique_lock<std::mutex> lock(mtx_);
  cb_arg_.rsp_received = false;
  cb_arg_.data_to_free = urpc_msg_buf;
  cb_arg_.allocator = urpc_allocator_;
  cb_arg_.mtx = &mtx_;
  cb_arg_.cv = &cv_;
  lock.unlock();

  struct urpc_send_wr send_wr = {};
  send_wr.func_id = msg->func_id_;
  send_wr.send_mode = URPC_SEND_MODE_ASYNC;
  send_wr.req = &sgl;
  send_wr.async.cb.wo_ctx = urpc_rsp_cb;
  send_wr.async.cb_arg = &cb_arg_;

  if (urpc_send_request_func(urpc_session_, &send_wr, nullptr) < 0) {
    MS_LOG(EXCEPTION) << "Failed to send request to server.";
    return;
  }
}

bool RDMAClient::Flush(const std::string &dst_url) {
  std::unique_lock<std::mutex> lock(mtx_);
  cv_.wait(lock, [this]() { return cb_arg_.rsp_received; });
  return true;
}

void RDMAClient::urpc_rsp_cb(struct urpc_sgl *rsp, int err, void *arg) {
  MS_ERROR_IF_NULL_WO_RET_VAL(rsp);
  MS_ERROR_IF_NULL_WO_RET_VAL(arg);
  if (err != kURPCSuccess) {
    MS_LOG(ERROR) << "Error code: " << err;
    return;
  }

  auto rsp_data = reinterpret_cast<char *>(rsp->sge[0].addr);
  MS_LOG(INFO) << "Server response message is " << rsp_data;
  struct req_cb_arg *cb_arg = static_cast<struct req_cb_arg *>(arg);
  std::unique_lock<std::mutex> lock(*(cb_arg->mtx));
  cb_arg->rsp_received = true;
  cb_arg->allocator->free(rsp_data);
  cb_arg->allocator->free(cb_arg->data_to_free);
  cb_arg->cv->notify_all();
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
