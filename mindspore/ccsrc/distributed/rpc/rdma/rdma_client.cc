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

#include "distributed/rpc/rdma/rdma_client.h"

#include <string>

namespace mindspore {
namespace distributed {
namespace rpc {
bool RDMAClient::Initialize() { return true; }

void RDMAClient::Finalize() {
  if (urpc_session_ != nullptr) {
    urpc_close_func(urpc_session_);
    urpc_uninit_func();
  }
}

bool RDMAClient::Connect(const std::string &dst_url, size_t retry_count, const MemFreeCallback &free_cb) {
  std::string ip;
  uint16_t port;
  if (!ParseURL(dst_url, &ip, &port)) {
    MS_LOG(EXCEPTION) << "Failed to parse url " << dst_url;
  }
  ip_addr_ = const_cast<char *>(ip.c_str());
  port_ = port;

  // Init URPC after destination is specified.
  struct urpc_config urpc_cfg = {};
  urpc_cfg.mode = URPC_MODE_CLIENT;
  urpc_cfg.cfeature = 0;
  urpc_cfg.polling_num = kClientPollingThreadNum;
  urpc_cfg.transport.dev_name = dev_name_;
  urpc_cfg.transport.ip_addr = ip_addr_;
  urpc_cfg.transport.port = port_;
  urpc_cfg.transport.max_sge = 0;
  urpc_cfg.allocator = nullptr;
  if (urpc_init_func(&urpc_cfg) != kURPCSuccess) {
    MS_LOG(EXCEPTION) << "Failed to call urpc_init. Device name: " << dev_name_ << ", ip address: " << ip_addr_
                      << ", port: " << port_ << ". Please refer to URPC log directory: /var/log/umdk/urpc.";
  }

  urpc_session_ = urpc_connect_func(ip_addr_, port_, nullptr);
  if (urpc_session_ == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to call urpc_connect to " << ip_addr_ << ":" << port_;
  }
  return true;
}

bool RDMAClient::IsConnected(const std::string &dst_url) { return false; }

bool RDMAClient::Disconnect(const std::string &dst_url, size_t timeout_in_sec) { return true; }

bool RDMAClient::SendSync(std::unique_ptr<MessageBase> &&msg, size_t *const send_bytes) {
  MS_EXCEPTION_IF_NULL(msg);
  size_t msg_size = msg->size;
  void *msg_buf = msg->data;
  MS_EXCEPTION_IF_NULL(msg_buf);

  struct urpc_sgl sgl;
  sgl.sge[0].addr = reinterpret_cast<uintptr_t>(msg_buf);
  sgl.sge[0].length = msg_size;
  sgl.sge[0].flag = URPC_SGE_FLAG_ZERO_COPY;
  sgl.sge_num = 1;

  int rsp_received = 0;
  struct req_cb_arg cb_arg = {};
  cb_arg.rsp_received = &rsp_received;
  cb_arg.allocator = urpc_allocator_;

  struct urpc_send_wr send_wr = {};
  send_wr.func_id = kInterProcessDataHandleID;
  send_wr.send_mode = URPC_SEND_MODE_ASYNC;
  send_wr.req = &sgl;
  send_wr.async.cb.wo_ctx = urpc_rsp_cb;
  send_wr.async.cb_arg = &cb_arg;

  if (urpc_send_request_func(urpc_session_, &send_wr, nullptr) != kURPCSuccess) {
    MS_LOG(ERROR) << "Failed to send request.";
    return false;
  }

  size_t sleep_time_us = 200000;
  // Wait till server responds.
  while (rsp_received == 0) {
    usleep(sleep_time_us);
  }
  return true;
}

void RDMAClient::SendAsync(std::unique_ptr<MessageBase> &&msg) {}

bool RDMAClient::Flush(const std::string &dst_url) { return true; }

void RDMAClient::urpc_rsp_cb(struct urpc_sgl *rsp, int err, void *arg) {
  MS_ERROR_IF_NULL_WO_RET_VAL(rsp);
  MS_ERROR_IF_NULL_WO_RET_VAL(arg);
  if (err != kURPCSuccess) {
    MS_LOG(ERROR) << "Error code: " << err;
    return;
  }

  MS_LOG(INFO) << "Server response message is " << reinterpret_cast<char *>(rsp->sge[0].addr);
  struct req_cb_arg *cb_arg = static_cast<struct req_cb_arg *>(arg);
  *(cb_arg->rsp_received) = 1;
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
