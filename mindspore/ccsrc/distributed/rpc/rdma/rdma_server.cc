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

#include "distributed/rpc/rdma/rdma_server.h"

#include <string>

namespace mindspore {
namespace distributed {
namespace rpc {
bool RDMAServer::Initialize(const std::string &url, const MemAllocateCallback &allocate_cb) {
  std::string ip;
  uint16_t port;
  if (!ParseURL(url, &ip, &port)) {
    MS_LOG(EXCEPTION) << "Failed to parse url " << url;
  }
  ip_addr_ = const_cast<char *>(ip.c_str());
  port_ = port;

  // Init URPC for RMDA server.
  struct urpc_config urpc_cfg = {};
  urpc_cfg.mode = URPC_MODE_SERVER;
  urpc_cfg.sfeature = 0;
  urpc_cfg.model = URPC_THREAD_MODEL_R2C;
  urpc_cfg.worker_num = kServerWorkingThreadNum;
  urpc_cfg.transport.dev_name = dev_name_;
  urpc_cfg.transport.ip_addr = ip_addr_;
  urpc_cfg.transport.port = port_;
  urpc_cfg.transport.max_sge = 0;
  urpc_cfg.allocator = nullptr;
  if (urpc_init_func(&urpc_cfg) != kURPCSuccess) {
    MS_LOG(EXCEPTION) << "Failed to call urpc_init. Device name: " << dev_name_ << ", ip address: " << ip_addr_
                      << ", port: " << port_ << ". Please refer to URPC log directory: /var/log/umdk/urpc.";
  }
  return true;
}

void RDMAServer::Finalize() {
  if (message_handler_) {
    urpc_unregister_handler_func(nullptr, kInterProcessDataHandleID);
    urpc_uninit_func();
  }
}

void RDMAServer::SetMessageHandler(const MessageHandler &handler) {
  if (!handler) {
    MS_LOG(EXCEPTION) << "The handler of RDMAServer is empty.";
  }
  message_handler_ = handler;

  if (urpc_register_raw_handler_explicit_func(urpc_req_handler, this, urpc_rsp_handler, urpc_allocator_,
                                              kInterProcessDataHandleID) != kURPCSuccess) {
    MS_LOG(EXCEPTION) << "Failed to set handler for RDMAServer.";
  }
}

std::string RDMAServer::GetIP() const { return ""; }

uint32_t RDMAServer::GetPort() const { return 0; }

void RDMAServer::urpc_req_handler(struct urpc_sgl *req, void *arg, struct urpc_sgl *rsp) {
  MS_ERROR_IF_NULL_WO_RET_VAL(req);
  MS_ERROR_IF_NULL_WO_RET_VAL(arg);
  MS_ERROR_IF_NULL_WO_RET_VAL(rsp);

  MessageBase *msg = new (std::nothrow) MessageBase();
  MS_ERROR_IF_NULL_WO_RET_VAL(msg);
  msg->data = reinterpret_cast<void *>(req->sge[0].addr);
  msg->size = req->sge[0].length;

  RDMAServer *server = static_cast<RDMAServer *>(arg);
  MessageHandler message_handler = server->message_handler_;
  (void)message_handler(msg);

  auto rsp_buf = server->urpc_allocator_->alloc(msg->size);
  std::string rsp_msg = "Hello client!";
  (void)memcpy_s(rsp_buf, msg->size, rsp_msg.c_str(), rsp_msg.size());
  rsp->sge[0].addr = reinterpret_cast<uintptr_t>(rsp_buf);
  rsp->sge[0].length = msg->size;
  rsp->sge[0].flag = URPC_SGE_FLAG_ZERO_COPY;
  rsp->sge_num = 1;
}

void RDMAServer::urpc_rsp_handler(struct urpc_sgl *rsp, void *arg) {
  MS_ERROR_IF_NULL_WO_RET_VAL(rsp);
  MS_ERROR_IF_NULL_WO_RET_VAL(arg);

  auto urpc_allocator = static_cast<struct urpc_buffer_allocator *>(arg);
  MS_ERROR_IF_NULL_WO_RET_VAL(urpc_allocator);
  urpc_allocator->free(reinterpret_cast<void *>(rsp->sge[0].addr));
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
