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

#include "include/backend/distributed/rpc/rdma/rdma_server.h"

#include <string>

namespace mindspore {
namespace distributed {
namespace rpc {
bool RDMAServer::Initialize(const std::string &url, const MemAllocateCallback &allocate_cb) {
  if (!ParseURL(url, &ip_addr_, &port_)) {
    MS_LOG(EXCEPTION) << "Failed to parse url " << url;
  }

  return InitializeURPC(dev_name_, ip_addr_, port_);
}

void RDMAServer::Finalize() {
  if (message_handler_) {
    urpc_unregister_handler_func(nullptr, func_id_);
  }
  if (kURPCInited) {
    urpc_uninit_func();
    kURPCInited = false;
  }
}

void RDMAServer::SetMessageHandler(const MessageHandler &handler, uint32_t func_id) {
  if (!handler) {
    MS_LOG(EXCEPTION) << "The handler of RDMAServer is empty.";
  }
  message_handler_ = handler;
  func_id_ = func_id;

  if (urpc_register_raw_handler_explicit_func(urpc_req_handler, this, urpc_rsp_handler, urpc_allocator_, func_id_) !=
      kURPCSuccess) {
    MS_LOG(EXCEPTION) << "Failed to set handler for RDMAServer of func_id: " << func_id_;
  }
}

std::string RDMAServer::GetIP() const { return ip_addr_; }

uint32_t RDMAServer::GetPort() const { return static_cast<uint32_t>(port_); }

void RDMAServer::urpc_req_handler(struct urpc_sgl *req, void *arg, struct urpc_sgl *rsp) {
  MS_ERROR_IF_NULL_WO_RET_VAL(req);
  MS_ERROR_IF_NULL_WO_RET_VAL(arg);
  MS_ERROR_IF_NULL_WO_RET_VAL(rsp);

  MessageBase *msg = new (std::nothrow) MessageBase();
  MS_ERROR_IF_NULL_WO_RET_VAL(msg);
  // Pay attention: when client send one message with URPC_SGE_FLAG_RENDEZVOUS, the data is stored in sge[1].
  msg->data = reinterpret_cast<void *>(req->sge[1].addr);
  msg->size = req->sge[1].length;

  RDMAServer *server = static_cast<RDMAServer *>(arg);
  MessageHandler message_handler = server->message_handler_;
  (void)message_handler(msg);

  std::string rsp_msg = "Client calls " + std::to_string(server->func_id()) + " function.";
  auto rsp_buf = server->urpc_allocator()->alloc(rsp_msg.size());
  if (memcpy_s(rsp_buf, rsp_msg.size(), rsp_msg.c_str(), rsp_msg.size()) != EOK) {
    MS_LOG(EXCEPTION) << "Failed to memcpy_s for response message.";
  }
  rsp->sge[0].addr = reinterpret_cast<uintptr_t>(rsp_buf);
  rsp->sge[0].length = rsp_msg.size();
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
