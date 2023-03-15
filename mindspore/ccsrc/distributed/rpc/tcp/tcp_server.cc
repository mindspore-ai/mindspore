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

#include "include/backend/distributed/rpc/tcp/tcp_server.h"
#include "distributed/rpc/tcp/tcp_comm.h"

namespace mindspore {
namespace distributed {
namespace rpc {
TCPServer::TCPServer(bool enable_ssl) : RPCServerBase(enable_ssl), tcp_comm_(nullptr) {}
TCPServer::~TCPServer() {}

bool TCPServer::Initialize(const std::string &url, const MemAllocateCallback &allocate_cb) {
  return InitializeImpl(url, allocate_cb);
}

bool TCPServer::Initialize(const MemAllocateCallback &allocate_cb) { return InitializeImpl("", allocate_cb); }

void TCPServer::Finalize() {
  if (tcp_comm_ != nullptr) {
    tcp_comm_->Finalize();
    tcp_comm_.reset();
    tcp_comm_ = nullptr;
  }
}

void TCPServer::SetMessageHandler(const MessageHandler &handler, uint32_t) { tcp_comm_->SetMessageHandler(handler); }

std::string TCPServer::GetIP() const { return ip_; }

uint32_t TCPServer::GetPort() const { return port_; }

bool TCPServer::InitializeImpl(const std::string &url, const MemAllocateCallback &allocate_cb) {
  if (tcp_comm_ == nullptr) {
    tcp_comm_ = std::make_unique<TCPComm>(enable_ssl_);
    MS_EXCEPTION_IF_NULL(tcp_comm_);
    bool rt = tcp_comm_->Initialize();
    if (!rt) {
      MS_LOG(EXCEPTION) << "Failed to initialize tcp comm";
    }
    if (url != "") {
      rt = tcp_comm_->StartServerSocket(url, allocate_cb);
      ip_ = SocketOperation::GetIP(url);
    } else {
      rt = tcp_comm_->StartServerSocket(allocate_cb);
      ip_ = SocketOperation::GetLocalIP();
    }
    auto server_fd = tcp_comm_->GetServerFd();
    port_ = SocketOperation::GetPort(server_fd);

    return rt;
  } else {
    return true;
  }
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
