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

#include "distributed/rpc/tcp/tcp_server.h"

namespace mindspore {
namespace distributed {
namespace rpc {
bool TCPServer::Initialize(const std::string &url) {
  if (tcp_comm_ == nullptr) {
    tcp_comm_ = std::make_unique<TCPComm>();
    MS_EXCEPTION_IF_NULL(tcp_comm_);
    bool rt = tcp_comm_->Initialize();
    if (!rt) {
      MS_LOG(EXCEPTION) << "Failed to initialize tcp comm";
    }
    rt = tcp_comm_->StartServerSocket(url);
    return rt;
  } else {
    return true;
  }
}

void TCPServer::Finalize() {
  if (tcp_comm_ != nullptr) {
    tcp_comm_.reset();
    tcp_comm_ = nullptr;
  }
}

void TCPServer::SetMessageHandler(MessageHandler handler) { tcp_comm_->SetMessageHandler(handler); }
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
