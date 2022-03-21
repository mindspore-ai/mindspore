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

#include "distributed/rpc/tcp/tcp_client.h"

namespace mindspore {
namespace distributed {
namespace rpc {
bool TCPClient::Initialize() {
  bool rt = false;
  if (tcp_comm_ == nullptr) {
    tcp_comm_ = std::make_unique<TCPComm>();
    MS_EXCEPTION_IF_NULL(tcp_comm_);
    rt = tcp_comm_->Initialize();
  } else {
    rt = true;
  }
  return rt;
}

void TCPClient::Finalize() {
  if (tcp_comm_ != nullptr) {
    tcp_comm_->Finalize();
    tcp_comm_.reset();
    tcp_comm_ = nullptr;
  }
}

bool TCPClient::Connect(const std::string &dst_url, size_t timeout_in_sec) {
  bool rt = false;
  tcp_comm_->Connect(dst_url);

  size_t timeout_in_ms = timeout_in_sec * 1000;
  size_t sleep_in_ms = 100;
  useconds_t sleep_in_us = 100000;

  while (true) {
    if (tcp_comm_->IsConnected(dst_url)) {
      rt = true;
      break;
    }
    if (timeout_in_ms > sleep_in_ms) {
      timeout_in_ms -= sleep_in_ms;
    } else {
      break;
    }
    (void)usleep(sleep_in_us);
  }
  return rt;
}

bool TCPClient::Disconnect(const std::string &dst_url, size_t timeout_in_sec) {
  bool rt = false;
  tcp_comm_->Disconnect(dst_url);

  size_t timeout_in_ms = timeout_in_sec * 1000;
  size_t sleep_in_ms = 100;
  useconds_t sleep_in_us = 100000;

  while (true) {
    if (!tcp_comm_->IsConnected(dst_url)) {
      rt = true;
      break;
    }
    if (timeout_in_ms > sleep_in_ms) {
      timeout_in_ms -= sleep_in_ms;
    } else {
      break;
    }
    (void)usleep(sleep_in_us);
  }
  return rt;
}

int TCPClient::SendSync(std::unique_ptr<MessageBase> &&msg) { return tcp_comm_->Send(msg.release(), true); }

void TCPClient::SendAsync(std::unique_ptr<MessageBase> &&msg) { (void)tcp_comm_->Send(msg.release(), false); }
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
