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

#include "include/backend/distributed/rpc/tcp/tcp_client.h"
#include "ps/core/communicator/ssl_client.h"
#include "distributed/rpc/tcp/tcp_comm.h"

namespace mindspore {
namespace distributed {
namespace rpc {
TCPClient::TCPClient(bool enable_ssl) : RPCClientBase(enable_ssl), tcp_comm_(nullptr), received_message_(nullptr) {}
TCPClient::~TCPClient() {}

bool TCPClient::Initialize() {
  bool rt = false;
  if (tcp_comm_ == nullptr) {
    if (enable_ssl_) {
      (void)ps::core::SSLClient::GetInstance().GetSSLCtx();
    }
    tcp_comm_ = std::make_unique<TCPComm>(enable_ssl_);
    MS_EXCEPTION_IF_NULL(tcp_comm_);

    // This message handler is used to accept and maintain the received message from the tcp server.
    tcp_comm_->SetMessageHandler([this](MessageBase *const message) -> MessageBase *const {
      // Wait for the previous received message has been handled.
      const int sleep_time = 10;
      while (received_message_ != nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
      }
      std::unique_lock<std::mutex> lock(mutex_);
      received_message_ = message;
      wait_msg_cond_.notify_one();
      return NULL_MSG;
    });
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

bool TCPClient::Connect(const std::string &dst_url, size_t retry_count, const MemFreeCallback &free_cb) {
  unsigned int interval = 2;
  for (size_t i = 0; i < retry_count; ++i) {
    if (tcp_comm_->Connect(dst_url, free_cb)) {
      MS_LOG(INFO) << "Connected to the tcp server " << dst_url << " successfully.";
      return true;
    } else {
      MS_LOG(WARNING) << "Failed to connect to the tcp server : " << dst_url << ", retry to reconnect(" << (i + 1)
                      << "/" << retry_count << ")...";
      if (!tcp_comm_->Disconnect(dst_url)) {
        MS_LOG(ERROR) << "Can not disconnect from the server: " << dst_url;
        return false;
      }
      (void)sleep(interval);
    }
  }
  return false;
}

bool TCPClient::IsConnected(const std::string &dst_url) { return tcp_comm_->IsConnected(dst_url); }

bool TCPClient::Disconnect(const std::string &dst_url, size_t timeout_in_sec) {
  bool rt = false;
  if (!tcp_comm_->Disconnect(dst_url)) {
    MS_LOG(ERROR) << "Can not disconnect from the server: " << dst_url;
    return false;
  }

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

bool TCPClient::SendSync(std::unique_ptr<MessageBase> &&msg, size_t *const send_bytes) {
  return tcp_comm_->Send(msg.release(), send_bytes, true);
}

void TCPClient::SendAsync(std::unique_ptr<MessageBase> &&msg) { (void)tcp_comm_->Send(msg.release(), nullptr, false); }

MessageBase *TCPClient::ReceiveSync(std::unique_ptr<MessageBase> &&msg, uint32_t timeout) {
  bool retval = tcp_comm_->Send(msg.release(), nullptr, true);
  if (retval) {
    std::unique_lock<std::mutex> lock(mutex_);
    received_message_ = nullptr;
    bool res =
      wait_msg_cond_.wait_for(lock, std::chrono::seconds(timeout), [this] { return received_message_ != nullptr; });
    if (res) {
      // Clear the address of received message before returning this address to the caller, because the next
      // `ReceiveSync` call will block on the received message's condition variable.
      MessageBase *message = received_message_;
      return message;
    }
  }
  return NULL_MSG;
}

bool TCPClient::Flush(const std::string &dst_url) { return tcp_comm_->Flush(dst_url); }
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
