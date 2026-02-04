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

#include "include/cluster/rpc/tcp_server.h"
#include <memory>
#include <string>
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "cluster/rpc/tcp/tcp_comm.h"

namespace mindspore {
namespace distributed {
namespace rpc {
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
  if (event_loop_group_) {
    event_loop_group_->Finalize();
  }
}

void TCPServer::SetMessageHandler(const MessageHandler &handler, uint32_t) { tcp_comm_->SetMessageHandler(handler); }

std::string TCPServer::GetIP() const { return ip_; }

uint32_t TCPServer::GetPort() const { return port_; }

bool TCPServer::InitializeImpl(const std::string &url, const MemAllocateCallback &allocate_cb) {
  if (tcp_comm_ == nullptr) {
    const auto &value = runtime::GetRuntimeConfigValue(runtime::kRuntimeClusterThreadNum);
    MS_LOG(INFO) << "Config cluster threads num : " << value << ".";
    // Enable multi threads by default.
    if (value.size() == 0 || (value != "False" && value != "false" && value != "FALSE")) {
      size_t cluster_thread_num = 0;
      std::stringstream sstream(value);
      size_t config_value = 0;
      sstream >> config_value;
      if (config_value != 0) {
        cluster_thread_num = config_value;
      }
      // Enable event loop group when cluster thread num more than 1.
      const size_t event_loop_group_min_thread_num = 1;
      if (cluster_thread_num > event_loop_group_min_thread_num) {
        event_loop_group_ = std::make_shared<EventLoopGroup>(cluster_thread_num);
      }
    }

    tcp_comm_ = std::make_unique<TCPComm>(enable_ssl_, event_loop_group_);
    MS_EXCEPTION_IF_NULL(tcp_comm_);
    bool rt = tcp_comm_->Initialize();
    if (!rt) {
      MS_LOG(EXCEPTION) << "Failed to initialize tcp comm";
    }
    if (event_loop_group_) {
      event_loop_group_->Initialize();
    }
    if (url != "") {
      rt = (tcp_comm_->StartServerSocket(url, allocate_cb) == 0) ? true : false;
      ip_ = SocketOperation::GetIP(url);
    } else {
      rt = StartSocketWithinPortRange(allocate_cb);
      ip_ = SocketOperation::GetLocalIP();
    }
    auto server_fd = tcp_comm_->GetServerFd();
    port_ = SocketOperation::GetPort(server_fd);

    return rt;
  } else {
    return true;
  }
}

bool TCPServer::StartSocketWithinPortRange(const MemAllocateCallback &allocate_cb) {
  uint32_t current_port = port_range_.first;
  int result;
  std::string new_url;
  do {
    if (port_range_.first > port_range_.second) {
      MS_LOG(EXCEPTION) << "The port range " << port_range_.first << " to " << port_range_.second << " is invalid.";
    }
    new_url = SocketOperation::GetLocalIP() + ":" + std::to_string(current_port);
    result = tcp_comm_->StartServerSocket(new_url, allocate_cb);

    // Return value kAddressInUseError means this port is in use, so we increase the port by 1 and retry.
    if (result == kAddressInUseError) {
      current_port++;
      MS_LOG(WARNING) << "The address " << new_url
                      << " is already in use. Select another url and increase port to: " << current_port;
      if (current_port > port_range_.second) {
        MS_LOG(EXCEPTION) << "Port range " << port_range_.first << " to " << port_range_.second
                          << " are all in use already. You can run 'netstat -anp|grep <port number>' command to check "
                             "which process occupies the port.";
      }
    } else if (result == -1) {
      return false;
    } else {
      return true;
    }
  } while (result == kAddressInUseError);
  return true;
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
