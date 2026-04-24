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

#include <utility>
#include <string>
#include "utils/ms_exception.h"
#include "include/cluster/topology/cluster_context.h"
#include "plugin/cpu/res_manager/collective/ms_collective_node.h"

namespace mindspore {
namespace ps {
namespace core {
constexpr char kRankIdPrefix[] = "MCCL_COLLECTIVE_RANK_";
using ClusterContext = mindspore::distributed::cluster::ClusterContext;

namespace {
// Offset to skip the closing bracket ']' and the colon ':' in IPv6 address format
constexpr size_t kIPv6BracketOffset = 2;

// Parses "ip:port" or "[ipv6]:port" format. Returns (ip, port) or ("", 0) on failure.
std::pair<std::string, uint16_t> ParseAddress(const std::string &address) {
  if (address.empty()) {
    return {"", 0};
  }
  std::string ip;
  uint16_t port = 0;
  if (address.front() == '[') {
    // IPv6 with brackets: [fe80::1]:8080
    size_t bracket_end = address.find(']');
    if (bracket_end == std::string::npos || bracket_end + kIPv6BracketOffset > address.size() ||
        address[bracket_end + 1] != ':') {
      return {"", 0};
    }
    ip = address.substr(1, bracket_end - 1);
    try {
      port = static_cast<uint16_t>(std::stoul(address.substr(bracket_end + kIPv6BracketOffset)));
    } catch (const std::exception &) {
      return {"", 0};
    }
  } else {
    // IPv4 or IPv6 without brackets: 192.168.1.1:8080 or fe80::1:8080
    size_t colon_pos = address.rfind(':');
    if (colon_pos == std::string::npos || colon_pos + 1 >= address.size()) {
      return {"", 0};
    }
    ip = address.substr(0, colon_pos);
    try {
      port = static_cast<uint16_t>(std::stoul(address.substr(colon_pos + 1)));
    } catch (const std::exception &) {
      return {"", 0};
    }
  }
  return {ip, port};
}

// Formats ip and port for registration. IPv6 addresses use brackets: [fe80::1]:8080
std::string FormatAddress(const std::string &ip, uint16_t port) {
  if (ip.find(':') != std::string::npos) {
    return "[" + ip + "]:" + std::to_string(port);
  }
  return ip + ":" + std::to_string(port);
}
}  // namespace

bool CollectiveNode::Start(const uint32_t &timeout) {
  InitNodeNum();
  config_ = std::make_unique<FileConfiguration>(PSContext::instance()->config_file_path());
  MS_EXCEPTION_IF_NULL(config_);
  if (!config_->Initialize()) {
    MS_LOG(INFO) << "Failed to initialize the configuration for this mccl collective node.";
  }

  InitServerHandler();
  CreateTcpServer(ClusterContext::instance()->port_range());
  InitNodeInfo(NodeRole::WORKER);
  InitCommandHandler();

  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Failed to initialize the common tcp client.";
  }
  is_already_stopped_ = false;

  SynchronizeAddresses();
  MS_EXCEPTION_IF_NULL(client_node_);
  node_info_.rank_id_ = client_node_->rank_id();
  MS_LOG(INFO) << "The cpu collective rank " << node_info_.rank_id_ << " has been started successfully.";
  is_initialized_.store(true);
  return true;
}

bool CollectiveNode::InitClientToScheduler() {
  // Create the TCP client to scheduler.
  client_to_scheduler_ = std::make_shared<TcpClient>(scheduler_ip_, 0, NodeRole::SCHEDULER);
  MS_EXCEPTION_IF_NULL(client_to_scheduler_);
  client_to_scheduler_->Init();

  client_to_scheduler_thread_ = std::make_unique<std::thread>([this]() { client_to_scheduler_->Start(); });
  return true;
}

bool CollectiveNode::Finish(const uint32_t &timeout) {
  MS_LOG(INFO) << "Begin to finish the cpu collective node.";
  is_initialized_.store(false);
  is_already_finished_ = true;
  if (is_already_stopped_) {
    return true;
  }
  MS_LOG(INFO) << "The cpu collective node has been finished successfully.";
  return true;
}

bool CollectiveNode::Stop() {
  MS_ERROR_IF_NULL_W_RET_VAL(client_to_scheduler_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(server_, false);
  if (!is_already_stopped_.load()) {
    MS_LOG(INFO) << "Stop worker node!";
    is_ready_ = true;
    is_finish_ = true;
    client_to_scheduler_->Stop();
    if (!connected_nodes_.empty()) {
      for (auto &connected_node : connected_nodes_) {
        connected_node.second->Stop();
      }
    }
    server_->Stop();
    is_already_stopped_ = true;
  }
  return true;
}

void CollectiveNode::SynchronizeAddresses() {
  if (client_node_ == nullptr) {
    return;
  }

  // Register the address of this node.
  auto rank_id = kRankIdPrefix + client_node_->role() + "_" + std::to_string(client_node_->rank_id());
  auto address = FormatAddress(node_info_.ip_, static_cast<uint16_t>(node_info_.port_));

  const size_t interval = 3;
  const size_t max_retry = 20;
  size_t retry = max_retry;
  bool success = false;
  while (!success && --retry > 0) {
    success = client_node_->PutMetadata(rank_id, address);
    if (!success) {
      MS_LOG(WARNING) << "Retry to register the address of rank " << rank_id << "...";
      (void)sleep(interval);
    } else {
      MS_LOG(INFO) << "The address of rank " << rank_id << " has been registered successfully.";
      break;
    }
    MsException::Instance().CheckException();
  }

  if (!success) {
    MS_LOG(EXCEPTION) << "Failed to register the address of this mccl collective node(rank id: " << rank_id << ").";
  }

  // Get the addresses of other nodes.
  nodes_address_.clear();
  auto node_num = ClusterContext::instance()->node_num(client_node_->role());
  for (size_t i = 0; i < node_num; ++i) {
    auto other_rank_id = kRankIdPrefix + client_node_->role() + "_" + std::to_string(i);
    bool fetch_success = FetchNodeAddress(other_rank_id, i);
    if (!fetch_success) {
      MS_LOG(EXCEPTION) << "Failed to fetch the address of the rank " << other_rank_id << " for mccl collective nodes.";
    }
  }
}

bool CollectiveNode::FetchNodeAddress(const std::string &other_rank_id, size_t node_index) {
  const size_t interval = 3;
  const size_t max_retry = 20;
  size_t retry = max_retry;
  while (retry-- > 0) {
    auto other_address = client_node_->GetMetadata(other_rank_id);
    if (other_address.empty()) {
      MS_LOG(INFO) << "Waiting for the address of rank " << other_rank_id << " to be registered, retry " << retry
                   << " times.";
      (void)sleep(interval);
      MsException::Instance().CheckException();
      continue;
    }

    auto [ip, port] = ParseAddress(other_address);
    if (ip.empty()) {
      MS_LOG(WARNING) << "Failed to parse address " << other_address << " for rank " << other_rank_id;
      MsException::Instance().CheckException();
      continue;
    }

    nodes_address_[std::make_pair(NodeRole::WORKER, node_index)] = std::make_pair(ip, port);
    MS_LOG(INFO) << "The address of rank " << other_rank_id << " has been fetched successfully: " << other_address;
    return true;
  }
  return false;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
