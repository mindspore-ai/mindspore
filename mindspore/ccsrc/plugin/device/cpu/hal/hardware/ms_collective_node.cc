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
#include "utils/ms_exception.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "plugin/device/cpu/hal/hardware/ms_collective_node.h"

namespace mindspore {
namespace ps {
namespace core {
constexpr char kRankIdPrefix[] = "MCCL_COLLECTIVE_RANK_";
using ClusterContext = mindspore::distributed::cluster::ClusterContext;

bool CollectiveNode::Start(const uint32_t &timeout) {
  InitNodeNum();
  config_ = std::make_unique<FileConfiguration>(PSContext::instance()->config_file_path());
  MS_EXCEPTION_IF_NULL(config_);
  if (!config_->Initialize()) {
    MS_LOG(WARNING) << "Failed to initialize the configuration for this mccl collective node.";
  }

  InitServerHandler();
  CreateTcpServer();
  InitNodeInfo(NodeRole::WORKER);
  InitCommandHandler();

  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Failed to initialize the common tcp client.";
  }
  is_already_stopped_ = false;

  SynchronizeAddresses();
  MS_EXCEPTION_IF_NULL(cgn_);
  node_info_.rank_id_ = cgn_->rank_id();
  MS_LOG(INFO) << "The cpu collective rank " << node_info_.rank_id_ << " has been started successfully.";
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
  is_already_finished_ = true;
  if (is_already_stopped_) {
    return true;
  }
  MS_LOG(INFO) << "The cpu collective node has been finished successfully.";
  return true;
}

void CollectiveNode::SynchronizeAddresses() {
  if (cgn_ != nullptr) {
    // Register the address of this node.
    auto rank_id = kRankIdPrefix + cgn_->role() + "_" + std::to_string(cgn_->rank_id());
    auto address = node_info_.ip_ + ":" + std::to_string(node_info_.port_);

    const size_t interval = 3;
    const size_t max_retry = 20;
    size_t retry = max_retry;
    bool success = false;
    while (!success && --retry > 0) {
      success = cgn_->PutMetadata(rank_id, address);
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
    auto node_num = ClusterContext::instance()->node_num(cgn_->role());
    for (size_t i = 0; i < node_num; ++i) {
      success = false;
      retry = max_retry;
      auto other_rank_id = kRankIdPrefix + cgn_->role() + "_" + std::to_string(i);
      while (!success && --retry > 0) {
        auto other_address = cgn_->GetMetadata(other_rank_id);
        if (other_address != "") {
          auto ip = other_address.substr(0, other_address.find(":"));
          auto port =
            std::stoi(other_address.substr(other_address.find(":") + 1, other_address.length() - ip.length()));
          nodes_address_[std::make_pair(NodeRole::WORKER, i)] = std::make_pair(ip, port);
          success = true;
        } else {
          MS_LOG(INFO) << "Waiting for the address of rank " << other_rank_id << " to be registered, retry " << retry
                       << " times.";
          (void)sleep(interval);
        }
        MsException::Instance().CheckException();
      }
      if (!success) {
        MS_LOG(EXCEPTION) << "Failed to fetch the address of the rank " << other_rank_id
                          << " for mccl collective nodes.";
      }
    }
  }
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
