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
#include "plugin/device/cpu/hal/hardware/ms_collective_node.h"

namespace mindspore {
namespace ps {
namespace core {
constexpr char kRankIdPrefix[] = "MCCL_COLLECTIVE_RANK_";

bool CollectiveNode::Start(const uint32_t &timeout) {
  InitNodeNum();
  config_ = std::make_unique<FileConfiguration>(PSContext::instance()->config_file_path());
  MS_EXCEPTION_IF_NULL(config_);
  if (config_->Initialize() && !Recover()) {
    MS_LOG(INFO) << "Failed to recover the mccl collective node.";
  }

  InitServerHandler();
  CreateTcpServer();
  InitNodeInfo(NodeRole::WORKER);
  InitCommandHandler();

  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Failed to initialize the common tcp client.";
  }
  is_already_stopped_ = false;

  if (cgn_ != nullptr) {
    // Register the address of this node.
    auto rank_id = kRankIdPrefix + std::to_string(cgn_->rank_id());
    auto address = node_info_.ip_ + ":" + std::to_string(node_info_.port_);
    cgn_->PutMetadata(rank_id, address, false);

    // Get the addresses of other nodes.
    const size_t interval = 3;
    nodes_address_.clear();
    for (size_t i = 0; i < worker_num_; ++i) {
      bool success = false;
      while (!success) {
        auto other_rank_id = kRankIdPrefix + std::to_string(i);
        auto other_address = cgn_->GetMetadata(other_rank_id);
        if (other_address != "") {
          auto ip = other_address.substr(0, other_address.find(":"));
          auto port =
            std::stoi(other_address.substr(other_address.find(":") + 1, other_address.length() - ip.length()));
          nodes_address_[std::make_pair(NodeRole::WORKER, i)] = std::make_pair(ip, port);
          success = true;
        } else {
          MS_LOG(INFO) << "Waiting for the address of rank " << other_rank_id << " to be registered";
          sleep(interval);
        }
      }
    }
  }
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
}  // namespace core
}  // namespace ps
}  // namespace mindspore
