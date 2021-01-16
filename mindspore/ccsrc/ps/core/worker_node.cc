/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ps/core/worker_node.h"

namespace mindspore {
namespace ps {
namespace core {
WorkerNode::~WorkerNode() {
  MS_LOG(INFO) << "Stop worker node!";
  Stop();
}
bool WorkerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "Starting worker node!";
  Initialize();
  Register(client_to_scheduler_);
  StartHeartbeatTimer(client_to_scheduler_);

  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start Worker node timeout!";
    return false;
  }
  MS_LOG(INFO) << "The node is ready to fetch servers!";

  // If the cluster is ready to use, then Get the address of all the servers
  if (!is_timeout_.load()) {
    FetchServers(client_to_scheduler_);
    MS_LOG(INFO) << "Worker node get all the servers address successful!";
  }
  MS_LOG(INFO) << "The Worker node has successfully started.";
  return true;
}

void WorkerNode::Initialize() {
  is_already_stopped_ = false;
  node_info_.node_id_ = CommUtil::GenerateUUID();
  node_info_.node_role_ = NodeRole::WORKER;
  MS_LOG(INFO) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id is:" << node_info_.node_id_;
  InitCommandHandler();
  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Worker node init client timeout!";
  }
  MS_LOG(INFO) << "Worker node init client successful!";
}

bool WorkerNode::Stop() {
  MS_LOG(INFO) << "Stop worker node!";
  if (!is_already_stopped_.load()) {
    is_ready_ = true;
    is_timeout_ = true;
    is_finish_ = true;
    heart_beat_thread_->join();
    client_to_scheduler_->Stop();
    if (!connected_nodes_.empty()) {
      for (auto &connected_node : connected_nodes_) {
        connected_node.second->Stop();
      }
    }
    client_to_scheduler_thread_->join();
    is_already_stopped_ = true;
  }
  return true;
}

bool WorkerNode::Finish(const uint32_t &timeout) {
  std::lock_guard<std::mutex> lock(finish_mutex_);
  if (is_already_finished_) {
    MS_LOG(INFO) << "Worker node already finish!";
    return true;
  }
  MS_LOG(INFO) << "Finish worker node!";
  is_already_finished_ = true;
  return Disconnect(client_to_scheduler_, timeout);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
