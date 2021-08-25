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
#include "ps/core/comm_util.h"

namespace mindspore {
namespace ps {
namespace core {
bool WorkerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "[Worker start]: 1. Begin to start worker node!";
  Initialize();
  Register(client_to_scheduler_);
  MS_LOG(INFO) << "[Worker start]: 4. The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " successfully registered to the scheduler!";

  StartHeartbeatTimer(client_to_scheduler_);
  MS_LOG(INFO) << "[Worker start]: 5. Worker start heartbeat timer!";

  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start Worker node timeout!";
    return false;
  }

  MsException::Instance().CheckException();
  MS_LOG(INFO) << "[Worker start]: 6. Successfully start worker node!";
  return true;
}

void WorkerNode::Initialize() {
  config_ = std::make_unique<FileConfiguration>(PSContext::instance()->config_file_path());
  MS_EXCEPTION_IF_NULL(config_);
  if (!config_->Initialize()) {
    MS_LOG(INFO) << "The config file is empty, then init node by context.";
    InitNodeNum();
  } else {
    if (!Recover()) {
      MS_LOG(WARNING) << "Recover the worker node is failed.";
    }
  }
  InitServerHandler();
  CreateTcpServer();
  InitNodeInfo(NodeRole::WORKER);

  MS_LOG(INFO) << "[Worker start]: 2. Worker node create tcp server successful!";

  InitCommandHandler();
  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Worker node connect to scheduler timeout!";
  }
  is_already_stopped_ = false;
  MS_LOG(INFO) << "[Worker start]: 3. Worker node crete tcp client to scheduler successful!";
}

bool WorkerNode::Stop() {
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

bool WorkerNode::Finish(const uint32_t &timeout) {
  if (is_already_finished_) {
    MS_LOG(INFO) << "Worker node already finish!";
    return true;
  }
  MS_LOG(INFO) << "[Worker finish]: 1. Begin to finish worker node!";
  is_already_finished_ = true;
  if (is_already_stopped_) {
    MS_LOG(INFO) << "The node is already stop.";
    return true;
  }
  bool res = Disconnect(client_to_scheduler_, timeout);
  if (res) {
    MS_LOG(INFO) << "[Worker finish]: 2. Successfully finish worker node!";
  } else {
    MS_LOG(WARNING) << "[Worker finish]: 2. finish worker node timeout!";
  }

  return res;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
