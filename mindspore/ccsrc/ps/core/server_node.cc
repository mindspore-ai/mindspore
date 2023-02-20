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
#include "ps/core/server_node.h"

#include <atomic>
#include <map>
#include <string>
#include <utility>

#include "proto/comm.pb.h"
#include "ps/core/comm_util.h"
#include "ps/core/communicator/tcp_client.h"
#include "ps/core/communicator/tcp_server.h"
#include "ps/core/configuration.h"
#include "ps/core/file_configuration.h"
#include "ps/core/node_info.h"
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace ps {
namespace core {
bool ServerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "[Server start]: 1. Begin to start server node!";
  Initialize();
  Register(client_to_scheduler_);
  MS_LOG(INFO) << "[Server start]: 4. The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " successfully registered to the scheduler!";

  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start server node timeout!";
    return false;
  }
  is_recover = false;
  MsException::Instance().CheckException();
  MS_LOG(INFO) << "[Server start]: 5. Successfully start server node!";
  return true;
}

void ServerNode::Initialize() {
  config_ = std::make_unique<FileConfiguration>(PSContext::instance()->config_file_path());
  MS_EXCEPTION_IF_NULL(config_);
  InitNodeNum();
  bool is_recover = false;
  if (!config_->Initialize()) {
    MS_LOG(WARNING) << "The config file is empty.";
  } else {
    is_recover = Recover();
    if (!is_recover) {
      MS_LOG(DEBUG) << "Recover the server node is failed.";
    }
  }
  InitServerHandler();
  CreateTcpServer();
  InitNodeInfo(NodeRole::SERVER);

  MS_LOG(INFO) << "[Server start]: 2. Server node create tcp server successful!";

  InitCommandHandler();
  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Server node connect to scheduler timedout!";
  }
  InitClientToServer();
  is_already_stopped_ = false;
  if (is_recover) {
    std::string node_role = CommUtil::NodeRoleToString(node_info_.node_role_);
    SendFailMessageToScheduler(node_role, "Node restart");
  }
  MS_LOG(INFO) << "[Server start]: 3. Server node crete tcp client to scheduler successful!";
}

bool ServerNode::Stop() {
  MS_LOG(INFO) << "Stop server node!";
  MS_ERROR_IF_NULL_W_RET_VAL(client_to_scheduler_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(server_, false);
  if (!is_already_stopped_.load()) {
    is_already_stopped_ = true;
    is_finish_ = true;
    client_to_scheduler_->Stop();
    if (!connected_nodes_.empty()) {
      for (auto &connected_node : connected_nodes_) {
        connected_node.second->Stop();
      }
    }
    server_->Stop();
  }
  return true;
}

bool ServerNode::Finish(const uint32_t &timeout) {
  if (is_already_finished_) {
    MS_LOG(INFO) << "Server node already finish!";
    return true;
  }
  is_already_finished_ = true;
  if (is_already_stopped_) {
    MS_LOG(INFO) << "The node has already stopped.";
    return true;
  }

  if (client_to_scheduler_->connection_status() != 1) {
    MS_LOG(INFO) << "[Server finish]: Not connect to scheduler, no need to disconnect!";
    return true;
  }

  MS_LOG(INFO) << "[Server finish]: 1. Begin to finish server node!";
  bool res = Disconnect(client_to_scheduler_, timeout);
  if (res) {
    MS_LOG(INFO) << "[Server finish]: 2. Successfully finish server node!";
  } else {
    MS_LOG(WARNING) << "[Server finish]: 2. finish server node timeout!";
  }
  return res;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
