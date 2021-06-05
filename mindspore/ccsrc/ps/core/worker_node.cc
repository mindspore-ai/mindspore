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
  MS_LOG(INFO) << "Starting worker node!";
  Initialize();
  Register(client_to_scheduler_);
  StartHeartbeatTimer(client_to_scheduler_);

  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start Worker node timeout!";
    return false;
  }
  MS_LOG(INFO) << "The node is ready to fetch servers!";

  MsException::Instance().CheckException();
  MS_LOG(INFO) << "The Worker node has successfully started.";
  return true;
}

void WorkerNode::Initialize() {
  is_already_stopped_ = false;
  InitServerHandler();
  CreateTcpServer();
  InitNodeInfo(NodeRole::WORKER);
  InitNodeNum();
  InitCommandHandler();
  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Worker node init client timeout!";
  }
  MS_LOG(INFO) << "Worker node init client successful!";
}

void WorkerNode::CreateTcpServer() {
  std::string interface;
  std::string server_ip;
  CommUtil::GetAvailableInterfaceAndIP(&interface, &server_ip);
  server_ = std::make_shared<TcpServer>(server_ip, 0);
  server_->SetMessageCallback([&](std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta,
                                  const Protos &protos, const void *data, size_t size) {
    if (server_handler_.count(meta->cmd()) == 0) {
      MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
    }
    const auto &handler_ptr = server_handler_[meta->cmd()];
    (this->*handler_ptr)(conn, meta, protos, data, size);
  });
  server_->Init();
  server_thread_ = std::make_unique<std::thread>([&]() {
    MS_LOG(INFO) << "The worker node start a tcp server!";
    server_->Start();
  });
  server_thread_->detach();
}

bool WorkerNode::Stop() {
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
  MS_LOG(INFO) << "Finish worker node!";
  is_already_finished_ = true;
  return Disconnect(client_to_scheduler_, timeout);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
