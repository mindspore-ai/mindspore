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

namespace mindspore {
namespace ps {
namespace core {
ServerNode::~ServerNode() {
  MS_LOG(INFO) << "Stop server node!";
  if (!is_already_stopped_.load()) {
    server_->Stop();
    client_to_scheduler_->Stop();
    client_to_scheduler_->StopEventBase();
    if (server_thread_->joinable()) {
      server_thread_->join();
    }
    if (client_to_scheduler_thread_->joinable()) {
      client_to_scheduler_thread_->join();
    }
    is_already_stopped_ = true;
  }
}

bool ServerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "Start server node!";
  Initialize();
  Register(client_to_scheduler_);
  StartHeartbeatTimer(client_to_scheduler_);

  if (!WaitForStart(timeout)) {
    MS_LOG(EXCEPTION) << "Start Worker node timeout!";
  }
  MS_LOG(INFO) << "The cluster is ready to use!";

  // If the cluster is ready to use, then Get the address of all the servers
  if (!is_timeout_.load()) {
    FetchServers(client_to_scheduler_);
    MS_LOG(INFO) << "Server node get all the servers address successful!";
  }
  MS_LOG(INFO) << "Start the node is successful!";
  return true;
}

void ServerNode::set_handler(const RequestHandler &handler) { request_handler_ = handler; }

void ServerNode::Response(const TcpServer &server, const TcpConnection &conn, const MessageMeta &message_meta,
                          const std::string &message) {
  auto &meta = const_cast<MessageMeta &>(message_meta);
  meta.set_role(node_info_.node_role_);
  meta.set_rank_id(node_info_.rank_id_);
  CommMessage comm_message;
  *comm_message.mutable_pb_meta() = {meta};
  comm_message.set_data(message);

  const_cast<TcpServer &>(server).SendMessage(conn, comm_message);
}

void ServerNode::CreateTcpServer() {
  std::string interface;
  std::string server_ip;
  CommUtil::GetAvailableInterfaceAndIP(&interface, &server_ip);
  server_ = std::make_shared<TcpServer>(server_ip, 0);
  server_->SetMessageCallback([&](const TcpServer &server, const TcpConnection &conn, const CommMessage &message) {
    switch (message.pb_meta().cmd()) {
      case NodeCommand::SEND_DATA:
        ProcessSendData(server, conn, message);
        break;
      default:
        MS_LOG(EXCEPTION) << "The cmd:" << message.pb_meta().cmd() << " is not supported!";
    }
  });
  server_->Init();
  server_thread_ = std::make_unique<std::thread>([&]() {
    MS_LOG(INFO) << "The server node start a tcp server!";
    server_->Start();
  });
  server_thread_->detach();
}

void ServerNode::Initialize() {
  CreateTcpServer();
  is_already_stopped_ = false;
  node_info_.node_id_ = CommUtil::GenerateUUID();
  node_info_.node_role_ = NodeRole::SERVER;
  node_info_.ip_ = server_->BoundIp();
  node_info_.port_ = server_->BoundPort();
  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " is generate uuid is:" << node_info_.node_id_;
  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Server node init client timeout!";
  }
  MS_LOG(INFO) << "Server node init client successful!";
}

void ServerNode::ProcessSendData(const TcpServer &server, const TcpConnection &conn, const CommMessage &message) {
  if (request_handler_) {
    request_handler_(server, conn, message.pb_meta(), message.data());
  }
}

bool ServerNode::Stop() {
  MS_LOG(INFO) << "Stop server node!";
  if (!is_already_stopped_.load()) {
    server_->Stop();
    client_to_scheduler_->Stop();
    client_to_scheduler_->StopEventBase();
    if (server_thread_->joinable()) {
      server_thread_->join();
    }
    if (client_to_scheduler_thread_->joinable()) {
      client_to_scheduler_thread_->join();
    }
    if (heart_beat_thread_->joinable()) {
      heart_beat_thread_->join();
    }
    is_already_stopped_ = true;
  }
  return true;
}

bool ServerNode::Finish(const uint32_t &timeout) {
  std::lock_guard<std::mutex> lock(finish_mutex_);
  if (is_already_finished_) {
    MS_LOG(INFO) << "Server node already finish!";
    return true;
  }
  is_already_finished_ = true;
  return Disconnect(client_to_scheduler_, timeout);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
