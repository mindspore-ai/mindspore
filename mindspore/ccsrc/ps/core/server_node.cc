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
bool ServerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "Start server node!";
  Initialize();
  Register(client_to_scheduler_);
  StartHeartbeatTimer(client_to_scheduler_);

  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start server node timeout!";
    return false;
  }
  MS_LOG(INFO) << "The cluster is ready to use!";

  // If the cluster is ready to use, then Get the address of all the servers
  if (!is_timeout_.load()) {
    FetchServers(client_to_scheduler_);
    MS_LOG(INFO) << "Server node get all the servers address successful!";
  }
  MsException::Instance().CheckException();
  MS_LOG(INFO) << "Start the node is successful!";
  return true;
}

void ServerNode::set_handler(const RequestHandler &handler) { request_handler_ = handler; }

void ServerNode::Response(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta, const void *data,
                          size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  meta->set_role(node_info_.node_role_);
  meta->set_rank_id(node_info_.rank_id_);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << meta->request_id();
  server_->SendMessage(conn, meta, Protos::RAW, data, size);
}

void ServerNode::CreateTcpServer() {
  std::string interface;
  std::string server_ip;
  CommUtil::GetAvailableInterfaceAndIP(&interface, &server_ip);
  server_ = std::make_shared<TcpServer>(server_ip, 0);
  server_->SetMessageCallback([&](std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta,
                                  const Protos &protos, const void *data, size_t size) {
    switch (meta->cmd()) {
      case NodeCommand::SEND_DATA:
        ProcessSendData(conn, meta, protos, data, size);
        break;
      case NodeCommand::COLLECTIVE_SEND_DATA:
        ProcessCollectiveSendData(conn, meta, data, size);
        RunReceiveCallback(meta, protos, data, size);
        break;
      default:
        MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
    }
  });
  server_->Init();
  server_thread_ = std::make_unique<std::thread>([&]() {
    MS_LOG(INFO) << "The server node start a tcp server!";
    server_->Start();
  });
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
  InitCommandHandler();
  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Server node init client timeout!";
  }
  MS_LOG(INFO) << "Server node init client successful!";
}

void ServerNode::ProcessSendData(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta,
                                 const Protos &protos, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  std::shared_ptr<unsigned char[]> res(new unsigned char[size]);
  size_t dest_size = size;
  size_t src_size = size;
  auto ret = memcpy_s(res.get(), dest_size, data, src_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << meta->request_id()
                << " the current time is:"
                << std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now())
                     .time_since_epoch()
                     .count();
  request_handler_(conn, meta, res, size);
}

void ServerNode::ProcessCollectiveSendData(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta,
                                           const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  server_->SendMessage(conn, meta, Protos::RAW, data, size);
}

bool ServerNode::Stop() {
  MS_LOG(INFO) << "Stop server node!";
  if (!is_already_stopped_.load()) {
    is_already_stopped_ = true;
    is_finish_ = true;
    if (heart_beat_thread_->joinable()) {
      heart_beat_thread_->join();
    }
    client_to_scheduler_->Stop();
    if (!connected_nodes_.empty()) {
      for (auto &connected_node : connected_nodes_) {
        connected_node.second->Stop();
      }
    }
    if (client_to_scheduler_thread_->joinable()) {
      client_to_scheduler_thread_->join();
    }
    server_->Stop();
    server_thread_->join();
  }
  return true;
}

bool ServerNode::Finish(const uint32_t &timeout) {
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
