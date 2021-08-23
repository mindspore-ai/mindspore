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
#include "ps/core/communicator/tcp_communicator.h"
#include "ps/core/communicator/http_communicator.h"

namespace mindspore {
namespace ps {
namespace core {
bool ServerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "[Server start]: 1. Begin to start server node!";
  Initialize();
  Register(client_to_scheduler_);
  MS_LOG(INFO) << "[Server start]: 4. The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " successfully registered to the scheduler!";

  StartHeartbeatTimer(client_to_scheduler_);
  MS_LOG(INFO) << "[Server start]: 5. Server start heartbeat timer!";

  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start server node timeout!";
    return false;
  }

  MsException::Instance().CheckException();
  MS_LOG(INFO) << "[Server start]: 6. Successfully start server node!";
  return true;
}

void ServerNode::set_handler(const RequestHandler &handler) { request_handler_ = handler; }

void ServerNode::Response(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                          const void *data, size_t size) {
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
  server_->SetMessageCallback([&](const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                  const Protos &protos, const void *data, size_t size) {
    if (server_handler_.count(meta->cmd()) == 0) {
      MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
    }

    if (meta->cmd() == NodeCommand::COLLECTIVE_SEND_DATA) {
      ProcessCollectiveSendData(conn, meta, data, size);
      RunReceiveCallback(meta, protos, data, size);
    } else if (meta->cmd() == NodeCommand::SEND_DATA) {
      ProcessSendData(conn, meta, protos, data, size);
    } else {
      const auto &handler_ptr = server_handler_[meta->cmd()];
      (this->*handler_ptr)(conn, meta, protos, data, size);
    }
  });
  server_->Init();
  server_thread_ = std::make_unique<std::thread>([this]() {
    MS_LOG(INFO) << "The server node start a tcp server!";
    this->server_->Start();
  });
  server_thread_->detach();
}

void ServerNode::Initialize() {
  config_ = std::make_unique<FileConfiguration>(PSContext::instance()->config_file_path());
  if (!config_->Initialize()) {
    MS_LOG(INFO) << "The config file is empty, then init node by context.";
    InitNodeNum();
  } else {
    if (!Recover()) {
      MS_LOG(WARNING) << "Recover the server node is failed.";
    }
  }
  InitServerHandler();
  CreateTcpServer();
  is_already_stopped_ = false;
  InitNodeInfo(NodeRole::SERVER);

  MS_LOG(INFO) << "[Server start]: 2. Server node create tcp server successful!";

  InitCommandHandler();
  if (!InitClientToScheduler()) {
    MS_LOG(EXCEPTION) << "Server node connect to scheduler timedout!";
  }
  MS_LOG(INFO) << "[Server start]: 3. Server node crete tcp client to scheduler successful!";
}

void ServerNode::ProcessSendData(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                 const Protos &, const void *data, size_t size) {
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

void ServerNode::ProcessCollectiveSendData(const std::shared_ptr<TcpConnection> &conn,
                                           const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  server_->SendMessage(conn, meta, Protos::RAW, data, size);
}

std::shared_ptr<CommunicatorBase> ServerNode::GetOrCreateHttpComm(const std::string &ip, uint16_t port,
                                                                  const std::shared_ptr<TaskExecutor> &task_executor) {
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (!communicators_.count(kHttpCommunicator)) {
    MS_LOG(INFO) << "Create Http communicator.";
    auto http_comm = std::make_shared<HttpCommunicator>(ip, port, task_executor);
    MS_EXCEPTION_IF_NULL(http_comm);
    communicators_[kHttpCommunicator] = http_comm;
  }
  return communicators_[kHttpCommunicator];
}

std::shared_ptr<CommunicatorBase> ServerNode::GetOrCreateTcpComm(const std::string &scheduler_ip,
                                                                 uint16_t scheduler_port, uint32_t worker_num,
                                                                 uint32_t server_num,
                                                                 const std::shared_ptr<TaskExecutor> &task_executor) {
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (!communicators_.count(kTcpCommunicator)) {
    MS_LOG(INFO) << "Create Tcp communicator.";
    auto tcp_comm = std::make_shared<TcpCommunicator>(task_executor, this);
    MS_EXCEPTION_IF_NULL(tcp_comm);
    PSContext::instance()->cluster_config().scheduler_host = scheduler_ip;
    PSContext::instance()->cluster_config().scheduler_port = scheduler_port;
    PSContext::instance()->cluster_config().initial_worker_num = worker_num;
    PSContext::instance()->cluster_config().initial_server_num = server_num;
    MS_LOG(INFO) << "Initialize cluster metadata for server. Worker number:" << worker_num
                 << ", Server number:" << server_num << ", Scheduler ip:" << scheduler_ip
                 << ", Scheduler port:" << scheduler_port;
    communicators_[kTcpCommunicator] = tcp_comm;
  }
  return communicators_[kTcpCommunicator];
}

bool ServerNode::Stop() {
  MS_LOG(INFO) << "Stop server node!";
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
