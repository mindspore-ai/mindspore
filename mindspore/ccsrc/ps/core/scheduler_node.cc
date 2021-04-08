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

#include "ps/core/scheduler_node.h"

namespace mindspore {
namespace ps {
namespace core {
SchedulerNode::~SchedulerNode() {
  MS_LOG(INFO) << "Stop scheduler node!";
  Stop();
}

bool SchedulerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "Start scheduler node!";
  Initialize();
  StartUpdateClusterStateTimer();
  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start Scheduler node timeout!";
    return false;
  }
  MS_LOG(INFO) << "Start the scheduler node is successful!";
  return true;
}

void SchedulerNode::ProcessHeartbeat(std::shared_ptr<TcpServer> server, std::shared_ptr<TcpConnection> conn,
                                     std::shared_ptr<MessageMeta> meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  HeartbeatMessage heartbeat_message;
  heartbeat_message.ParseFromArray(data, size);

  node_manager_.UpdateHeartbeat(heartbeat_message.node_id());

  if (heartbeat_message.is_node_finish()) {
    node_manager_.UpdateNodeFinishState(heartbeat_message.node_id());
  }

  if (heartbeat_message.is_node_finish() && node_manager_.CheckNodesFinishState()) {
    MS_LOG(INFO) << "The scheduler node receive all the finish cmd!";
    is_finish_ = true;
    wait_finish_cond_.notify_all();
  }

  HeartbeatRespMessage heartbeat_resp_message;
  heartbeat_resp_message.set_is_cluster_ready(node_manager_.is_cluster_ready());
  heartbeat_resp_message.set_is_cluster_finish(node_manager_.is_cluster_finish());
  heartbeat_resp_message.set_is_cluster_timeout(node_manager_.is_cluster_timeout());
  heartbeat_resp_message.set_is_node_timeout(node_manager_.is_node_timeout());

  server->SendMessage(conn, meta, Protos::PROTOBUF, heartbeat_resp_message.SerializeAsString().data(),
                      heartbeat_resp_message.ByteSizeLong());
}

void SchedulerNode::Initialize() {
  InitCommandHandler();
  CreateTcpServer();
  is_already_stopped_ = false;
  node_info_.node_id_ = CommUtil::GenerateUUID();
  node_info_.node_role_ = NodeRole::SCHEDULER;
  MS_LOG(INFO) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id is:" << node_info_.node_id_;
}

void SchedulerNode::InitCommandHandler() {
  handlers_[NodeCommand::HEARTBEAT] = &SchedulerNode::ProcessHeartbeat;
  handlers_[NodeCommand::REGISTER] = &SchedulerNode::ProcessRegister;
  handlers_[NodeCommand::FINISH] = &SchedulerNode::ProcessFinish;
  handlers_[NodeCommand::FETCH_SERVER] = &SchedulerNode::ProcessFetchServers;
}

void SchedulerNode::CreateTcpServer() {
  node_manager_.InitNodeNum();

  std::string scheduler_host = ClusterMetadata::instance()->scheduler_host();
  uint32_t scheduler_port = ClusterMetadata::instance()->scheduler_port();
  server_ = std::make_shared<TcpServer>(scheduler_host, scheduler_port);
  server_->SetMessageCallback([&](std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta,
                                  const Protos &protos, const void *data, size_t size) {
    if (handlers_.count(meta->cmd()) == 0) {
      MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
    }
    const auto &handler_ptr = handlers_[meta->cmd()];
    (this->*handler_ptr)(server_, conn, meta, data, size);
  });

  server_->Init();

  scheduler_thread_ = std::make_unique<std::thread>([&]() {
    MS_LOG(INFO) << "The scheduler node start a tcp server!";
    server_->Start();
  });
}

void SchedulerNode::ProcessRegister(std::shared_ptr<TcpServer> server, std::shared_ptr<TcpConnection> conn,
                                    std::shared_ptr<MessageMeta> meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  MS_LOG(INFO) << "The scheduler process a register message!";
  RegisterMessage register_message;
  register_message.ParseFromArray(data, size);

  // assign worker node and server node rank id
  int rank_id = node_manager_.NextRankId(register_message);
  if (rank_id < 0) {
    MS_LOG(WARNING) << "The rank id is wrong!";
  }
  const std::string &node_id = register_message.node_id();
  node_manager_.UpdateHeartbeat(node_id);

  RegisterRespMessage register_resp_message;
  register_resp_message.set_node_id(node_id);
  register_resp_message.set_rank_id(rank_id);

  server->SendMessage(conn, meta, Protos::PROTOBUF, register_resp_message.SerializeAsString().data(),
                      register_resp_message.ByteSizeLong());
}

void SchedulerNode::ProcessFinish(std::shared_ptr<TcpServer> server, std::shared_ptr<TcpConnection> conn,
                                  std::shared_ptr<MessageMeta> meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  auto finish_message = std::make_unique<std::string>(reinterpret_cast<const char *>(data), size);
  node_manager_.AddFinishNode(*finish_message);
  MS_LOG(INFO) << "Process finish message from node id:" << *finish_message;
  server->SendMessage(conn, meta, Protos::PROTOBUF, data, size);
}

void SchedulerNode::ProcessFetchServers(std::shared_ptr<TcpServer> server, std::shared_ptr<TcpConnection> conn,
                                        std::shared_ptr<MessageMeta> meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  FetchServersRespMessage fetch_servers_message;
  std::vector<ServersMeta> servers_meta_list = node_manager_.FetchServersMeta();

  *fetch_servers_message.mutable_servers_meta() = {servers_meta_list.begin(), servers_meta_list.end()};

  server->SendMessage(conn, meta, Protos::PROTOBUF, fetch_servers_message.SerializeAsString().data(),
                      fetch_servers_message.ByteSizeLong());
}

void SchedulerNode::StartUpdateClusterStateTimer() {
  MS_LOG(WARNING) << "The scheduler start a heartbeat timer!";
  update_state_thread_ = std::make_unique<std::thread>([&]() {
    auto start_time = std::chrono::steady_clock::now();
    while (!is_finish_.load()) {
      // 1. update cluster timeout
      if (!node_manager_.is_cluster_ready() &&
          (std::chrono::steady_clock::now() - start_time >
           std::chrono::seconds(ClusterMetadata::instance()->cluster_available_timeout()))) {
        node_manager_.CheckClusterTimeout();
      }

      // 2. update cluster state
      std::this_thread::sleep_for(std::chrono::seconds(ClusterMetadata::instance()->heartbeat_interval()));
      node_manager_.UpdateClusterState();
      if (node_manager_.is_cluster_ready()) {
        is_ready_ = true;
        wait_start_cond_.notify_all();
      }
      if (node_manager_.is_cluster_finish()) {
        std::this_thread::sleep_for(std::chrono::seconds(ClusterMetadata::instance()->heartbeat_interval() * 2));
        is_finish_ = true;
        wait_finish_cond_.notify_all();
      }
    }
  });
}

bool SchedulerNode::Stop() {
  MS_LOG(INFO) << "Stop scheduler node!";
  if (!is_already_stopped_) {
    is_already_stopped_ = true;
    update_state_thread_->join();
    server_->Stop();
    scheduler_thread_->join();
    is_ready_ = true;
  }
  return true;
}

bool SchedulerNode::Finish(const uint32_t &timeout) {
  MS_LOG(INFO) << "Finish scheduler node!";
  std::unique_lock<std::mutex> lock(wait_finish_mutex_);
  wait_finish_cond_.wait(lock, [&] {
    if (is_finish_.load()) {
      MS_LOG(INFO) << "The scheduler finish success!";
    }
    return is_finish_.load();
  });
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
