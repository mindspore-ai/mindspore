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

  HeartbeatRespMessage heartbeat_resp_message;

  heartbeat_resp_message.set_cluster_state(node_manager_.GetClusterState());

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
  handlers_[NodeCommand::FETCH_METADATA] = &SchedulerNode::ProcessFetchMetadata;
}

void SchedulerNode::CreateTcpServer() {
  node_manager_.InitNode();

  std::string scheduler_host = PSContext::instance()->cluster_config().scheduler_host;
  uint32_t scheduler_port = PSContext::instance()->cluster_config().scheduler_port;
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

  if (node_manager_.IsAllNodesRegistered()) {
    is_ready_ = true;
    auto node_infos = node_manager_.nodes_info();
    for (const auto &kvs : node_infos) {
      auto client = GetOrCreateClient(kvs.second);
      SendMetadata(client);
    }
    current_cluster_state_ = ClusterState::CLUSTER_READY;
    wait_start_cond_.notify_all();
  }
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
  if (node_manager_.IsAllNodesFinished()) {
    auto node_infos = node_manager_.nodes_info();
    for (const auto &kvs : node_infos) {
      auto client = GetOrCreateClient(kvs.second);
      SendFinish(client);
    }
    is_finish_ = true;
    current_cluster_state_ = ClusterState::CLUSTER_FINISH;
    wait_finish_cond_.notify_all();
  }
}

void SchedulerNode::ProcessFetchMetadata(std::shared_ptr<TcpServer> server, std::shared_ptr<TcpConnection> conn,
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

void SchedulerNode::SendMetadata(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  message_meta->set_cmd(NodeCommand::SEND_METADATA);

  SendMetadataMessage send_metadata_message;
  std::vector<ServersMeta> servers_meta_list = node_manager_.FetchServersMeta();

  *send_metadata_message.mutable_servers_meta() = {servers_meta_list.begin(), servers_meta_list.end()};

  if (!SendMessageAsync(client, message_meta, Protos::PROTOBUF, send_metadata_message.SerializeAsString().data(),
                        send_metadata_message.ByteSizeLong())) {
    MS_LOG(EXCEPTION) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << " the node id:" << node_info_.node_id_ << " send metadata timeout!";
  }

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << "is sending metadata to workers and servers!";
}

void SchedulerNode::SendFinish(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  message_meta->set_cmd(NodeCommand::FINISH);

  // The scheduler does not need to bring any data when sending the finish command
  std::string resp_data;

  if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, resp_data.data(), resp_data.size())) {
    MS_LOG(EXCEPTION) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << " the node id:" << node_info_.node_id_ << " send finish timeout!";
  }

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << "is sending finish to workers and servers!";
}

void SchedulerNode::StartUpdateClusterStateTimer() {
  MS_LOG(WARNING) << "The scheduler start a heartbeat timer!";
  update_state_thread_ = std::make_unique<std::thread>([&]() {
    auto start_time = std::chrono::steady_clock::now();
    while (!is_finish_.load()) {
      // 1. update cluster timeout
      if (!is_ready_ && (std::chrono::steady_clock::now() - start_time >
                         std::chrono::seconds(PSContext::instance()->cluster_config().cluster_available_timeout))) {
        node_manager_.CheckClusterTimeout();
      }
      std::this_thread::sleep_for(std::chrono::seconds(PSContext::instance()->cluster_config().heartbeat_interval));
      node_manager_.UpdateCluster();

      if (node_manager_.GetClusterState() == ClusterState::CLUSTER_FINISH) {
        std::this_thread::sleep_for(
          std::chrono::seconds(PSContext::instance()->cluster_config().heartbeat_interval * 2));
        is_finish_ = true;
        wait_finish_cond_.notify_all();
      }
    }
  });
}

const std::shared_ptr<TcpClient> &SchedulerNode::GetOrCreateClient(const NodeInfo &node_info) {
  if (connected_nodes_.count(node_info.node_id_)) {
    return connected_nodes_[node_info.node_id_];
  } else {
    std::string ip = node_info.ip_;
    uint16_t port = node_info.port_;
    auto client = std::make_shared<TcpClient>(ip, port);
    client->SetMessageCallback([&](std::shared_ptr<MessageMeta> meta, const Protos &protos, const void *data,
                                   size_t size) { NotifyMessageArrival(meta); });
    client->Init();
    if (is_client_started_ == false) {
      is_client_started_ = true;
      client_thread_ = std::make_unique<std::thread>([&]() {
        MS_LOG(INFO) << "The node start a tcp client!";
        client->Start();
      });
    }
    connected_nodes_[node_info.node_id_] = client;
    return connected_nodes_[node_info.node_id_];
  }
}

bool SchedulerNode::Stop() {
  MS_LOG(INFO) << "Stop scheduler node!";
  if (!is_already_stopped_) {
    is_already_stopped_ = true;
    update_state_thread_->join();
    server_->Stop();
    scheduler_thread_->join();
    if (!connected_nodes_.empty()) {
      for (auto &connected_node : connected_nodes_) {
        connected_node.second->Stop();
      }
    }
    client_thread_->join();
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
