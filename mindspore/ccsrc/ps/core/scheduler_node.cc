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
#include "ps/core/scheduler_recovery.h"

namespace mindspore {
namespace ps {
namespace core {
SchedulerNode::~SchedulerNode() {
  MS_LOG(INFO) << "Stop scheduler node!";
  if (!Stop()) {
    MS_LOG(WARNING) << "Scheduler node stop failed.";
  }
}

bool SchedulerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "[Scheduler start]: 1. Begin to start scheduler node!";
  config_ = std::make_unique<FileConfiguration>(PSContext::instance()->config_file_path());
  MS_EXCEPTION_IF_NULL(config_);
  if (!config_->Initialize()) {
    MS_LOG(INFO) << "The config file is empty, then init node by context.";
    InitNodeMetaData();
  } else {
    if (!RecoverScheduler()) {
      MS_LOG(WARNING) << "Recover the server node is failed.";
    }
  }

  if (PSContext::instance()->scheduler_manage_port() != 0) {
    MS_LOG(WARNING) << "Start the restful scheduler http service, the ip is 127.0.0.1 "
                    << ", the port:" << PSContext::instance()->scheduler_manage_port();
    StartRestfulServer(kLocalIp, PSContext::instance()->scheduler_manage_port(), 1);
  }
  Initialize();
  StartUpdateClusterStateTimer();
  RunRecovery();

  if (is_worker_timeout_) {
    BroadcastTimeoutEvent();
  }
  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start Scheduler node timeout!";
    return false;
  }
  node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);

  StartUpdatePersistentCommandTimer();
  MS_LOG(INFO) << "[Scheduler start]: 4. Successfully start scheduler, there are " << node_manager_.worker_num()
               << " workers and " << node_manager_.server_num() << " servers registered.";
  return true;
}

void SchedulerNode::RunRecovery() {
  core::ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
  // create tcp client to myself in case of event dispatch failed when Send reconnect msg to server failed
  client_to_scheduler_ =
    std::make_shared<TcpClient>(clusterConfig.scheduler_host, clusterConfig.scheduler_port, config_.get());
  MS_EXCEPTION_IF_NULL(client_to_scheduler_);
  client_to_scheduler_->Init();
  client_thread_ = std::make_unique<std::thread>([&]() {
    MS_LOG(INFO) << "The node start a tcp client!";
    client_to_scheduler_->Start();
  });
  MS_EXCEPTION_IF_NULL(client_thread_);

  auto initial_node_infos = clusterConfig.initial_registered_nodes_infos;
  if (initial_node_infos.empty()) {
    MS_LOG(WARNING) << "There is no registered nodes in scheduler!";
    return;
  }
  MS_LOG(INFO) << "The scheduler start run recovery!";
  int worker_num = clusterConfig.initial_worker_num;
  int server_num = clusterConfig.initial_server_num;

  node_manager_.set_worker_num(worker_num);
  node_manager_.set_server_num(server_num);
  node_manager_.set_next_worker_rank_id(clusterConfig.initial_next_worker_rank_id);
  node_manager_.set_next_server_rank_id(clusterConfig.initial_next_server_rank_id);
  node_manager_.set_total_node_num(clusterConfig.initial_total_node_num);

  for (const auto &kvs : initial_node_infos) {
    auto client = std::make_shared<TcpClient>(kvs.second.ip_, kvs.second.port_, config_.get());
    client->SetMessageCallback(
      [&](const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data, size_t size) {
        MS_LOG(INFO) << "received the response. ";
        NotifyMessageArrival(meta);
      });
    client->Init();
    MS_EXCEPTION_IF_NULL(client);

    auto message_meta = std::make_shared<MessageMeta>();
    MS_EXCEPTION_IF_NULL(message_meta);
    message_meta->set_cmd(NodeCommand::SCHEDULER_RECOVERY);

    int rank_id = kvs.second.rank_id_;
    SendMetadataMessage scheduler_recovery_message;
    scheduler_recovery_message.set_worker_num(worker_num);
    scheduler_recovery_message.set_server_num(server_num);
    scheduler_recovery_message.set_rank_id(rank_id);
    if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, scheduler_recovery_message.SerializeAsString().data(),
                         scheduler_recovery_message.ByteSizeLong())) {
      if (kvs.second.node_role_ == NodeRole::WORKER) {
        is_worker_timeout_ = true;
        break;
      }
      MS_LOG(WARNING) << "Scheduler send recovery msg to " << kvs.first << " timeout!";
    } else {
      MS_LOG(INFO) << "Scheduler send recovery msg to " << kvs.first << " successful.";
    }
  }
  MS_LOG(INFO) << "Scheduler recovery finish.";
}

void SchedulerNode::ProcessHeartbeat(const std::shared_ptr<TcpServer> &server,
                                     const std::shared_ptr<TcpConnection> &conn,
                                     const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  HeartbeatMessage heartbeat_message;
  CHECK_RETURN_TYPE(heartbeat_message.ParseFromArray(data, SizeToInt(size)));

  std::string node_id = heartbeat_message.node_id();
  node_manager_.UpdateHeartbeat(node_id);
  MS_LOG(DEBUG) << "The scheduler get a heartbeat from node id :" << heartbeat_message.node_id();

  HeartbeatRespMessage heartbeat_resp_message;
  heartbeat_resp_message.set_persistent_cmd(PersistentCommand::DEFAULT);

  NodeRole node_role = (node_manager_.QueryNodeInfo(node_id)).node_role_;
  // The worker role does not support disaster recovery for the time being.
  if (node_role == NodeRole::SERVER && persistent_cmd_ == PersistentCommand::BEGIN_PERSIST) {
    if (!node_manager_.IsNodePersisting(node_id)) {
      heartbeat_resp_message.set_persistent_cmd(PersistentCommand::BEGIN_PERSIST);
      node_manager_.AddPersistingNode(node_id);
    }
    if (node_manager_.IsAllNodeInPersisting()) {
      persistent_cmd_ = PersistentCommand::DEFAULT;
    }
  }

  MS_LOG(DEBUG) << "The cluster state:" << CommUtil::ClusterStateToString(node_manager_.GetClusterState());
  heartbeat_resp_message.set_cluster_state(node_manager_.GetClusterState());

  std::vector<ServersMeta> servers_meta_list = node_manager_.FetchAllNodesMeta();

  *heartbeat_resp_message.mutable_servers_meta() = {servers_meta_list.begin(), servers_meta_list.end()};
  heartbeat_resp_message.set_is_worker(node_manager_.IsWorker());

  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, heartbeat_resp_message.SerializeAsString().data(),
                           heartbeat_resp_message.ByteSizeLong())) {
    MS_LOG(WARNING) << "Send heart beat failed.";
  }
}

void SchedulerNode::Initialize() {
  InitCommandHandler();
  CreateTcpServer();
  is_already_stopped_ = false;
  if (PSContext::instance()->node_id().empty() && config_->Exists(kNodeId)) {
    node_info_.node_id_ = config_->Get(kNodeId, "");
  } else {
    node_info_.node_id_ = PSContext::instance()->node_id();
  }

  if (node_info_.node_id_.empty()) {
    node_info_.node_id_ = CommUtil::GenerateUUID();
  }
  node_info_.rank_id_ = 0;
  node_info_.node_role_ = NodeRole::SCHEDULER;
  leader_scaler_ = std::make_unique<LeaderScaler>(this);
  MS_EXCEPTION_IF_NULL(leader_scaler_);
  instance_manager_ = std::make_unique<InstanceManager>(this);
  MS_LOG(INFO) << "[Scheduler start]: 2. The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id is:" << node_info_.node_id_ << " create a tcp server.";
}

void SchedulerNode::InitCommandHandler() {
  handlers_[NodeCommand::HEARTBEAT] = &SchedulerNode::ProcessHeartbeat;
  handlers_[NodeCommand::REGISTER] = &SchedulerNode::ProcessRegister;
  handlers_[NodeCommand::FINISH] = &SchedulerNode::ProcessFinish;
  handlers_[NodeCommand::FETCH_METADATA] = &SchedulerNode::ProcessFetchMetadata;
  handlers_[NodeCommand::SCALE_OUT_DONE] = &SchedulerNode::ProcessScaleOutDone;
  handlers_[NodeCommand::SCALE_IN_DONE] = &SchedulerNode::ProcessScaleInDone;
  handlers_[NodeCommand::SEND_EVENT] = &SchedulerNode::ProcessSendEvent;
}

void SchedulerNode::CreateTcpServer() {
  node_manager_.InitNode();

  std::string scheduler_host = PSContext::instance()->cluster_config().scheduler_host;
  uint32_t scheduler_port = PSContext::instance()->cluster_config().scheduler_port;
  MS_LOG(INFO) << "scheduler ip: " << scheduler_host << ", scheduler ip: " << scheduler_port;
  server_ = std::make_shared<TcpServer>(scheduler_host, scheduler_port, config_.get());
  MS_EXCEPTION_IF_NULL(server_);
  server_->SetMessageCallback([&](const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                  const Protos &, const void *data, size_t size) {
    if (handlers_.count(meta->cmd()) == 0) {
      MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
    }
    const auto &handler_ptr = handlers_[meta->cmd()];
    (this->*handler_ptr)(server_, conn, meta, data, size);
  });

  const auto client_disconn = [&](const TcpServer &, const TcpConnection &conn) {
    int fd = conn.GetFd();
    if (register_connection_fd_.count(fd) <= 0) {
      return;
    }
    MS_LOG(WARNING) << "remove client fd:" << fd << ", remove client id:" << register_connection_fd_[fd];
    register_connection_fd_.erase(fd);
    MS_LOG(WARNING) << "Register node number is:" << register_connection_fd_.size()
                    << ", total node num is:" << node_manager_.total_node_num()
                    << ", scale in node size is: " << scale_in_node_ids_.size();
  };
  server_->SetServerCallback(nullptr, client_disconn, nullptr);
  server_->Init();

  scheduler_thread_ = std::make_unique<std::thread>([this]() {
    MS_LOG(INFO) << "The scheduler node start a tcp server!";
    this->server_->Start();
  });
  MS_EXCEPTION_IF_NULL(scheduler_thread_);
}

void SchedulerNode::ProcessRegister(const std::shared_ptr<TcpServer> &server,
                                    const std::shared_ptr<TcpConnection> &conn,
                                    const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  RegisterMessage register_message;
  CHECK_RETURN_TYPE(register_message.ParseFromArray(data, SizeToInt(size)));
  const std::string &node_id = register_message.node_id();

  MS_LOG(INFO) << "The node id:" << node_id << " is registering to scheduler.";
  client_mutex_.lock();
  if (node_manager_.IsNodeRegistered(node_id)) {
    MS_LOG(INFO) << "The node id is registered.";
    if (connected_nodes_.count(node_id)) {
      (void)connected_nodes_.erase(node_id);
    }
  }
  client_mutex_.unlock();

  // assign worker node and server node rank id
  uint32_t rank_id = node_manager_.NextRankId(register_message, meta);
  if (rank_id == UINT32_MAX) {
    MS_LOG(ERROR) << "The rank id is wrong, return register rejected message!";
    RegisterRespMessage register_rejected_message;
    register_rejected_message.set_node_id(node_id);
    register_rejected_message.set_rank_id(rank_id);
    if (!server->SendMessage(conn, meta, Protos::PROTOBUF, register_rejected_message.SerializeAsString().data(),
                             register_rejected_message.ByteSizeLong())) {
      MS_LOG(WARNING) << "Server response message failed.";
    }
    return;
  }

  node_manager_.UpdateHeartbeat(node_id);
  RegisterRespMessage register_resp_message;
  register_resp_message.set_node_id(node_id);
  register_resp_message.set_rank_id(rank_id);
  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, register_resp_message.SerializeAsString().data(),
                           register_resp_message.ByteSizeLong())) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  SetRegisterConnectionFd(conn, node_id);

  if (node_manager_.IsAllNodesRegistered()) {
    is_ready_ = true;
    MS_LOG(INFO) << "There are " << node_manager_.worker_num() << " workers and " << node_manager_.server_num()
                 << " servers registered to scheduer, so the scheduler send meta data to worker/server.";
    if (node_manager_.GetClusterState() == ClusterState::CLUSTER_SCALE_IN) {
      auto nodes = node_manager_.nodes_info();
      for (const auto &id : scale_in_node_ids_) {
        MS_LOG(INFO) << "The scheduler send metadata to scale in node:" << id;
        if (nodes.count(id)) {
          auto scale_in_client = GetOrCreateClient(nodes[id]);
          SendMetadata(scale_in_client, nodes[id].rank_id_);
          node_manager_.UpdateHeartbeat(id);
        }
      }
    }
    node_manager_.UpdateNodesInfo();
    auto node_infos = node_manager_.nodes_info();
    bool res = SendPrepareBuildingNetwork(node_infos);
    if (!res) {
      MS_LOG(WARNING) << "Prepare for building network failed!";
      return;
    }
    MS_LOG(INFO) << "Prepare for building network success.";
    for (const auto &kvs : node_infos) {
      auto client = GetOrCreateClient(kvs.second);
      MS_EXCEPTION_IF_NULL(client);
      SendMetadata(client, kvs.second.rank_id_);
      node_manager_.UpdateHeartbeat(kvs.first);
    }
    node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
    PersistMetaData();
    wait_start_cond_.notify_all();
  }
}

void SchedulerNode::ProcessFinish(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                                  const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  auto finish_message = std::make_unique<std::string>(reinterpret_cast<const char *>(data), size);
  MS_EXCEPTION_IF_NULL(finish_message);
  std::string node_id = *finish_message;
  MS_LOG(INFO) << "Process finish message from node id:" << node_id;
  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }

  auto iter = std::find_if(scale_in_node_ids_.begin(), scale_in_node_ids_.end(), [node_id](auto item) {
    if (node_id == item) {
      MS_LOG(INFO) << "The finish node is a scale in node.";
      return true;
    }
    return false;
  });
  if (iter != scale_in_node_ids_.end()) {
    return;
  }

  node_manager_.AddFinishNode(node_id);
  if (node_manager_.IsAllNodesFinished()) {
    auto node_infos = node_manager_.nodes_info();
    for (const auto &kvs : node_infos) {
      auto client = GetOrCreateClient(kvs.second);
      SendFinish(client);
    }
    is_finish_ = true;
    node_manager_.UpdateClusterState(ClusterState::CLUSTER_EXIT);
    wait_finish_cond_.notify_all();
  }
}

void SchedulerNode::ProcessFetchMetadata(const std::shared_ptr<TcpServer> &server,
                                         const std::shared_ptr<TcpConnection> &conn,
                                         const std::shared_ptr<MessageMeta> &meta, const void *data, size_t) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  FetchServersRespMessage fetch_servers_message;
  std::vector<ServersMeta> servers_meta_list = node_manager_.FetchAllNodesMeta();

  *fetch_servers_message.mutable_servers_meta() = {servers_meta_list.begin(), servers_meta_list.end()};

  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, fetch_servers_message.SerializeAsString().data(),
                           fetch_servers_message.ByteSizeLong())) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
}

void SchedulerNode::ProcessScaleOutDone(const std::shared_ptr<TcpServer> &server,
                                        const std::shared_ptr<TcpConnection> &conn,
                                        const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  ScaleOutDoneMessage scale_out_done_message;
  scale_out_done_message.ParseFromArray(data, SizeToInt(size));
  std::string node_id = scale_out_done_message.node_id();
  MS_LOG(INFO) << "The scheduler process a scale_out_done message from node id:" << node_id;
  node_manager_.AddScaleOutDoneNode(node_id);

  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }

  if (node_manager_.IsAllNodesScaleOutDone()) {
    auto node_infos = node_manager_.nodes_info();
    for (const auto &kvs : node_infos) {
      auto client = GetOrCreateClient(kvs.second);
      SendScaleOutDone(client);
    }
    is_ready_ = true;
    node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
    PersistMetaData();
  }
}

void SchedulerNode::ProcessScaleInDone(const std::shared_ptr<TcpServer> &server,
                                       const std::shared_ptr<TcpConnection> &conn,
                                       const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  ScaleInDoneMessage scale_in_done_message;
  scale_in_done_message.ParseFromArray(data, SizeToInt(size));
  std::string node_id = scale_in_done_message.node_id();
  MS_LOG(INFO) << "The scheduler process a scale_in_done message from node id:" << node_id;
  node_manager_.AddScaleInDoneNode(node_id);

  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }

  if (node_manager_.IsAllNodesScaleInDone()) {
    auto node_infos = node_manager_.nodes_info();
    for (const auto &kvs : node_infos) {
      auto client = GetOrCreateClient(kvs.second);
      SendScaleInDone(client);
    }
    is_ready_ = true;
    node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
    PersistMetaData();
  }
}

void SchedulerNode::ProcessSendEvent(const std::shared_ptr<TcpServer> &server,
                                     const std::shared_ptr<TcpConnection> &conn,
                                     const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  EventMessage event_message;
  event_message.ParseFromArray(data, SizeToInt(size));
  std::string node_id = event_message.node_id();
  uint32_t event = event_message.event();
  MS_LOG(DEBUG) << "The scheduler process a event message from node id:" << node_id;

  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }

  auto node_infos = node_manager_.nodes_info();
  for (const auto &kvs : node_infos) {
    auto client = GetOrCreateClient(kvs.second);
    SendEvent(client, event);
  }
}

bool SchedulerNode::SendPrepareBuildingNetwork(const std::unordered_map<std::string, NodeInfo> &node_infos) {
  std::string timeoutNodeId = "";
  for (const auto &kvs : node_infos) {
    auto client = GetOrCreateClient(kvs.second);
    auto message_meta = std::make_shared<MessageMeta>();
    MS_EXCEPTION_IF_NULL(message_meta);
    message_meta->set_cmd(NodeCommand::PREPARE_BUILDING_NETWORK);

    SendMetadataMessage send_metadata_message;
    send_metadata_message.set_rank_id(kvs.second.rank_id_);
    if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, send_metadata_message.SerializeAsString().data(),
                         send_metadata_message.ByteSizeLong(), kCommTimeoutInThreeSeconds)) {
      MS_LOG(ERROR) << "The node role:" << CommUtil::NodeRoleToString(kvs.second.node_role_)
                    << " the node id:" << kvs.second.node_id_ << " prepare building network timeout!";
      timeoutNodeId += kvs.second.node_id_ + " ";
    }
  }
  return timeoutNodeId.empty();
}

void SchedulerNode::SendMetadata(const std::shared_ptr<TcpClient> &client, uint32_t rank_id) {
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_METADATA);

  SendMetadataMessage send_metadata_message;
  std::vector<ServersMeta> servers_meta_list = node_manager_.FetchAllNodesMeta();
  send_metadata_message.set_worker_num(node_manager_.worker_num());
  send_metadata_message.set_server_num(node_manager_.server_num());
  send_metadata_message.set_cluster_state(node_manager_.GetClusterState());
  send_metadata_message.set_rank_id(rank_id);

  *send_metadata_message.mutable_servers_meta() = {servers_meta_list.begin(), servers_meta_list.end()};

  if (!SendMessageAsync(client, message_meta, Protos::PROTOBUF, send_metadata_message.SerializeAsString().data(),
                        send_metadata_message.ByteSizeLong())) {
    MS_LOG(EXCEPTION) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << " the node id:" << node_info_.node_id_ << " send metadata timeout!";
  }

  MS_LOG(DEBUG) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << " the node id:" << node_info_.node_id_ << "is sending metadata to workers and servers!";
}

void SchedulerNode::SendFinish(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
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

void SchedulerNode::SendScaleOutDone(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SCALE_OUT_DONE);

  // The scheduler does not need to bring any data when sending the scale_out_done command
  std::string resp_data;

  if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, resp_data.data(), resp_data.size())) {
    MS_LOG(EXCEPTION) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << " the node id:" << node_info_.node_id_ << " send scale_out_done timeout!";
  }

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << "is sending scale_out_done to workers and servers!";
}

void SchedulerNode::SendScaleInDone(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SCALE_IN_DONE);

  // The scheduler does not need to bring any data when sending the scale_in_done command
  std::string resp_data;

  if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, resp_data.data(), resp_data.size())) {
    MS_LOG(EXCEPTION) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << " the node id:" << node_info_.node_id_ << " send scale_in_done timeout!";
  }

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << "is sending scale_in_done to workers and servers!";
}

void SchedulerNode::SendEvent(const std::shared_ptr<TcpClient> &client, const uint32_t &event) {
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_EVENT);

  EventRespMessage event_resp_message;
  event_resp_message.set_event(event);

  if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, event_resp_message.SerializeAsString().data(),
                       event_resp_message.ByteSizeLong())) {
    MS_LOG(ERROR) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                  << " the node id:" << node_info_.node_id_ << " send event resp timeout!";
    return;
  }

  MS_LOG(DEBUG) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << " the node id:" << node_info_.node_id_ << "is sending event resp to workers and servers!";
}

void SchedulerNode::StartUpdateClusterStateTimer() {
  MS_LOG(INFO) << "[Scheduler start]: 3. The scheduler start a heartbeat timer!";
  node_manager_.setPersistCallback([&]() { PersistMetaData(); });
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

      if (node_manager_.GetClusterState() == ClusterState::CLUSTER_EXIT) {
        std::this_thread::sleep_for(
          std::chrono::seconds(PSContext::instance()->cluster_config().heartbeat_interval * kHeartbeatTimes));
        is_finish_ = true;
        wait_finish_cond_.notify_all();
      }
    }
  });
  MS_EXCEPTION_IF_NULL(update_state_thread_);
}

void SchedulerNode::StartUpdatePersistentCommandTimer() {
  if (!EnableRecovery()) {
    return;
  }

  update_persistent_cmd_thread_ = std::make_unique<std::thread>([&]() {
    while (!is_finish_.load()) {
      std::unique_lock<std::mutex> locker(persistent_cmd_mutex_);
      (void)persistent_cmd_cv_.wait_for(
        locker, std::chrono::seconds(PSContext::instance()->cluster_config().persistent_interval));
      persistent_cmd_ = PersistentCommand::BEGIN_PERSIST;
    }
  });

  MS_EXCEPTION_IF_NULL(update_persistent_cmd_thread_);
}

const std::shared_ptr<TcpClient> &SchedulerNode::GetOrCreateClient(const NodeInfo &node_info) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  if (connected_nodes_.count(node_info.node_id_)) {
    return connected_nodes_[node_info.node_id_];
  } else {
    if (config_ == nullptr) {
      MS_LOG(EXCEPTION) << "The config is empty.";
    }
    std::string ip = node_info.ip_;
    uint16_t port = node_info.port_;
    MS_LOG(DEBUG) << "ip:" << ip << ", port:" << port << ", node id:" << node_info.node_id_;
    auto client = std::make_shared<TcpClient>(ip, port, config_.get());
    MS_EXCEPTION_IF_NULL(client);
    client->SetMessageCallback(
      [&](const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data, size_t size) {
        switch (meta->cmd()) {
          case NodeCommand::SEND_DATA:
            ProcessSendDataResp(meta, protos, data, size);
            RunMessageCallback(meta->request_id());
            break;
          default:
            MS_LOG(DEBUG) << "The cmd:" << meta->cmd();
        }
        NotifyMessageArrival(meta);
      });
    client->Init();
    connected_nodes_[node_info.node_id_] = client;
    return connected_nodes_[node_info.node_id_];
  }
}

bool SchedulerNode::Stop() {
  MS_LOG(INFO) << "Stop scheduler node!";
  if (!is_already_stopped_) {
    MS_ERROR_IF_NULL_W_RET_VAL(update_state_thread_, false);
    MS_ERROR_IF_NULL_W_RET_VAL(server_, false);
    MS_ERROR_IF_NULL_W_RET_VAL(scheduler_thread_, false);
    is_already_stopped_ = true;
    update_state_thread_->join();
    server_->Stop();
    scheduler_thread_->join();
    if (!connected_nodes_.empty()) {
      for (auto &connected_node : connected_nodes_) {
        auto client = connected_node.second;
        MS_ERROR_IF_NULL_W_RET_VAL(client, false);
        client->Stop();
      }
    }
    if (client_to_scheduler_ != nullptr) {
      client_to_scheduler_->Stop();
    }
    if (client_thread_ != nullptr && client_thread_->joinable()) {
      client_thread_->join();
    }
    is_ready_ = true;

    if (update_persistent_cmd_thread_ && update_persistent_cmd_thread_->joinable()) {
      persistent_cmd_cv_.notify_one();
      update_persistent_cmd_thread_->join();
    }
  }
  if (PSContext::instance()->scheduler_manage_port() != 0) {
    MS_LOG(WARNING) << "Stop the restful scheduler http service, the ip is 127.0.0.1 "
                    << ", the port:" << PSContext::instance()->scheduler_manage_port();
    StopRestfulServer();
  }

  return true;
}

bool SchedulerNode::Finish(const uint32_t &) {
  MS_LOG(INFO) << "[Scheduler finish]: 1. Begin to listen finish scheduler node!";
  std::unique_lock<std::mutex> lock(wait_finish_mutex_);
  wait_finish_cond_.wait(lock, [this] {
    if (this->is_finish_.load()) {
      MS_LOG(INFO) << "[Scheduler finish]: 2. Successfully finish scheduler!";
    }
    return this->is_finish_.load();
  });
  return true;
}

void SchedulerNode::ProcessScaleoutRollback(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);
  RequestProcessResult status(RequestProcessResultCode::kSuccess);
  if (node_manager_.GetClusterState() != ClusterState::CLUSTER_SCALE_OUT) {
    std::string message = "The cluster state is not CLUSTER_SCALE_OUT, does not need to rollback.";
    ERROR_STATUS(status, RequestProcessResultCode::kSystemError, message);
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }
  // set the last worker num and last server num
  ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
  node_manager_.set_worker_num(clusterConfig.initial_worker_num);
  node_manager_.set_server_num(clusterConfig.initial_server_num);
  node_manager_.set_total_node_num(clusterConfig.initial_total_node_num);

  MS_LOG(INFO) << "After scale out rollback, the last worker num:" << clusterConfig.initial_worker_num
               << ", the last server num:" << clusterConfig.initial_server_num;

  auto node_infos = node_manager_.nodes_info();
  node_manager_.ResetMetadata();
  for (const auto &kvs : node_infos) {
    auto client = GetOrCreateClient(kvs.second);
    MS_EXCEPTION_IF_NULL(client);
    MS_EXCEPTION_IF_NULL(leader_scaler_);
    leader_scaler_->ScaleOutAsync(client, node_manager_);
  }

  MS_LOG(INFO) << "Scheduler send scale out rollback successful.";
  node_manager_.UpdateClusterState(ClusterState::CLUSTER_SCALE_OUT);
  nlohmann::json js;
  js["message"] = "Cluster scale out rollback success.";
  js["code"] = kSuccessCode;
  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::ProcessScaleOut(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);
  RequestProcessResult status(RequestProcessResultCode::kSuccess);
  status = resp->ParsePostMessageToJson();
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  int32_t scale_worker_num = 0;
  status = resp->ParseValueFromKey(kWorkerNum, &scale_worker_num);
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  int32_t scale_server_num = 0;
  status = resp->ParseValueFromKey(kServerNum, &scale_server_num);
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  status = CheckIfClusterReady();
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  int32_t total_worker_num = scale_worker_num + node_manager_.worker_num();
  int32_t total_server_num = scale_server_num + node_manager_.server_num();

  MS_LOG(INFO) << "After scale out, the total worker num:" << total_worker_num
               << ", the total server num:" << total_server_num;

  node_manager_.set_worker_num(total_worker_num);
  node_manager_.set_server_num(total_server_num);
  node_manager_.set_total_node_num(total_worker_num + total_server_num);

  node_manager_.UpdateClusterState(ClusterState::CLUSTER_SCALE_OUT);
  auto node_infos = node_manager_.nodes_info();
  node_manager_.ResetMetadata();
  for (const auto &kvs : node_infos) {
    auto client = GetOrCreateClient(kvs.second);
    MS_EXCEPTION_IF_NULL(client);
    MS_EXCEPTION_IF_NULL(leader_scaler_);
    leader_scaler_->ScaleOutAsync(client, node_manager_);
  }
  MS_LOG(INFO) << "Scheduler send scale out successful.";

  nlohmann::json js;
  js["message"] = "Cluster begin to scale out.";
  js["code"] = kSuccessCode;
  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

/*
 * The response body format.
 * {
 *    "node_ids": ["node_id1", "node_id2"]
 * }
 */
void SchedulerNode::ProcessScaleIn(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);
  RequestProcessResult status(RequestProcessResultCode::kSuccess);
  status = resp->ParsePostMessageToJson();
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
  }

  status = CheckIfClusterReady();
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  scale_in_node_ids_.clear();
  status = resp->ParseNodeIdsFromKey(kNodesIds, &scale_in_node_ids_);
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  status = CheckIfNodeIdLegal(scale_in_node_ids_);
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  MS_LOG(WARNING) << "The scale in node ids:" << scale_in_node_ids_;

  std::unordered_map<std::string, bool> scale_in_nodes;

  int32_t scale_worker_num = 0;
  int32_t scale_server_num = 0;
  auto node_infos = node_manager_.nodes_info();
  node_manager_.UpdateClusterState(ClusterState::CLUSTER_SCALE_IN);
  node_manager_.ResetMetadata(scale_in_node_ids_);
  for (auto const &val : scale_in_node_ids_) {
    if (node_infos.count(val)) {
      scale_in_nodes[val] = true;
      NodeInfo info = node_infos[val];
      if (info.node_role_ == NodeRole::WORKER) {
        scale_worker_num++;
      } else if (info.node_role_ == NodeRole::SERVER) {
        scale_server_num++;
      }
    }
  }

  MS_LOG(INFO) << "The scale worker num:" << scale_worker_num << ", the scale server num:" << scale_server_num;

  int32_t total_worker_num = node_manager_.worker_num() - scale_worker_num;
  int32_t total_server_num = node_manager_.server_num() - scale_server_num;

  node_manager_.set_worker_num(total_worker_num);
  node_manager_.set_server_num(total_server_num);
  node_manager_.set_total_node_num(total_worker_num + total_server_num);
  for (const auto &kvs : node_infos) {
    auto client = GetOrCreateClient(kvs.second);
    bool is_node_scale_in = false;
    if (scale_in_nodes.count(kvs.first)) {
      is_node_scale_in = true;
    }
    MS_EXCEPTION_IF_NULL(leader_scaler_);
    leader_scaler_->ScaleInAsync(client, node_manager_, is_node_scale_in);
  }

  nlohmann::json js;
  js["message"] = "Cluster begin to scale in.";
  js["code"] = kSuccessCode;
  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

/*
 * The response body format.
 * {
 *    "message": "Get nodes info successful.",
 *    "node_ids": [
 *        {
 *            "node_id": "node_id1",
 *            "rank_id": "0",
 *            "role": "SERVER"
 *        },
 *        {
 *            "node_id": "node_id2",
 *            "rank_id": "1",
 *            "role": "WORKER"
 *        }
 *    ]
 * }
 */
void SchedulerNode::ProcessGetNodesInfo(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);
  nlohmann::json js;
  js["message"] = "Get nodes info successful.";
  js["code"] = kSuccessCode;
  auto node_infos = node_manager_.registered_nodes_info();
  for (const auto &kvs : node_infos) {
    std::unordered_map<std::string, std::string> res;
    res["nodeId"] = kvs.second.node_id_;
    res["rankId"] = std::to_string(kvs.second.rank_id_);
    res["role"] = CommUtil::NodeRoleToString(kvs.second.node_role_);
    res["alive"] = kvs.second.is_alive ? "true" : "false";
    js["nodeIds"].push_back(res);
  }

  std::unordered_map<std::string, std::string> scheduler_info;
  scheduler_info["nodeId"] = node_info_.node_id_;
  scheduler_info["rankId"] = std::to_string(node_info_.rank_id_);
  scheduler_info["role"] = CommUtil::NodeRoleToString(node_info_.node_role_);
  scheduler_info["alive"] = "true";
  js["nodeIds"].push_back(scheduler_info);

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

/*
 * The response body format.
 * {
 *    "message": "Get cluster state successful.",
 *    "cluster_state": "CLUSTER_READY"
 * }
 */
void SchedulerNode::ProcessGetClusterState(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);
  nlohmann::json js;
  js["message"] = "Get cluster state successful.";
  auto cluster_state = node_manager_.GetClusterState();
  js["cluster_state"] = CommUtil::ClusterStateToString(cluster_state);
  js["code"] = kSuccessCode;

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::ProcessNewInstance(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);

  RequestProcessResult status(RequestProcessResultCode::kSuccess);

  status = CheckIfClusterReady();
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  status = resp->ParsePostMessageToJson();
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  node_manager_.UpdateClusterState(ClusterState::CLUSTER_NEW_INSTANCE);

  std::string body = resp->request_message().dump();

  uint64_t request_id = AddMessageTrack(node_manager_.server_num());

  std::unordered_map<uint32_t, VectorPtr> outputs;

  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    outputs = receive_messages_[request_id];
    (void)receive_messages_.erase(request_id);
    receive_messages_mutex_.unlock();
  });

  auto node_infos = node_manager_.nodes_info();
  for (const auto &kvs : node_infos) {
    if (kvs.second.node_role_ == NodeRole::SERVER) {
      auto client = GetOrCreateClient(kvs.second);
      MS_EXCEPTION_IF_NULL(client);
      MS_EXCEPTION_IF_NULL(instance_manager_);
      instance_manager_->NewInstanceAsync(client, node_manager_, body, request_id, node_info_);
    }
  }
  bool res = Wait(request_id);
  if (!res) {
    ERROR_STATUS(status, RequestProcessResultCode::kInvalidInputs, "The new instance is timeout.");
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
    return;
  }

  node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
  nlohmann::json js;
  js["message"] = "Start new instance successful.";
  js["code"] = kSuccessCode;
  for (const auto &output : outputs) {
    std::string data = std::string(reinterpret_cast<char *>(output.second->data()), output.second->size());
    js["result"][output.first] = data;
  }

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::ProcessQueryInstance(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);

  RequestProcessResult status(RequestProcessResultCode::kSuccess);

  status = CheckIfClusterReady();
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  uint64_t request_id = AddMessageTrack(node_manager_.server_num());

  std::unordered_map<uint32_t, VectorPtr> outputs;

  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    outputs = receive_messages_[request_id];
    (void)receive_messages_.erase(request_id);
    receive_messages_mutex_.unlock();
  });

  auto node_infos = node_manager_.nodes_info();
  for (const auto &kvs : node_infos) {
    if (kvs.second.node_role_ == NodeRole::SERVER) {
      auto client = GetOrCreateClient(kvs.second);
      MS_EXCEPTION_IF_NULL(client);
      MS_EXCEPTION_IF_NULL(instance_manager_);
      instance_manager_->QueryInstanceAsync(client, node_manager_, request_id, node_info_);
    }
  }
  bool res = Wait(request_id);
  if (!res) {
    ERROR_STATUS(status, RequestProcessResultCode::kInvalidInputs, "The query instance is timeout.");
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  nlohmann::json js;
  js["message"] = "Query Instance successful.";
  js["code"] = kSuccessCode;
  for (const auto &output : outputs) {
    std::string data = std::string(reinterpret_cast<char *>(output.second->data()), output.second->size());
    nlohmann::json dataJson = nlohmann::json::parse(data);
    js["result"] = dataJson;
    break;
  }

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::ProcessEnableFLS(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);

  RequestProcessResult status(RequestProcessResultCode::kSuccess);

  status = CheckIfClusterReady();
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  node_manager_.UpdateClusterState(ClusterState::CLUSTER_ENABLE_FLS);

  uint64_t request_id = AddMessageTrack(node_manager_.server_num());

  std::unordered_map<uint32_t, VectorPtr> outputs;

  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    outputs = receive_messages_[request_id];
    (void)receive_messages_.erase(request_id);
    receive_messages_mutex_.unlock();
  });

  auto node_infos = node_manager_.nodes_info();
  for (const auto &kvs : node_infos) {
    if (kvs.second.node_role_ == NodeRole::SERVER) {
      auto client = GetOrCreateClient(kvs.second);
      MS_EXCEPTION_IF_NULL(client);
      MS_EXCEPTION_IF_NULL(instance_manager_);
      instance_manager_->EnableFLSAsync(client, node_manager_, request_id, node_info_);
    }
  }
  bool res = Wait(request_id);
  if (!res) {
    ERROR_STATUS(status, RequestProcessResultCode::kInvalidInputs, "The enable FLS is timeout.");
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
    return;
  }

  node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
  nlohmann::json js;
  js["message"] = "start enabling FL-Server successful.";
  js["code"] = kSuccessCode;
  for (const auto &output : outputs) {
    std::string data = std::string(reinterpret_cast<char *>(output.second->data()), output.second->size());
    js["result"][output.first] = data;
  }

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::ProcessDisableFLS(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);

  RequestProcessResult status(RequestProcessResultCode::kSuccess);

  status = CheckIfClusterReady();
  if (status != RequestProcessResultCode::kSuccess) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }

  node_manager_.UpdateClusterState(ClusterState::CLUSTER_DISABLE_FLS);

  uint64_t request_id = AddMessageTrack(node_manager_.server_num());

  std::unordered_map<uint32_t, VectorPtr> outputs;

  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    outputs = receive_messages_[request_id];
    (void)receive_messages_.erase(request_id);
    receive_messages_mutex_.unlock();
  });

  auto node_infos = node_manager_.nodes_info();
  for (const auto &kvs : node_infos) {
    if (kvs.second.node_role_ == NodeRole::SERVER) {
      auto client = GetOrCreateClient(kvs.second);
      MS_EXCEPTION_IF_NULL(client);
      MS_EXCEPTION_IF_NULL(instance_manager_);
      instance_manager_->DisableFLSAsync(client, node_manager_, request_id, node_info_);
    }
  }
  bool res = Wait(request_id);
  if (!res) {
    ERROR_STATUS(status, RequestProcessResultCode::kInvalidInputs, "The disable FLS is timeout.");
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
    return;
  }

  node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
  nlohmann::json js;
  js["message"] = "start disabling FL-Server successful.";
  js["code"] = kSuccessCode;
  for (const auto &output : outputs) {
    std::string data = std::string(reinterpret_cast<char *>(output.second->data()), output.second->size());
    js["result"][output.first] = data;
  }

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

RequestProcessResult SchedulerNode::CheckIfClusterReady() {
  RequestProcessResult result(RequestProcessResultCode::kSuccess);
  if (node_manager_.GetClusterState() != ClusterState::CLUSTER_READY || CheckIfNodeDisconnected()) {
    std::string message = "The cluster is not ready.";
    ERROR_STATUS(result, RequestProcessResultCode::kSystemError, message);
    return result;
  }
  return result;
}

RequestProcessResult SchedulerNode::CheckIfNodeIdLegal(const std::vector<std::string> &node_ids) {
  RequestProcessResult result(RequestProcessResultCode::kSuccess);
  if (node_ids.size() == 0) {
    std::string message = "The node ids should not be empty.";
    ERROR_STATUS(result, RequestProcessResultCode::kInvalidInputs, message);
    return result;
  }

  auto node_infos = node_manager_.nodes_info();

  for (auto val : node_ids) {
    if (!node_infos.count(val)) {
      std::string message = "The node id:" + val + " is illegal.";
      MS_LOG(ERROR) << message;
      ERROR_STATUS(result, RequestProcessResultCode::kInvalidInputs, message);
      return result;
    }

    if (node_infos[val].node_role_ == NodeRole::SERVER && node_infos[val].rank_id_ == 0) {
      std::string error_message = "The node id:" + val + " is rank 0 of server, should not be scale in.";
      MS_LOG(ERROR) << error_message;
      ERROR_STATUS(result, RequestProcessResultCode::kInvalidInputs, error_message);
      return result;
    }

    if (node_infos[val].node_role_ == NodeRole::WORKER) {
      std::string error_message = "The node id:" + val + " is the role of worker, should not be scale in.";
      MS_LOG(ERROR) << error_message;
      ERROR_STATUS(result, RequestProcessResultCode::kInvalidInputs, error_message);
      return result;
    }
  }

  return result;
}

void SchedulerNode::StartRestfulServer(const std::string &address, std::uint16_t port, size_t thread_num) {
  MS_LOG(INFO) << "Scheduler start https server.";
  http_server_ = std::make_shared<HttpServer>(address, port, thread_num);
  MS_EXCEPTION_IF_NULL(http_server_);

  OnRequestReceive scale_out = std::bind(&SchedulerNode::ProcessScaleOut, this, std::placeholders::_1);
  callbacks_["/scaleout"] = scale_out;
  http_server_->RegisterRoute("/scaleout", &callbacks_["/scaleout"]);

  OnRequestReceive scale_in = std::bind(&SchedulerNode::ProcessScaleIn, this, std::placeholders::_1);
  callbacks_["/scalein"] = scale_in;
  http_server_->RegisterRoute("/scalein", &callbacks_["/scalein"]);

  OnRequestReceive nodes = std::bind(&SchedulerNode::ProcessGetNodesInfo, this, std::placeholders::_1);
  callbacks_["/nodes"] = nodes;
  http_server_->RegisterRoute("/nodes", &callbacks_["/nodes"]);

  OnRequestReceive cluster_state = std::bind(&SchedulerNode::ProcessGetClusterState, this, std::placeholders::_1);
  callbacks_["/state"] = cluster_state;
  http_server_->RegisterRoute("/state", &callbacks_["/state"]);

  OnRequestReceive new_instance = std::bind(&SchedulerNode::ProcessNewInstance, this, std::placeholders::_1);
  callbacks_["/newInstance"] = new_instance;
  (void)http_server_->RegisterRoute("/newInstance", &callbacks_["/newInstance"]);

  OnRequestReceive query_instance = std::bind(&SchedulerNode::ProcessQueryInstance, this, std::placeholders::_1);
  callbacks_["/queryInstance"] = query_instance;
  (void)http_server_->RegisterRoute("/queryInstance", &callbacks_["/queryInstance"]);

  OnRequestReceive enable_fls = std::bind(&SchedulerNode::ProcessEnableFLS, this, std::placeholders::_1);
  callbacks_["/enableFLS"] = enable_fls;
  (void)http_server_->RegisterRoute("/enableFLS", &callbacks_["/enableFLS"]);

  OnRequestReceive disable_fls = std::bind(&SchedulerNode::ProcessDisableFLS, this, std::placeholders::_1);
  callbacks_["/disableFLS"] = disable_fls;
  (void)http_server_->RegisterRoute("/disableFLS", &callbacks_["/disableFLS"]);

  OnRequestReceive scale_out_rollback = std::bind(&SchedulerNode::ProcessScaleoutRollback, this, std::placeholders::_1);
  callbacks_["/scaleoutRollback"] = scale_out_rollback;
  (void)http_server_->RegisterRoute("/scaleoutRollback", &callbacks_["/scaleoutRollback"]);

  if (!http_server_->InitServer()) {
    MS_LOG(EXCEPTION) << "The scheduler init http server failed.";
  }

  if (!http_server_->Start(false)) {
    MS_LOG(EXCEPTION) << "The scheduler start http server failed.";
  }
  restful_thread_ = std::make_unique<std::thread>([&]() { http_server_->Wait(); });
  MS_EXCEPTION_IF_NULL(restful_thread_);
}

void SchedulerNode::StopRestfulServer() {
  MS_LOG(INFO) << "Scheduler stop https server.";
  MS_ERROR_IF_NULL_WO_RET_VAL(http_server_);
  MS_ERROR_IF_NULL_WO_RET_VAL(restful_thread_);
  if (!http_server_->Stop()) {
    MS_LOG(WARNING) << "Scheduler stop https server failed.";
  }
  if (restful_thread_ != nullptr && restful_thread_->joinable()) {
    restful_thread_->join();
  }
}

void SchedulerNode::InitNodeMetaData() {
  ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
  clusterConfig.scheduler_host = PSContext::instance()->scheduler_host();
  clusterConfig.scheduler_port = PSContext::instance()->scheduler_port();
  clusterConfig.initial_worker_num = PSContext::instance()->initial_worker_num();
  clusterConfig.initial_server_num = PSContext::instance()->initial_server_num();
  MS_LOG(INFO) << "The cluster worker num:" << clusterConfig.initial_worker_num
               << ", the server num:" << clusterConfig.initial_server_num
               << ", the scheduler ip:" << clusterConfig.scheduler_host
               << ", the scheduler port:" << clusterConfig.scheduler_port;
}

bool SchedulerNode::RecoverScheduler() {
  MS_EXCEPTION_IF_NULL(config_);
  if (config_->Exists(kKeyRecovery)) {
    MS_LOG(INFO) << "The scheduler node is support recovery.";
    scheduler_recovery_ = std::make_unique<SchedulerRecovery>();
    MS_EXCEPTION_IF_NULL(scheduler_recovery_);
    (void)scheduler_recovery_->Initialize(config_->Get(kKeyRecovery, ""));
    (void)scheduler_recovery_->InitializeNodes(config_->Get(kKeyRecovery, ""));

    return scheduler_recovery_->Recover();
  }
  return false;
}

void SchedulerNode::PersistMetaData() {
  if (scheduler_recovery_ == nullptr) {
    MS_LOG(WARNING) << "scheduler recovery is null, so don't persist meta data";
    return;
  }
  if (config_->Exists(kKeyRecovery)) {
    ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
    clusterConfig.initial_worker_num = node_manager_.worker_num();
    clusterConfig.initial_server_num = node_manager_.server_num();
    clusterConfig.initial_total_node_num = node_manager_.total_node_num();
    clusterConfig.initial_next_worker_rank_id = node_manager_.next_worker_rank_id();
    clusterConfig.initial_next_server_rank_id = node_manager_.next_server_rank_id();
    clusterConfig.initial_registered_nodes_infos.clear();
    clusterConfig.initial_registered_nodes_infos = node_manager_.registered_nodes_info();

    scheduler_recovery_->Persist(clusterConfig);
    scheduler_recovery_->PersistNodesInfo(clusterConfig);
  }
}

bool SchedulerNode::CheckIfNodeDisconnected() const {
  return UintToInt(register_connection_fd_.size()) != node_manager_.total_node_num();
}

void SchedulerNode::BroadcastTimeoutEvent() {
  core::ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
  auto initial_node_infos = clusterConfig.initial_registered_nodes_infos;
  const uint32_t event = static_cast<uint32_t>(ps::UserDefineEvent::kNodeTimeout);
  MS_LOG(INFO) << "Broad timeout event:" << event;
  for (const auto &kvs : initial_node_infos) {
    auto client = GetOrCreateClient(kvs.second);
    SendEvent(client, event);
  }
  MS_LOG(INFO) << "Broad timeout event finish.";
}

void SchedulerNode::SetRegisterConnectionFd(const std::shared_ptr<TcpConnection> &conn, const std::string &node_id) {
  int fd = conn->GetFd();
  if (register_connection_fd_.count(fd) > 0) {
    MS_LOG(WARNING) << "This server has contained the fd:" << fd;
    return;
  }
  MS_LOG(INFO) << "register client fd:" << fd << ", register client id:" << node_id;
  register_connection_fd_[fd] = node_id;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
