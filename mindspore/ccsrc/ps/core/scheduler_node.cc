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

#include <string>

#include "include/common/debug/common.h"
#include "ps/core/common.h"
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
  InitNodeMetaData();
  bool is_recover = false;
  if (!config_->Initialize()) {
    MS_LOG(WARNING) << "The config file is empty.";
  } else {
    InitEventTxtFile();
    is_recover = RecoverScheduler();
    if (!is_recover) {
      MS_LOG(DEBUG) << "Recover the server node is failed.";
    }
  }
  RecoverFromPersistence();

  Initialize();
  StartUpdateClusterStateTimer();
  RunRecovery();

  if (is_recover) {
    RecordSchedulerRestartInfo();
  }

  if (is_worker_timeout_) {
    BroadcastTimeoutEvent();
  }
  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start Scheduler node timeout!";
    return false;
  }

  StartUpdatePersistentCommandTimer();
  MS_LOG(INFO) << "[Scheduler start]: 4. Successfully start scheduler, there are " << node_manager_.worker_num()
               << " workers and " << node_manager_.server_num() << " servers registered.";

  InitializeActorRouteTableService();
  return true;
}

void SchedulerNode::RunRecovery() {
  core::ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
  // create tcp client to myself in case of event dispatch failed when Send reconnect msg to server failed
  client_to_scheduler_ =
    std::make_shared<TcpClient>(clusterConfig.scheduler_host, clusterConfig.scheduler_port, NodeRole::SCHEDULER);
  MS_EXCEPTION_IF_NULL(client_to_scheduler_);
  client_to_scheduler_->Init();
  client_thread_ = std::make_unique<std::thread>([this]() {
    MS_LOG(INFO) << "The node start a tcp client!";
    client_to_scheduler_->Start();
  });
  MS_EXCEPTION_IF_NULL(client_thread_);

  auto initial_node_infos = clusterConfig.initial_registered_nodes_infos;
  if (initial_node_infos.empty()) {
    MS_LOG(WARNING) << "There is no registered nodes in scheduler!";
    return;
  }
  MS_LOG(INFO) << "The scheduler start run recovery!"
               << " The worker num:" << clusterConfig.initial_worker_num
               << ", the server num:" << clusterConfig.initial_server_num
               << ", the scheduler ip:" << clusterConfig.scheduler_host
               << ", the scheduler port:" << clusterConfig.scheduler_port
               << ", the initial total node num:" << clusterConfig.initial_total_node_num
               << ", the initial next worker rank id:" << clusterConfig.initial_next_worker_rank_id
               << ", the initial next server rank id:" << clusterConfig.initial_next_server_rank_id
               << ", the initial cluster state:" << kClusterState.at(clusterConfig.initial_cluster_state);

  if (!clusterConfig.initial_registered_nodes_infos.empty()) {
    for (const auto kvs : clusterConfig.initial_registered_nodes_infos) {
      MS_LOG(INFO) << "The ip:" << kvs.second.ip_ << ", the port:" << kvs.second.port_
                   << ", the node_id:" << kvs.second.node_id_
                   << ", the node_role:" << CommUtil::NodeRoleToString(kvs.second.node_role_)
                   << ", the rank_id_:" << kvs.second.rank_id_
                   << ", the is_alive:" << CommUtil::BoolToString(kvs.second.is_alive);
    }
  }

  uint32_t worker_num = clusterConfig.initial_worker_num;
  uint32_t server_num = clusterConfig.initial_server_num;

  node_manager_.set_worker_num(worker_num);
  node_manager_.set_server_num(server_num);
  node_manager_.set_next_worker_rank_id(clusterConfig.initial_next_worker_rank_id);
  node_manager_.set_next_server_rank_id(clusterConfig.initial_next_server_rank_id);
  node_manager_.set_total_node_num(clusterConfig.initial_total_node_num);
  if (clusterConfig.initial_cluster_state == ClusterState::CLUSTER_DISABLE_FLS) {
    MS_LOG(WARNING) << "Scheduler recover and update cluster state from recovery file, cluster state is "
                    << CommUtil::ClusterStateToString(clusterConfig.initial_cluster_state);
    node_manager_.UpdateClusterState(clusterConfig.initial_cluster_state);
  }

  for (const auto &kvs : initial_node_infos) {
    auto &node_id = kvs.first;
    auto &node_info = kvs.second;
    auto client = std::make_shared<TcpClient>(node_info.ip_, node_info.port_, node_info.node_role_);
    client->SetMessageCallback([this](const std::shared_ptr<MessageMeta> &meta, const Protos &, const void *, size_t) {
      MS_LOG(INFO) << "received the response. ";
      NotifyMessageArrival(meta);
    });
    client->Init();
    MS_EXCEPTION_IF_NULL(client);

    auto message_meta = std::make_shared<MessageMeta>();
    MS_EXCEPTION_IF_NULL(message_meta);
    message_meta->set_cmd(NodeCommand::SCHEDULER_RECOVERY);

    auto rank_id = node_info.rank_id_;
    SendMetadataMessage scheduler_recovery_message;
    scheduler_recovery_message.set_worker_num(worker_num);
    scheduler_recovery_message.set_server_num(server_num);
    scheduler_recovery_message.set_rank_id(rank_id);
    if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, scheduler_recovery_message.SerializeAsString().data(),
                         scheduler_recovery_message.ByteSizeLong())) {
      if (node_info.node_role_ == NodeRole::WORKER && PSContext::instance()->server_mode() == kServerModePS) {
        is_worker_timeout_ = true;
        break;
      }
      MS_LOG(WARNING) << "Scheduler send recovery msg to " << node_id << " timeout!";
    } else {
      MS_LOG(INFO) << "Scheduler send recovery msg to " << node_id << " successful.";
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

  HeartbeatRespMessage heartbeat_resp_message;
  heartbeat_resp_message.set_persistent_cmd(PersistentCommand::DEFAULT);

  NodeInfo nodeInfo = node_manager_.QueryNodeInfo(node_id);
  if (!nodeInfo.node_id_.empty()) {
    // The worker role does not support disaster recovery for the time being.
    NodeRole node_role = nodeInfo.node_role_;
    if (node_role == NodeRole::SERVER && persistent_cmd_ == PersistentCommand::BEGIN_PERSIST) {
      if (!node_manager_.IsNodePersisting(node_id)) {
        heartbeat_resp_message.set_persistent_cmd(PersistentCommand::BEGIN_PERSIST);
        node_manager_.AddPersistingNode(node_id);
      }
      if (node_manager_.IsAllNodeInPersisting()) {
        persistent_cmd_ = PersistentCommand::DEFAULT;
      }
    }
    node_manager_.UpdateHeartbeat(node_id);
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

  if (node_manager_.IsAllNodesRegistered()) {
    return;
  }

  uint32_t rank_id = UINT32_MAX;
  // Re-Add the missing node into node manager.
  if (heartbeat_message.has_address() &&
      node_manager_.ReAddNodeIfNotExists(node_id, heartbeat_message.ip(), heartbeat_message.port(), &rank_id)) {
    SetRegisterConnectionFd(conn, node_id);
    HandleNodeRecoverByHeartBeat(rank_id);

    if (node_manager_.IsAllNodesRegistered()) {
      is_ready_ = true;
      MS_LOG(INFO) << "There are " << node_manager_.worker_num() << " workers and " << node_manager_.server_num()
                   << " servers registered to scheduer, so the scheduler send meta data to worker/server.";
      node_manager_.UpdateNodesInfo();

      auto node_infos = node_manager_.nodes_info();
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
  MS_LOG(INFO) << "[Scheduler start]: 2. The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id is:" << node_info_.node_id_ << " create a tcp server.";
}

void SchedulerNode::InitCommandHandler() {
  handlers_[NodeCommand::HEARTBEAT] = &SchedulerNode::ProcessHeartbeat;
  handlers_[NodeCommand::REGISTER] = &SchedulerNode::ProcessRegister;
  handlers_[NodeCommand::FINISH] = &SchedulerNode::ProcessFinish;
  handlers_[NodeCommand::FETCH_METADATA] = &SchedulerNode::ProcessFetchMetadata;
  handlers_[NodeCommand::SEND_EVENT] = &SchedulerNode::ProcessSendEvent;
  handlers_[NodeCommand::FAILURE_EVENT_INFO] = &SchedulerNode::ProcessFailureEvent;
  RegisterActorRouteTableServiceHandler();
  RegisterInitCollectCommServiceHandler();
  RegisterRecoveryServiceHandler();
}

void SchedulerNode::RegisterActorRouteTableServiceHandler() {
  handlers_[NodeCommand::REGISTER_ACTOR_ROUTE] = &SchedulerNode::ProcessRegisterActorRoute;
  handlers_[NodeCommand::DELETE_ACTOR_ROUTE] = &SchedulerNode::ProcessDeleteActorRoute;
  handlers_[NodeCommand::LOOKUP_ACTOR_ROUTE] = &SchedulerNode::ProcessLookupActorRoute;
}

void SchedulerNode::InitEventTxtFile() {
  MS_LOG(DEBUG) << "Start init event txt";
  ps::core::FileConfig event_file_config;
  if (!ps::core::CommUtil::ParseAndCheckConfigJson(config_.get(), kFailureEvent, &event_file_config)) {
    MS_LOG(EXCEPTION) << "Parse and checkout config json failed";
    return;
  }

  std::lock_guard<std::mutex> lock(event_txt_file_mtx_);
  event_file_path_ = event_file_config.storage_file_path;
  auto realpath = Common::CreatePrefixPath(event_file_path_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(EXCEPTION) << "Creating path for " << event_file_path_ << " failed.";
    return;
  }
  event_txt_file_.open(realpath.value(), std::ios::out | std::ios::app);
  event_txt_file_.close();
  MS_LOG(DEBUG) << "Init event txt success!";
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

  const auto client_disconn = [this](const TcpServer &, const TcpConnection &conn) {
    int fd = conn.GetFd();
    if (register_connection_fd_.count(fd) > 0) {
      MS_LOG(WARNING) << "remove client fd:" << fd << ", remove client id:" << register_connection_fd_[fd];
      (void)register_connection_fd_.erase(fd);
      MS_LOG(INFO) << "Register node number is:" << register_connection_fd_.size()
                   << ", total node num is:" << node_manager_.total_node_num()
                   << ", scale in node size is: " << scale_in_node_ids_.size();
    }
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
    NodeInfo node_info = node_manager_.QueryNodeInfo(node_id);
    if (NeedRejectRegister(node_info)) {
      MS_LOG(WARNING) << "The node(id: " << node_id << ") is alive, register is rejected!";
      RegisterRespMessage register_rejected_message;
      register_rejected_message.set_node_id(node_id);
      register_rejected_message.set_rank_id(UINT32_MAX);
      if (!server->SendMessage(conn, meta, Protos::PROTOBUF, register_rejected_message.SerializeAsString().data(),
                               register_rejected_message.ByteSizeLong())) {
        MS_LOG(WARNING) << "Server response rejected message failed.";
      }
      client_mutex_.unlock();
      return;
    }

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

  auto node_infos = node_manager_.nodes_info();
  if (node_manager_.GetClusterState() == ClusterState::CLUSTER_SCALE_OUT && register_message.is_recover() &&
      node_infos.count(node_id) > 0) {
    ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
    // set the last worker num and last server num and start cluster scale out rollback
    node_manager_.set_worker_num(clusterConfig.initial_worker_num);
    node_manager_.set_server_num(clusterConfig.initial_server_num);
    node_manager_.set_total_node_num(clusterConfig.initial_total_node_num);

    node_manager_.UpdateClusterState(ClusterState::CLUSTER_SCALE_OUT_ROLLBACK);
    MS_LOG(INFO) << "Trigger scale out rollback because the node is recovery, the last worker num:"
                 << clusterConfig.initial_worker_num << ", the last server num:" << clusterConfig.initial_server_num;
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
    if (!node_manager_.IsAllNodesAlive()) {
      MS_LOG(ERROR)
        << "Do not broadcast nodes info because some server nodes are not alive, and cluster will exit later.";
      return;
    }
    if (!BuildingNetwork()) {
      MS_LOG(ERROR) << "Building network failed! Cluster will exit later.";
    }
  }
}

bool SchedulerNode::BuildingNetwork() {
  if (node_manager_.GetClusterState() == ClusterState::CLUSTER_SCALE_IN) {
    auto nodes = node_manager_.nodes_info();
    for (const auto &id : scale_in_node_ids_) {
      MS_LOG(INFO) << "The scheduler send metadata to scale in node:" << id;
      if (nodes.count(id)) {
        auto scale_in_client = GetOrCreateClient(nodes[id]);
        SendMetadata(scale_in_client, nodes[id].rank_id_);
        node_manager_.UpdateHeartbeat(id);
      }
      if (connected_nodes_.count(id)) {
        MS_LOG(INFO) << "remove scale in node id: " << id << " connection.";
        connected_nodes_.erase(id);
      }
    }
  }
  node_manager_.UpdateNodesInfo();
  auto node_infos = node_manager_.nodes_info();
  bool res = SendPrepareBuildingNetwork(node_infos);
  if (!res) {
    MS_LOG(ERROR) << "Prepare for building network failed!";
    return false;
  }
  is_ready_ = true;
  MS_LOG(INFO) << "Prepare for building network success. There are " << node_manager_.worker_num() << " workers and "
               << node_manager_.server_num()
               << " servers registered to scheduer, so the scheduler send meta data to worker/server.";

  for (const auto &kvs : node_infos) {
    auto client = GetOrCreateClient(kvs.second);
    MS_EXCEPTION_IF_NULL(client);
    SendMetadata(client, kvs.second.rank_id_);
    node_manager_.UpdateHeartbeat(kvs.first);
  }

  if (node_manager_.GetClusterState() == ClusterState::CLUSTER_DISABLE_FLS) {
    MS_LOG(WARNING)
      << "Cluster state is CLUSTER_DISABLE_FLS, do not need to change to CLUSTER_READY when building network.";
  } else {
    node_manager_.UpdateClusterState(ClusterState::CLUSTER_READY);
  }
  PersistMetaData();
  wait_start_cond_.notify_all();
  return true;
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

void SchedulerNode::ProcessRegisterActorRoute(const std::shared_ptr<TcpServer> &server,
                                              const std::shared_ptr<TcpConnection> &conn,
                                              const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);
  MS_ERROR_IF_NULL_WO_RET_VAL(actor_route_table_service_);
  ActorAddress actor_address;
  actor_address.ParseFromArray(data, SizeToInt(size));
  std::string actor_id = actor_address.actor_id();

  std::string error = "";
  bool ret = actor_route_table_service_->RegisterRoute(actor_id, actor_address, &error);
  GeneralResponse(server, conn, meta, ret, error);
}

void SchedulerNode::ProcessDeleteActorRoute(const std::shared_ptr<TcpServer> &server,
                                            const std::shared_ptr<TcpConnection> &conn,
                                            const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);
  MS_ERROR_IF_NULL_WO_RET_VAL(actor_route_table_service_);
  std::string actor_id(static_cast<const char *>(data), size);

  std::string error = "";
  bool ret = actor_route_table_service_->DeleteRoute(actor_id, &error);
  GeneralResponse(server, conn, meta, ret, error);
}

void SchedulerNode::ProcessLookupActorRoute(const std::shared_ptr<TcpServer> &server,
                                            const std::shared_ptr<TcpConnection> &conn,
                                            const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);
  MS_ERROR_IF_NULL_WO_RET_VAL(actor_route_table_service_);
  std::string actor_id(static_cast<const char *>(data), size);

  std::string error = "";
  ActorAddress address = actor_route_table_service_->LookupRoute(actor_id, &error);
  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, address.SerializeAsString().data(), address.ByteSizeLong())) {
    MS_LOG(ERROR) << "Scheduler failed to respond message for lookup route.";
  }
}

void SchedulerNode::ProcessFailureEvent(const std::shared_ptr<TcpServer> &server,
                                        const std::shared_ptr<TcpConnection> &conn,
                                        const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);
  FailureEventMessage failure_event_message;
  failure_event_message.ParseFromArray(data, SizeToInt(size));
  std::string node_role = failure_event_message.node_role();
  std::string ip = failure_event_message.ip();
  uint32_t port = failure_event_message.port();
  std::string time = failure_event_message.time();
  std::string event = failure_event_message.event();
  std::lock_guard<std::mutex> lock(event_txt_file_mtx_);
  event_txt_file_.open(event_file_path_, std::ios::out | std::ios::app);
  if (!event_txt_file_.is_open()) {
    MS_LOG(EXCEPTION) << "The event txt file is not open";
    return;
  }
  std::string event_info = "nodeRole:" + node_role + "," + ip + ":" + std::to_string(port) + "," +
                           "currentTime:" + time + "," + "event:" + event + ";";
  event_txt_file_ << event_info << "\n";
  (void)event_txt_file_.flush();
  event_txt_file_.close();
  MS_LOG(INFO) << "Process failure event success!";
}

bool SchedulerNode::SendPrepareBuildingNetwork(const std::unordered_map<std::string, NodeInfo> &node_infos) {
  uint64_t request_id = AddMessageTrack(node_infos.size());
  for (const auto &kvs : node_infos) {
    auto client = GetOrCreateClient(kvs.second);
    MS_ERROR_IF_NULL_W_RET_VAL(client, false);
    auto message_meta = std::make_shared<MessageMeta>();
    MS_EXCEPTION_IF_NULL(message_meta);
    message_meta->set_request_id(request_id);
    message_meta->set_cmd(NodeCommand::PREPARE_BUILDING_NETWORK);

    std::string req_data;
    if (!client->SendMessage(message_meta, Protos::RAW, req_data.data(), req_data.length())) {
      MS_LOG(ERROR) << "The node role:" << CommUtil::NodeRoleToString(kvs.second.node_role_)
                    << " the node id:" << kvs.second.node_id_ << " prepare building network timeout!";
    }
  }
  return Wait(request_id);
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

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " send metadata to workers and servers!";
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

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " send event resp to workers and servers!";
}

void SchedulerNode::StartUpdateClusterStateTimer() {
  MS_LOG(INFO) << "[Scheduler start]: 3. The scheduler start a heartbeat timer!";
  node_manager_.setPersistCallback([this]() { PersistMetaData(); });
  update_state_thread_ = std::make_unique<std::thread>([&]() {
    auto start_time = std::chrono::steady_clock::now();
    while (!is_finish_.load()) {
      // 1. update cluster timeout
      if (!is_ready_ && (std::chrono::steady_clock::now() - start_time >
                         std::chrono::seconds(PSContext::instance()->cluster_config().cluster_available_timeout))) {
        node_manager_.UpdateClusterState(ClusterState::CLUSTER_EXIT);
      }
      std::this_thread::sleep_for(std::chrono::seconds(PSContext::instance()->cluster_config().heartbeat_interval));
      node_manager_.UpdateCluster(is_ready_);
      HandleNodeTimeoutForRecovery(node_manager_.QueryTimeOutNodesInfo());

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

void SchedulerNode::InitializeActorRouteTableService() {
  actor_route_table_service_ = std::make_unique<ActorRouteTableService>();
  MS_EXCEPTION_IF_NULL(actor_route_table_service_);
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
    MS_LOG(INFO) << "ip:" << ip << ", port:" << port << ", node id:" << node_info.node_id_;
    auto client = std::make_shared<TcpClient>(ip, port, node_info.node_role_);
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

RequestProcessResult SchedulerNode::CheckIfClusterReady() {
  RequestProcessResult result(RequestProcessResultCode::kSuccess);
  if (node_manager_.GetClusterState() != ClusterState::CLUSTER_READY) {
    std::string message = "The cluster is not ready.";
    ERROR_STATUS(result, RequestProcessResultCode::kSystemError, message);
    return result;
  }

  if (CheckIfNodeDisconnected()) {
    ERROR_STATUS(result, RequestProcessResultCode::kSystemError, kClusterNotReady);
    return result;
  }
  return result;
}

RequestProcessResult SchedulerNode::CheckIfNodeIdLegal(const std::vector<std::string> &node_ids) {
  RequestProcessResult result(RequestProcessResultCode::kSuccess);
  if (node_ids.empty()) {
    std::string message = "The node ids should not be empty.";
    ERROR_STATUS(result, RequestProcessResultCode::kInvalidInputs, message);
    return result;
  }

  auto node_infos = node_manager_.nodes_info();

  for (const auto &val : node_ids) {
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
    scheduler_recovery_ = std::make_shared<SchedulerRecovery>();
    MS_EXCEPTION_IF_NULL(scheduler_recovery_);
    bool ret = scheduler_recovery_->Initialize(config_->Get(kKeyRecovery, ""));
    bool ret_node = scheduler_recovery_->InitializeNodes(config_->Get(kKeyRecovery, ""));
    if (ret && ret_node) {
      MS_LOG(INFO) << "Scheduler recovery initialize successful.";
      return scheduler_recovery_->Recover();
    }
  }
  return false;
}

void SchedulerNode::RecordSchedulerRestartInfo() {
  MS_LOG(DEBUG) << "Start to record scheduler restart error message";
  std::lock_guard<std::mutex> lock(event_txt_file_mtx_);
  event_txt_file_.open(event_file_path_, std::ios::out | std::ios::app);
  if (!event_txt_file_.is_open()) {
    MS_LOG(EXCEPTION) << "The event txt file is not open";
    return;
  }
  std::string node_role = CommUtil::NodeRoleToString(node_info_.node_role_);
  auto scheduler_recovery_ptr = std::dynamic_pointer_cast<SchedulerRecovery>(scheduler_recovery_);
  MS_EXCEPTION_IF_NULL(scheduler_recovery_ptr);
  auto ip = scheduler_recovery_ptr->GetMetadata(kRecoverySchedulerIp);
  auto port = scheduler_recovery_ptr->GetMetadata(kRecoverySchedulerPort);
  std::string time = ps::core::CommUtil::GetNowTime().time_str_mill;
  std::string event = "Node restart";
  std::string event_info =
    "nodeRole:" + node_role + "," + ip + ":" + port + "," + "currentTime:" + time + "," + "event:" + event + ";";
  event_txt_file_ << event_info << "\n";
  (void)event_txt_file_.flush();
  event_txt_file_.close();
  MS_LOG(DEBUG) << "Record scheduler node restart info " << event_info << " success!";
}

void SchedulerNode::PersistMetaData() {
  if (scheduler_recovery_ == nullptr) {
    MS_LOG(WARNING) << "scheduler recovery is null, do not persist meta data";
    return;
  }
  if (!is_ready_) {
    MS_LOG(WARNING) << "Cluster is not building network successful, do not persist meta data";
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
    clusterConfig.initial_cluster_state = node_manager_.GetClusterState();

    scheduler_recovery_->Persist(clusterConfig);
    scheduler_recovery_->PersistNodesInfo(clusterConfig);
  }
}

bool SchedulerNode::CheckIfNodeDisconnected() const {
  return UlongToUint(register_connection_fd_.size()) != node_manager_.total_node_num();
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

void SchedulerNode::GeneralResponse(const std::shared_ptr<TcpServer> &server,
                                    const std::shared_ptr<TcpConnection> &conn,
                                    const std::shared_ptr<MessageMeta> &meta, bool is_success,
                                    const std::string &error) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);

  GeneralResponseMsg general_response_message;
  general_response_message.set_is_success(is_success);
  general_response_message.set_error(error);

  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, general_response_message.SerializeAsString().data(),
                           general_response_message.ByteSizeLong())) {
    MS_LOG(ERROR) << "Scheduler failed to respond message.";
    return;
  }
  return;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
