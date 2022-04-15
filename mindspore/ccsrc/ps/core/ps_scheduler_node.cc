/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <memory>
#include "ps/core/ps_scheduler_node.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ps {
namespace core {
constexpr char kActorSetNames[] = "actor_set_names";
constexpr char kRecoveryStorage[] = "scheduler_persistent.json";

void PSSchedulerNode::RunRecovery() {
  const auto &clusterConfig = PSContext::instance()->cluster_config();
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

  const auto &initial_node_infos = clusterConfig.initial_registered_nodes_infos;
  if (initial_node_infos.empty()) {
    MS_LOG(WARNING) << "There is no registered nodes in scheduler!";
    return;
  }
  MS_LOG(INFO) << "The scheduler start run recovery!";
  uint32_t worker_num = clusterConfig.initial_worker_num;
  uint32_t server_num = clusterConfig.initial_server_num;

  node_manager_.set_worker_num(worker_num);
  node_manager_.set_server_num(server_num);
  node_manager_.set_next_worker_rank_id(clusterConfig.initial_next_worker_rank_id);
  node_manager_.set_next_server_rank_id(clusterConfig.initial_next_server_rank_id);
  node_manager_.set_total_node_num(clusterConfig.initial_total_node_num);

  MS_LOG(INFO) << "Scheduler recovery finish.";
}

void PSSchedulerNode::RegisterInitCollectCommServiceHandler() {
  handlers_[NodeCommand::SEND_HOST_NAME] = static_cast<ResponseHandler>(&PSSchedulerNode::ProcessSendHostName);
  handlers_[NodeCommand::QUERY_HOST_NAMES] = static_cast<ResponseHandler>(&PSSchedulerNode::ProcessQueryHostNames);
  handlers_[NodeCommand::SEND_UNIQUE_ID] = static_cast<ResponseHandler>(&PSSchedulerNode::ProcessSendUniqueID);
  handlers_[NodeCommand::QUERY_UNIQUE_ID] = static_cast<ResponseHandler>(&PSSchedulerNode::ProcessQueryUniqueID);
}

void PSSchedulerNode::RegisterRecoveryServiceHandler() {
  handlers_[NodeCommand::SEND_FINISH_TRANSFORM] =
    static_cast<ResponseHandler>(&PSSchedulerNode::ProcessSendFinishTransform);
  handlers_[NodeCommand::QUERY_FINISH_TRANSFORM] =
    static_cast<ResponseHandler>(&PSSchedulerNode::ProcessQueryFinishTransform);
}

void PSSchedulerNode::ProcessSendHostName(const std::shared_ptr<TcpServer> &server,
                                          const std::shared_ptr<TcpConnection> &conn,
                                          const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);

  SendHostHashNameMessage send_host_name_msg;
  send_host_name_msg.ParseFromArray(data, SizeToInt(size));
  std::string node_id = send_host_name_msg.node_id();
  uint32_t rank_id = send_host_name_msg.rank_id();
  size_t host_hash_name = send_host_name_msg.host_hash_name();
  MS_LOG(INFO) << "Receive send host name request, node id: " << node_id << ", rank id: " << rank_id;

  bool ret = true;
  std::string error_message = "";
  NodeInfo node_info = node_manager_.QueryNodeInfo(node_id);
  if (node_info.node_id_.empty()) {
    ret = false;
    error_message = "The node info is empty";
  }
  auto node_role = node_info.node_role_;

  if (!ret) {
    MS_LOG(ERROR) << error_message;
  } else if (rank_id >= node_nums_[node_role]) {
    error_message =
      "The rank id: " + std::to_string(rank_id) + " should be less than: " + std::to_string(node_nums_[node_role]);
    MS_LOG(ERROR) << error_message;
    ret = false;
  } else {
    if (host_hash_names_.find(node_role) == host_hash_names_.end()) {
      host_hash_names_[node_role].resize(node_nums_[node_role]);
    }
    host_hash_names_[node_role][rank_id] = host_hash_name;
    (void)recv_rank_ids_send_host_name_[node_role].insert(rank_id);
    ret = true;
  }

  GeneralResponse(server, conn, meta, ret, error_message);
  MS_LOG(INFO) << "Respond send host name request, node id: " << node_id << ", rank id: " << rank_id;
}

void PSSchedulerNode::ProcessQueryHostNames(const std::shared_ptr<TcpServer> &server,
                                            const std::shared_ptr<TcpConnection> &conn,
                                            const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);

  GeneralQueryMessage query_msg;
  query_msg.ParseFromArray(data, SizeToInt(size));
  std::string node_id = query_msg.node_id();
  uint32_t rank_id = query_msg.rank_id();
  MS_LOG(INFO) << "Receive query host name request, node id: " << node_id << ", rank id: " << rank_id;

  NodeInfo node_info = node_manager_.QueryNodeInfo(node_id);
  if (node_info.node_id_.empty()) {
    MS_LOG(ERROR) << "The node info is empty";
    return;
  }

  NodeRole node_role = node_info.node_role_;
  auto iter = host_hash_names_.find(node_role);
  bool is_success =
    (iter != host_hash_names_.end()) && (recv_rank_ids_send_host_name_[node_role].size() == node_nums_[node_role]);
  QueryHostHashNameRespMessage resp_msg;
  resp_msg.set_is_success(is_success);
  if (is_success) {
    *resp_msg.mutable_host_hash_names() = {iter->second.begin(), iter->second.end()};
  }
  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, resp_msg.SerializeAsString().data(),
                           resp_msg.ByteSizeLong())) {
    MS_LOG(ERROR) << "Scheduler failed to respond message.";
    return;
  }
  MS_LOG(INFO) << "Respond query host name request, node id: " << node_id << ", rank id: " << rank_id;

  if (is_success) {
    (void)recv_rank_ids_query_host_name_[node_role].insert(rank_id);

    if (recv_rank_ids_query_host_name_[node_role].size() == recv_rank_ids_send_host_name_[node_role].size()) {
      recv_rank_ids_send_host_name_[node_role].clear();
      recv_rank_ids_query_host_name_[node_role].clear();
      node_timeout_ = false;
    }
  }
}

void PSSchedulerNode::ProcessSendUniqueID(const std::shared_ptr<TcpServer> &server,
                                          const std::shared_ptr<TcpConnection> &conn,
                                          const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);

  SendUniqueIDMessage send_unique_id_msg;
  send_unique_id_msg.ParseFromArray(data, SizeToInt(size));
  std::string node_id = send_unique_id_msg.node_id();
  uint32_t rank_id = send_unique_id_msg.rank_id();
  std::string group_name = send_unique_id_msg.group_name();
  MS_LOG(INFO) << "Receive send unique id request, group name: " << group_name << ", node id: " << node_id
               << ", group rank id: " << rank_id;

  bool ret = true;
  std::string error_message = "";
  NodeInfo node_info = node_manager_.QueryNodeInfo(node_id);
  if (node_info.node_id_.empty()) {
    ret = false;
    error_message = "The node info is empty";
  }

  if (!ret) {
    MS_LOG(ERROR) << error_message;
  } else {
    unique_id_groups_[node_info.node_role_][group_name] = send_unique_id_msg.unique_id();
  }

  GeneralResponse(server, conn, meta, ret, error_message);
  MS_LOG(INFO) << "Respond send unique id request, group name: " << group_name << ", node id: " << node_id
               << ", group rank id: " << rank_id;
}

void PSSchedulerNode::ProcessQueryUniqueID(const std::shared_ptr<TcpServer> &server,
                                           const std::shared_ptr<TcpConnection> &conn,
                                           const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);

  QueryUniqueIDMessage query_msg;
  query_msg.ParseFromArray(data, SizeToInt(size));
  std::string node_id = query_msg.node_id();
  std::string group_name = query_msg.group_name();
  MS_LOG(INFO) << "Receive query unique id request, group name: " << group_name << ", node id: " << node_id;

  NodeInfo node_info = node_manager_.QueryNodeInfo(node_id);
  if (node_info.node_id_.empty()) {
    MS_LOG(ERROR) << "The node info is empty";
    return;
  }

  bool is_success = false;
  std::string unique_id;
  auto node_role = node_info.node_role_;
  auto role_iter = unique_id_groups_.find(node_role);
  if (role_iter != unique_id_groups_.end()) {
    const auto &unique_id_groups = role_iter->second;
    auto group_iter = unique_id_groups.find(group_name);
    if (group_iter != unique_id_groups.end()) {
      is_success = true;
      unique_id = group_iter->second;
    }
  }

  QueryUniqueIDRespMessage resp_msg;
  resp_msg.set_is_success(is_success);
  if (is_success) {
    resp_msg.set_unique_id(unique_id);
  }

  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, resp_msg.SerializeAsString().data(),
                           resp_msg.ByteSizeLong())) {
    MS_LOG(ERROR) << "Scheduler failed to respond message.";
    return;
  }

  MS_LOG(INFO) << "Respond query unique id request, group name: " << group_name << ", node id: " << node_id;
}

void PSSchedulerNode::ProcessSendFinishTransform(const std::shared_ptr<TcpServer> &server,
                                                 const std::shared_ptr<TcpConnection> &conn,
                                                 const std::shared_ptr<MessageMeta> &meta, const void *data,
                                                 size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);

  SendFinishTransformMessage send_finish_transform_msg;
  send_finish_transform_msg.ParseFromArray(data, SizeToInt(size));
  std::string node_id = send_finish_transform_msg.node_id();
  uint32_t rank_id = send_finish_transform_msg.rank_id();
  std::string actor_set_name = send_finish_transform_msg.actor_set_name();
  MS_LOG(INFO) << "Receive send finish transform request, node id: " << node_id << ", rank id: " << rank_id;
  bool is_ready = send_finish_transform_msg.is_ready();
  if (is_ready) {
    std::unique_lock<std::mutex> lock(nodes_finish_trans_mutex_);
    if (nodes_finish_trans_.count(actor_set_name) == 0) {
      MS_ERROR_IF_NULL_WO_RET_VAL(recovery_storage_);
      std::vector<std::string> actor_set_names;
      if (recovery_storage_->Exists(kActorSetNames)) {
        actor_set_names = recovery_storage_->GetValue<std::vector<std::string>>(kActorSetNames);
      }
      actor_set_names.push_back(actor_set_name);
      recovery_storage_->PutValue(kActorSetNames, actor_set_names);
    }

    (void)nodes_finish_trans_[actor_set_name].insert(rank_id);
  }

  GeneralResponse(server, conn, meta, true, "");
  MS_LOG(INFO) << "Respond send finish transform request, node id: " << node_id << ", rank id: " << rank_id;
}

void PSSchedulerNode::ProcessQueryFinishTransform(const std::shared_ptr<TcpServer> &server,
                                                  const std::shared_ptr<TcpConnection> &conn,
                                                  const std::shared_ptr<MessageMeta> &meta, const void *data,
                                                  size_t size) {
  MS_ERROR_IF_NULL_WO_RET_VAL(server);
  MS_ERROR_IF_NULL_WO_RET_VAL(conn);
  MS_ERROR_IF_NULL_WO_RET_VAL(meta);
  MS_ERROR_IF_NULL_WO_RET_VAL(data);

  QueryFinishTransformMessage query_msg;
  query_msg.ParseFromArray(data, SizeToInt(size));
  std::string node_id = query_msg.node_id();
  uint32_t rank_id = query_msg.rank_id();
  std::string actor_set_name = query_msg.actor_set_name();
  MS_LOG(INFO) << "Receive query finish transform request, node id: " << node_id << ", rank id: " << rank_id;

  NodeInfo node_info = node_manager_.QueryNodeInfo(node_id);
  if (node_info.node_id_.empty()) {
    MS_LOG(ERROR) << "The node info is empty";
    return;
  }
  auto node_role = node_info.node_role_;
  std::unique_lock<std::mutex> lock(nodes_finish_trans_mutex_);
  bool is_ready = nodes_finish_trans_[actor_set_name].size() == node_nums_[node_role];

  QueryFinishTransformRespMessage resp_msg;
  resp_msg.set_is_ready(is_ready);

  if (node_timeout_) {
    (void)resp_msg.set_is_worker_timeout(true);
  } else {
    resp_msg.set_is_worker_timeout(false);
  }

  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, resp_msg.SerializeAsString().data(),
                           resp_msg.ByteSizeLong())) {
    MS_LOG(ERROR) << "Scheduler failed to respond message.";
    return;
  }
  MS_LOG(INFO) << "Respond query finish transform request, node id: " << node_id << ", rank id: " << rank_id;
}

void PSSchedulerNode::HandleNodeTimeoutForRecovery(
  const std::unordered_map<std::string, NodeInfo> &timeout_nodes_infos) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_RECOVERY)) {
    return;
  }

  if (timeout_nodes_infos.empty()) {
    return;
  }

  std::unique_lock<std::mutex> lock(nodes_finish_trans_mutex_);
  node_timeout_ = true;
  for (const auto &item : timeout_nodes_infos) {
    for (auto &node_item : nodes_finish_trans_) {
      (void)node_item.second.erase(item.second.rank_id_);
    }
  }
}

void PSSchedulerNode::HandleNodeRecoverByHeartBeat(uint32_t rank_id) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_RECOVERY)) {
    return;
  }

  std::unique_lock<std::mutex> lock(nodes_finish_trans_mutex_);
  for (auto &node_item : nodes_finish_trans_) {
    (void)node_item.second.insert(rank_id);
  }
}

void PSSchedulerNode::RecoverFromPersistence() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_RECOVERY)) {
    return;
  }

  if (recovery_storage_ == nullptr) {
    if (!config_->Exists(kKeyRecovery)) {
      MS_LOG(EXCEPTION) << "The " << kKeyRecovery << " is not existed.";
    }

    nlohmann::json recovery_json;
    try {
      recovery_json = nlohmann::json::parse(config_->Get(kKeyRecovery, ""));
    } catch (nlohmann::json::exception &e) {
      MS_LOG(EXCEPTION) << "Parse the json failed.";
    }

    if (!recovery_json.contains(kStoreFilePath)) {
      MS_LOG(EXCEPTION) << "The " << kStoreFilePath << " is not existed.";
    }
    std::string storage_file_path = recovery_json.at(kStoreFilePath);
    std::string storage_file_dir = storage_file_path.substr(0, storage_file_path.rfind('/') + 1);

    recovery_storage_ = std::make_unique<FileConfiguration>(storage_file_dir + kRecoveryStorage);
    (void)recovery_storage_->Initialize();
  }

  if (recovery_storage_->Exists(kActorSetNames)) {
    std::vector<std::string> actor_set_names = recovery_storage_->GetValue<std::vector<std::string>>(kActorSetNames);
    std::unique_lock<std::mutex> lock(nodes_finish_trans_mutex_);
    (void)std::for_each(actor_set_names.begin(), actor_set_names.end(), [this](const std::string &name) {
      if (nodes_finish_trans_.count(name) == 0) {
        nodes_finish_trans_.emplace(name, std::set<uint32_t>());
      }
    });
  }
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
