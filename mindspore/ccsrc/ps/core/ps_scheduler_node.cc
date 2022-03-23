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

#include <memory>
#include "ps/core/ps_scheduler_node.h"

namespace mindspore {
namespace ps {
namespace core {
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
  MS_LOG(INFO) << "Received send host name request, node id: " << node_id << ", rank id: " << rank_id;

  bool ret = false;
  std::string error = "";
  if (rank_id >= worker_num_) {
    error = "The rank id: " + std::to_string(rank_id) + " should be less than: " + std::to_string(worker_num_);
    MS_LOG(ERROR) << error;
  } else {
    host_hash_names_[rank_id] = host_hash_name;
    (void)recv_rank_id_send_host_name_.insert(rank_id);
    ret = true;
  }

  GeneralResponse(server, conn, meta, ret, error);
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
  MS_LOG(INFO) << "Received query host name request, node id: " << node_id << ", rank id: " << rank_id;

  bool is_success = recv_rank_id_send_host_name_.size() == host_hash_names_.size();
  QueryHostHashNameRespMessage resp_msg;
  resp_msg.set_is_success(is_success);
  if (is_success) {
    *resp_msg.mutable_host_hash_names() = {host_hash_names_.begin(), host_hash_names_.end()};
  }
  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, resp_msg.SerializeAsString().data(),
                           resp_msg.ByteSizeLong())) {
    MS_LOG(ERROR) << "Scheduler failed to respond message.";
    return;
  }
  MS_LOG(INFO) << "Respond query host name request, node id: " << node_id << ", rank id: " << rank_id;

  if (is_success) {
    (void)recv_rank_id_query_host_name_.insert(rank_id);

    if (recv_rank_id_query_host_name_.size() == recv_rank_id_send_host_name_.size()) {
      recv_rank_id_send_host_name_.clear();
      recv_rank_id_query_host_name_.clear();
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
  MS_LOG(INFO) << "Received send unique id request, group name: " << group_name << ", node id: " << node_id
               << ", rank id: " << rank_id;

  bool ret = false;
  std::string error = "";
  if (rank_id != 0) {
    error = "The rank id: " + std::to_string(rank_id) + " of worker which sends unique id should be 0";
    MS_LOG(ERROR) << error;
  } else {
    unique_id_group_[group_name] = send_unique_id_msg.unique_id();
    ret = true;
  }

  GeneralResponse(server, conn, meta, ret, error);
  MS_LOG(INFO) << "Respond send unique id request, group name: " << group_name << ", node id: " << node_id
               << ", rank id: " << rank_id;
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
  uint32_t rank_id = query_msg.rank_id();
  std::string group_name = query_msg.group_name();
  MS_LOG(INFO) << "Received query unique id request, group name: " << group_name << ", node id: " << node_id
               << ", rank id: " << rank_id;

  auto iter = unique_id_group_.find(group_name);
  bool is_success = (iter != unique_id_group_.end());

  QueryUniqueIDRespMessage resp_msg;
  resp_msg.set_is_success(is_success);
  if (is_success) {
    resp_msg.set_unique_id(iter->second);
  }

  if (!server->SendMessage(conn, meta, Protos::PROTOBUF, resp_msg.SerializeAsString().data(),
                           resp_msg.ByteSizeLong())) {
    MS_LOG(ERROR) << "Scheduler failed to respond message.";
    return;
  }

  MS_LOG(INFO) << "Respond query unique id request, group name: " << group_name << ", node id: " << node_id
               << ", rank id: " << rank_id;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
