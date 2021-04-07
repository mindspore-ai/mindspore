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

#include "ps/core/node_manager.h"

namespace mindspore {
namespace ps {
namespace core {
void NodeManager::InitNodeNum() {
  total_node_num_ = ClusterMetadata::instance()->total_server_num() + ClusterMetadata::instance()->total_worker_num();
}

int NodeManager::NextRankId(const RegisterMessage &register_message) {
  std::lock_guard<std::mutex> lock(assign_rank_id_mutex_);
  int rank_id = -1;

  const std::string &node_id = register_message.node_id();
  if (nodes_info_.find(node_id) != nodes_info_.end()) {
    rank_id = nodes_info_[node_id].rank_id_;
    MS_LOG(INFO) << "The node id: " << node_id << " is already assigned!";
    return rank_id;
  }

  if (register_message.role() == NodeRole::SERVER) {
    const std::string &ip = register_message.ip();
    uint32_t port = register_message.port();

    rank_id = ++next_server_rank_id_;
    if (IntToUint(rank_id) >= ClusterMetadata::instance()->total_server_num()) {
      MS_LOG(WARNING) << "The rank id is greater than the number of servers.";
      rank_id = -1;
      --next_server_rank_id_;
    }
    NodeInfo node_info;
    node_info.node_role_ = NodeRole::SERVER;
    node_info.node_id_ = node_id;
    node_info.rank_id_ = rank_id;
    node_info.ip_ = ip;
    node_info.port_ = port;
    nodes_info_[node_id] = node_info;
    MS_LOG(INFO) << "The server node id:" << node_id << ",node ip: " << node_info.ip_ << ",node port:" << port
                 << " assign rank id:" << rank_id;
  } else if (register_message.role() == NodeRole::WORKER) {
    rank_id = ++next_worker_rank_id_;
    if (IntToUint(rank_id) >= ClusterMetadata::instance()->total_worker_num()) {
      MS_LOG(WARNING) << "The rank id is greater than the number of workers.";
      rank_id = -1;
      --next_worker_rank_id_;
    }
    NodeInfo node_info;
    node_info.node_role_ = NodeRole::WORKER;
    node_info.node_id_ = node_id;
    node_info.rank_id_ = rank_id;
    nodes_info_[node_id] = node_info;
    MS_LOG(INFO) << "The worker node id:" << node_id << " assign rank id:" << rank_id;
  }
  return rank_id;
}

void NodeManager::UpdateHeartbeat(const std::string &node_id) {
  std::lock_guard<std::mutex> lock(heartbeat_mutex_);
  NodeInfo node_info = nodes_info_[node_id];
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  heartbeats_[node_id] = current_time;
  MS_LOG(DEBUG) << "The node role: " << CommUtil::NodeRoleToString(node_info.node_role_) << ", the node id:" << node_id
                << ", the node rank id:" << node_info.rank_id_ << " the current time is: " << current_time.tv_sec;
}

void NodeManager::UpdateNodeFinishState(const std::string &node_id) { heartbeats_finish_nodes_.insert(node_id); }

bool NodeManager::CheckNodesFinishState() { return heartbeats_finish_nodes_.size() == nodes_info_.size(); }

std::vector<ServersMeta> NodeManager::FetchServersMeta() {
  std::vector<ServersMeta> servers_meta_list;
  for (auto it = nodes_info_.begin(); it != nodes_info_.end(); ++it) {
    if (it->second.node_role_ == NodeRole::SERVER) {
      ServersMeta servers_meta;
      servers_meta.set_rank_id(it->second.rank_id_);
      servers_meta.set_ip(it->second.ip_);
      servers_meta.set_port(it->second.port_);
      servers_meta_list.push_back(servers_meta);
    }
  }
  return servers_meta_list;
}

void NodeManager::UpdateClusterState() {
  // 1. update cluster timeout state
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  timeout_nodes_info_.clear();
  for (auto it = heartbeats_.begin(); it != heartbeats_.end(); ++it) {
    if (it->second.tv_sec + ClusterMetadata::instance()->heartbeat_timeout() < current_time.tv_sec) {
      MS_LOG(WARNING) << "The node id:" << it->first << " is timeout!";
      timeout_nodes_info_[it->first] = nodes_info_[it->first];
    }
  }
  if (!timeout_nodes_info_.empty()) {
    is_node_timeout_ = true;
    for (auto it = timeout_nodes_info_.begin(); it != timeout_nodes_info_.end(); ++it) {
      finish_nodes_id_.insert(it->first);
    }
  }

  // 2. update cluster finish state
  if (finish_nodes_id_.size() == total_node_num_ || SizeToInt(finish_nodes_id_.size()) == current_node_num_) {
    is_cluster_finish_ = true;
    is_cluster_ready_ = true;
  }

  // 3. update cluster ready state
  if (nodes_info_.size() == total_node_num_) {
    is_cluster_ready_ = true;
  }
}

void NodeManager::CheckClusterTimeout() {
  if (total_node_num_ != nodes_info_.size()) {
    MS_LOG(WARNING) << "The cluster is not ready after " << ClusterMetadata::instance()->cluster_available_timeout()
                    << " seconds,so finish the cluster, and change total node number from " << total_node_num_ << " to "
                    << nodes_info_.size();
    current_node_num_ = nodes_info_.size();
    is_cluster_timeout_ = true;
  }
}

void NodeManager::AddFinishNode(const std::string &finish_message) { finish_nodes_id_.insert(finish_message); }

std::unordered_map<std::string, NodeInfo> NodeManager::nodes_info() { return nodes_info_; }

bool NodeManager::is_cluster_finish() { return is_cluster_finish_.load(); }

bool NodeManager::is_cluster_ready() { return is_cluster_ready_.load(); }

bool NodeManager::is_cluster_timeout() { return is_cluster_timeout_.load(); }

bool NodeManager::is_node_timeout() { return is_node_timeout_.load(); }

void NodeManager::set_cluster_timeout(bool is_cluster_timeout) { is_cluster_timeout_ = is_cluster_timeout; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
