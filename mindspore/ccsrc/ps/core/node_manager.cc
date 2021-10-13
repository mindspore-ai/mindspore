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
void NodeManager::InitNode() {
  initial_total_node_num_ = PSContext::instance()->cluster_config().initial_server_num +
                            PSContext::instance()->cluster_config().initial_worker_num;
  meta_data_ = std::make_unique<ClusterMetadata>(PSContext::instance()->cluster_config().initial_worker_num,
                                                 PSContext::instance()->cluster_config().initial_server_num);
  MS_EXCEPTION_IF_NULL(meta_data_);
  total_node_num_ = UintToInt(initial_total_node_num_);
}

uint32_t NodeManager::NextRankId(const RegisterMessage &register_message, const std::shared_ptr<MessageMeta> &meta) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(meta_data_);
  std::lock_guard<std::mutex> lock(assign_rank_id_mutex_);
  uint32_t rank_id = UINT_MAX;

  const std::string &node_id = register_message.node_id();
  if (registered_nodes_info_.find(node_id) != registered_nodes_info_.end()) {
    const std::string &new_ip = register_message.ip();
    uint32_t new_port = register_message.port();
    rank_id = registered_nodes_info_[node_id].rank_id_;
    registered_nodes_info_[node_id].is_alive = true;
    registered_nodes_info_[node_id].ip_ = new_ip;
    registered_nodes_info_[node_id].port_ = static_cast<uint16_t>(new_port);
    MS_LOG(INFO) << "The node id: " << node_id << " is already assigned!";
    return rank_id;
  }

  if (register_message.role() == NodeRole::SERVER) {
    const std::string &ip = register_message.ip();
    uint32_t port = register_message.port();

    auto rank_it = std::find_if(registered_nodes_info_.begin(), registered_nodes_info_.end(), [&rank_id](auto item) {
      bool res = item.second.is_alive == false && item.second.node_role_ == NodeRole::SERVER;
      if (res) {
        MS_LOG(INFO) << "The server node id:" << item.first << " rank id:" << item.second.rank_id_ << " is not alive.";
        rank_id = item.second.rank_id_;
      }
      return res;
    });
    if (rank_it == registered_nodes_info_.end()) {
      if (meta->rank_id() != UINT32_MAX && UintToInt(meta->rank_id()) <= next_server_rank_id_) {
        rank_id = meta->rank_id();
        MS_LOG(INFO) << "Use the old rank id:" << rank_id;
      } else {
        rank_id = IntToUint(++next_server_rank_id_);
      }
    } else {
      registered_nodes_info_.erase((*rank_it).first);
    }

    if (rank_id >= meta_data_->server_num) {
      MS_LOG(WARNING) << "The rank id is greater than the number of servers:" << meta_data_->server_num;
      rank_id = UINT_MAX;
      --next_server_rank_id_;
    }
    NodeInfo node_info;
    node_info.node_role_ = NodeRole::SERVER;
    node_info.node_id_ = node_id;
    node_info.rank_id_ = rank_id;
    node_info.ip_ = ip;
    node_info.port_ = static_cast<uint16_t>(port);
    node_info.is_alive = true;
    registered_nodes_info_[node_id] = node_info;
    MS_LOG(INFO) << "The server node id:" << node_id << ",node ip: " << node_info.ip_ << ",node port:" << port
                 << " assign rank id:" << rank_id;
  } else if (register_message.role() == NodeRole::WORKER) {
    const std::string &ip = register_message.ip();
    uint32_t port = register_message.port();

    auto worker_rank_it =
      std::find_if(registered_nodes_info_.begin(), registered_nodes_info_.end(), [&rank_id](auto item) {
        bool res = item.second.is_alive == false && item.second.node_role_ == NodeRole::WORKER;
        if (res) {
          MS_LOG(INFO) << "The worker node id:" << item.first << " rank id:" << rank_id << " is not alive.";
          rank_id = item.second.rank_id_;
        }
        return res;
      });
    if (worker_rank_it == registered_nodes_info_.end()) {
      if (meta->rank_id() != UINT32_MAX && UintToInt(meta->rank_id()) <= next_worker_rank_id_) {
        rank_id = meta->rank_id();
        MS_LOG(INFO) << "Use the old rank id:" << rank_id;
      } else {
        rank_id = IntToUint(++next_worker_rank_id_);
      }
    } else {
      registered_nodes_info_.erase((*worker_rank_it).first);
    }

    if (rank_id >= meta_data_->worker_num) {
      MS_LOG(WARNING) << "The rank id is greater than the number of workers:" << meta_data_->worker_num;
      rank_id = UINT_MAX;
      --next_worker_rank_id_;
    }
    NodeInfo node_info;
    node_info.node_role_ = NodeRole::WORKER;
    node_info.node_id_ = node_id;
    node_info.rank_id_ = rank_id;
    node_info.ip_ = ip;
    node_info.port_ = static_cast<uint16_t>(port);
    node_info.is_alive = true;
    registered_nodes_info_[node_id] = node_info;
    MS_LOG(INFO) << "The worker node id:" << node_id << " assign rank id:" << rank_id;
  }
  return rank_id;
}

void NodeManager::UpdateHeartbeat(const std::string &node_id) {
  std::lock_guard<std::mutex> lock(heartbeat_mutex_);
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  heartbeats_[node_id] = current_time;
}

std::vector<ServersMeta> NodeManager::FetchServersMeta() {
  std::vector<ServersMeta> servers_meta_list;
  for (auto it = registered_nodes_info_.begin(); it != registered_nodes_info_.end(); ++it) {
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

std::vector<ServersMeta> NodeManager::FetchAllNodesMeta() {
  std::vector<ServersMeta> servers_meta_list;
  for (auto it = registered_nodes_info_.begin(); it != registered_nodes_info_.end(); ++it) {
    ServersMeta servers_meta;
    servers_meta.set_rank_id(it->second.rank_id_);
    servers_meta.set_ip(it->second.ip_);
    servers_meta.set_port(it->second.port_);
    servers_meta.set_is_alive(it->second.is_alive);
    servers_meta.set_role(it->second.node_role_);
    servers_meta.set_node_id(it->second.node_id_);
    servers_meta_list.push_back(servers_meta);
  }
  return servers_meta_list;
}

void NodeManager::UpdateCluster() {
  // 1. update cluster timeout state
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  timeout_nodes_info_.clear();
  for (auto it = heartbeats_.begin(); it != heartbeats_.end(); ++it) {
    if (it->second.tv_sec + PSContext::instance()->cluster_config().heartbeat_timeout < current_time.tv_sec) {
      if (registered_nodes_info_.count(it->first)) {
        MS_LOG(WARNING) << "The node id:" << it->first << " is timeout!";
        timeout_nodes_info_[it->first] = registered_nodes_info_[it->first];
        registered_nodes_info_[it->first].is_alive = false;
      }
    }
  }

  if (!timeout_nodes_info_.empty()) {
    UpdateClusterState(ClusterState::NODE_TIMEOUT);
    for (auto iter = timeout_nodes_info_.begin(); iter != timeout_nodes_info_.end(); ++iter) {
      (void)heartbeats_.erase(iter->first);
      finish_nodes_id_.insert(iter->first);
    }
  }

  // 2. update cluster finish state
  if (SizeToInt(finish_nodes_id_.size()) == total_node_num_ ||
      SizeToInt(finish_nodes_id_.size()) == current_node_num_) {
    UpdateClusterState(ClusterState::CLUSTER_EXIT);
  }
}

void NodeManager::CheckClusterTimeout() {
  if (total_node_num_ != SizeToInt(registered_nodes_info_.size())) {
    MS_LOG(WARNING) << "The cluster is not ready after "
                    << PSContext::instance()->cluster_config().cluster_available_timeout
                    << " seconds,so finish the cluster, and change total node number from " << total_node_num_ << " to "
                    << registered_nodes_info_.size();
    current_node_num_ = SizeToInt(registered_nodes_info_.size());
    UpdateClusterState(ClusterState::NODE_TIMEOUT);
  }
}

void NodeManager::AddFinishNode(const std::string &finish_message) { finish_nodes_id_.insert(finish_message); }

void NodeManager::AddScaleOutDoneNode(const std::string &node_id) { scale_out_done_nodes_id_.insert(node_id); }

void NodeManager::AddScaleInDoneNode(const std::string &node_id) { scale_in_done_nodes_id_.insert(node_id); }

bool NodeManager::IsAllNodesRegistered() const {
  int32_t num = std::count_if(registered_nodes_info_.begin(), registered_nodes_info_.end(),
                              [](auto item) { return item.second.is_alive == true; });
  return num == total_node_num_;
}

bool NodeManager::IsAllNodesFinished() const { return SizeToInt(finish_nodes_id_.size()) == total_node_num_; }

bool NodeManager::IsAllNodesScaleOutDone() const {
  return SizeToInt(scale_out_done_nodes_id_.size()) == total_node_num_;
}

bool NodeManager::IsAllNodesScaleInDone() const { return SizeToInt(scale_in_done_nodes_id_.size()) == total_node_num_; }

const std::unordered_map<std::string, NodeInfo> &NodeManager::nodes_info() const { return nodes_info_; }

const std::unordered_map<std::string, NodeInfo> &NodeManager::registered_nodes_info() const {
  return registered_nodes_info_;
}

void NodeManager::UpdateNodesInfo() {
  MS_LOG(INFO) << "Update nodes info.";
  nodes_info_.clear();
  nodes_info_ = registered_nodes_info_;
}

void NodeManager::UpdateNodeState(const NodeState &state) {
  std::lock_guard<std::mutex> lk(node_mutex_);
  node_state_ = state;
}

void NodeManager::UpdateClusterState(const ClusterState &state) {
  std::lock_guard<std::mutex> lk(cluster_mutex_);
  MS_LOG(INFO) << "[state]: Scheduler change state from:" << CommUtil::ClusterStateToString(cluster_state_) << " to "
               << CommUtil::ClusterStateToString(state);
  cluster_state_ = state;
}

NodeState NodeManager::GetNodeState() {
  std::lock_guard<std::mutex> lk(node_mutex_);
  return node_state_;
}

ClusterState NodeManager::GetClusterState() {
  std::lock_guard<std::mutex> lk(cluster_mutex_);
  return cluster_state_;
}

void NodeManager::ResetMetadata(const std::vector<std::string> &scale_in_nodes) {
  MS_LOG(WARNING) << "Reset metadata.";
  std::vector<uint32_t> server_rank_ids;
  if (GetClusterState() == ClusterState::CLUSTER_SCALE_IN) {
    for (const auto &item : scale_in_nodes) {
      if (registered_nodes_info_.count(item)) {
        server_rank_ids.push_back(registered_nodes_info_[item].rank_id_);
      }
    }
    auto min_rank_id = std::min_element(server_rank_ids.begin(), server_rank_ids.end());
    next_server_rank_id_ = UintToInt(*min_rank_id - 1);
    MS_LOG(INFO) << "The next server rank id:" << next_server_rank_id_;
  }
  registered_nodes_info_.clear();
  heartbeats_.clear();
}

bool NodeManager::IsWorkerOrServer0() {
  bool res = std::any_of(registered_nodes_info_.begin(), registered_nodes_info_.end(), [](auto item) {
    if (item.second.node_role_ == NodeRole::WORKER && item.second.is_alive == false) {
      return true;
    }

    if (item.second.node_role_ == NodeRole::SERVER && item.second.is_alive == false && item.second.rank_id_ == 0) {
      return true;
    }

    return false;
  });

  return res;
}

bool NodeManager::IsNodeRegistered(const std::string &node_id) {
  if (registered_nodes_info_.find(node_id) != registered_nodes_info_.end()) {
    return true;
  }
  return false;
}

void NodeManager::set_total_node_num(const int32_t &node_num) { total_node_num_ = node_num; }

const int32_t &NodeManager::total_node_num() const { return total_node_num_; }

void NodeManager::set_worker_num(const int32_t &worker_num) { meta_data_->worker_num = IntToUint(worker_num); }

void NodeManager::set_server_num(const int32_t &server_num) { meta_data_->server_num = IntToUint(server_num); }

int32_t NodeManager::worker_num() const { return UintToInt(meta_data_->worker_num); }

int32_t NodeManager::server_num() const { return UintToInt(meta_data_->server_num); }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
