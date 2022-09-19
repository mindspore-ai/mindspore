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
#include "utils/ms_context.h"

namespace mindspore {
namespace ps {
namespace core {
void NodeManager::InitNode() {
  initial_total_node_num_ = PSContext::instance()->cluster_config().initial_server_num +
                            PSContext::instance()->cluster_config().initial_worker_num;
  meta_data_ = std::make_unique<ClusterMetadata>(PSContext::instance()->cluster_config().initial_worker_num,
                                                 PSContext::instance()->cluster_config().initial_server_num);
  MS_EXCEPTION_IF_NULL(meta_data_);
  total_node_num_ = initial_total_node_num_;
}

uint32_t NodeManager::checkIfRankIdExist(const RegisterMessage &register_message) {
  uint32_t rank_id = UINT_MAX;
  const std::string &node_id = register_message.node_id();
  if (registered_nodes_info_.find(node_id) != registered_nodes_info_.end()) {
    const std::string &new_ip = register_message.ip();
    uint32_t new_port = register_message.port();

    rank_id = registered_nodes_info_[node_id].rank_id_;
    registered_nodes_info_[node_id].is_alive = true;
    registered_nodes_info_[node_id].ip_ = new_ip;
    registered_nodes_info_[node_id].port_ = static_cast<uint16_t>(new_port);
    MS_LOG(WARNING) << "The node id: " << node_id << " is already assigned!"
                    << ", ip: " << register_message.ip() << ", port: " << register_message.port()
                    << ", rank id: " << rank_id << ", alive: " << registered_nodes_info_[node_id].is_alive
                    << ", the node_role:" << CommUtil::NodeRoleToString(registered_nodes_info_[node_id].node_role_);
    return rank_id;
  }

  (void)ReAddNodeIfNotExists(node_id, register_message.ip(), register_message.port(), &rank_id);
  return rank_id;
}

bool NodeManager::ReAddNodeIfNotExists(const std::string &node_id, const std::string &ip, uint32_t port,
                                       uint32_t *rank_id) {
  core::ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
  std::unordered_map<std::string, NodeInfo> recovery_node_infos = clusterConfig.initial_registered_nodes_infos;

  if (registered_nodes_info_.find(node_id) == registered_nodes_info_.end() &&
      recovery_node_infos.find(node_id) != recovery_node_infos.end()) {
    if (rank_id != nullptr) {
      *rank_id = recovery_node_infos[node_id].rank_id_;
    }
    recovery_node_infos[node_id].is_alive = true;
    recovery_node_infos[node_id].ip_ = ip;
    recovery_node_infos[node_id].port_ = static_cast<uint16_t>(port);
    registered_nodes_info_[node_id] = recovery_node_infos[node_id];
    MS_LOG(INFO) << "The node id: " << node_id << " is recovery successful!"
                 << ", ip: " << ip << ", port: " << port;
    return true;
  }
  return false;
}

uint32_t NodeManager::NextRankId(const RegisterMessage &register_message, const std::shared_ptr<MessageMeta> &meta) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(meta_data_);
  std::lock_guard<std::mutex> lock(assign_rank_id_mutex_);
  uint32_t rank_id = checkIfRankIdExist(register_message);
  if (rank_id != UINT_MAX) {
    return rank_id;
  }
  if (total_node_num_ == SizeToUint(registered_nodes_info_.size())) {
    MS_LOG(WARNING) << "There are enough nodes registering to scheduler.";
    return UINT_MAX;
  }

  const std::string &node_id = register_message.node_id();
  // create new rank id
  if (register_message.role() == NodeRole::SERVER) {
    const std::string &ip = register_message.ip();
    uint32_t port = register_message.port();

    if (meta->rank_id() != UINT32_MAX && meta->rank_id() < next_server_rank_id_) {
      rank_id = meta->rank_id();
      MS_LOG(INFO) << "Use the old rank id:" << rank_id;
    } else {
      rank_id = next_server_rank_id_;
      next_server_rank_id_ += 1;
    }

    if (rank_id >= meta_data_->server_num) {
      MS_LOG(ERROR) << "The rank id is greater than the number of servers:" << meta_data_->server_num;
      rank_id = UINT_MAX;
      next_server_rank_id_ -= 1;
      return rank_id;
    }
    NodeInfo node_info;
    node_info.node_role_ = NodeRole::SERVER;
    node_info.node_id_ = node_id;
    node_info.rank_id_ = rank_id;
    node_info.ip_ = ip;
    node_info.port_ = port;
    node_info.is_alive = true;
    registered_nodes_info_[node_id] = node_info;
    MS_LOG(INFO) << "The server node id:" << node_id << ", node ip: " << node_info.ip_ << ", node port:" << port
                 << " assign rank id:" << rank_id << ", " << (meta_data_->server_num - next_server_rank_id_)
                 << " servers still need to be registered.";
  } else if (register_message.role() == NodeRole::WORKER) {
    const std::string &ip = register_message.ip();
    uint32_t port = register_message.port();

    if (meta->rank_id() != UINT32_MAX && meta->rank_id() < next_worker_rank_id_) {
      rank_id = meta->rank_id();
      MS_LOG(INFO) << "Use the old rank id:" << rank_id;
    } else {
      rank_id = next_worker_rank_id_;
      next_worker_rank_id_ += 1;
    }

    if (rank_id >= meta_data_->worker_num) {
      MS_LOG(ERROR) << "The rank id is greater than the number of workers:" << meta_data_->worker_num;
      rank_id = UINT_MAX;
      next_worker_rank_id_ -= 1;
      return rank_id;
    }
    NodeInfo node_info;
    node_info.node_role_ = NodeRole::WORKER;
    node_info.node_id_ = node_id;
    node_info.rank_id_ = rank_id;
    node_info.ip_ = ip;
    node_info.port_ = port;
    node_info.is_alive = true;
    registered_nodes_info_[node_id] = node_info;
    MS_LOG(INFO) << "The worker node id:" << node_id << ", node ip: " << node_info.ip_ << ", node port:" << port
                 << " assign rank id:" << rank_id << ", " << (meta_data_->worker_num - next_worker_rank_id_)
                 << " workers still need to be registered.";
  }
  return rank_id;
}

void NodeManager::UpdateHeartbeat(const std::string &node_id) {
  std::lock_guard<std::mutex> lock(heartbeat_mutex_);
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  if (registered_nodes_info_.count(node_id) > 0) {
    heartbeats_[node_id] = current_time;
  }
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

const std::unordered_map<std::string, NodeInfo> &NodeManager::QueryTimeOutNodesInfo() const {
  return timeout_nodes_info_;
}

void NodeManager::UpdateCluster(bool is_cluster_ready) {
  // 1. update cluster timeout state
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  timeout_nodes_info_.clear();
  std::lock_guard<std::mutex> lock(heartbeat_mutex_);
  for (auto it = heartbeats_.begin(); it != heartbeats_.end(); ++it) {
    if (it->second.tv_sec + PSContext::instance()->cluster_config().heartbeat_timeout < current_time.tv_sec) {
      if (registered_nodes_info_.count(it->first)) {
        MS_LOG(WARNING) << "The node id:" << it->first << " is timeout!";
        timeout_nodes_info_[it->first] = registered_nodes_info_[it->first];
        registered_nodes_info_[it->first].is_alive = false;
      }
    } else {
      if (registered_nodes_info_.count(it->first) && !registered_nodes_info_[it->first].is_alive) {
        MS_LOG(WARNING) << registered_nodes_info_[it->first].node_id_ << " is alive.";
        registered_nodes_info_[it->first].is_alive = true;
      }
    }
  }

  if (!timeout_nodes_info_.empty()) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_RECOVERY)) {
      for (auto iter = timeout_nodes_info_.begin(); iter != timeout_nodes_info_.end(); ++iter) {
        (void)heartbeats_.erase(iter->first);
        finish_nodes_id_.insert(iter->first);
      }
    }
    if (cluster_state_ != ClusterState::CLUSTER_DISABLE_FLS && cluster_state_ != ClusterState::CLUSTER_SCALE_OUT) {
      UpdateClusterState(ClusterState::NODE_TIMEOUT);
    }
  } else if (SizeToUint(heartbeats_.size()) == total_node_num_ && cluster_state_ == ClusterState::NODE_TIMEOUT) {
    if (is_cluster_ready) {
      UpdateClusterState(ClusterState::CLUSTER_READY);
    } else {
      UpdateClusterState(ClusterState::CLUSTER_STARTING);
    }
  }

  // 2. update cluster finish state
  if (SizeToUint(finish_nodes_id_.size()) == total_node_num_) {
    UpdateClusterState(ClusterState::CLUSTER_EXIT);
  }
}

void NodeManager::AddFinishNode(const std::string &finish_message) { finish_nodes_id_.insert(finish_message); }

void NodeManager::AddScaleOutDoneNode(const std::string &node_id) { scale_out_done_nodes_id_.insert(node_id); }

void NodeManager::AddScaleInDoneNode(const std::string &node_id) { scale_in_done_nodes_id_.insert(node_id); }

bool NodeManager::IsAllNodesAlive() const {
  uint32_t num = std::count_if(registered_nodes_info_.begin(), registered_nodes_info_.end(), [](auto item) {
    if (!item.second.is_alive) {
      MS_LOG(ERROR) << item.second.node_id_ << " is not alive.";
      return false;
    }
    return true;
  });
  return num == total_node_num_;
}

bool NodeManager::IsAllNodesRegistered() const { return SizeToUint(registered_nodes_info_.size()) == total_node_num_; }

bool NodeManager::IsAllNodesFinished() const { return SizeToUint(finish_nodes_id_.size()) == total_node_num_; }

bool NodeManager::IsAllNodesScaleOutDone() const {
  return SizeToUint(scale_out_done_nodes_id_.size()) == total_node_num_;
}

bool NodeManager::IsAllNodesScaleInDone() const {
  return SizeToUint(scale_in_done_nodes_id_.size()) == total_node_num_;
}

const std::unordered_map<std::string, NodeInfo> &NodeManager::nodes_info() const { return nodes_info_; }

const std::unordered_map<std::string, NodeInfo> &NodeManager::registered_nodes_info() const {
  return registered_nodes_info_;
}

void NodeManager::set_registered_nodes_info(const std::unordered_map<std::string, NodeInfo> registered_nodes_info) {
  this->registered_nodes_info_ = registered_nodes_info;
}

void NodeManager::UpdateNodesInfo() {
  MS_LOG(INFO) << "Update nodes info.";
  nodes_info_.clear();
  nodes_info_ = registered_nodes_info_;
}

void NodeManager::UpdateClusterState(const ClusterState &state) {
  std::lock_guard<std::mutex> lk(cluster_mutex_);
  std::string state_str = CommUtil::ClusterStateToString(state);
  if (state_str.empty() || state == cluster_state_) {
    return;
  }
  MS_LOG(INFO) << "[state]: Cluster state change from:" << CommUtil::ClusterStateToString(cluster_state_) << " to "
               << state_str;
  cluster_state_ = state;
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
    next_server_rank_id_ = *min_rank_id;
    MS_LOG(INFO) << "The next server rank id:" << next_server_rank_id_;
  }
  registered_nodes_info_.clear();
  ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
  clusterConfig.initial_registered_nodes_infos.clear();
  heartbeats_.clear();
}

void NodeManager::SaveRecoveryRankId(const NodeInfo &info) {
  if (info.node_role_ == NodeRole::SERVER) {
    recovery_server_rank_id_.push_back(info.rank_id_);
  } else if (info.node_role_ == NodeRole::WORKER) {
    recovery_worker_rank_id_.push_back(info.rank_id_);
  }
}

bool NodeManager::IsWorker() const {
  bool res = std::any_of(registered_nodes_info_.begin(), registered_nodes_info_.end(), [](auto item) {
    if (item.second.node_role_ == NodeRole::WORKER && item.second.is_alive == false) {
      return true;
    }
    return false;
  });

  return res;
}

bool NodeManager::IsNodeRegistered(const std::string &node_id) {
  if (registered_nodes_info_.find(node_id) != registered_nodes_info_.end()) {
    MS_LOG(WARNING) << "The node id " << node_id << " has been registered.";
    return true;
  }
  return false;
}

const NodeInfo NodeManager::QueryNodeInfo(const std::string &node_id) const {
  auto iter = registered_nodes_info_.find(node_id);
  if (iter == registered_nodes_info_.end()) {
    return NodeInfo();
  }
  return iter->second;
}

bool NodeManager::IsNodePersisting(const std::string &node_id) const {
  return nodes_persisting_.find(node_id) != nodes_persisting_.end();
}

void NodeManager::AddPersistingNode(const std::string &node_id) { (void)nodes_persisting_.insert(node_id); }

bool NodeManager::IsAllNodeInPersisting() {
  // The worker role does not support disaster recovery currently.
  if (SizeToUint(nodes_persisting_.size()) == server_num()) {
    nodes_persisting_.clear();
    return true;
  }
  return false;
}

void NodeManager::set_total_node_num(const uint32_t &node_num) { total_node_num_ = node_num; }

const uint32_t &NodeManager::total_node_num() const { return total_node_num_; }

void NodeManager::set_worker_num(const uint32_t &worker_num) { meta_data_->worker_num = worker_num; }

void NodeManager::set_server_num(const uint32_t &server_num) { meta_data_->server_num = server_num; }

uint32_t NodeManager::worker_num() const { return meta_data_->worker_num; }

uint32_t NodeManager::server_num() const { return meta_data_->server_num; }

uint32_t NodeManager::next_worker_rank_id() const { return next_worker_rank_id_.load(); }

uint32_t NodeManager::next_server_rank_id() const { return next_server_rank_id_.load(); }

void NodeManager::set_next_worker_rank_id(const uint32_t &next_worker_rank_id) {
  this->next_worker_rank_id_ = next_worker_rank_id;
}
void NodeManager::set_next_server_rank_id(const uint32_t &next_server_rank_id) {
  this->next_server_rank_id_ = next_server_rank_id;
}
void NodeManager::setPersistCallback(const OnPersist &onPersist) { this->onPersist_ = onPersist; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
