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

#ifndef MINDSPORE_CCSRC_PS_CORE_NODE_MANAGER_H_
#define MINDSPORE_CCSRC_PS_CORE_NODE_MANAGER_H_

#include <atomic>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <condition_variable>
#include <unordered_set>
#include <deque>
#include <algorithm>

#include "ps/core/node.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "ps/core/cluster_metadata.h"

namespace mindspore {
namespace ps {
namespace core {
class NodeManager {
 public:
  NodeManager()
      : initial_total_node_num_(0),
        total_node_num_(0),
        next_worker_rank_id_(0),
        next_server_rank_id_(0),
        meta_data_(nullptr),
        cluster_state_(ClusterState::CLUSTER_STARTING) {}
  virtual ~NodeManager() = default;
  using OnPersist = std::function<void()>;
  // When initializing nodes, the initial number of nodes will be assigned to the total number of nodes.
  void InitNode();
  uint32_t NextRankId(const RegisterMessage &register_message, const std::shared_ptr<MessageMeta> &meta);
  uint32_t checkIfRankIdExist(const RegisterMessage &register_message);

  // Re-Add the server or worker node into the registered node list if the node do not existed in the scheduler.
  bool ReAddNodeIfNotExists(const std::string &node_id, const std::string &ip, uint32_t port,
                            uint32_t *rank_id = nullptr);

  void UpdateHeartbeat(const std::string &node_id);
  std::vector<ServersMeta> FetchServersMeta();
  // Fetch metadata information of all nodes.
  std::vector<ServersMeta> FetchAllNodesMeta();

  void UpdateCluster(bool is_cluster_ready);
  void AddFinishNode(const std::string &finish_message);

  // After the scheduler receives the scale_out_done node, it will save this node.
  void AddScaleOutDoneNode(const std::string &node_id);
  // After the scheduler receives the scale_in_done node, it will save this node.
  void AddScaleInDoneNode(const std::string &node_id);

  // When workers and servers registered to scheduler, the scheduler will collect the number of registered
  // nodes and Determine whether the registered number of worker and server is equal to total_node_num_.
  bool IsAllNodesRegistered() const;
  // When workers and servers send a finish message to the scheduler, the scheduler will collect the number of
  // finish nodes and Determine whether the finished nodes are equal to total_node_num_.
  bool IsAllNodesFinished() const;

  // When workers and servers send a scale_out_done message to the scheduler, the scheduler will collect the number of
  // nodes and Determine whether the nodes are equal to total_node_num_.
  bool IsAllNodesScaleOutDone() const;
  // When workers and servers send a scale_in_done message to the scheduler, the scheduler will collect the number of
  // nodes and Determine whether the nodes are equal to total_node_num_.
  bool IsAllNodesScaleInDone() const;

  const std::unordered_map<std::string, NodeInfo> &nodes_info() const;
  const std::unordered_map<std::string, NodeInfo> &registered_nodes_info() const;
  // After all the nodes are registered successfully, the nodes info can be updated.
  void UpdateNodesInfo();

  void set_total_node_num(const uint32_t &node_num);
  const uint32_t &total_node_num() const;
  void set_worker_num(const uint32_t &worker_num);
  void set_server_num(const uint32_t &server_num);
  uint32_t worker_num() const;
  uint32_t server_num() const;

  uint32_t next_worker_rank_id() const;
  uint32_t next_server_rank_id() const;

  void UpdateClusterState(const ClusterState &state);
  ClusterState GetClusterState();

  // When the scheduler receives the scale out or scale in message, the metadata needs to be reset, because all nodes
  // will re-register.
  void ResetMetadata(const std::vector<std::string> &scale_in_nodes = {});

  void SaveRecoveryRankId(const NodeInfo &info);

  bool IsWorker() const;

  // Determine whether the node id has been registered.
  bool IsNodeRegistered(const std::string &node_id);

  void set_registered_nodes_info(const std::unordered_map<std::string, NodeInfo> registered_nodes_info);
  void set_next_worker_rank_id(const uint32_t &next_worker_rank_id);
  void set_next_server_rank_id(const uint32_t &next_server_rank_id);
  void setPersistCallback(const OnPersist &onPersist);

  // Query node information by node id.
  const NodeInfo QueryNodeInfo(const std::string &node_id) const;

  // Determine whether the node is in persistent.
  bool IsNodePersisting(const std::string &node_id) const;

  // Record the node that is persisting.
  void AddPersistingNode(const std::string &node_id);

  // Determine whether all nodes that need to be persisted are in persistence.
  bool IsAllNodeInPersisting();

  bool IsAllNodesAlive() const;

  const std::unordered_map<std::string, NodeInfo> &QueryTimeOutNodesInfo() const;

 private:
  std::mutex node_mutex_;
  std::mutex cluster_mutex_;

  uint32_t initial_total_node_num_;
  uint32_t total_node_num_;
  uint32_t current_node_num_;

  std::atomic<uint32_t> next_worker_rank_id_;
  std::atomic<uint32_t> next_server_rank_id_;

  // Whenever a node is registered, it will be stored in this map.
  std::unordered_map<std::string, NodeInfo> registered_nodes_info_;
  // When all nodes are registered successfully, then all nodes info will be stored in this map. In other words, the
  // nodes_info_ is a snapshot of the registered_nodes_info_.
  std::unordered_map<std::string, NodeInfo> nodes_info_;

  // Record the node that is persisting.
  std::set<std::string> nodes_persisting_;

  std::mutex assign_rank_id_mutex_;
  std::mutex heartbeat_mutex_;

  std::unordered_map<std::string, timeval> heartbeats_;
  // timeout nodes
  std::unordered_map<std::string, NodeInfo> timeout_nodes_info_;
  std::unordered_set<std::string> finish_nodes_id_;

  // The scheduler aggregates scale_out_done messages from workers/servers
  std::unordered_set<std::string> scale_out_done_nodes_id_;
  // The scheduler aggregates scale_in_done messages from workers/servers
  std::unordered_set<std::string> scale_in_done_nodes_id_;

  // Cluster metadata information can be dynamically changed
  std::unique_ptr<ClusterMetadata> meta_data_;

  ClusterState cluster_state_;

  std::deque<uint32_t> recovery_worker_rank_id_;
  std::deque<uint32_t> recovery_server_rank_id_;

  OnPersist onPersist_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_NODE_MANAGER_H_
