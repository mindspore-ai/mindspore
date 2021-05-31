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
        total_node_num_(-1),
        current_node_num_(-1),
        next_worker_rank_id_(-1),
        next_server_rank_id_(-1),
        meta_data_(nullptr),
        node_state_(NodeState::NODE_STARTING),
        cluster_state_(ClusterState::ClUSTER_STARTING) {}
  virtual ~NodeManager() = default;

  // When initializing nodes, the initial number of nodes will be assigned to the total number of nodes.
  void InitNode();
  int NextRankId(const RegisterMessage &register_message);

  void UpdateHeartbeat(const std::string &node_id);
  bool CheckNodesScaluOutState();
  void UpdateNodeScaleInState(const std::string &node_id);
  bool CheckNodesScaleInState();

  std::vector<ServersMeta> FetchServersMeta();
  void UpdateCluster();
  void CheckClusterTimeout();
  void AddFinishNode(const std::string &finish_message);

  // When workers and servers registered to scheduler, the scheduler will collect the number of registered
  // nodes and Determine whether the registered number of worker and server is equal to total_node_num_.
  bool IsAllNodesRegistered();
  // When workers and servers send a finish message to the scheduler, the scheduler will collect the number of
  // finish nodes and Determine whether the finished nodes are equal to total_node_num_.
  bool IsAllNodesFinished();

  std::unordered_map<std::string, NodeInfo> &nodes_info();

  void set_total_node_num(const int32_t &node_num);
  const int32_t &total_node_num();
  void set_worker_num(const int32_t &worker_num);
  void set_server_num(const int32_t &server_num);
  int32_t worker_num() const;
  int32_t server_num() const;

  void UpdateNodeState(const NodeState &state);
  void UpdateClusterState(const ClusterState &state);
  NodeState GetNodeState();
  ClusterState GetClusterState();

  // When the scheduler receives the scale out or scale in message, the metadata needs to be reset, because all nodes
  // will re-register.
  void ResetMetadata();

 private:
  std::mutex node_mutex_;
  std::mutex cluster_mutex_;

  uint32_t initial_total_node_num_;
  int32_t total_node_num_;
  int32_t current_node_num_;

  std::atomic<int> next_worker_rank_id_;
  std::atomic<int> next_server_rank_id_;

  // worker nodes and server nodes
  std::unordered_map<std::string, NodeInfo> nodes_info_;
  std::mutex assign_rank_id_mutex_;
  std::mutex heartbeat_mutex_;

  std::unordered_map<std::string, timeval> heartbeats_;
  std::unordered_set<std::string> heartbeats_finish_nodes_;
  std::unordered_set<std::string> heartbeats_scale_out_nodes_;
  std::unordered_set<std::string> heartbeats_scale_in_nodes_;
  // timeout nodes
  std::unordered_map<std::string, NodeInfo> timeout_nodes_info_;
  std::unordered_set<std::string> finish_nodes_id_;

  // Cluster metadata information can be dynamically changed
  std::unique_ptr<ClusterMetadata> meta_data_;

  NodeState node_state_;
  ClusterState cluster_state_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_NODE_MANAGER_H_
