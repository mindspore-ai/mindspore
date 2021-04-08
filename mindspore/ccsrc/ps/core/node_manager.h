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

namespace mindspore {
namespace ps {
namespace core {
class NodeManager {
 public:
  NodeManager()
      : is_cluster_ready_(false),
        is_cluster_finish_(false),
        is_cluster_timeout_(false),
        is_node_timeout_(false),
        total_node_num_(0),
        current_node_num_(-1),
        next_worker_rank_id_(-1),
        next_server_rank_id_(-1) {}
  virtual ~NodeManager() = default;

  enum ClusterState { STARTING, STARTED, FAILED, STOPPING, STOPPED };

  void InitNodeNum();
  int NextRankId(const RegisterMessage &register_message);
  void UpdateHeartbeat(const std::string &node_id);
  void UpdateNodeFinishState(const std::string &node_id);
  bool CheckNodesFinishState();
  std::vector<ServersMeta> FetchServersMeta();
  void UpdateClusterState();
  void CheckClusterTimeout();
  void AddFinishNode(const std::string &finish_message);
  std::unordered_map<std::string, NodeInfo> nodes_info();
  bool is_cluster_ready();
  bool is_cluster_finish();
  bool is_cluster_timeout();
  bool is_node_timeout();
  void set_cluster_timeout(bool is_cluster_timeout);

 private:
  std::atomic<bool> is_cluster_ready_;
  std::atomic<bool> is_cluster_finish_;
  std::atomic<bool> is_cluster_timeout_;
  std::atomic<bool> is_node_timeout_;
  uint32_t total_node_num_;
  int32_t current_node_num_;
  std::atomic<int> next_worker_rank_id_;
  std::atomic<int> next_server_rank_id_;
  // worker nodes and server nodes
  std::unordered_map<std::string, NodeInfo> nodes_info_;
  std::mutex assign_rank_id_mutex_;
  std::mutex heartbeat_mutex_;
  std::unordered_map<std::string, timeval> heartbeats_;
  std::unordered_set<std::string> heartbeats_finish_nodes_;
  // timeout nodes
  std::unordered_map<std::string, NodeInfo> timeout_nodes_info_;
  std::unordered_set<std::string> finish_nodes_id_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_NODE_MANAGER_H_
