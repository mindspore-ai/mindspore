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

#ifndef MINDSPORE_CCSRC_PS_CORE_PS_SCHEDULER_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_PS_SCHEDULER_NODE_H_

#include <map>
#include <unordered_map>
#include <memory>
#include <vector>
#include <set>
#include <string>

#include "ps/core/scheduler_node.h"
#include "ps/core/node_info.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace ps {
namespace core {
// This class is a derived class of SchedulerNode specialized for Parameter Server. It is used to rewrite the specific
// logic for Parameter Server mode training in SchedulerNode. For example, the Scheduler of Parameter Server will reject
// the registration request of alive nodes.
class PSSchedulerNode : public SchedulerNode {
 public:
  PSSchedulerNode() {
    node_nums_[NodeRole::WORKER] = ps::PSContext::instance()->worker_num();
    node_nums_[NodeRole::SERVER] = ps::PSContext::instance()->server_num();
  }

  ~PSSchedulerNode() override = default;

 protected:
  // Override the scheduler node to remove the nofification from scheduler to other nodes.
  void RunRecovery() override;

 private:
  // Determine whether the registration request of the node should be rejected, the registration of the
  // alive node should be rejected.
  bool NeedRejectRegister(const NodeInfo &node_info) override { return node_info.is_alive; }

  bool SendPrepareBuildingNetwork(const std::unordered_map<std::string, NodeInfo> &node_infos) override {
    return true;
  };

  // Register collective communication initialization service.
  void RegisterInitCollectCommServiceHandler() override;

  // Register recovery service.
  void RegisterRecoveryServiceHandler() override;

  // Process message for sending node's host name.
  void ProcessSendHostName(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                           const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process message for querying all nodes' host name.
  void ProcessQueryHostNames(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                             const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process message for send unique id.
  void ProcessSendUniqueID(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                           const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process message for querying unique id.
  void ProcessQueryUniqueID(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                            const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process message for sending the ready status to finish transform graph of computed node,
  void ProcessSendFinishTransform(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                                  const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process message for querying the ready status to finish transform graph of computed node,
  void ProcessQueryFinishTransform(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                                   const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Handle node timeout info and update nodes which finish transform graph.
  void HandleNodeTimeoutForRecovery(const std::unordered_map<std::string, NodeInfo> &timeout_nodes_infos) override;

  // Recover finish transform nodes info when nodes recover heartbeat.
  void HandleNodeRecoverByHeartBeat(uint32_t rank_id) override;

  void RecoverFromPersistence() override;

  void InitEventTxtFile() override {}

  void RecordSchedulerRestartInfo() override {}

  // Record received host hash name from workers or servers.
  std::map<NodeRole, std::vector<size_t>> host_hash_names_;
  // Record rank id of the nodes which sended host name.
  std::map<NodeRole, std::set<uint32_t>> recv_rank_ids_send_host_name_;
  // Record rank id of the nodes which queried host name.
  std::map<NodeRole, std::set<uint32_t>> recv_rank_ids_query_host_name_;

  // Record unique id of every group of every node role, key: node role, value: {key: group name, value: unique id}.
  std::map<NodeRole, std::map<std::string, std::string>> unique_id_groups_;

  // Record node number of each node role.
  std::map<NodeRole, uint32_t> node_nums_;

  std::mutex nodes_finish_trans_mutex_;
  // Key: actor set name, value: the set of rank ids of nodes who finish transform this actor.
  std::map<std::string, std::set<uint32_t>> nodes_finish_trans_;
  std::atomic_bool node_timeout_{false};

  std::unique_ptr<FileConfiguration> recovery_storage_{nullptr};
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_PS_SCHEDULER_NODE_H_
