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
class BACKEND_EXPORT PSSchedulerNode : public SchedulerNode {
 public:
  PSSchedulerNode() : worker_num_(ps::PSContext::instance()->worker_num()) { host_hash_names_.resize(worker_num_); }
  ~PSSchedulerNode() override = default;

 protected:
  // Override the scheduler node to remove the nofification from scheduler to other nodes.
  void RunRecovery() override;

 private:
  // Determine whether the registration request of the node should be rejected, the registration of the
  // alive node should be rejected.
  bool NeedRejectRegister(const NodeInfo &node_info) override { return node_info.is_alive; }

  // Register collective communication initialization service.
  void RegisterInitCollectCommServiceHandler() override;

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

  // Record received host hash name from workers.
  std::vector<size_t> host_hash_names_;
  // Record rank id of the nodes which sended host name.
  std::set<uint32_t> recv_rank_id_send_host_name_;
  // Record rank id of the nodes which queried host name.
  std::set<uint32_t> recv_rank_id_query_host_name_;

  // Record unique id of every group, key: group name, value: unique id.
  std::map<std::string, std::string> unique_id_group_;

  uint32_t worker_num_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_PS_SCHEDULER_NODE_H_
