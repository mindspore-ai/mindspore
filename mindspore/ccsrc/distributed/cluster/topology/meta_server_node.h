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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_META_SERVER_NODE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_META_SERVER_NODE_H_

#include <string>
#include <memory>
#include <map>
#include <thread>
#include <shared_mutex>
#include <unordered_map>
#include "include/backend/distributed/rpc/tcp/tcp_server.h"
#include "distributed/recovery/configuration.h"
#include "include/backend/distributed/cluster/topology/node_base.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// Indicates the state of compute graph node.
enum class NodeState {
  // This node is newly created and unauthenticated.
  kNew = 0,

  // This node has finished registration from meta server.
  kRegistered,

  // This node has finished unregistration from meta server.
  kUnregistered,

  // This node has timed out because there's no heartbeat message after `kNodeTimeout`.
  kTimeout
};

// Record the state of the compute graph node.
struct NodeInfo {
  explicit NodeInfo(const std::string &id) { node_id = id; }
  std::string node_id;

  // The local host name of this cluster node.
  std::string host_name;

  // The role name of this cluster node.
  std::string role;

  // The rank id of this cluster node(only for compute graph node).
  uint32_t rank_id{0};

  // The timestamp of last heartbeat.
  // This timestamp is considered the health state of the node.
  time_t last_update{0};

  // Maintain the state of the node.
  NodeState state{NodeState::kNew};
};

// The MetaServerNode is a separate process representing the meta server node which stores all the metadata and status
// of computation graph nodes.
class MetaServerNode : public NodeBase {
 public:
  explicit MetaServerNode(const std::string &node_id, const std::string &role, const size_t &node_num,
                          uint64_t node_timeout = kDefaultNodeTimeout)
      : NodeBase(node_id, role),
        total_node_num_(node_num),
        abnormal_node_num_(0),
        enable_monitor_(true),
        node_timeout_(node_timeout) {}
  ~MetaServerNode() override;

  bool Initialize() override;
  bool Initialized() override;

  bool Finalize(bool force = false) override;

  // Get the current topology state.
  TopoState TopologyState() const;

  // Get the number of alive compute graph node.
  size_t GetAliveNodeNum();

  // Register the message handler for the user defined message which is specified by the `name` parameter.
  bool RegisterMessageHandler(const std::string &name,
                              const std::shared_ptr<std::function<std::string(const std::string &)>> &handler);

 private:
  // Create and init the tcp server.
  bool InitTCPServer();

  // Handle the message received by the tcp server.
  MessageBase *const HandleMessage(MessageBase *const message);

  // Process the received register message sent from compute graph nodes.
  MessageBase *const ProcessRegister(MessageBase *const message);

  // Process the received unregister message sent from compute graph nodes.
  MessageBase *const ProcessUnregister(MessageBase *const message);

  // Process the received heartbeat message sent from compute graph nodes.
  MessageBase *const ProcessHeartbeat(MessageBase *const message);

  // Process user-defined metadata writing and reading requests.
  MessageBase *const ProcessWriteMetadata(MessageBase *const message);
  MessageBase *const ProcessReadMetadata(MessageBase *const message);
  MessageBase *const ProcessDeleteMetadata(MessageBase *const message);

  // Gather all the hostname of registered compute graph nodes.
  MessageBase *const ProcessGetHostNames(MessageBase *const message);

  // Maintain the state which is type of `TopoState` of this cluster topology.
  void UpdateTopoState();

  // Try to transition the state of cluster to be initialized.
  bool TransitionToInitialized();

  // For each computing graph node, port range should be assigned by meta server node for rpc servers to bind.
  void AssignPortRange();

  // Recover metadata from the configuration if recovery is enabled.
  bool Recovery();

  // Allocate a new valid rank id for new registered compute graph node.
  uint32_t AllocateRankId(const std::string &role);

  // Persist the required metadata of cluster into storage through configuration.
  bool Persist();

  // The meta server address used to manage the tcp server.
  MetaServerAddress meta_server_addr_;

  // The TCP server is used to process messages sent from compute graph nodes.
  std::unique_ptr<rpc::TCPServer> tcp_server_;

  // All the handlers for compute graph node's system messages processing.
  // The `system` means the built-in messages used for cluster topology construction.
  std::map<MessageName, MessageHandler> system_msg_handlers_;

  // All the handlers for compute graph node's user-defined messages processing.
  // The `user-defined` means that this kind of message is user defined and has customized message handler.
  std::map<std::string, std::shared_ptr<std::function<std::string(const std::string &)>>> message_handlers_;

  // Stores the registered compute graph nodes.
  std::map<std::string, std::shared_ptr<NodeInfo>> nodes_;

  mutable std::shared_mutex nodes_mutex_;

  // The total legal number of compute graph nodes.
  size_t total_node_num_;

  // The total number of abnormal(eg. timeout) compute graph nodes.
  size_t abnormal_node_num_;

  // The monitor thread for update the topo state.
  std::thread topo_monitor_;

  // The switch for the topo monitor thread.
  std::atomic<bool> enable_monitor_;

  // The metadata written and read by users.
  std::map<std::string, std::string> metadata_;

  mutable std::shared_mutex meta_mutex_;

  uint64_t node_timeout_;

  // A key-value pairs metadata config used for failover recovery if enabled.
  std::unique_ptr<recovery::Configuration> configuration_;

  // The next valid rank id for compute graph nodes.
  // Note that each role(group) has it's own rank id.
  std::map<std::string, std::atomic<uint32_t>> next_rank_ids_;
  mutable std::shared_mutex rank_mutex_;
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_META_SERVER_NODE_H_
