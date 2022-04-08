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

#include <time.h>
#include <string>
#include <memory>
#include <map>
#include <thread>
#include <chrono>
#include <shared_mutex>
#include "distributed/cluster/topology/common.h"
#include "distributed/rpc/tcp/tcp_server.h"
#include "distributed/cluster/topology/node_base.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// Record the state of the compute graph node.
struct ComputeGraphNodeState {
  explicit ComputeGraphNodeState(std::string id) { node_id = id; }
  std::string node_id;

  // The timestamp of last heartbeat.
  // This timestamp is considered the health state of the node.
  time_t last_update;
};

// Indicates the state of the cluster physical topology.
enum class TopoState {
  // All the nodes of this cluster are in the process of starting up.
  kInitializing = 0,

  // All the nodes of this cluster has been started and registered to the meta server node successfully.
  kInitialized = 1,

  // The topo of this cluster failed to construct at specified time.
  kFailed = 2
};

// The MetaServerNode is a separate process representing the meta server node which stores all the metadata and status
// of computation graph nodes.
class MetaServerNode : public NodeBase {
 public:
  explicit MetaServerNode(const std::string &node_id, const size_t &node_num)
      : NodeBase(node_id), total_node_num_(node_num), topo_state_(TopoState::kInitializing), enable_monitor_(true) {}
  ~MetaServerNode() override = default;

  bool Initialize() override;
  bool Finalize() override;

  // Get the current topology state.
  TopoState TopologyState();

  // Get the number of alive compute graph node.
  size_t GetAliveNodeNum();

  // Register the message handler for the user defined message which is specified by the `name` parameter.
  bool RegisterMessageHandler(const std::string &name,
                              std::shared_ptr<std::function<std::string(const std::string &)>> handler);

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

  // Maintain the state which is type of `TopoState` of this cluster topology.
  void UpdateTopoState();

  // The meta server address used to manage the tcp server.
  MetaServerAddress meta_server_addr_;

  // The TCP server is used to process messages sent from compute graph nodes.
  std::unique_ptr<rpc::TCPServer> tcp_server_;

  // All the handlers for compute graph node's system messages processing.
  // The `system` means the built-in messages used for cluster topology construction.
  std::map<MessageName, rpc::MessageHandler> system_msg_handlers_;

  // All the handlers for compute graph node's user-defined messages processing.
  // The `user-defined` means that this kind of message is user defined and has customized message handler.
  std::map<std::string, std::shared_ptr<std::function<std::string(const std::string &)>>> message_handlers_;

  // Stores the registered compute graph nodes.
  std::map<std::string, std::shared_ptr<ComputeGraphNodeState>> nodes_;

  mutable std::shared_mutex nodes_mutex_;

  // The total legal number of compute graph nodes.
  size_t total_node_num_;

  // The state of the topology consisting of compute graph nodes.
  TopoState topo_state_;

  // The monitor thread for update the topo state.
  std::thread topo_monitor_;

  // The switch for the topo monitor thread.
  std::atomic<bool> enable_monitor_;

  // The start time of this meta server node.
  std::chrono::high_resolution_clock::time_point start_time_;
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_META_SERVER_NODE_H_
