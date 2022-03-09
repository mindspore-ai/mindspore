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

// The MetaServerNode is a separate process representing the meta server node which stores all the metadata and status
// of computation graph nodes.
class MetaServerNode : public NodeBase {
 public:
  explicit MetaServerNode(const std::string &node_id) : NodeBase(node_id) {}
  ~MetaServerNode() override = default;

  bool Initialize() override;
  bool Finalize() override;

 private:
  // Create and init the tcp server.
  bool InitTCPServer();

  // Handle the message received by the tcp server.
  void HandleMessage(const std::shared_ptr<MessageBase> &message);

  // Process the received register message sent from compute graph nodes.
  void ProcessRegister(const std::shared_ptr<MessageBase> &message);

  // Process the received heartbeat message sent from compute graph nodes.
  void ProcessHeartbeat(const std::shared_ptr<MessageBase> &message);

  // The meta server address used to manage the tcp server.
  MetaServerAddress meta_server_addr_;

  // The TCP server is used to process messages sent from compute graph nodes.
  std::unique_ptr<rpc::TCPServer> tcp_server_;

  // All the handlers for compute graph node's messages processing.
  std::map<MessageName, rpc::MessageHandler> message_handlers_;

  // Stores the registered compute graph nodes.
  std::map<std::string, std::shared_ptr<ComputeGraphNodeState>> nodes_;
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_META_SERVER_NODE_H_
