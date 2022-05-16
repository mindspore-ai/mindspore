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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMPUTE_GRAPH_NODE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMPUTE_GRAPH_NODE_H_

#include <string>
#include <memory>
#include <thread>
#include "distributed/cluster/topology/common.h"
#include "distributed/rpc/tcp/tcp_client.h"
#include "distributed/cluster/topology/node_base.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// The ComputeGraphNode is a separate process representing a sub-graph of the distributed computation graph.
class ComputeGraphNode : public NodeBase {
 public:
  explicit ComputeGraphNode(const std::string &node_id) : NodeBase(node_id), authenticated_(false), enable_hb_(false) {}
  ~ComputeGraphNode() override = default;

  bool Initialize() override;
  bool Initialized() override;

  bool Finalize(bool force = false) override;

  // Send the specified message to the meta server node.
  bool SendMessageToMSN(const std::string msg_name, const std::string &msg_body);

  // Query the specified message from the meta server node according to the given message name.
  // Returns nullptr if no message returned after timeout.
  std::shared_ptr<std::string> RetrieveMessageFromMSN(const std::string &msg_name, uint32_t timeout = 5);

  // Write and read user defined metadata to the meta server node.
  bool PutMetadata(const std::string &name, const std::string &value);
  std::string GetMetadata(const std::string &name, uint32_t timeout = 5);

 private:
  // Send the register message to the meta server node when this node process startup.
  bool Register();

  // Send the unregister message to the meta server node.
  bool Unregister();

  // Send the heartbeat message to the meta server node.
  bool Heartbeat();

  // The meta server address used to synchronize metadata with other compute graph nodes.
  MetaServerAddress meta_server_addr_;

  // The TCP client is used to send messages to meta server node.
  std::unique_ptr<rpc::TCPClient> tcp_client_;

  // Incidate whether this node is authenticated by meta server node.
  std::atomic<bool> authenticated_;

  // The heartbeat thread from compute graph node to meta server node.
  std::thread heartbeat_;

  // Indicate whether the heartbeat thread is running.
  bool enable_hb_;
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMPUTE_GRAPH_NODE_H_
