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
#include <vector>
#include <map>
#include <shared_mutex>
#include "include/backend/distributed/cluster/topology/common.h"
#include "include/backend/distributed/rpc/tcp/tcp_client.h"
#include "include/backend/distributed/cluster/topology/node_base.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// The ComputeGraphNode is a separate process representing a sub-graph of the distributed computation graph.
class BACKEND_EXPORT ComputeGraphNode : public NodeBase {
 public:
  ComputeGraphNode(const std::string &node_id, const std::string &role)
      : NodeBase(node_id, role), authenticated_(false), enable_hb_(false) {}
  ~ComputeGraphNode() override;

  bool Initialize() override;
  bool Initialized() override;

  bool Finalize(bool force = false) override;

  // Send the specified message to the meta server node.
  bool SendMessageToMSN(const std::string msg_name, const std::string &msg_body, bool sync = true);

  // Query the specified message from the meta server node according to the given message name.
  // Returns nullptr if no message returned after timeout.
  std::shared_ptr<std::string> RetrieveMessageFromMSN(const std::string &msg_name, uint32_t timeout = 5);

  // Write and read user defined metadata to the meta server node.
  bool PutMetadata(const std::string &name, const std::string &value, bool sync = true);
  bool PutMetadata(const std::string &name, const void *value, const size_t &size);

  std::string GetMetadata(const std::string &name, uint32_t timeout = 5);

  bool DeleteMetadata(const std::string &name, uint32_t timeout = 5);

  // Exchange metadata(name:value) between all the compute graph nodes.
  // The transaction of the exchange process is guaranteed.
  bool ExchangeMetadata(const std::string &biz, const size_t &rank_size, const std::vector<std::string> &names_prefix,
                        const std::vector<std::string> &values, std::map<std::string, std::string> *results,
                        uint32_t timeout = 90);

  // Get all the hostnames of compute graph nodes.
  std::vector<std::string> GetHostNames(const std::string &role);

  void set_abnormal_callback(std::shared_ptr<std::function<void(void)>> abnormal_callback) override;

 private:
  // Send the register message to the meta server node when this node process startup.
  bool Register();

  // Send the unregister message to the meta server node.
  bool Unregister();

  // Send the heartbeat message to the meta server node.
  bool Heartbeat();

  // Call the `Reconnect` function if the input func execution failed.
  bool ReconnectIfNeeded(const std::function<bool(void)> &func, const std::string &error, size_t retry);

  // Reconnect to the meta server node.
  bool Reconnect();

  std::shared_ptr<std::string> RetrieveMessageFromMSN(const std::string &msg_name, const std::string &msg_body,
                                                      uint32_t timeout = 5);

  // The meta server address used to synchronize metadata with other compute graph nodes.
  MetaServerAddress meta_server_addr_;

  // The TCP client is used to send messages to meta server node.
  std::unique_ptr<rpc::TCPClient> tcp_client_;

  // The TCP client used to send heartbeat to meta server.
  std::unique_ptr<rpc::TCPClient> hb_client_;

  // Incidate whether this node is authenticated by meta server node.
  std::atomic<bool> authenticated_;

  // The heartbeat thread from compute graph node to meta server node.
  std::thread heartbeat_;

  // Indicate whether the heartbeat thread is running.
  bool enable_hb_;

  std::shared_ptr<std::function<void(void)>> abnormal_callback_;

  mutable std::shared_mutex exchange_meta_mutex_;
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMPUTE_GRAPH_NODE_H_
