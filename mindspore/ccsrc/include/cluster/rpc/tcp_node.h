/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_TCP_NODE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_TCP_NODE_H_

#include <string>
#include <memory>
#include <thread>
#include <vector>
#include <map>
#include <shared_mutex>
#include <limits>
#include "include/cluster/topology/common.h"
#include "include/cluster/rpc/tcp_client.h"
#include "include/cluster/topology/node_base.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
class CLUSTER_EXPORT TcpNodeBase : public NodeBase {
 public:
  TcpNodeBase(const std::string &node_id, const std::string &role, const std::string &address = "",
              size_t timeout = kDefaultTopoTimeOut)
      : NodeBase(node_id, role) {
    address_id_ = address;
    timeout_ = timeout;
  }
  ~TcpNodeBase() override;
  bool Initialize() override;
  bool Initialized() override;
  bool Finalize(bool force = false) override;
  // Send the specified message to the meta server node.
  bool SendMessageToMSN(const std::string msg_name, const std::string &msg_body, bool sync = true);

  // Write and read user defined metadata to the meta server node. value is string.
  bool PutMetadata(const std::string &name, const std::string &value, bool sync = true);
  // Write and read user defined metadata to the meta server node. value is ptr.
  bool PutMetadata(const std::string &name, const void *value, const size_t &size);
  // Get metadata from the meta server node.
  std::string GetMetadata(const std::string &name, uint32_t timeout = 5);
  // Get metadata from the meta server node by polling.
  std::string ReTryGetMetadata(const std::string &name, uint32_t timeout = 5);
  // Delete metadata from the meta server node.
  bool DeleteMetadata(const std::string &name, uint32_t timeout = 5);
  // Accumulate the values corresponding to the key. value is a numerical will exists as a string.
  int64_t AddMetadata(const std::string &name, int64_t value);
  // Send the register message to the meta server node when this node process startup.
  bool RegisterTcpClient();

  bool ReConnectWithTimeout(const std::function<bool(void)> &func, const std::string &error, size_t time_out);
  // ReConnect to the meta server node.
  bool ReConnect();

  // Return client ip of this node which is used for cluster building.
  const std::string &client_ip() const { return client_ip_; }

  // Return tcp client which sends message to server node.
  const std::unique_ptr<rpc::TCPClient> &tcp_client() const { return tcp_client_; }

  // Get all the hostnames of one type of roles.
  std::vector<std::string> GetHostNames(const std::string &role);

  // Query the specified message from the meta server node according to the given message name.
  // Returns nullptr if no message returned after timeout.
  std::shared_ptr<std::string> RetrieveMessageFromMSN(const std::string &msg_name, uint32_t timeout = 5);
  std::shared_ptr<std::string> RetrieveMessageFromMSN(const std::string &msg_name, const std::string &msg_body,
                                                      uint32_t timeout = 5);

  // The meta server address used to synchronize metadata with other compute graph nodes.
  MetaServerAddress meta_server_addr_;

  // The TCP client is used to send messages to server node.
  std::unique_ptr<rpc::TCPClient> tcp_client_;
  std::string address_id_;
  // unit: milliseconds
  size_t timeout_;
  std::mutex mutex_;
  std::string client_ip_;
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_TCP_NODE_H_
