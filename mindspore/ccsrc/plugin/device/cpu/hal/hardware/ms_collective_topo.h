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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MS_COLLECTIVE_TOPO_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MS_COLLECTIVE_TOPO_H_

#include <string>
#include <memory>
#include <queue>
#include <map>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "actor/msg.h"
#include "include/backend/distributed/rpc/tcp/tcp_client.h"
#include "include/backend/distributed/rpc/tcp/tcp_server.h"
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"

namespace mindspore {
namespace device {
namespace cpu {
class TopologyNode {
 public:
  TopologyNode(size_t total_node_num, const std::shared_ptr<distributed::cluster::topology::ComputeGraphNode> &cgn)
      : rank_id_(-1), total_node_num_(total_node_num), cgn_(cgn), initialized_(false) {}
  ~TopologyNode() = default;

  // Init this topology node includes build tcp clients and server.
  bool Initialize();

  // Indicates whether this topo node has been initialized successfully.
  bool Initialized();

  // Destroy tcp clients and the tcp server.
  bool Finalize();

  // Send data asynchronously to the specified rank node.
  bool SendAsync(size_t rank_id, const void *data, size_t size);

  // Wait for all the pending sending tasks to the rank_id to be finished.
  bool WaitForSend(size_t rank_id);

  // Receive data asynchronously from the specified rank node.
  bool Receive(size_t rank_id, MessageBase **message, size_t timeout = 15);

  size_t rank_id() const;

  size_t rank_size() const;

 private:
  // Handle the message received by the tcp server.
  MessageBase *const HandleMessage(MessageBase *const message);

  // The rank id of this node in the collective communication topology.
  size_t rank_id_;

  // The total topology node number.
  size_t total_node_num_;

  // The received messages sent from other rank nodes.
  std::map<size_t, std::queue<MessageBase *> *> received_messages_;

  // Synchronizer for receive message queue reads and writes.
  std::mutex cond_mutex_;
  std::condition_variable cond_var_;

  // The tcp clients for other ranks, each client is responsible for sending message to the specified rank node.
  std::map<size_t, distributed::rpc::TCPClient *> tcp_clients_;

  // Maintain the tcp addresses for other nodes if needed.
  std::map<size_t, std::string> node_addresses_;

  // The tcp server which is responsible for receiving messages from other rank nodes.
  std::unique_ptr<distributed::rpc::TCPServer> tcp_server_;

  // The compute grpah node used to exchange the topology meta info(eg. ip:port) between topology nodes.
  std::shared_ptr<distributed::cluster::topology::ComputeGraphNode> cgn_;

  std::atomic<bool> initialized_;

  std::thread init_thread_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MS_COLLECTIVE_TOPO_H_
