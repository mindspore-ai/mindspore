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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_NODE_BASE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_NODE_BASE_H_

#include <chrono>
#include <string>
#include <memory>
#include "include/backend/distributed/cluster/topology/common.h"
#include "include/backend/distributed/cluster/topology/utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// A node represents a separate process which is one node of the distributed computation graph or the meta-server
// process. The node abstraction is for the dynamic networking of the distributed computation graph, allowing
// distributed computation graphs to communicate with each other during runtime and automatic recovery of node
// processes.
class NodeBase {
 public:
  explicit NodeBase(const std::string &node_id, const std::string &role)
      : node_id_(node_id),
        rank_id_(-1),
        role_(role),
        finalized_(false),
        start_time_(Now()),
        topo_state_(TopoState::kInitializing) {}
  virtual ~NodeBase() = default;

  // Prepare the resources hold in this node.
  virtual bool Initialize() = 0;

  // Returns whether all the initialization work has been completed.
  virtual bool Initialized() = 0;

  // Release the resources hold in this node.
  // If the parameter force is set to true, this node will be finalized without waiting for unregister of all the
  // compute graph node.
  virtual bool Finalize(bool force = false) = 0;

  // Set the callback which will be called when the state of the cluster is abnormal.
  virtual void set_abnormal_callback(std::shared_ptr<std::function<void(void)>> abnormal_callback) {}

  std::string node_id() const { return node_id_; }

  void set_rank_id(uint32_t rank_id) { rank_id_ = rank_id; }
  uint32_t rank_id() const { return rank_id_; }

  std::string role() const { return role_; }

 protected:
  // Each node process has a unique node id which is immutable during the life cycle of this node.
  // The node id is used for identify authentication during networking and process recovery.
  std::string node_id_;

  // The rank id of this compute graph node process in the cluster.
  // The rank id is assigned by meta server node and starts from 0 to (node_num - 1).
  uint32_t rank_id_;

  // The role name of this node specified by the environment variable.
  std::string role_;

  // Indicates whether the finalize method of this node has been called.
  bool finalized_;

  // The start time of this meta server node.
  std::chrono::high_resolution_clock::time_point start_time_;

  // The state of the topology consisting of compute graph nodes.
  TopoState topo_state_;
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_NODE_BASE_H_
