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

#include <string>

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
  explicit NodeBase(const std::string &node_id) : node_id_(node_id) {}
  virtual ~NodeBase() = default;

 protected:
  // Each node process has a unique node id which is immutable during the life cycle of this node.
  // The node id is used for identify authentication during networking and process recovery.
  std::string node_id_;
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_NODE_BASE_H_
