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
#include "distributed/cluster/topology/node_base.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// The ComputeGraphNode is a separate process representing a sub-graph of the distributed computation graph.
class ComputeGraphNode : public NodeBase {
 public:
  explicit ComputeGraphNode(const std::string &node_id) : NodeBase(node_id) {}
  ~ComputeGraphNode() override = default;

 private:
  // Send the heartbeat message to the meta server node.
  bool Heartbeat();
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMPUTE_GRAPH_NODE_H_
