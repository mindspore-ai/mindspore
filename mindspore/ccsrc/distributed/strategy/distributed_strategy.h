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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_STRATEGY_DISTRIBUTED_STRATEGY_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_STRATEGY_DISTRIBUTED_STRATEGY_H_

#include <map>
#include <tuple>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "ir/func_graph.h"
#include "include/backend/distributed/constants.h"

namespace mindspore {
namespace distributed {
namespace strategy {
// Base class for different execution modes. It builds distributed graphs, optimize execution performance, etc.
class DistributedStrategy {
 public:
  // Pass the dyed graph, node labels, process's role and rank id to construct execution mode.
  explicit DistributedStrategy(const FuncGraphPtr &func_graph, NodeLabels *node_labels, uint32_t rank_id,
                               const std::string &role)
      : func_graph_(func_graph), node_labels_(node_labels), rank_id_(rank_id), role_(role) {}
  virtual ~DistributedStrategy() = default;

  // Prebuild the distributed graph to prepare for splitting graph. For example,adding extra accumulation nodes, replace
  // gradient input of optimizer nodes, dying new created nodes so that common split implementation could applied.
  // Input 'node_labels' represents node labels of the origin graph. This method could modify this map.
  virtual void PreBuildDistributedGraph() {}

  // Do rpc node fusion to decrease the overhead of network communication.
  virtual FusedInterProcessOpPairMap DoRpcNodeFusion(InterProcessOpEdgesInfo *comm_edges_ptr) { return {}; }

  // Postbuild the distributed graph after splitting graph. For example, adding extra edges to the split graph.
  // Input 'node_labels' represents node labels of the split graph.
  // Input 'comm_edges' represents the inter-process edges generated after splitting the graph.
  virtual void PostBuildDistributedGraph(const InterProcessOpEdgesInfo &comm_edges) {}
  virtual void PostBuildDistributedGraph(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs) {}

 protected:
  FuncGraphPtr func_graph_;

  // The node label set by graph splitter. It could be modified by DistributedStrategy.
  NodeLabels *node_labels_;

  // Rank id and node role of this process. They are used to dye graph with different labels, help build split graph,
  // etc.
  uint32_t rank_id_;
  std::string role_;
};
}  // namespace strategy
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_STRATEGY_DISTRIBUTED_STRATEGY_H_
