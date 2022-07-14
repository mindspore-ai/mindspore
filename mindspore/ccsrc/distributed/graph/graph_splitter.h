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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_GRAPH_GRAPH_SPLITTER_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_GRAPH_GRAPH_SPLITTER_H_

#include <map>
#include <tuple>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "distributed/graph/graph_utils.h"
#include "distributed/strategy/distributed_strategy.h"

namespace mindspore {
namespace distributed {
namespace graph {
using distributed::strategy::DistributedStrategy;

// The class is used as an action in pipeline. It will process the graph and split the nodes to each process in the
// cluster.
class GraphSplitter {
 public:
  GraphSplitter(const FuncGraphPtr &func_graph, uint32_t rank_id, const std::string &role);
  ~GraphSplitter();

  // Launch the action.
  void Run();

 private:
  // Dyeing the func_graph according to the split label passed by frontend. Only nodes with the same label will be dyed
  // with the same 'color'.
  void DyeGraph();

  // Create the execution mode.
  void CreateExecutionMode();

  // Traverse all nodes and split these nodes to multiple segments according to the split label.
  std::vector<SplitGraphSegment> GenerateSplitSegments();

  // Generate Send-Recv pairs for the nodes which has different split.
  // Because nodes with different split label from this proccess's with be on another machine, we use Send-Recv pairs to
  // do network communication.
  InterProcessOpEdgesInfo GenerateInterProcessOperators();

  // Eliminate nodes which are on other machine's graphs and add control edges for nodes of this process's graph.
  void SplitGraph(const std::vector<SplitGraphSegment> &segments, const InterProcessOpEdgesInfo &comm_edges);
  void SplitGraph(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs);

  // Split the graph but don't eliminate the nodes so that a global graph ir could be exported.
  void DumpDistributedGraph(const InterProcessOpEdgesInfo &comm_edges);

  // Return the split label of this node. Only CNode is supported for now.
  // If the node has no split label, return the label of this process, which means this node should be in this process's
  // graph.
  OperatorLabel GetSplitLabel(const AnfNodePtr &node);

  // Consider Node-X is the split node. Node-In is Node-X's one input, Node-Out takes Node-X as one input.
  // So the graph should be like this:
  // Node-In-->Node-X-->Node-Out.
  // After send and recv op is inserted, the graph should be:
  // Node-In-->Send-->Recv-->Node-X-->Send-->Recv-->Node-Out.
  // So method GenerateInterProcessOpsForNodeInputs is for generating Send-Recv pair between Node-In and Node-X.
  InterProcessOpEdgesInfo GenerateInterProcessOpsForNodeInputs(const AnfNodePtr &node);

  InterProcessEdgeLabel GenerateEdgeLabel(const AnfNodePtr &src_node, const AnfNodePtr &dst_node);

  // Segments will be independent with each other after the graph is cut, so in-degrees and out-degrees of each segment
  // should be connected with control edges in case that the nodes are optimized out.
  std::vector<AnfNodePtr> FindInterProcessInDegree(const std::vector<AnfNodePtr> &nodes,
                                                   const InterProcessOpEdgesInfo &comm_edges);
  std::vector<AnfNodePtr> FindInterProcessOutDegree(const std::vector<AnfNodePtr> &nodes,
                                                    const InterProcessOpEdgesInfo &comm_edges);

  // Generate in and out degrees list of the segments to add dependency between segments.
  InOutDegreeList GenerateInOutDegreeList(const std::vector<SplitGraphSegment> &segments,
                                          const InterProcessOpEdgesInfo &comm_edges);

  // For the segments on this process, dependency edges should be created so that they won't be optimized out.
  void AddDependencyBetweenSegments(const InOutDegreeList &in_out_degree_list);

  // Replace nodes inputs with Recv nodes to eliminate extra nodes not on this process.
  void EliminateExtraNodes(const InterProcessOpEdgesInfo &comm_edges);

  // Replace nodes inputs with Recv nodes.
  void ReplaceOriginNodesWithRecv(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs);

  // Add outputs edges for send nodes so that they won't be optimized out.
  void AddDependencyForSend(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs);

  // Judge whether two nodes have the same distributed label.
  bool IsNodesWithSameLabel(const AnfNodePtr &node1, const AnfNodePtr &node2);

  // Check whether need split distributed graph.
  bool NeedSplitGraph() const;

  FuncGraphPtr func_graph_;

  // Rank id and node role of this process. They are used to dye graph with different labels, help build split graph,
  // etc.
  uint32_t rank_id_;
  std::string role_;

  // Created according to the execution mode. Used to build the distributed graph.
  distributed::DistExecutionMode mode_;
  std::unique_ptr<DistributedStrategy> exec_mode_;

  // The label of this process which consists of its rank and role.
  OperatorLabel this_process_label_;

  // For each mode, there is a default label. Every node in the graph should be launched on the process with this label
  // defaultly unless it has a different split label.
  OperatorLabel default_label_;

  // The map of all nodes in the graph to their distributed split label.
  NodeLabels node_labels_;

  // Whether need to fuse rpc nodes.
  bool need_fuse_rpc_nodes_;
};
using GraphSplitterPtr = std::shared_ptr<GraphSplitter>;
}  // namespace graph
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_GRAPH_GRAPH_SPLITTER_H_
