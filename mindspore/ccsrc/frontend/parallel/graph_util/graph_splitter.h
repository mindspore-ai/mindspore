/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_SPLITTER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_SPLITTER_H_

#include <map>
#include <tuple>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "ir/value.h"
#include "ir/graph_utils.h"
#include "base/base.h"
#include "ir/func_graph.h"
#include "distributed/constants.h"

namespace mindspore {
namespace parallel {
// The distributed label of the operators(kernel) used to split graph with send/recv nodes.
struct OperatorLabel {
  uint32_t rank_id;
  std::string ms_role;

  bool operator<(const OperatorLabel &label) const;
  bool operator==(const OperatorLabel &label) const;
  bool operator!=(const OperatorLabel &label) const;
  std::string to_string() const;
};

// The judging functions for different modes because the logic will change under different execution modes. If labels
// are not matched, the send and recv nodes should be inserted.
using LabelMatchingFunc = std::function<bool(const OperatorLabel &, const OperatorLabel &)>;
inline bool MatchLabelForPSMode(const OperatorLabel &label1, const OperatorLabel &label2) {
  // In Parameter Server training mode, Workers have the same labels regardless of their rank id.
  bool both_worker = (label1.ms_role == label2.ms_role) && (label1.ms_role == distributed::kEnvRoleOfWorker);
  bool all_match = (label1.rank_id == label2.rank_id) && (label1.ms_role == label2.ms_role);
  if (both_worker || all_match) {
    return true;
  }
  return false;
}
const std::map<distributed::DistributedExecutionMode, LabelMatchingFunc> kLabelMatchingFuncMap = {
  {distributed::DistributedExecutionMode::kPSMode, MatchLabelForPSMode}};

// Split graph segment which is generated according to the topo sort of the graph.
struct SplitGraphSegment {
  std::vector<AnfNodePtr> nodes;
  OperatorLabel label;
};

// The cross-process data flow edge.
struct InterProcessOpEdge {
  AnfNodePtr from_node;
  AnfNodePtr to_node;

  bool operator<(const InterProcessOpEdge &e) const { return to_string() < e.to_string(); }

  std::string to_string() const { return from_node->fullname_with_scope() + "->" + to_node->fullname_with_scope(); }
};

// The connection relationship for Send and Recv nodes.
// First element represents the Send node.
// Second element represents the Recv node.
// Third element represents a node which uses the Recv node as a input.
// Fourth element represents the input index of the user node.
using InterProcessOpPair = std::tuple<CNodePtr, CNodePtr, CNodePtr, int>;
using InterProcessOpEdgesInfo = std::map<InterProcessOpEdge, InterProcessOpPair>;

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

  // Traverse all nodes and split these nodes to multiple segments according to the split label.
  std::vector<SplitGraphSegment> GenerateSplitSegments();

  // Generate Send-Recv pairs for the nodes which has different split.
  // Because nodes with different split label from this proccess's with be on another machine, we use Send-Recv pairs to
  // do network communication.
  InterProcessOpEdgesInfo GenerateInterProcessOperators();

  // Eliminate nodes which are on other machine's graphs and add control edges for nodes of this process's graph.
  void SplitGraph(const std::vector<SplitGraphSegment> &segments, const InterProcessOpEdgesInfo &comm_edges);

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

  // The inter-process edge between two nodes should be like this:
  // input-->Send-->Recv-->peer.
  // Send node takes 'input' node as one input, its output's abstract is the same as a scalar value tensor's to save
  // memory. Recv node takes a scalar value tensor as one input, its output's abstract is the same as the 'input'
  // node's.
  CNodePtr GenerateSendNode(const AnfNodePtr &input, const AnfNodePtr &peer);
  CNodePtr GenerateRecvNode(const AnfNodePtr &input, const AnfNodePtr &peer);

  // Set attributes for send and recv node. These attributes is used in other stages like graph compiling, rpc route,
  // etc.
  void SetSendNodeAttr(const AnfNodePtr &send_node, const AnfNodePtr &send_from_node, const AnfNodePtr &send_to_node);
  void SetRecvNodeAttr(const AnfNodePtr &recv_node, const AnfNodePtr &recv_from_node, const AnfNodePtr &recv_to_node);

  // Segments will be independent with each other after the graph is cut, so in-degrees and out-degrees of each segment
  // should be connected with control edges in case that the nodes are optimized out.
  std::vector<AnfNodePtr> FindInterProcessInDegree(const std::vector<AnfNodePtr> &nodes,
                                                   const InterProcessOpEdgesInfo &comm_edges);
  std::vector<AnfNodePtr> FindInterProcessOutDegree(const std::vector<AnfNodePtr> &nodes,
                                                    const InterProcessOpEdgesInfo &comm_edges);

  // Judge whether two nodes have the same distributed label.
  bool IsNodesWithSameLabel(const AnfNodePtr &node1, const AnfNodePtr &node2);

  // This method creates a scalar tensor. Its type is the same as the origin_node's output if use_origin_node is set
  // true.
  // Normally it is used to connect the edges for send/recv nodes.
  ValueNodePtr GenerateMockValueNode(bool use_origin_node, const AnfNodePtr &origin_node = nullptr);

  FuncGraphPtr func_graph_;

  // The label of this process which consists of its rank and role.
  OperatorLabel this_process_label_;

  // For each mode, there is a default label. Every node in the graph should be launched on the process with this label
  // defaultly unless it has a different split label.
  OperatorLabel default_label_;

  // The map of all nodes in the graph to their distributed split label.
  std::map<AnfNodePtr, OperatorLabel> node_labels_;
};
using GraphSplitterPtr = std::shared_ptr<GraphSplitter>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_SPLITTER_H_
