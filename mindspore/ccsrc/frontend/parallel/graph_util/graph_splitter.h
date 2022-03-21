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
#include "include/common/utils/utils.h"
#include "ir/func_graph.h"
#include "distributed/constants.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "distributed/cluster/cluster_context.h"
#else
#include "distributed/cluster/dummy_cluster_context.h"
#endif

namespace mindspore {
namespace parallel {
using distributed::cluster::ClusterContext;

// The distributed label of the operators(kernel) used to split graph with send/recv nodes.
struct OperatorLabel {
  uint32_t rank_id;
  std::string ms_role;

  bool operator<(const OperatorLabel &label) const;
  bool operator==(const OperatorLabel &label) const;
  bool operator!=(const OperatorLabel &label) const;
  std::string to_string() const;
};

// The map of all nodes in the graph to their distributed split label.
using NodeLabels = std::map<AnfNodePtr, OperatorLabel>;

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
  AnfNodePtr src_node;
  OperatorLabel src_label;
  AnfNodePtr dst_node;
  OperatorLabel dst_label;

  bool operator==(const InterProcessOpEdge &e) const { return to_string() == e.to_string(); }

  bool operator<(const InterProcessOpEdge &e) const { return to_string() < e.to_string(); }

  std::string to_string() const {
    return src_node->fullname_with_scope() + "_" + src_label.to_string() + "->" + dst_node->fullname_with_scope() +
           "_" + dst_label.to_string();
  }
};

// The connection relationship for Send and Recv nodes.
// First element represents the Send node.
// Second element represents the Recv node.
// Third element represents a node which uses the Recv node as a input.
// Fourth element represents the input index of the user node.
using InterProcessOpPair = std::tuple<CNodePtr, CNodePtr, CNodePtr, int>;
using InterProcessOpEdgesInfo = std::map<InterProcessOpEdge, InterProcessOpPair>;

constexpr char kAttrUpdateParameter[] = "update_parameter";
constexpr char kAttrParameterInputIndex[] = "parameter_input_index";
constexpr char kAttrGradientInputIndex[] = "gradient_input_index";
constexpr char kAttrIndicesInputIndex[] = "indices_input_index";

constexpr char kAttrGradientType[] = "gradient_type";
constexpr char kDenseGradient[] = "dense_gradient";
constexpr char kSparseGradient[] = "sparse_gradient";
// The accumulator operator names for different gradient types.
const std::map<std::string, std::string> kGradTypeToAccumOpName = {
  {kDenseGradient, kAddNOpName},
  {kSparseGradient, kConcatOpName},
};

// Node which is not physically on this process should be created for splitting graph implementation. This could be
// considered as a virtual node which will be elimimated after splitting graph. For example, for server in PS mode, some
// virtual nodes which are launched on the workers should be created as gradient accumulation nodes' inputs:
// VirtualNode  VirtualNode  RealBackwardNode
//      |            |               |
//      |            |               |
//      |            |               |
//       \           |               /
//        \          |              /
//         \         |             /
//          \        |            /
//         GradientAccumulationNode
constexpr char kVirtualNode[] = "VirtualNode";

// This method creates a fake tensor. Its type is the same as the origin_node's output if use_origin_node is set
// true.
// Normally it is used to connect the edges for send/recv nodes.
ValueNodePtr CreateFakeValueNode(bool use_origin_node, const AnfNodePtr &origin_node = nullptr);

// Set attributes for send and recv node. These attributes is used in other stages like graph compiling, rpc route,
// etc.
void SetSendNodeAttr(const AnfNodePtr &send_node, const InterProcessOpEdge &inter_process_edge);
void SetRecvNodeAttr(const AnfNodePtr &recv_node, const InterProcessOpEdge &inter_process_edge);

// The inter-process edge between two nodes should be like this:
// input-->Send-->Recv-->peer.
// Send node takes 'input' node as one input, its output's abstract is the same as a scalar value tensor's to save
// memory. Recv node takes a scalar value tensor as one input, its output's abstract is the same as the 'input'
// node's.
CNodePtr CreateSendNode(const FuncGraphPtr &func_graph, const InterProcessOpEdge &inter_process_edge);
CNodePtr CreateRecvNode(const FuncGraphPtr &func_graph, const InterProcessOpEdge &inter_process_edge);

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

  // Segments will be independent with each other after the graph is cut, so in-degrees and out-degrees of each segment
  // should be connected with control edges in case that the nodes are optimized out.
  std::vector<AnfNodePtr> FindInterProcessInDegree(const std::vector<AnfNodePtr> &nodes,
                                                   const InterProcessOpEdgesInfo &comm_edges);
  std::vector<AnfNodePtr> FindInterProcessOutDegree(const std::vector<AnfNodePtr> &nodes,
                                                    const InterProcessOpEdgesInfo &comm_edges);

  // Judge whether two nodes have the same distributed label.
  bool IsNodesWithSameLabel(const AnfNodePtr &node1, const AnfNodePtr &node2);

  FuncGraphPtr func_graph_;

  // The label of this process which consists of its rank and role.
  OperatorLabel this_process_label_;

  // For each mode, there is a default label. Every node in the graph should be launched on the process with this label
  // defaultly unless it has a different split label.
  OperatorLabel default_label_;

  // The map of all nodes in the graph to their distributed split label.
  NodeLabels node_labels_;
};
using GraphSplitterPtr = std::shared_ptr<GraphSplitter>;

// Base class for different execution modes. It builds distributed graphs, optimize execution performance, etc.
class DistributedExecutionMode {
 public:
  // Pass the dyed graph, node labels, process's role and rank id to construct execution mode.
  explicit DistributedExecutionMode(const FuncGraphPtr &func_graph, NodeLabels *node_labels, uint32_t rank_id,
                                    const std::string &role)
      : func_graph_(func_graph), node_labels_(node_labels), rank_id_(rank_id), role_(role) {}
  virtual ~DistributedExecutionMode() = default;

  // Prebuild the distributed graph to prepare for splitting graph. For example,adding extra accumulation nodes, replace
  // gradient input of optimizer nodes, dying new created nodes so that common split implementation could applied.
  // Input 'node_labels' represents node labels of the origin graph. This method could modify this map.
  virtual void PreBuildDistributedGraph() {}

  // Postbuild the distributed graph after splitting graph. For example, adding extra edges to the split graph.
  // Input 'node_labels' represents node labels of the split graph.
  // Input 'comm_edges' represents the inter-process edges generated after splitting the graph.
  virtual void PostBuildDistributedGraph(const InterProcessOpEdgesInfo &comm_edges) {}

 protected:
  FuncGraphPtr func_graph_;

  // The node label set by graph splitter. It could be modified by DistributedExecutionMode.
  NodeLabels *node_labels_;

  // Rank id and node role of this process. They are used to dye graph with different labels, help build split graph,
  // etc.
  uint32_t rank_id_;
  std::string role_;
};

// Gradient accumulation node is needed when the worker number is equal to or greater than 2.
constexpr uint32_t kMinGradAccumWorkerNum = 2;

// The execution of Parameter Server mode.
class ParameterServerMode : public DistributedExecutionMode {
 public:
  explicit ParameterServerMode(const FuncGraphPtr &func_graph, NodeLabels *node_labels, uint32_t rank_id,
                               const std::string &role)
      : DistributedExecutionMode(func_graph, node_labels, rank_id, role) {}
  ~ParameterServerMode() = default;

  void PreBuildDistributedGraph() override;
  void PostBuildDistributedGraph(const InterProcessOpEdgesInfo &comm_edges) override;

 private:
  // Process optimizers split to the parameter server.
  void ProcessForSplittedOptimizer();

  // Filter out all optimizer nodes which are set on parameter server from the graph.
  std::vector<CNodePtr> FilterServerAwareOptimizerList(const std::vector<AnfNodePtr> &nodes);

  // Create gradients accumulator with mean operator for the given optimizer. It could be sparse or dense gradients.
  // 'total_gradient_number' represents how many workers' gradients will be accumulated for this optimizer.
  // The return value is a pair of accumulation node to RealDiv node.
  std::pair<CNodePtr, CNodePtr> CreateNodesForGradAccumulation(const AnfNodePtr &gradient_input,
                                                               size_t gradient_input_index,
                                                               const std::string &gradient_type,
                                                               size_t total_gradient_number);

  // Normally after gradients accumulation, the mean value should be calculated.
  CNodePtr CreateGradMeanNode(const AnfNodePtr &gradient, size_t divisor);

  // Create node with multiple inputs. Some of the inputs could be fake nodes.
  // 'many_to_one_node_name' represents the name of the node to be created.
  // 'real_input' represents the input which is already in the func_graph_. Other inputs will be created as this input.
  // 'index_of_real_input': the input index of 'real_input' of this new created node: 'many_to_one_node_name'.
  // 'total_inputs_number': the total inputs number of the created node.
  CNodePtr CreateNodeWithInterProcessEdgeOnPServer(const std::string &many_to_one_node_name,
                                                   const AnfNodePtr &real_input, size_t index_of_real_input,
                                                   uint32_t total_inputs_number);
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_SPLITTER_H_
