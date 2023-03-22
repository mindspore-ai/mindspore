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
#include <set>
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
#include "include/backend/distributed/constants.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/cluster/cluster_context.h"
#else
#include "include/backend/distributed/cluster/dummy_cluster_context.h"
#endif
#include "frontend/parallel/cache_embedding/ps_embedding_cache_inserter.h"

namespace mindspore {
namespace parallel {
using distributed::cluster::ClusterContext;
using AnfNodePtrSet = std::set<AnfNodePtr>;

constexpr char kEnvNeedFusion[] = "fusion";

// The distributed label of the operators(kernel) used to split graph with send/recv nodes.
struct OperatorLabel {
  uint32_t rank_id;
  std::string ms_role;

  bool operator<(const OperatorLabel &label) const;
  bool operator==(const OperatorLabel &label) const;
  bool operator!=(const OperatorLabel &label) const;

  // Judge whether the labels are equal but with looser conditions according to different modes. For example, this
  // method returns true when comparing the workers in PS mode.
  bool LooseEqual(const OperatorLabel &label, distributed::DistExecutionMode mode) const;

  std::string to_string() const;
};

// The label for inter-process edges. This is used for classify the edges.
// For example, only edges with same label should be fused.
struct InterProcessEdgeLabel {
  std::string label_name;
};

// The map of all nodes in the graph to their distributed split label.
using NodeLabels = std::map<AnfNodePtr, OperatorLabel>;

// The list of data-sync node pairs.
using DataSyncNodePairList = std::vector<std::pair<CNodePtr, CNodePtr>>;

// The pair of control edge nodes.
using ControlEdgeNodePair = std::pair<CNodePtr, CNodePtr>;
// The pair list of control edge nodes.
using ControlEdgeNodePairList = std::vector<std::pair<CNodePtr, CNodePtr>>;

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
inline bool MatchLabelForParallelMode(const OperatorLabel &label1, const OperatorLabel &label2) {
  // When parallel mode is enabled by using MindSpore cluster, processes with the same role has the same label
  // regardless of their rank id.
  return (label1.ms_role == label2.ms_role);
}

const std::map<distributed::DistExecutionMode, LabelMatchingFunc> kLabelMatchingFuncMap = {
  {distributed::DistExecutionMode::kPSMode, MatchLabelForPSMode},
  {distributed::DistExecutionMode::kEmbeddingCacheMode, MatchLabelForPSMode},
  {distributed::DistExecutionMode::kParallelMode, MatchLabelForParallelMode}};

// Split graph segment which is generated according to the topo sort of the graph.
struct SplitGraphSegment {
  std::vector<AnfNodePtr> nodes;
  OperatorLabel label;
};

// The inter-process edge with nodes. This represents the edge between two nodes on two processes.
struct InterProcessOpEdge {
  // The peers of this edge with nodes and their labels.
  AnfNodePtr src_node;
  OperatorLabel src_label;
  AnfNodePtr dst_node;
  OperatorLabel dst_label;

  // The label of this inter-process edge.
  InterProcessEdgeLabel edge_label;

  bool operator==(const InterProcessOpEdge &e) const { return to_string() == e.to_string(); }

  bool operator<(const InterProcessOpEdge &e) const { return to_string() < e.to_string(); }

  std::string to_string() const {
    return src_node->fullname_with_scope() + "_" + src_label.to_string() + "->" + dst_node->fullname_with_scope() +
           "_" + dst_label.to_string();
  }
};

// The inter-process edge without nodes. This just represents communication edge between two processes.
struct InterProcessEdgeWithIndex {
  OperatorLabel src_label;
  OperatorLabel dst_label;

  // If there are multiple independent edges between two processes, after rpc node fusion with segments, multiple
  // InterProcessEdgeWithIndex will be generated. Index represents the segment index in this case.
  size_t index;

  bool operator==(const InterProcessEdgeWithIndex &e) const { return to_string() == e.to_string(); }

  bool operator<(const InterProcessEdgeWithIndex &e) const { return to_string() < e.to_string(); }

  std::string to_string() const {
    return src_label.to_string() + "->" + dst_label.to_string() + "_" + std::to_string(index);
  }
};

// The connection relationship for Send and Recv nodes.
// First element represents the Send node.
// Second element represents the Recv node.
// Third element represents a node which uses the Recv node as a input.
// Fourth element represents the input index of the user node.
using InterProcessOpPair = std::tuple<CNodePtr, CNodePtr, CNodePtr, int>;
using InterProcessOpEdgesInfo = std::map<InterProcessOpEdge, InterProcessOpPair>;

// The connection relationship for fused Send and Recv nodes.
// First element represents the fused Send node.
// Second element represents the fused Recv node.
// Third element represents the output index of the fused Recv node.
// Third element represents the user node which uses the fused Recv node output as an input.
// Fourth element represents the input index of the user node.
using FusedInterProcessOpPair = std::tuple<CNodePtr, CNodePtr, int, CNodePtr, int>;
using InterProcessOpPairMap = std::map<InterProcessEdgeWithIndex, std::vector<InterProcessOpPair>>;
using FusedInterProcessOpPairMap = std::map<InterProcessEdgeWithIndex, std::vector<FusedInterProcessOpPair>>;

// The list of in and out degrees of one segment.
using InOutDegreeList = std::vector<std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>>>;

constexpr char kPSOptimizerEdgeLabel[] = "ps_optimizer_edge_label";

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
ValueNodePtr CreateFakeValueNode(bool use_origin_node, const AnfNodePtr &origin_node = nullptr,
                                 bool use_fake_shape = true);

// Create a TupleGetItem node from a node with tuple output.
CNodePtr CreateTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node_with_tuple_output,
                                size_t item_index);

// Create a MakeTuple node from multiple inputs.
CNodePtr CreateMakeTupleNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &tuple_inputs);

// For some processes, the original output should be replaced with a node with the same abstract so error won't be
// raised in Python layer.
AnfNodePtr CreateReplacedOutputNode(const FuncGraphPtr &func_graph, const AnfNodePtr &origin_output);

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

// Calculate the index to segment number map.
std::map<size_t, size_t> GetRealIndexToSeg(const std::vector<size_t> &split_segment, size_t real_size);

bool IsOneOfRealGraphInput(const FuncGraphPtr &func_graph, const AnfNodePtr &input);

/**
 * @description: Generate the distributed strategy according to user configuration.
 * @return {distributed::DistExecutionMode}: The distributed strategy enum.
 */
distributed::DistExecutionMode GenerateStrategy();

/**
 * @description: Transform primal attributes of cnode to normal attributes.
 * @param {CNodePtr} &cnode: The cnode which has the primal attributes.
 * @return {void}
 */
void TransformPrimAttrToAttr(const CNodePtr &cnode);

/**
 * @description: Judge whether this node has label.
 * @param {AnfNodePtr} &node: AnfNode in a func_graph.
 * @return {bool}: Whether this node has label.
 */
bool NodeHasLabel(const AnfNodePtr &node);

/**
 * @description: Judge whether this graph has any label.
 * @param {FuncGraphPtr} &func_graph: The func_graph.
 * @return {bool}: Whether this graph has label.
 */
bool GraphHasLabel(const FuncGraphPtr &func_graph);

/**
 * @description: Get node list of side effect nodes in the func_graph.
 * @param {AnfNodePtrList} &nodes: All nodes of the func_graph.
 * @return {CNodePtrList}: Side effect node list.
 */
CNodePtrList GetSideEffectNodeList(const AnfNodePtrList &nodes);

/**
 * @description: Get reference inputs of the cnode.
 * @param {CNodePtr} &cnode: Node with side effect.
 * @return {AnfNodePtrList}: The reference inputs node list.
 */
AnfNodePtrList GetRefInputs(const CNodePtr &cnode);

/**
 * @description: Find the UpdateState node which is the user of the input cnode.
 * @param {FuncGraphPtr} &func_graph: The graph.
 * @param {CNodePtr} &cnode: The node with side effect.
 * @return {CNodePtr}: UpdateState node.
 */
CNodePtr FindNextUpdateStateNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

/**
 * @description: Create 'U' node which is the input of user created side effect nodes like RpcRecv, UpdateState, etc.
 * @return {CNodePtr}: UMonad node.
 */
ValueNodePtr CreateUMonadNode();

/**
 * @description: Create UpdateState node manually.
 * @param {FuncGraphPtr} &func_graph: The func_graph.
 * @param {AnfNodePtrList} &update_state_inputs: Inputs of UpdateState node. Normally first is UMonadNode, which is
 * created inside this method. the others are other side effect nodes passed by caller.
 * @return {CNodePtr}: UpdateState node.
 */
CNodePtr CreateUpdateStateNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &update_state_inputs);

// Filter out 'func_graph' nodes' dependency matrix to the specified target nodes set.
std::map<AnfNodePtr, AnfNodePtrSet> FilterDependencyToTargetNode(const FuncGraphPtr &func_graph,
                                                                 const AnfNodePtrSet &target_nodes);

// After a new node is added, the depended set should be updated to keep the minimal dependencies.
AnfNodePtrSet UpdateDependedSet(const AnfNodePtr &new_node, const AnfNodePtrSet &old_depended_set,
                                const std::map<AnfNodePtr, AnfNodePtrSet> &node_dependency);

// Connect hung nodes to output in case they are optimized out.
void HandleHungNodes(const FuncGraphPtr &func_graph, const NodeLabels &node_labels, OperatorLabel process_label,
                     const AnfNodePtrList &hung_nodes_list);

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

  // Do rpc node fusion to decrease the overhead of network communication.
  virtual FusedInterProcessOpPairMap DoRpcNodeFusion(InterProcessOpEdgesInfo *comm_edges_ptr) { return {}; }

  // Postbuild the distributed graph after splitting graph. For example, adding extra edges to the split graph.
  // Input 'node_labels' represents node labels of the split graph.
  // Input 'comm_edges' represents the inter-process edges generated after splitting the graph.
  virtual void PostBuildDistributedGraph(const InterProcessOpEdgesInfo &comm_edges) {}
  virtual void PostBuildDistributedGraph(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs) {}

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
  FusedInterProcessOpPairMap DoRpcNodeFusion(InterProcessOpEdgesInfo *comm_edges_ptr) override;
  void PostBuildDistributedGraph(const InterProcessOpEdgesInfo &comm_edges) override;
  void PostBuildDistributedGraph(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs) override;

 private:
  // Process optimizers split to the parameter server.
  void ProcessForSplitOptimizer();

  // Filter out all optimizer nodes which are set on parameter server from the graph.
  std::vector<CNodePtr> FilterServerAwareOptimizerList();

  // Create gradients accumulator with mean operator for the given optimizer. It could be sparse or dense gradients.
  // 'total_gradient_number' represents how many workers' gradients will be accumulated for this optimizer.
  // The return value is a pair of accumulation node to RealDiv node.
  std::pair<CNodePtr, CNodePtr> CreateNodesForGradAccumulation(const AnfNodePtr &gradient_input,
                                                               size_t gradient_input_index,
                                                               const std::string &gradient_type,
                                                               size_t total_gradient_number);

  // Normally after gradients accumulation, the mean value should be calculated.
  CNodePtr CreateGradMeanNode(const AnfNodePtr &gradient, size_t divisor);

  // Create MakeTupe and TupleGetItem nodes for the multiple inputs.
  std::pair<CNodePtr, CNodePtr> CreateNodesForMakeTuple(const AnfNodePtr &input, size_t total_inputs_number);

  // Create node with multiple inputs. Some of the inputs could be fake nodes.
  // 'many_to_one_node_name' represents the name of the node to be created.
  // 'real_input' represents the input which is already in the func_graph_. Other inputs will be created as this input.
  // 'index_of_real_input': the input index of 'real_input' of this new created node: 'many_to_one_node_name'.
  // 'total_inputs_number': the total inputs number of the created node.
  CNodePtr CreateNodeWithInterProcessEdgeOnPServer(const std::string &many_to_one_node_name,
                                                   const AnfNodePtr &real_input, size_t index_of_real_input,
                                                   uint32_t total_inputs_number);

  // Fuse RpcSend and RpcRecv nodes for Parameter Server optimizers. Only one fused send node should be corresponding to
  // one fused recv node, vice versa.
  FusedInterProcessOpPairMap FuseRpcNodesForSplitOptimizer(
    const InterProcessOpEdgesInfo &comm_edges_of_server_optimizer);

  // Filter out all communication edges related to optimizers on Parameter Server.
  InterProcessOpEdgesInfo FilterCommEdgesOfServerOptimizer(const InterProcessOpEdgesInfo &comm_edges) const;

  // Filter out all communication edges which are not related to any Parameter Server optimizers and convert them to
  // FusedInterProcessOpPairMap.
  FusedInterProcessOpPairMap FilterNotServerOptimizerEdges(const InterProcessOpEdgesInfo &comm_edges) const;

  // Fuse the given rpc send nodes list. Only nodes which send data to the same peer can be fused.
  CNodePtr FuseRpcSendNodes(const std::vector<CNodePtr> &rpc_send_nodes);

  // Fuse the given rpc recv nodes list. Only nodes which recv data from the same peer can be fused.
  CNodePtr FuseRpcRecvNodes(const std::vector<CNodePtr> &rpc_recv_nodes);

  // Fuse communication edges with same peers.
  std::vector<FusedInterProcessOpPair> FuseCommEdges(const std::vector<InterProcessOpPair> &inter_process_pairs);

  // The fusion config for rpc nodes connected with optimizers on Parameter Server. This is similar to
  // 'all_reduce_fusion_split_indices'.
  std::vector<size_t> ps_optimizer_fusion_segments_;
};

class EmbeddingCacheMode : public DistributedExecutionMode {
 public:
  explicit EmbeddingCacheMode(const FuncGraphPtr &func_graph, NodeLabels *node_labels, uint32_t rank_id,
                              const std::string &role)
      : DistributedExecutionMode(func_graph, node_labels, rank_id, role) {}
  ~EmbeddingCacheMode() = default;

  void PreBuildDistributedGraph() override;

 private:
  void AddEmbeddingCacheOps() const;

  OperatorLabel GetNodeLabel(const AnfNodePtr &node) const;
};

// Users may want to simply split a training graph into multiple devices without other extra features. GeneralMode is
// for this scenario.
class GeneralMode : public DistributedExecutionMode {
 public:
  explicit GeneralMode(const FuncGraphPtr &func_graph, NodeLabels *node_labels, uint32_t rank_id,
                       const std::string &role)
      : DistributedExecutionMode(func_graph, node_labels, rank_id, role) {}
  ~GeneralMode() = default;
};

// The mode applied when AutoParallel/SemiAutoParallel feature is enabled.
class ParallelMode : public DistributedExecutionMode {
 public:
  explicit ParallelMode(const FuncGraphPtr &func_graph, NodeLabels *node_labels, uint32_t rank_id,
                        const std::string &role)
      : DistributedExecutionMode(func_graph, node_labels, rank_id, role) {}
  ~ParallelMode() = default;
};

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

  /**
   * @description: Reassign the operator label for 'TupleGetItem' nodes. This is an optimization for nodes with multiple
   * outputs.
   * @return {void}
   */
  void ReassignTupleGetItemNodeLabel();

  /**
   * @description: Recursively visit TupeGetItem nodes and set their labels.
   * @param {CNodePtr} &tuple_get_item_node: The TupeGetItem node.
   * @return {OperatorLabel}: The OperatorLabel of this TupeGetItem node.
   */
  OperatorLabel RecursiveSetTupeGetItemLabel(const CNodePtr &tuple_get_item_node);

  /**
   * @description: Add data-sync node pairs for reference nodes like trainable parameters. These nodes are used to
   * synchronize updates of parameters between nodes.
   * @return {void}
   */
  void ProcessRefNodes();

  /**
   * @description: Add some extra control edges between nodes with different labels to keep the consistency of
   * topo-sort.
   * @return {void}
   */
  void AddExtraControlEdgeAcrossProcess();

  // Generate Send-Recv pairs for the nodes which has different split.
  // Because nodes with different split label from this proccess's with be on another machine, we use Send-Recv pairs to
  // do network communication.
  InterProcessOpEdgesInfo GenerateInterProcessOperators();

  // Eliminate nodes which are on other machine's graphs and add control edges for nodes of this process's graph.
  void SplitGraph(const std::vector<SplitGraphSegment> &segments, const InterProcessOpEdgesInfo &comm_edges);
  void SplitGraph(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs);

  /**
   * @description: Add data-sync nodes for reference nodes. To ensure the control edges, the data-sync nodes should be
   * add to the UpdateState node's input:
   * SideEffectNode(Ref1, Ref2, U)
   *              |             |
   *              |             |
   *              |        DataSyncSrcNode(Ref1, Ref2)
   *              |             |
   *              |        DataSyncDstNode(Ref1, Ref2)
   * UpdateState(U, SideEffectNode, DataSyncDstNode)
   *
   * The topology relationship is shown above: After SideEffectNode is launched and could have updated Ref1 and Ref2,
   * data-sync nodes are inserted and connected to UpdateState node so that nodes after UpdateState will not be launched
   * until the data is synchronized.
   * @param {CNodePtr} &update_state_node: The update state node which is the reference node's user.
   * @param {AnfNodePtrList} &ref_nodes: Reference nodes need to be synchronized.
   * @return {void}
   */
  void AddDataSyncNode(const CNodePtr &side_effect_node, const CNodePtr &update_state_node,
                       const AnfNodePtrList &ref_nodes);

  /**
   * @description: Create data-sync node pairs for the reference node. It may need to be synchronized to multiple
   * processes.
   * @param {CNodePtr} &side_effect_node: The node with side effect using reference node as input.
   * @param {AnfNodePtr} &ref: The reference node.
   * @param {vector<OperatorLabel>} &diff_labels: The operator label list of each process to which the reference node
   * data will be synchronized.
   * @return {DataSyncNodePairList}: The list of data-sync nodes.
   */
  DataSyncNodePairList CreateDataSyncNodes(const CNodePtr &side_effect_node, const AnfNodePtr &ref,
                                           const std::set<OperatorLabel> &diff_labels);

  /**
   * @description: For processes without any indegree, control edge should be connected from process with default label.
   * This is to avoid these processes
   * @return {void}
   */
  void AddControlEdgeForProcessWithoutIndegree();

  /**
   * @description: Create src and dst node of a control edge with the specified src and dst operator labels.
   *              ControlSrc(1.0)
   *                   |
   *                   |
   *              ControlDst()
   * @param {OperatorLabel} &src_label: Control edge src label.
   * @param {OperatorLabel} &dst_label: Control edge dst label.
   * @return {ControlEdgeNodePair}: The nodes pair.
   */
  ControlEdgeNodePair CreateControlEdgeNode(const OperatorLabel &src_label, const OperatorLabel &dst_label);

  /**
   * @description: The data-sync nodes and control-edge nodes should be eliminated at the end of the splitting process.
   * These nodes are just for graph splitting and have no corresponding backend kernels.
   * @return {void}
   */
  void EliminateDataSyncNode();
  void EliminateControlEdgeNode();

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

  InterProcessEdgeLabel GenerateEdgeLabel(const AnfNodePtr &src_node, const AnfNodePtr &dst_node) const;

  // Segments will be independent with each other after the graph is cut, so in-degrees and out-degrees of each segment
  // should be connected with control edges in case that the nodes are optimized out.
  std::vector<AnfNodePtr> FindInterProcessInDegree(const std::vector<AnfNodePtr> &nodes,
                                                   const InterProcessOpEdgesInfo &comm_edges);
  std::vector<AnfNodePtr> FindInterProcessOutDegree(const std::vector<AnfNodePtr> &nodes,
                                                    const InterProcessOpEdgesInfo &comm_edges);

  // Generate in and out degrees list of the segments to add dependency between segments.
  InOutDegreeList GenerateInOutDegreeList(const std::vector<SplitGraphSegment> &segments,
                                          const InterProcessOpEdgesInfo &comm_edges);

  //  Must add extra dependency edge for RpcSend and RpcRecv nodes in case they are optimized out or lose explicit
  //  dependencies.
  void AddDependencyBetweenEdges(const InterProcessOpEdgesInfo &comm_edges);
  // For the segments on this process, dependency edges should be created so that they won't be optimized out.
  void AddDependencyBetweenSegments(const InOutDegreeList &in_out_degree_list);

  // Replace nodes inputs with Recv nodes to eliminate extra nodes not on this process.
  void EliminateExtraNodes(const InterProcessOpEdgesInfo &comm_edges);

  // Replace nodes inputs with Recv nodes.
  void ReplaceOriginNodesWithRecv(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs);

  void AddSendRecvDependency(const InterProcessOpEdgesInfo &in_degree_comm_edges, const AnfNodePtrSet &send_src_nodes,
                             const std::map<AnfNodePtr, AnfNodePtrSet> &src_nodes_to_send_nodes,
                             const std::map<AnfNodePtr, AnfNodePtrSet> &node_dependency,
                             std::map<AnfNodePtr, bool> *is_send_node_hung);

  // Add outputs edges for send nodes so that they won't be optimized out.
  void AddDependencyForSend(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs);

  // Judge whether two nodes have the same distributed label.
  bool IsNodesWithSameLabel(const AnfNodePtr &node1, const AnfNodePtr &node2);

  // Check whether need split distributed graph.
  bool NeedSplitGraph() const;

  // Return whether this node has corresponding label stored in node_labels_.
  bool NodeHasLabel(const AnfNodePtr &node);

  FuncGraphPtr func_graph_;

  // Rank id and node role of this process. They are used to dye graph with different labels, help build split graph,
  // etc.
  uint32_t rank_id_;
  std::string role_;

  // Created according to the execution mode. Used to build the distributed graph.
  distributed::DistExecutionMode mode_;
  std::unique_ptr<DistributedExecutionMode> exec_mode_;

  // The label of this process which consists of its rank and role.
  OperatorLabel this_process_label_;

  // For each mode, there is a default label. Every node in the graph should be launched on the process with this label
  // defaultly unless it has a different split label.
  OperatorLabel default_label_;

  // The map of all nodes in the graph to their distributed split label.
  NodeLabels node_labels_;

  // All labels in the graph.
  std::set<OperatorLabel> all_labels_;

  // Whether need to fuse rpc nodes.
  bool need_fuse_rpc_nodes_;

  // Visited TupleGetItem nodes when recursively setting their labels.
  std::map<AnfNodePtr, bool> visited_tuple_get_item_nodes_;
};
using GraphSplitterPtr = std::shared_ptr<GraphSplitter>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_SPLITTER_H_
