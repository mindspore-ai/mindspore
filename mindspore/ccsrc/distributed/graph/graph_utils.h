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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_GRAPH_GRAPH_UTILS_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_GRAPH_GRAPH_UTILS_H_

#include <map>
#include <tuple>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "base/base.h"
#include "ir/value.h"
#include "ir/graph_utils.h"
#include "ir/func_graph.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace distributed {
namespace graph {
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
  bool LooseEqual(const OperatorLabel &label) const;

  std::string to_string() const;
};

// The label for inter-process edges. This is used for classify the edges.
// For example, only edges with same label should be fused.
struct InterProcessEdgeLabel {
  std::string label_name;
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
const std::map<distributed::DistExecutionMode, LabelMatchingFunc> kLabelMatchingFuncMap = {
  {distributed::DistExecutionMode::kPSMode, MatchLabelForPSMode}};

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
ValueNodePtr CreateFakeValueNode(bool use_origin_node, const AnfNodePtr &origin_node = nullptr);

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
}  // namespace graph
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_GRAPH_GRAPH_UTILS_H_
