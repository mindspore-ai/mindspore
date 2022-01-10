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

#include "frontend/parallel/graph_util/graph_splitter.h"
#include <unordered_map>
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "utils/utils.h"
#include "base/core_ops.h"

namespace mindspore {
namespace parallel {
bool OperatorLabel::operator==(const OperatorLabel &label) const {
  auto mode = distributed::DistributedExecutionMode::kPSMode;
  if (kLabelMatchingFuncMap.count(mode) == 0) {
    MS_LOG(ERROR) << "The mode " << mode << " is invalid.";
    return false;
  }
  return kLabelMatchingFuncMap.at(mode)(label, *this);
}

bool OperatorLabel::operator!=(const OperatorLabel &label) const { return !(*this == label); }

std::string OperatorLabel::to_string() const { return std::to_string(rank_id) + "_" + ms_role; }

GraphSplitter::GraphSplitter(const FuncGraphPtr &func_graph, uint32_t rank_id, const std::string &role)
    : func_graph_(func_graph), this_process_label_({rank_id, role}) {
  default_label_ = {0, distributed::kEnvRoleOfWorker};
}

GraphSplitter::~GraphSplitter() { node_labels_.clear(); }

void GraphSplitter::Run() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  MS_EXCEPTION_IF_NULL(func_graph_->manager());

  DyeGraph();

  std::vector<SplitGraphSegment> segments = GenerateSplitSegments();
  // If the segment number is 0, there will be no distributed execution.
  if (segments.empty()) {
    return;
  }
  InterProcessOpEdgesInfo comm_edges = GenerateInterProcessOperators();

  SplitGraph(segments, comm_edges);
}

void GraphSplitter::DyeGraph() {
  MS_EXCEPTION_IF_NULL(func_graph_);

  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  (void)std::for_each(all_nodes.begin(), all_nodes.end(), [this](AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    // Mark all nodes with original label at the beginning.
    node_labels_[node] = default_label_;
    if (node->isa<CNode>()) {
      // For CNodes, mark them with the label passed by frontend if has one.
      CNodePtr cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      OperatorLabel label = GetSplitLabel(cnode);
      node_labels_[node] = label;
    }
  });
}

std::vector<SplitGraphSegment> GraphSplitter::GenerateSplitSegments() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  auto return_node = func_graph_->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> nodes = func_graph_->TopoSort(return_node);

  std::vector<SplitGraphSegment> results = {};
  SplitGraphSegment segment;
  OperatorLabel last_label = default_label_;
  segment.label = last_label;
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto cnode_split_label = node_labels_[n];
    // If this node's label is not the same as last node's, create a segment from 'segment_nodes'.
    if (cnode_split_label != last_label && !segment.nodes.empty()) {
      (void)results.emplace_back(segment);
      segment.nodes.clear();
    }
    // Mark the last label.
    last_label = cnode_split_label;
    segment.label = cnode_split_label;
    (void)segment.nodes.emplace_back(n);
  }

  // Add the last segment.
  (void)results.emplace_back(segment);
  MS_LOG(INFO) << "Segments number with different distributed split labels is " << results.size();
  return results;
}

InterProcessOpEdgesInfo GraphSplitter::GenerateInterProcessOperators() {
  InterProcessOpEdgesInfo comm_edges = {};
  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    // Only support to split CNode to other process.
    if (!node->isa<CNode>()) {
      continue;
    }

    auto current_node_comm_edges = GenerateInterProcessOpsForNodeInputs(node);
    auto user_node_comm_edges = GenerateInterProcessOpsForNodeOutputs(node);
    // The edge with the same peers will be unique after inserting.
    comm_edges.insert(current_node_comm_edges.begin(), current_node_comm_edges.end());
    comm_edges.insert(user_node_comm_edges.begin(), user_node_comm_edges.end());
  }
  MS_LOG(INFO) << "The communication edge number is " << comm_edges.size();
  return comm_edges;
}

void GraphSplitter::SplitGraph(const std::vector<SplitGraphSegment> &segments,
                               const InterProcessOpEdgesInfo &comm_edges) {
  // Traverse all the segments to add Depend for this process's graph.
  for (const auto &segment : segments) {
    // If this segment should be on current process, continue.
    if (segment.label == this_process_label_) {
      continue;
    }
    std::vector<AnfNodePtr> nodes = segment.nodes;
    if (nodes.empty()) {
      MS_LOG(EXCEPTION) << "This segment is empty.";
      return;
    }
    if (node_labels_[nodes[0]] != segment.label) {
      MS_LOG(EXCEPTION) << "Node label " << node_labels_[nodes[0]].to_string() << " is not the same as segment label "
                        << segment.label.to_string();
      return;
    }

    // Add Depend between in-degree and out-degree of this segment because the execution order should be kept
    // consistent.
    std::vector<AnfNodePtr> concerned_in_degree_nodes = FindInterProcessInDegree(nodes, comm_edges);
    std::vector<AnfNodePtr> concerned_out_degree_nodes = FindInterProcessOutDegree(nodes, comm_edges);

    std::vector<AnfNodePtr> make_tuple_send_input = {NewValueNode(prim::kPrimMakeTuple)};
    (void)make_tuple_send_input.insert(make_tuple_send_input.end(), concerned_in_degree_nodes.begin(),
                                       concerned_in_degree_nodes.end());
    auto make_tuple = func_graph_->NewCNode(make_tuple_send_input);
    if (concerned_out_degree_nodes.empty()) {
      std::vector<AnfNodePtr> out = {NewValueNode(prim::kPrimDepend)};
      out.push_back(make_tuple_send_input.back());
      out.push_back(make_tuple);
      auto out_node = func_graph_->NewCNode(out);
      (void)func_graph_->manager()->Replace(func_graph_->output(), out_node);
    } else {
      for (auto &recv : concerned_out_degree_nodes) {
        std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), recv->cast<CNodePtr>()->inputs()[1],
                                                make_tuple};
        auto depend = func_graph_->NewCNode(depend_input);
        depend->set_abstract(recv->cast<CNodePtr>()->inputs()[1]->abstract());
        func_graph_->manager()->SetEdge(recv, 1, depend);
      }
    }
  }

  // Eliminate nodes which should be launched by other processes by set output edge.
  for (auto &edge : comm_edges) {
    InterProcessOpPair send_recv_pair = edge.second;
    auto send_node = std::get<0>(send_recv_pair);
    auto recv_node = std::get<1>(send_recv_pair);
    auto user_node = std::get<2>(send_recv_pair);
    int user_node_index = std::get<3>(send_recv_pair);

    OperatorLabel send_label = node_labels_[send_node];
    OperatorLabel recv_label = node_labels_[recv_node];
    if (send_label == recv_label) {
      MS_LOG(EXCEPTION) << "The Send and Recv must have different label. But got Send: " << send_label.to_string()
                        << ", Recv: " << recv_label.to_string();
      return;
    }

    if (recv_label == this_process_label_) {
      func_graph_->manager()->SetEdge(user_node, user_node_index, recv_node);
    }
  }
}

void GraphSplitter::DumpDistributedGraph(const std::vector<SplitGraphSegment> &segments,
                                         const InterProcessOpEdgesInfo &comm_edges) {
  // Traverse all the segments to add Depend for this process's graph.
  for (const auto &edge : comm_edges) {
    auto send_recv_pair = edge.second;
    auto send_node = std::get<0>(send_recv_pair);
    auto recv_node = std::get<1>(send_recv_pair);
    auto user_node = std::get<2>(send_recv_pair);
    auto user_node_index = std::get<3>(send_recv_pair);
    func_graph_->manager()->SetEdge(recv_node, 1, send_node);
    func_graph_->manager()->SetEdge(user_node, user_node_index, recv_node);
  }
  MS_LOG(INFO) << "Cut graph without eliminating nodes.";
  func_graph_->DumpFuncGraph("./single_node_graph.dot");
}

OperatorLabel GraphSplitter::GetSplitLabel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only CNode has distributed split label.";
    return default_label_;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  auto prim_node = cnode->input(0);
  if (IsValueNode<Primitive>(prim_node)) {
    auto prim = GetValueNode<PrimitivePtr>(prim_node);
    MS_EXCEPTION_IF_NULL(prim);
    if (prim->HasAttr(distributed::kOpLabelRankId) && prim->HasAttr(distributed::kOpLabelRole)) {
      MS_LOG(INFO) << "CNode which has distributed split label: " << cnode->fullname_with_scope();
      uint32_t rank_id = static_cast<uint32_t>(GetValue<int64_t>(prim->GetAttr(distributed::kOpLabelRankId)));
      std::string ms_role = GetValue<std::string>(prim->GetAttr(distributed::kOpLabelRole));
      return {rank_id, ms_role};
    }
  }
  return default_label_;
}

InterProcessOpEdgesInfo GraphSplitter::GenerateInterProcessOpsForNodeInputs(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  InterProcessOpEdgesInfo comm_edges = {};
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_i = cnode->inputs()[i];
    MS_EXCEPTION_IF_NULL(input_i);

    // If the input's label is the same as this node's, there's no need to add communication nodes.
    if (IsNodesWithSameLabel(input_i, cnode)) {
      continue;
    }

    // Create Send node.
    auto send_node = GenerateSendNode(input_i);
    MS_EXCEPTION_IF_NULL(send_node);
    // Create Recv node.
    auto recv_node = GenerateRecvNode(input_i, cnode);
    MS_EXCEPTION_IF_NULL(recv_node);

    InterProcessOpEdge comm_edge = {input_i, cnode};
    auto comm_node_pair = std::make_tuple(send_node, recv_node, cnode, SizeToInt(i));
    (void)comm_edges.insert(std::make_pair(comm_edge, comm_node_pair));
  }
  return comm_edges;
}

InterProcessOpEdgesInfo GraphSplitter::GenerateInterProcessOpsForNodeOutputs(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  InterProcessOpEdgesInfo comm_edges = {};
  auto users = func_graph_->manager()->node_users()[node];
  for (auto &u : users) {
    auto user_node = u.first->cast<CNodePtr>();
    int index = u.second;

    // If the user's label is the same as this node's, there's no need to add communication nodes.
    if (IsNodesWithSameLabel(user_node, cnode)) {
      continue;
    }

    // Create Send node.
    auto send_node = GenerateSendNode(cnode);
    MS_EXCEPTION_IF_NULL(send_node);
    // Create Recv node.
    auto recv_node = GenerateRecvNode(cnode, user_node);
    MS_EXCEPTION_IF_NULL(recv_node);

    InterProcessOpEdge comm_edge = {cnode, user_node};
    auto comm_node_pair = std::make_tuple(send_node, recv_node, user_node, index);
    (void)comm_edges.insert(std::make_pair(comm_edge, comm_node_pair));
  }
  return comm_edges;
}

CNodePtr GraphSplitter::GenerateSendNode(const AnfNodePtr &input) {
  std::vector<AnfNodePtr> send_inputs = {NewValueNode(prim::kPrimSend)};
  auto mock_tensor = std::make_shared<tensor::Tensor>(1.0);
  MS_EXCEPTION_IF_NULL(mock_tensor);
  auto mock_value = NewValueNode(mock_tensor);
  MS_EXCEPTION_IF_NULL(mock_value);
  mock_value->set_abstract(mock_tensor->ToAbstract());
  if (IsPrimitiveCNode(input, prim::kPrimUpdateState)) {
    send_inputs.push_back(mock_value);
    send_inputs.push_back(input);
  } else {
    send_inputs.push_back(input);
  }

  CNodePtr send_node = func_graph_->NewCNode(send_inputs);
  MS_EXCEPTION_IF_NULL(send_node);
  send_node->set_abstract(mock_value->abstract());
  // The label should be the same as the node which will 'launch' Send node.
  node_labels_[send_node] = node_labels_[input];
  return send_node;
}

CNodePtr GraphSplitter::GenerateRecvNode(const AnfNodePtr &input, const AnfNodePtr &peer) {
  std::vector<AnfNodePtr> recv_inputs = {NewValueNode(prim::kPrimReceive)};
  if (IsPrimitiveCNode(input, prim::kPrimUpdateState)) {
    auto mock_tensor = std::make_shared<tensor::Tensor>(1.0);
    MS_EXCEPTION_IF_NULL(mock_tensor);
    auto mock_value = NewValueNode(mock_tensor);
    MS_EXCEPTION_IF_NULL(mock_value);
    mock_value->set_abstract(mock_tensor->ToAbstract());

    recv_inputs.push_back(mock_value);
    recv_inputs.push_back(input);
  } else {
    auto input_abstract = input->abstract()->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(input_abstract);
    auto mock_tensor = std::make_shared<tensor::Tensor>(1.0, input_abstract->element()->BuildType());
    MS_EXCEPTION_IF_NULL(mock_tensor);
    auto mock_value = NewValueNode(mock_tensor);
    MS_EXCEPTION_IF_NULL(mock_value);
    mock_value->set_abstract(mock_tensor->ToAbstract());
    recv_inputs.push_back(mock_value);
  }

  CNodePtr recv_node = func_graph_->NewCNode(recv_inputs);
  MS_EXCEPTION_IF_NULL(recv_node);
  recv_node->set_abstract(input->abstract());
  // The label should be the same as the node which Receives the 'input'.
  node_labels_[recv_node] = node_labels_[peer];
  return recv_node;
}

std::vector<AnfNodePtr> GraphSplitter::FindInterProcessInDegree(const std::vector<AnfNodePtr> &nodes,
                                                                const InterProcessOpEdgesInfo &comm_edges) {
  std::vector<AnfNodePtr> results = {};
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }

    CNodePtr cnode = n->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      auto input_i = cnode->inputs()[i];
      if (comm_edges.count({input_i, cnode}) == 0) {
        MS_LOG(DEBUG) << input_i->fullname_with_scope() << " to " << cnode->fullname_with_scope()
                      << " is not a communication edge.";
        continue;
      }

      MS_LOG(INFO) << input_i->fullname_with_scope() << " to " << cnode->fullname_with_scope()
                   << " is a communication edge.";
      auto comm_node_pair = comm_edges.at({input_i, cnode});
      (void)results.emplace_back(std::get<0>(comm_node_pair));
    }
  }
  return results;
}

std::vector<AnfNodePtr> GraphSplitter::FindInterProcessOutDegree(const std::vector<AnfNodePtr> &nodes,
                                                                 const InterProcessOpEdgesInfo &comm_edges) {
  std::vector<AnfNodePtr> results = {};
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }

    CNodePtr cnode = n->cast<CNodePtr>();
    auto users = func_graph_->manager()->node_users()[cnode];
    for (auto &u : users) {
      auto user_node = u.first->cast<CNodePtr>();
      if (comm_edges.count({cnode, user_node}) == 0) {
        MS_LOG(DEBUG) << cnode->fullname_with_scope() << " to " << user_node->fullname_with_scope()
                      << " is not a communication edge.";
        continue;
      }

      MS_LOG(INFO) << cnode->fullname_with_scope() << " to " << user_node->fullname_with_scope()
                   << " is a communication edge.";
      auto comm_node_pair = comm_edges.at({cnode, user_node});
      (void)results.emplace_back(std::get<1>(comm_node_pair));
    }
  }
  return results;
}

bool GraphSplitter::IsNodesWithSameLabel(const AnfNodePtr &node1, const AnfNodePtr &node2) {
  if (node_labels_.count(node1) == 0 || node_labels_.count(node2) == 0) {
    MS_LOG(EXCEPTION) << "Either 'node1': " << node1->fullname_with_scope()
                      << " or 'node2': " << node2->fullname_with_scope() << " is not marked with split label.";
    return false;
  }
  return node_labels_[node1] == node_labels_[node2];
}
}  // namespace parallel
}  // namespace mindspore
