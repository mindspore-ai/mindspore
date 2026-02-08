/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/graph_fusion/core/transform_op_optimizer.h"

#include <algorithm>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <utility>
#include <string>
#include <tuple>
#include <functional>

#include "backend/ms_backend/graph_fusion/model/lite_graph.h"
#include "backend/ms_backend/graph_fusion/model/graph_builder.h"
#include "backend/ms_backend/graph_fusion/model/op_register.h"
#include "backend/ms_backend/graph_fusion/core/graph_builder.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "ir/func_graph_flag.h"

namespace mindspore::graphkernel {
constexpr const size_t num2 = 2;
struct Edge {
  size_t from;
  size_t to;
  bool operator<(const Edge &other) const { return from == other.from ? to < other.to : from < other.from; }
  friend std::ostream &operator<<(std::ostream &os, const Edge &e) {
    return os << "[" << e.from << " -> " << e.to << "]";
  }
};
inline std::ostream &operator<<(std::ostream &os, FormatType fmt) {
  return os << (fmt == FormatType::kFlexFormat ? "kFlexFormat"
                                               : (fmt == FormatType::kFormatA ? "kFormatA" : "kFormatB"));
}
inline std::ostream &operator<<(std::ostream &os, TransOpType trans) {
  return os << (trans == TransOpType::kTransAB ? "kTransAB" : "kTransBA");
}

// For format-inflexible nodes, index -1 represent its output field, and index 0~n represent its input field.
// for format-flexible nodes, only index -1 represent its all inputs and output fields.
using NodeWithIndex = std::pair<NodePtr, int>;
using NodeIdWithFormat = std::pair<size_t, FormatType>;

namespace {
constexpr size_t INF = static_cast<size_t>(1) << 30;
class MinCut {
 private:
  struct MinCutEdge {
    size_t to;
    size_t capacity;
  };

  struct Vertex {
    FormatType format{FormatType::kFormatB};
    size_t depth{0};
    std::vector<size_t> out_edges;
  };

  // Add the bidirectional edges for the vertex `from` and `to`.
  // the two edge ids are adjacent in vector, x and x+1 (x are 0,2,4,...)
  // we can use (i xor 1) to get the inverse edge for any edge i.
  // e.g. edge_0 and edge_1 are a couple, 0^1=1, 1^1=0.
  void AddEdge(size_t from, size_t to, size_t capacity, size_t inv_capacity) {
    (void)edges_.emplace_back(MinCutEdge{to, capacity});
    (void)nodes_[from].out_edges.emplace_back(edges_.size() - 1);
    // inverse edge
    (void)edges_.emplace_back(MinCutEdge{from, inv_capacity});
    (void)nodes_[to].out_edges.emplace_back(edges_.size() - 1);
  }

  bool BfsSetDepth() {
    std::queue<size_t> bfs_queue;
    for (auto &node : nodes_) {
      node.depth = 0;
    }
    nodes_[source_id_].depth = 1;
    bfs_queue.push(source_id_);
    while (!bfs_queue.empty()) {
      auto edge_from = bfs_queue.front();
      bfs_queue.pop();
      for (auto e_id : nodes_[edge_from].out_edges) {
        auto edge_to = edges_[e_id].to;
        if (edges_[e_id].capacity > 0 && nodes_[edge_to].depth == 0) {
          nodes_[edge_to].depth = nodes_[edge_from].depth + 1;
          bfs_queue.push(edge_to);
        }
      }
    }
    return nodes_[sink_id_].depth > 0;
  }

  size_t DfsMaxFlow(size_t node, size_t flow) {
    if (node == sink_id_) {
      return flow;
    }
    size_t max_flow = 0;
    for (size_t e_id : nodes_[node].out_edges) {
      if ((edges_[e_id].capacity > 0) && (nodes_[node].depth + 1 == nodes_[edges_[e_id].to].depth)) {
        auto tmp_flow = DfsMaxFlow(edges_[e_id].to, std::min(flow, edges_[e_id].capacity));
        if (tmp_flow > 0) {
          max_flow += tmp_flow;
          flow -= tmp_flow;
          edges_[e_id].capacity -= tmp_flow;
          edges_[e_id ^ 1].capacity += tmp_flow;
        }
      }
    }
    return max_flow;
  }

  void Dinic() {
    while (BfsSetDepth()) {
      (void)DfsMaxFlow(source_id_, INF);
    }
  }

  // set the nodes that connected with source node to kFormatA, the remaining nodes are seen as kFormatB.
  void SetFormat(size_t node_id) {
    nodes_[node_id].format = FormatType::kFormatA;
    MS_LOG(DEBUG) << "Set node_id " << node_id << " to kFormatA.";
    for (size_t i : nodes_[node_id].out_edges) {
      if (edges_[i].capacity > 0 && nodes_[edges_[i].to].format != FormatType::kFormatA) {
        SetFormat(edges_[i].to);
      }
    }
  }

  void BuildGraph(const std::vector<NodeIdWithFormat> &original_nodes) {
    for (size_t i = 0; i < origin_nodes_num_; ++i) {
      // link the source node to the nodes with FormatA,
      // link the nodes with FormatB to the sink node.
      if (original_nodes[i].second == FormatType::kFormatA) {
        AddEdge(source_id_, original_nodes[i].first, INF, 0);
      } else if (original_nodes[i].second == FormatType::kFormatB) {
        AddEdge(original_nodes[i].first, sink_id_, INF, 0);
      }
      // each nodes was split into two part, input part and output part.
      // the input part's id is the original node's id, the output part's id is input id + origin_nodes_num_.
      AddEdge(original_nodes[i].first, original_nodes[i].first + origin_nodes_num_, 1, 1);
    }
    for (auto e : original_edges_) {
      AddEdge(e.from + origin_nodes_num_, e.to, 1, 1);
    }
  }

 public:
  MinCut(const std::vector<NodeIdWithFormat> &original_nodes, const std::vector<Edge> &original_edges)
      : origin_nodes_num_(original_nodes.size()),
        source_id_(0),
        sink_id_(num2 * original_nodes.size() + 1),
        nodes_(num2 * original_nodes.size() + num2),  // double nodes, and source_node/sink_node
        original_edges_(original_edges) {
    BuildGraph(original_nodes);
  }
  ~MinCut() = default;

  void Run() {
    Dinic();
    SetFormat(source_id_);
  }

  std::vector<std::pair<size_t, TransOpType>> GetOneNodeOps() const {
    std::vector<std::pair<size_t, TransOpType>> one_node_ops;
    for (size_t i = 1; i <= origin_nodes_num_; ++i) {
      auto tmpi = i;  // to evade pclint warning "for statement index variable modified in body."
      if (nodes_[i].format == FormatType::kFormatA && nodes_[i + origin_nodes_num_].format != FormatType::kFormatA) {
        (void)one_node_ops.emplace_back(tmpi, TransOpType::kTransAB);
        MS_LOG(DEBUG) << "Inserted kTransAB for node_id " << tmpi;
      } else if (nodes_[i].format != FormatType::kFormatA &&
                 nodes_[i + origin_nodes_num_].format == FormatType::kFormatA) {
        (void)one_node_ops.emplace_back(tmpi, TransOpType::kTransBA);
        MS_LOG(DEBUG) << "Inserted kTransBA for node_id " << tmpi;
      }
    }
    return one_node_ops;
  }

  std::vector<std::pair<Edge, TransOpType>> GetTwoNodeOps() const {
    std::vector<std::pair<Edge, TransOpType>> two_node_ops;
    for (auto e : original_edges_) {
      if (nodes_[e.from + origin_nodes_num_].format == FormatType::kFormatA &&
          nodes_[e.to].format != FormatType::kFormatA) {
        (void)two_node_ops.emplace_back(e, TransOpType::kTransAB);
        MS_LOG(DEBUG) << "Inserted kTransAB for edge " << e;
      } else if (nodes_[e.from + origin_nodes_num_].format != FormatType::kFormatA &&
                 nodes_[e.to].format == FormatType::kFormatA) {
        (void)two_node_ops.emplace_back(e, TransOpType::kTransBA);
        MS_LOG(DEBUG) << "Inserted kTransBA for edge " << e;
      }
    }
    return two_node_ops;
  }

 private:
  size_t origin_nodes_num_;
  size_t source_id_;
  size_t sink_id_;
  std::vector<Vertex> nodes_;
  std::vector<MinCutEdge> edges_;
  std::vector<Edge> original_edges_;
};

bool IsDynamicShapeGraph(const inner::LiteGraphPtr &litegraph) {
  MS_EXCEPTION_IF_NULL(litegraph);
  for (auto &op : litegraph->ops()) {
    if (IsDynamic(op->shape)) {
      return true;
    }
  }
  return false;
}
}  // namespace

using inner::LiteGraph;
using inner::LiteGraphPtr;
using inner::NodePtrList;
using inner::NType;
using inner::PrimOp;
using inner::PrimOpPtr;

TransformOp::TransformOp(const NodePtr &node)
    : op_(node->As<PrimOp>()->op()), format_a_(node->input(0)->format), format_b_(node->format) {}

size_t TransformOp::Hash() const {
  // TransAB and TransBA are seen as the same trans op.
  auto fmt1 = format_a_;
  auto fmt2 = format_b_;
  if (fmt1 > fmt2) {
    std::swap(fmt1, fmt2);
  }
  return std::hash<std::string>{}(op_ + fmt1 + fmt2);
}

std::string TransformOp::GetFormat(const NodePtr &node) const { return node->format; }

bool TransformOp::IsTransformOp(const NodePtr &node) {
  if (node->NodeType() != NType::Primitive || node->As<PrimOp>()->op() != op_) {
    return false;
  }
  auto format_in = GetFormat(node->input(0));
  auto format_out = GetFormat(node);
  if (format_in == format_a_ && format_out == format_b_) {
    return true;
  } else if (format_in == format_b_ && format_out == format_a_) {
    return true;
  }
  return false;
}

bool TransformOp::NeedInsert(const NodePtr &input_node) const {
  // a trick, if the node's size of 1, it's not need to insert transform op.
  return input_node->tensor_size() != 1;
}

FormatType TransformOp::GetFormatType(const std::string &fmt) {
  // nodes that are not flexible and not FormatA will be set to FormatB (include "others" format)
  return fmt == format_a_ ? FormatType::kFormatA : FormatType::kFormatB;
}

void TransformOp::SetInput(const NodePtr &node, const NodePtr &input_node) { node->SetInputs({input_node}); }

bool TransformOpCreator::IsTransOp(const NodePtr &node) const {
  if (node->NodeType() == NType::Primitive) {
    if (node->As<PrimOp>()->op() == op_name_) {
      if (op_name_ == "Reshape") {
        return node->format == node->input(0)->format;
      }
      return true;
    }
  }
  return false;
}

class TransposeHandle : public TransformOp {
 public:
  using TransformOp::TransformOp;
  NodePtr GenTransformOp(const NodePtr &input_node, TransOpType trans_type) override {
    static std::map<std::tuple<size_t, std::string, std::string>, std::vector<int64_t>> perm_map = {
      // rank 3
      {{3, kOpFormat_NCHW, kOpFormat_NHWC}, {1, 2, 0}},
      {{3, kOpFormat_NHWC, kOpFormat_NCHW}, {2, 0, 1}},
      // rank 4
      {{4, kOpFormat_DEFAULT, kOpFormat_NHWC}, {0, 2, 3, 1}},
      {{4, kOpFormat_NCHW, kOpFormat_NHWC}, {0, 2, 3, 1}},
      {{4, kOpFormat_NHWC, kOpFormat_NCHW}, {0, 3, 1, 2}},
      {{4, kOpFormat_NHWC, kOpFormat_DEFAULT}, {0, 3, 1, 2}},
    };
    std::vector<int64_t> perm;
    std::string dst_format;
    auto rank = input_node->shape.size();
    if (trans_type == TransOpType::kTransAB) {
      perm = perm_map[{rank, format_a_, format_b_}];
      dst_format = format_b_;
    } else {
      perm = perm_map[{rank, format_b_, format_a_}];
      dst_format = format_a_;
    }
    if (perm.empty()) {
      MS_LOG(INFO) << "unsupported format: " << format_a_ << " to " << format_b_ << " of rank " << rank;
      return nullptr;
    }
    auto op = inner::OpRegistry::Instance().NewOp(op_);
    auto perm_tensor = tensor::from_vector(perm, kInt64);
    node_to_input_tensor_map_[op] = perm_tensor;
    op->SetAttr(kAttrDstFormat, MakeValue(dst_format));
    return op;
  }

  void SetInput(const NodePtr &node, const NodePtr &input_node) override {
    inner::GraphBuilder gb;
    auto iter = node_to_input_tensor_map_.find(node);
    if (iter == node_to_input_tensor_map_.end()) {
      MS_LOG(EXCEPTION) << "Can't find input valueptr for node: " << node->ToString();
    }
    auto perm_tensor = iter->second;
    auto perm_node = gb.Value(perm_tensor);
    node->SetInputs({input_node, perm_node});
  }

 private:
  std::map<NodePtr, tensor::TensorPtr> node_to_input_tensor_map_;
};

class ReshapeHandle : public TransformOp {
 public:
  explicit ReshapeHandle(const NodePtr &node) : TransformOp(node) {
    format_a_ = EncodeShape(node->input(0)->shape);
    format_b_ = EncodeShape(node->shape);
  }
  virtual ~ReshapeHandle() = default;

  std::string GetFormat(const NodePtr &node) const override {
    // Reshape op uses shape as format
    return EncodeShape(node->shape);
  }

  bool NeedInsert(const NodePtr &) const override {
    // Reshape op must be inserted, otherwise the out shape of a node may changed and users may need infer shape again.
    return true;
  }

  NodePtr GenTransformOp(const NodePtr &, TransOpType trans_type) override {
    auto op = inner::OpRegistry::Instance().NewOp(op_);
    auto out_format = trans_type == TransOpType::kTransAB ? format_b_ : format_a_;
    auto out_shape = DecodeShape(out_format);
    auto shape_tensor = tensor::from_vector(out_shape, kInt64);
    node_to_input_tensor_map_[op] = shape_tensor;
    return op;
  }

  void SetInput(const NodePtr &node, const NodePtr &input_node) override {
    inner::GraphBuilder gb;
    auto iter = node_to_input_tensor_map_.find(node);
    if (iter == node_to_input_tensor_map_.end()) {
      MS_LOG(EXCEPTION) << "Can't find input valueptr for node: " << node->ToString();
    }
    auto shape_tensor = iter->second;
    auto shape_node = gb.Value(shape_tensor);
    node->SetInputs({input_node, shape_node});
  }

 private:
  std::string EncodeShape(const ShapeVector &shape) const {
    std::string res;
    for (const auto &s : shape) {
      res += std::to_string(s) + "_";
    }
    return res;
  }

  ShapeVector DecodeShape(const std::string &shape) const {
    ShapeVector res;
    size_t l = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == '_' && i > l) {
        std::istringstream iss(shape.substr(l, i));
        l = i + 1;
        int64_t s;
        iss >> s;
        res.push_back(s);
      }
    }
    return res;
  }

  std::map<NodePtr, tensor::TensorPtr> node_to_input_tensor_map_;
};

constexpr int kOutputIndex = -1;
class Mutator {
 public:
  Mutator(const NodePtr &node, const TransformOpPtr &handle) : op_handle_(handle), basenode_(node), ori_node_(1) {}
  ~Mutator() = default;

  bool Run(std::set<NodePtr> *changed_nodes) {
    VisitNode(basenode_, kOutputIndex);
    if (flexible_ops_.empty() && trans_ops_.size() <= 1) {
      return false;
    }
    // remove transform ops in litegraph
    RemoveTransOp();
    GenFormatGraph();
    RebuildLiteGraph(changed_nodes);
    changed_nodes->insert(flexible_ops_.begin(), flexible_ops_.end());
    return true;
  }

  size_t new_trans_op_num() const { return new_trans_op_num_; }

 private:
  void VisitNode(const NodePtr &node, int index) {
    if (visited_.count(node) > 0 && inflexible_ops_.count(node) == 0) {
      return;
    }
    (void)visited_.insert(node);
    if (op_handle_->IsTransformOp(node)) {
      (void)trans_ops_.insert(node);
    } else if (!IsFlexibleOp(node)) {
      VisitInflexibleOp(node, index);
      return;
    } else {
      (void)flexible_ops_.insert(node);
      fmt_type[{node, kOutputIndex}] = FormatType::kFlexFormat;
    }
    // for trans op or format-flexible op, visit node bidirectionally.
    for (auto &input : node->inputs()) {
      if (input->NodeType() != NType::Tensor && input->NodeType() != NType::Scalar) {
        VisitNode(input, kOutputIndex);
      }
    }
    for (auto &user : node->users()) {
      for (auto user_idx : user.second) {
        VisitNode(user.first->shared_from_this(), SizeToInt(user_idx));
      }
    }
  }

  void VisitInflexibleOp(const NodePtr &node, int index) {
    auto &visited_index = inflexible_ops_[node];
    if (!visited_index.insert(index).second) {
      return;
    }
    if (visited_index.size() == 1) {
      if (node->NodeType() != NType::Output) {
        fmt_type[{node, kOutputIndex}] = op_handle_->GetFormatType(op_handle_->GetFormat(node));
      }
      if (node->NodeType() != NType::Parameter) {
        for (size_t i = 0; i < node->inputs().size(); i++) {
          if (node->input(i)->NodeType() != NType::Tensor && node->input(i)->NodeType() != NType::Scalar) {
            fmt_type[{node, i}] = op_handle_->GetFormatType(op_handle_->GetFormat(node->input(i)));
          }
        }
      }
    }
    // this node is visited from output direction, visit its other users
    if (index < 0) {
      for (const auto &user : node->users()) {
        for (auto user_idx : user.second) {
          VisitNode(user.first->shared_from_this(), SizeToInt(user_idx));
        }
      }
    }
  }

  void RemoveTransOp() {
    for (const auto &node : trans_ops_) {
      (void)visited_.erase(node);
      node->ReplaceWith(node->input(0));
      // clear inputs, so that the node will not be the basenode again.
      node->ClearInputs();
    }
    trans_ops_.clear();
  }

  void GenFormatGraph() {
    for (const auto &node : visited_) {
      if (node->NodeType() == NType::Parameter) {
        continue;
      }
      bool is_flexible = (flexible_ops_.find(node) != flexible_ops_.cend());
      size_t cur_id = 0;
      if (is_flexible) {
        cur_id = GetNodeId({node, kOutputIndex});
      }
      for (size_t i = 0; i < node->inputs().size(); i++) {
        if (visited_.count(node->input(i)) == 0) {
          continue;
        }
        if (!is_flexible) {
          cur_id = GetNodeId({node, SizeToInt(i)});
        }
        auto input_id = GetNodeId({node->input(i), kOutputIndex});
        (void)graph_edges_.emplace_back(Edge{input_id, cur_id});
      }
    }
  }

  NodePtr NewTransOp(const NodePtr &input, TransOpType trans_type, std::set<NodePtr> *changed_nodes) {
    if (!op_handle_->NeedInsert(input)) {
      return nullptr;
    }
    NodePtr trans_op = op_handle_->GenTransformOp(input, trans_type);
    MS_EXCEPTION_IF_NULL(trans_op);
    static size_t inc_id = 0;
    trans_op->SetDebugName("new_trans_op_" + std::to_string(inc_id++));
    MS_LOG(DEBUG) << "Create " << trans_op->debug_name() << " of " << trans_type << " with input node "
                  << input->debug_name();
    (void)changed_nodes->insert(trans_op);
    new_trans_op_num_++;
    return trans_op;
  }

  void RefineEdges(std::vector<std::pair<size_t, TransOpType>> *one_node_edge,
                   std::vector<std::pair<Edge, TransOpType>> *two_node_edge) const {
    std::map<size_t, TransOpType> one_node_edge_map;
    for (auto &one : *one_node_edge) {
      one_node_edge_map[one.first] = one.second;
    }
    std::set<Edge> removed_edges;
    std::set<size_t> removed_edges_from;
    for (auto iter = two_node_edge->begin(); iter != two_node_edge->end();) {
      if (one_node_edge_map.count(iter->first.from) == 0) {
        ++iter;
        continue;
      }
      auto from = iter->first.from;
      (void)removed_edges_from.insert(from);
      // remove node from one_node_edge.
      auto rm_iter = std::find_if(one_node_edge->begin(), one_node_edge->end(),
                                  [from](const std::pair<size_t, TransOpType> &no) { return from == no.first; });
      if (rm_iter != one_node_edge->end()) {
        (void)one_node_edge->erase(rm_iter);
        MS_LOG(DEBUG) << "Removed edge for node_id " << from;
      }
      // remove node from two_node_edge.
      (void)removed_edges.insert(iter->first);
      iter = two_node_edge->erase(iter);
      MS_LOG(DEBUG) << "Removed edge " << iter->first.from << " -> " << iter->first.to;
    }
    for (auto &e : graph_edges_) {
      if (removed_edges_from.count(e.from) != 0 && removed_edges.count(e) == 0) {
        two_node_edge->push_back(std::make_pair(e, one_node_edge_map[e.from]));
        MS_LOG(DEBUG) << "Inserted " << (one_node_edge_map[e.from] == TransOpType::kTransAB ? "kTransAB" : "kTransBA")
                      << " for edge " << e.from << " -> " << e.to;
      }
    }
  }

  void RebuildLiteGraph(std::set<NodePtr> *changed_nodes) {
    MinCut min_cut(graph_vertex_, graph_edges_);
    min_cut.Run();
    auto one_node_edge = min_cut.GetOneNodeOps();
    auto two_node_edge = min_cut.GetTwoNodeOps();
    RefineEdges(&one_node_edge, &two_node_edge);
    for (auto [node_id, trans_type] : one_node_edge) {
      if (ori_node_[node_id].second != kOutputIndex) {
        MS_LOG(EXCEPTION) << "OneNodeOp should be the output edge. node_id:" << node_id
                          << " index:" << ori_node_[node_id].second;
      }
      auto input_node = ori_node_[node_id].first;
      auto trans_op = NewTransOp(input_node, trans_type, changed_nodes);
      if (trans_op == nullptr) {
        continue;
      }
      input_node->ReplaceWith(trans_op);
      op_handle_->SetInput(trans_op, input_node);
      MS_LOG(DEBUG) << "Inserted " << trans_op->debug_name() << " after " << input_node->debug_name();
    }

    std::map<size_t, NodePtr> trans_op_cache;
    for (auto [insert_edge, trans_type] : two_node_edge) {
      if (ori_node_[insert_edge.from].second != kOutputIndex) {
        MS_LOG(EXCEPTION) << "node_from should be the output insert_edge. node_id:" << insert_edge.from
                          << " index:" << ori_node_[insert_edge.from].second;
      }
      auto node_from = ori_node_[insert_edge.from].first;
      auto node_to = ori_node_[insert_edge.to].first;
      if (trans_op_cache.count(insert_edge.from) == 0) {
        auto trans_op = NewTransOp(node_from, trans_type, changed_nodes);
        if (trans_op == nullptr) {
          continue;
        }
        trans_op_cache[insert_edge.from] = trans_op;
        op_handle_->SetInput(trans_op, node_from);
      }
      auto trans_op = trans_op_cache[insert_edge.from];
      if (ori_node_[insert_edge.to].second >= 0) {
        node_to->SetInput(IntToSize(ori_node_[insert_edge.to].second), trans_op);
        MS_LOG(DEBUG) << "Inserted " << trans_op->debug_name() << " before " << node_to->debug_name() << " (input "
                      << ori_node_[insert_edge.to].second << ")";
      } else {
        // "node_to" is flexible.
        for (size_t i = 0; i < node_to->inputs().size(); i++) {
          if (node_to->input(i) == node_from) {
            node_to->SetInput(i, trans_op);
            MS_LOG(DEBUG) << "Inserted " << trans_op->debug_name() << " before " << node_to->debug_name() << " (input "
                          << i << ")";
          }
        }
      }
    }
  }

  size_t GetNodeId(const NodeWithIndex &node_with_index) {
    // the nodes are indexed from 1 in the MinCut model.
    auto &id = node_id_[node_with_index];
    if (id == 0) {
      id = node_id_.size();
      ori_node_.push_back(node_with_index);
      // set format_type for new id.
      (void)graph_vertex_.emplace_back(id, fmt_type[node_with_index]);
      MS_LOG(DEBUG) << "Allot node_id " << id << " to " << node_with_index.first->debug_name() << " (index "
                    << node_with_index.second << ").";
    }
    return id;
  }

  bool IsFlexibleOp(const NodePtr &node) const {
    if (node->NodeType() != NType::Primitive) {
      return false;
    }
    if (node->As<PrimOp>()->compute_type() != PrimOp::ComputeType::ELEMWISE) {
      return false;
    }
    // check the input and output formats are all the same, except ConstValue.
    for (auto &inp : node->inputs()) {
      if (inp->NodeType() != NType::Tensor && inp->NodeType() != NType::Scalar &&
          op_handle_->GetFormat(inp) != op_handle_->GetFormat(node)) {
        return false;
      }
    }
    return true;
  }

  size_t new_trans_op_num_{0};

  TransformOpPtr op_handle_;
  NodePtr basenode_;
  std::set<NodePtr> flexible_ops_;
  std::set<NodePtr> trans_ops_;
  std::set<NodePtr> visited_;
  std::map<NodePtr, std::set<int>> inflexible_ops_;  // no transop and no flexibleop, record the visit index.

  std::map<NodeWithIndex, FormatType> fmt_type;
  std::map<NodeWithIndex, size_t> node_id_;
  std::vector<NodeWithIndex> ori_node_;  // node_id to NodePtr, this vector is indexed from 1
  std::vector<NodeIdWithFormat> graph_vertex_;
  std::vector<Edge> graph_edges_;
};

void TransformOpOptimizer::ReInfer(const LiteGraphPtr &litegraph, const std::set<NodePtr> &nodes_may_change) const {
  auto &new_ops = litegraph->GetOrderedNodes();
  MS_LOG(DEBUG) << "The changed graph before InferShape: \n" << litegraph->ToString();
  for (auto &op : new_ops) {
    if (nodes_may_change.count(op) != 0) {
      op->SetBaseInfo(op->As<PrimOp>()->Infer(op->inputs(), op->attrs()));
    }
  }
  MS_LOG(DEBUG) << "The changed graph after InferShape: \n" << litegraph->ToString();
}

bool TransformOpOptimizer::Process(const LiteGraphPtr &litegraph, const TransformOpPtr &op_handle) const {
  MS_LOG(DEBUG) << "Process begin, handle is " << *op_handle << ". litegraph: \n" << litegraph->ToString();
  auto ops = litegraph->ops();
  bool changed = false;
  auto check_is_trans_op = [&op_handle](const NodePtr &node) { return op_handle->IsTransformOp(node); };
  size_t ori_trans_op_num = static_cast<size_t>(std::count_if(ops.begin(), ops.end(), check_is_trans_op));
  size_t new_trans_op_num = 0;
  for (auto &op : ops) {
    if (check_is_trans_op(op) && !op->inputs().empty()) {
      if (op_handle->GetFormat(op->input(0)) != op_handle->GetFormat(op)) {
        std::set<NodePtr> nodes_may_change;
        auto mutator = Mutator(op, op_handle);
        MS_LOG(DEBUG) << "Run mutator with basenode " << op->debug_name();
        auto ret = mutator.Run(&nodes_may_change);
        MS_LOG(DEBUG) << "Run mutator result: " << ret;
        new_trans_op_num += mutator.new_trans_op_num();
        if (ret) {
          changed = true;
          ReInfer(litegraph, nodes_may_change);
        }
      }
    }
  }
  bool result = changed && new_trans_op_num < ori_trans_op_num;
  MS_LOG(DEBUG) << "Process result=" << result << ". changed=" << changed << ", new_trans_op_num=" << new_trans_op_num
                << ", ori_trans_op_num=" << ori_trans_op_num;
  return result;
}

void TransformOpOptimizer::Init() {
  (void)supported_ops_.emplace_back(TRANS_OP_CREATOR("Transpose", TransposeHandle));
  (void)supported_ops_.emplace_back(TRANS_OP_CREATOR("Reshape", ReshapeHandle));
}

std::vector<TransformOpPtr> TransformOpOptimizer::CreateOpHandles(const LiteGraphPtr &litegraph) const {
  HashSet<size_t> handle_hash;
  std::vector<TransformOpPtr> handles;
  for (auto &creator : supported_ops_) {
    if (creator.Name() == "Reshape" && IsDynamicShapeGraph(litegraph)) {
      // skip dynamic shape
      continue;
    }
    for (auto &op : litegraph->ops()) {
      if (creator.IsTransOp(op)) {
        auto handle = creator.CreateHandle(op);
        if (handle_hash.insert(handle->Hash()).second) {
          (void)handles.emplace_back(handle);
        }
      }
    }
  }
  return handles;
}

bool TransformOpOptimizer::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = GkUtils::GetGraphKernelNodes(func_graph);
  bool changed = false;
  for (auto node : todos) {
    MS_LOG(DEBUG) << "Run the node: " << node->fullname_with_scope();
    auto sub_func_graph = GetCNodeFuncGraph(node);
    auto node_name = sub_func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
    auto litegraph = GkUtils::AnfGraph2LiteGraph(sub_func_graph);
    auto handles = CreateOpHandles(litegraph);
    for (size_t i = 0; i < handles.size(); i++) {
      // rebuild litegraph for every process
      if (i > 0) {
        litegraph = GkUtils::AnfGraph2LiteGraph(GetCNodeFuncGraph(node));
      }
      bool result = false;
      try {
        MS_LOG_TRY_CATCH_SCOPE;
        result = Process(litegraph, handles[i]);
      } catch (std::exception &e) {
        result = false;
        MS_LOG(INFO) << "Process node " << node->DebugString() << " failed. message: " << e.what();
      }
      if (result) {
        changed = true;
        MS_LOG(DEBUG) << "Replace with graph:\n" << litegraph->ToString(true);
        auto new_funcgraph = GkUtils::LiteGraph2AnfGraph(litegraph, Callback::Instance());
        MS_EXCEPTION_IF_NULL(new_funcgraph);
        new_funcgraph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, node_name);
        auto cnode = node->cast<CNodePtr>();
        AnfNodePtrList inputs(cnode->inputs().begin() + 1, cnode->inputs().end());
        (void)ConvertTensorToParameter(new_funcgraph, &inputs);
        auto new_node = CreateNewFuseCNode(func_graph, new_funcgraph, inputs);
        (void)mng->Replace(node, new_node);
        node = new_node;
        mng->AddFuncGraph(new_funcgraph);
      }
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
