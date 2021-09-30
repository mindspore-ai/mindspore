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
#include "backend/optimizer/graph_kernel/transform_op_optimizer.h"
#include "base/core_ops.h"
#include "ir/graph_utils.h"
#include "debug/common.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/optimizer/graph_kernel/model/lite_graph.h"
#include "backend/optimizer/graph_kernel/model/op_register.h"

namespace mindspore {
namespace opt {
namespace {
enum FormatType { kFormatUnknown, kFormatA, kFormatB };
enum TransOpType { kTransAB, kTransBA };
struct Edge {
  size_t to;
  size_t capacity;
};

struct Vertex {
  FormatType format{kFormatB};
  size_t depth{0};
  std::vector<size_t> out_edges;
};

constexpr size_t INF = static_cast<size_t>(1) << 30;

class MinCut {
 private:
  // Add the bidirectional edges for the vertex `from` and `to`.
  // the two edge ids are adjacent in vector, x and x+1 (x are 0,2,4,...)
  // we can use (i xor 1) to get the inverse edge for any edge i.
  // e.g. edge_0 and edge_1 are a couple, 0^1=1, 1^1=0.
  void AddEdge(size_t from, size_t to, size_t capacity, size_t inv_capacity) {
    (void)edges_.emplace_back(Edge{to, capacity});
    (void)nodes_[from].out_edges.emplace_back(edges_.size() - 1);
    // inverse edge
    (void)edges_.emplace_back(Edge{from, inv_capacity});
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
    if (node == sink_id_) return flow;
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

  void SetFormat(size_t node_id) {
    nodes_[node_id].format = kFormatA;
    for (size_t i : nodes_[node_id].out_edges) {
      if (edges_[i].capacity > 0 && nodes_[edges_[i].to].format != kFormatA) {
        SetFormat(edges_[i].to);
      }
    }
  }

  void BuildGraph(const std::vector<std::pair<size_t, FormatType>> &original_nodes) {
    for (size_t i = 0; i < origin_nodes_num_; ++i) {
      // link the source node to the nodes with FormatA,
      // link the nodes with FormatB to the sink node.
      if (original_nodes[i].second == kFormatA) {
        AddEdge(source_id_, original_nodes[i].first, INF, 0);
      } else if (original_nodes[i].second == kFormatB) {
        AddEdge(original_nodes[i].first, sink_id_, INF, 0);
      }
      // each nodes was split into two part, input part and output part.
      // the input part's id is the original node's id, the output part's id is input id + origin_nodes_num_.
      AddEdge(original_nodes[i].first, original_nodes[i].first + origin_nodes_num_, 1, 1);
    }
    for (auto e : original_edges_) {
      auto from = e.first, to = e.second;
      AddEdge(from + origin_nodes_num_, to, 1, 1);
    }
  }

 public:
  MinCut(const std::vector<std::pair<size_t, FormatType>> &original_nodes,
         const std::vector<std::pair<size_t, size_t>> &original_edges)
      : origin_nodes_num_(original_nodes.size()),
        sink_id_(2 * original_nodes.size() + 1),  // source_id_ is 0
        nodes_(2 * original_nodes.size() + 2),    // double nodes, and source_node/sink_node
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
      if (nodes_[i].format == kFormatA && nodes_[i + origin_nodes_num_].format != kFormatA) {
        (void)one_node_ops.emplace_back(i, kTransAB);
      } else if (nodes_[i].format != kFormatA && nodes_[i + origin_nodes_num_].format == kFormatA) {
        (void)one_node_ops.emplace_back(i, kTransBA);
      }
    }
    return one_node_ops;
  }

  std::vector<std::pair<std::pair<size_t, size_t>, TransOpType>> GetTwoNodeOps() const {
    std::vector<std::pair<std::pair<size_t, size_t>, TransOpType>> two_node_ops;
    for (auto i : original_edges_) {
      if (nodes_[i.first + origin_nodes_num_].format == kFormatA && nodes_[i.second].format != kFormatA) {
        (void)two_node_ops.emplace_back(i, kTransAB);
      } else if (nodes_[i.first + origin_nodes_num_].format != kFormatA && nodes_[i.second].format == kFormatA) {
        (void)two_node_ops.emplace_back(i, kTransBA);
      }
    }
    return two_node_ops;
  }

 private:
  size_t origin_nodes_num_;
  size_t source_id_{0};
  size_t sink_id_;
  std::vector<Vertex> nodes_;
  std::vector<Edge> edges_;
  std::vector<std::pair<size_t, size_t>> original_edges_;
};
}  // namespace

using graphkernel::LiteGraph;
using graphkernel::LiteGraphPtr;
using graphkernel::Node;
using graphkernel::NodePtr;
using graphkernel::NodePtrList;
using graphkernel::NType;
using graphkernel::PrimOp;
using graphkernel::PrimOpPtr;

class TransformOp {
 public:
  explicit TransformOp(const NodePtr &node)
      : op_(node->As<PrimOp>()->op()), format_a_(node->input(0)->format), format_b_(node->format) {}
  ~TransformOp() = default;
  bool IsTransformOp(const NodePtr &node) {
    if (node->NodeType() != NType::Primitive || node->As<PrimOp>()->op() != op_) {
      return false;
    }
    if (node->input(0)->format == format_a_ && node->format == format_b_) {
      return true;
    } else if (node->input(0)->format == format_b_ && node->format == format_a_) {
      return true;
    }
    return false;
  }

  FormatType GetFormatType(const std::string &fmt) {
    return fmt == format_a_ ? FormatType::kFormatA : FormatType::kFormatB;
  }

  NodePtr GenTransformOp(TransOpType trans_type) {
    // Only support Transpose now
    static std::map<std::pair<std::string, std::string>, std::vector<int64_t>> perm_map = {
      {{kOpFormat_DEFAULT, kOpFormat_NHWC}, {0, 2, 3, 1}},
      {{kOpFormat_NCHW, kOpFormat_NHWC}, {0, 2, 3, 1}},
      {{kOpFormat_NHWC, kOpFormat_NCHW}, {0, 3, 1, 2}},
      {{kOpFormat_NHWC, kOpFormat_DEFAULT}, {0, 3, 1, 2}},
    };
    std::vector<int64_t> perm;
    if (trans_type == TransOpType::kTransAB) {
      perm = perm_map[{format_a_, format_b_}];
    } else {
      perm = perm_map[{format_b_, format_a_}];
    }
    if (perm.empty()) {
      MS_LOG(EXCEPTION) << "unsupported format: " << format_a_ << " to " << format_b_;
    }
    auto op = graphkernel::OpRegistry::Instance().NewOp("Transpose", "new_trans");
    op->SetAttr("perm", MakeValue(perm));
    return op;
  }

 private:
  std::string op_;
  std::string format_a_;
  std::string format_b_;
};

bool IsFlexibleOp(const NodePtr &node) {
  static const std::set<std::string> format_flexible_ops = {
    "Abs",  "Add",     "Sub",     "Mul",   "Round",   "Cast",         "Neg",  "Exp",       "Log",
    "Pow",  "Minimum", "Maximum", "Rsqrt", "Sqrt",    "Reciprocal",   "Tanh", "Sin",       "Cos",
    "Asin", "ACos",    "RealDiv", "Equal", "Greater", "GreaterEqual", "Less", "LessEqual", "Sign"};
  if (node->NodeType() != NType::Primitive) {
    return false;
  }
  if (format_flexible_ops.count(node->As<PrimOp>()->op()) == 0) {
    return false;
  }
  // check the input and output formats are all the same, except ConstValue.
  for (auto &inp : node->inputs()) {
    if (inp->NodeType() != NType::Value && inp->format != node->format) {
      return false;
    }
  }
  return true;
}

class Mutator {
 public:
  explicit Mutator(const NodePtr &node) : op_checker_(node), basenode_(node), ori_node_(1) {}
  ~Mutator() = default;
  bool Run() {
    VisitNode(basenode_);
    if (flexible_ops_.empty()) return false;
    // remove transform ops in litegraph
    RemoveTransOp();
    GenFormatGraph();
    RebuildLiteGraph();
    return true;
  }

 private:
  // visit nodes bidirectionally
  void VisitNode(const NodePtr &node) {
    if (visited_.count(node) > 0) return;
    (void)visited_.insert(node);
    if (op_checker_.IsTransformOp(node)) {
      (void)trans_ops_.insert(node);
    } else if (!IsFlexibleOp(node)) {
      if (node->NodeType() != NType::Output) {
        fmt_type[{node, -1}] = op_checker_.GetFormatType(node->format);
      }
      if (node->NodeType() != NType::Parameter) {
        for (size_t i = 0; i < node->inputs().size(); i++) {
          if (node->input(i)->NodeType() == NType::Value) {
            continue;
          }
          fmt_type[{node, i}] = op_checker_.GetFormatType(node->input(i)->format);
        }
      }
      return;
    } else {
      (void)flexible_ops_.insert(node);
      fmt_type[{node, -1}] = FormatType::kFormatUnknown;
    }

    for (auto &input : node->inputs()) {
      if (input->NodeType() != NType::Value) {
        VisitNode(input);
      }
    }
    for (auto &user : node->users()) {
      VisitNode(user.first->shared_from_this());
    }
  }

  void RemoveTransOp() {
    for (auto &node : trans_ops_) {
      (void)visited_.erase(node);
      node->ReplaceWith(node->input(0));
      // clear inputs, so that the node will not be the basenode again.
      node->SetInputs({});
    }
    trans_ops_.clear();
  }

  void GenFormatGraph() {
    for (auto &node : visited_) {
      if (node->NodeType() == NType::Parameter) continue;
      bool is_flexible = (flexible_ops_.find(node) != flexible_ops_.end());
      size_t cur_id = 0;
      if (is_flexible) {
        cur_id = GetId({node, -1});
      }
      for (size_t i = 0; i < node->inputs().size(); i++) {
        if (visited_.count(node->input(i)) == 0) continue;
        if (!is_flexible) {
          cur_id = GetId({node, SizeToInt(i)});
        }
        auto input_id = GetId({node->input(i), -1});
        (void)graph_edges_.emplace_back(input_id, cur_id);
      }
    }
  }

  void RebuildLiteGraph() {
    MinCut min_cut(graph_vertex_, graph_edges_);
    min_cut.Run();
    for (auto [node_id, trans_type] : min_cut.GetOneNodeOps()) {
      if (ori_node_[node_id].second != -1) {
        MS_LOG(EXCEPTION) << "OneNodeOp should be the output edge. node_id:" << node_id
                          << " index:" << ori_node_[node_id].second;
      }
      auto trans_op = op_checker_.GenTransformOp(trans_type);
      ori_node_[node_id].first->ReplaceWith(trans_op);
      trans_op->SetInputs({ori_node_[node_id].first});
    }

    std::map<size_t, NodePtr> trans_op_cache;
    for (auto [edge, trans_type] : min_cut.GetTwoNodeOps()) {
      auto node_id_from = edge.first;
      auto node_id_to = edge.second;
      if (ori_node_[node_id_from].second != -1) {
        MS_LOG(EXCEPTION) << "node_from should be the output edge. node_id:" << node_id_from
                          << " index:" << ori_node_[node_id_from].second;
      }
      auto node_from = ori_node_[node_id_from].first;
      auto node_to = ori_node_[node_id_to].first;
      auto &trans_op = trans_op_cache[node_id_from];
      if (trans_op == nullptr) {
        trans_op = op_checker_.GenTransformOp(trans_type);
        trans_op->SetInputs({node_from});
      }
      if (ori_node_[node_id_to].second >= 0) {
        node_to->SetInput(IntToSize(ori_node_[node_id_to].second), trans_op);
      } else {
        for (size_t i = 0; i < node_to->inputs().size(); i++) {
          if (node_to->input(i) == node_from) {
            node_to->SetInput(i, trans_op);
          }
        }
      }
    }
  }

  size_t GetId(const std::pair<NodePtr, int> &node) {
    // the nodes are indexed from 1 in the MinCut model.
    auto &id = node_id_[node];
    if (id == 0) {
      id = node_id_.size();
      ori_node_.push_back(node);
      // set format_type for new id.
      (void)graph_vertex_.emplace_back(id, fmt_type[node]);
    }
    return id;
  }

  TransformOp op_checker_;
  NodePtr basenode_;
  std::set<NodePtr> flexible_ops_;
  std::set<NodePtr> trans_ops_;
  std::set<NodePtr> visited_;

  std::map<std::pair<NodePtr, int>, FormatType> fmt_type;
  std::map<std::pair<NodePtr, int>, size_t> node_id_;
  std::vector<std::pair<NodePtr, int>> ori_node_;
  std::vector<std::pair<size_t, FormatType>> graph_vertex_;
  std::vector<std::pair<size_t, size_t>> graph_edges_;
};

bool TransformOpOptimizer::Process(const LiteGraphPtr &litegraph, const std::string &trans_op_name) {
  ori_trans_op_num_ = 0;
  auto &ops = litegraph->ops();
  bool changed = true;
  auto check_is_trans_op = [&trans_op_name](const NodePtr &node) { return node->As<PrimOp>()->op() == trans_op_name; };
  auto ori_trans_op_num = std::count_if(ops.begin(), ops.end(), check_is_trans_op);
  for (auto &op : ops) {
    if (check_is_trans_op(op) && !op->inputs().empty() && op->input(0)->format != op->format) {
      auto mutator = Mutator(op);
      changed = mutator.Run() || changed;
    }
  }
  if (!changed) return false;
  auto &new_ops = litegraph->GetOrderedNodes();
  auto new_trans_op_num = std::count_if(new_ops.begin(), new_ops.end(), check_is_trans_op);
  if (new_trans_op_num >= ori_trans_op_num) {
    return false;
  }
  for (auto &op : new_ops) {
    op->SetBaseInfo(op->As<PrimOp>()->Infer(op->inputs(), op->attrs()));
  }
  return true;
}

bool TransformOpOptimizer::Run(const FuncGraphPtr &kernel_graph) {
  auto mng = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(kernel_graph->get_return());
  bool changed = false;
  for (auto node : todos) {
    if (!AnfAlgo::IsGraphKernel(node)) continue;
    auto sub_func_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
    auto litegraph = AnfGraph2LiteGraph(sub_func_graph);
    if (Process(litegraph)) {
      changed = true;
      AnfNodePtrList outputs;
      auto new_funcgraph = LiteGraph2AnfGraph(litegraph, &outputs);
      new_funcgraph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, sub_func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
      auto cnode = node->cast<CNodePtr>();
      AnfNodePtrList inputs(cnode->inputs().begin() + 1, cnode->inputs().end());
      auto new_node = CreateNewFuseCNode(kernel_graph, new_funcgraph, inputs, outputs);
      SetNewKernelInfo(new_node, new_funcgraph, inputs, outputs);
      (void)mng->Replace(node, new_node);
      mng->AddFuncGraph(new_funcgraph);
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
