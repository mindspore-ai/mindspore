/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/core/transform_op_optimizer.h"
#include <algorithm>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <utility>
#include <string>
#include <tuple>
#include "mindspore/core/ops/core_ops.h"
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "backend/common/graph_kernel/model/lite_graph.h"
#include "backend/common/graph_kernel/model/op_register.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
namespace {
constexpr size_t INF = static_cast<size_t>(1) << 30;
class MinCut {
 private:
  struct Edge {
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

  void SetFormat(size_t node_id) {
    nodes_[node_id].format = FormatType::kFormatA;
    for (size_t i : nodes_[node_id].out_edges) {
      if (edges_[i].capacity > 0 && nodes_[edges_[i].to].format != FormatType::kFormatA) {
        SetFormat(edges_[i].to);
      }
    }
  }

  void BuildGraph(const std::vector<std::pair<size_t, FormatType>> &original_nodes) {
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
      auto from = e.first, to = e.second;
      AddEdge(from + origin_nodes_num_, to, 1, 1);
    }
  }

 public:
  MinCut(const std::vector<std::pair<size_t, FormatType>> &original_nodes,
         const std::vector<std::pair<size_t, size_t>> &original_edges)
      : origin_nodes_num_(original_nodes.size()),
        source_id_(0),
        sink_id_(2 * original_nodes.size() + 1),
        nodes_(2 * original_nodes.size() + 2),  // double nodes, and source_node/sink_node
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
      } else if (nodes_[i].format != FormatType::kFormatA &&
                 nodes_[i + origin_nodes_num_].format == FormatType::kFormatA) {
        (void)one_node_ops.emplace_back(tmpi, TransOpType::kTransBA);
      }
    }
    return one_node_ops;
  }

  std::vector<std::pair<std::pair<size_t, size_t>, TransOpType>> GetTwoNodeOps() const {
    std::vector<std::pair<std::pair<size_t, size_t>, TransOpType>> two_node_ops;
    for (auto i : original_edges_) {
      if (nodes_[i.first + origin_nodes_num_].format == FormatType::kFormatA &&
          nodes_[i.second].format != FormatType::kFormatA) {
        (void)two_node_ops.emplace_back(i, TransOpType::kTransAB);
      } else if (nodes_[i.first + origin_nodes_num_].format != FormatType::kFormatA &&
                 nodes_[i.second].format == FormatType::kFormatA) {
        (void)two_node_ops.emplace_back(i, TransOpType::kTransBA);
      }
    }
    return two_node_ops;
  }

 private:
  size_t origin_nodes_num_;
  size_t source_id_;
  size_t sink_id_;
  std::vector<Vertex> nodes_;
  std::vector<Edge> edges_;
  std::vector<std::pair<size_t, size_t>> original_edges_;
};
}  // namespace

using inner::LiteGraph;
using inner::LiteGraphPtr;
using inner::NodePtrList;
using inner::NType;
using inner::PrimOp;
using inner::PrimOpPtr;
using NodeWithIndex = std::pair<NodePtr, int>;

TransformOp::TransformOp(const NodePtr &node)
    : op_(node->As<PrimOp>()->op()), format_a_(node->input(0)->format), format_b_(node->format) {}

bool TransformOp::IsTransformOp(const NodePtr &node) {
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

FormatType TransformOp::GetFormatType(const std::string &fmt) {
  return fmt == format_a_ ? FormatType::kFormatA : FormatType::kFormatB;
}

bool TransformOpCreator::IsTransOp(const NodePtr &node) const {
  return node->NodeType() == NType::Primitive && node->As<PrimOp>()->op() == op_name_;
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
    op->SetAttr("perm", MakeValue(perm));
    op->SetAttr(kAttrDstFormat, MakeValue(dst_format));
    return op;
  }
};

class LayoutTransformHandle : public TransformOp {
 public:
  using TransformOp::TransformOp;
  NodePtr GenTransformOp(const NodePtr &, TransOpType trans_type) override {
    auto op = inner::OpRegistry::Instance().NewOp(op_);
    if (trans_type == TransOpType::kTransAB) {
      op->SetAttr(kAttrSrcFormat, MakeValue(format_a_));
      op->SetAttr(kAttrDstFormat, MakeValue(format_b_));
    } else {
      op->SetAttr(kAttrSrcFormat, MakeValue(format_b_));
      op->SetAttr(kAttrDstFormat, MakeValue(format_a_));
    }
    return op;
  }
};

bool IsFlexibleOp(const NodePtr &node) {
  if (node->NodeType() != NType::Primitive) {
    return false;
  }
  if (node->As<PrimOp>()->compute_type() != PrimOp::ComputeType::ELEMWISE) {
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

constexpr int kOutputIndex = -1;
class Mutator {
 public:
  enum class ResultStatus { kChanged, kUnchanged, kRollback };
  Mutator(const NodePtr &node, const TransformOpPtr &handle) : op_handle_(handle), basenode_(node), ori_node_(1) {}
  ~Mutator() = default;

  ResultStatus Run(std::set<NodePtr> *changed_nodes) {
    VisitNode(basenode_);
    if (flexible_ops_.empty() && trans_ops_.size() <= 1) {
      return ResultStatus::kUnchanged;
    }
    // remove transform ops in litegraph
    RemoveTransOp();
    GenFormatGraph();
    if (!RebuildLiteGraph(changed_nodes)) {
      return ResultStatus::kRollback;
    }
    changed_nodes->insert(flexible_ops_.begin(), flexible_ops_.end());
    return ResultStatus::kChanged;
  }

 private:
  // visit nodes bidirectionally
  void VisitNode(const NodePtr &node) {
    if (visited_.count(node) > 0) {
      return;
    }
    (void)visited_.insert(node);
    if (op_handle_->IsTransformOp(node)) {
      (void)trans_ops_.insert(node);
    } else if (!IsFlexibleOp(node)) {
      if (node->NodeType() != NType::Output) {
        fmt_type[{node, kOutputIndex}] = op_handle_->GetFormatType(node->format);
      }
      if (node->NodeType() != NType::Parameter) {
        for (size_t i = 0; i < node->inputs().size(); i++) {
          if (node->input(i)->NodeType() == NType::Value) {
            continue;
          }
          fmt_type[{node, i}] = op_handle_->GetFormatType(node->input(i)->format);
        }
      }
      return;
    } else {
      (void)flexible_ops_.insert(node);
      fmt_type[{node, kOutputIndex}] = FormatType::kFormatUnknown;
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
        cur_id = GetId({node, kOutputIndex});
      }
      for (size_t i = 0; i < node->inputs().size(); i++) {
        if (visited_.count(node->input(i)) == 0) {
          continue;
        }
        if (!is_flexible) {
          cur_id = GetId({node, SizeToInt(i)});
        }
        auto input_id = GetId({node->input(i), kOutputIndex});
        (void)graph_edges_.emplace_back(input_id, cur_id);
      }
    }
  }

  std::pair<bool, NodePtr> NewTransOp(const NodePtr &input, TransOpType trans_type, std::set<NodePtr> *changed_nodes) {
    NodePtr trans_op = nullptr;
    // a trick, if the node's size of 1, it's not need to insert transform op.
    if (input->tensor_size() <= 1) {
      return std::make_pair(true, trans_op);
    }
    trans_op = op_handle_->GenTransformOp(input, trans_type);
    if (trans_op == nullptr) {
      return std::make_pair(false, trans_op);
    }
    (void)changed_nodes->insert(trans_op);
    return std::make_pair(true, trans_op);
  }

  bool RebuildLiteGraph(std::set<NodePtr> *changed_nodes) {
    MinCut min_cut(graph_vertex_, graph_edges_);
    min_cut.Run();
    for (auto [node_id, trans_type] : min_cut.GetOneNodeOps()) {
      if (ori_node_[node_id].second != kOutputIndex) {
        MS_LOG(EXCEPTION) << "OneNodeOp should be the output edge. node_id:" << node_id
                          << " index:" << ori_node_[node_id].second;
      }
      auto input_node = ori_node_[node_id].first;
      auto [result, trans_op] = NewTransOp(input_node, trans_type, changed_nodes);
      if (!result) {
        return false;
      }
      if (trans_op == nullptr) {
        continue;
      }
      input_node->ReplaceWith(trans_op);
      trans_op->SetInputs({input_node});
    }

    std::map<size_t, NodePtr> trans_op_cache;
    for (auto [edge, trans_type] : min_cut.GetTwoNodeOps()) {
      auto node_id_from = edge.first;
      auto node_id_to = edge.second;
      if (ori_node_[node_id_from].second != kOutputIndex) {
        MS_LOG(EXCEPTION) << "node_from should be the output edge. node_id:" << node_id_from
                          << " index:" << ori_node_[node_id_from].second;
      }
      auto node_from = ori_node_[node_id_from].first;
      auto node_to = ori_node_[node_id_to].first;
      if (trans_op_cache.count(node_id_from) == 0) {
        auto [result, trans_op] = NewTransOp(node_from, trans_type, changed_nodes);
        if (!result) {
          return false;
        }
        if (trans_op == nullptr) {
          continue;
        }
        trans_op_cache[node_id_from] = trans_op;
        trans_op->SetInputs({node_from});
      }
      auto trans_op = trans_op_cache[node_id_from];
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
    return true;
  }

  size_t GetId(const NodeWithIndex &node_with_index) {
    // the nodes are indexed from 1 in the MinCut model.
    auto &id = node_id_[node_with_index];
    if (id == 0) {
      id = node_id_.size();
      ori_node_.push_back(node_with_index);
      // set format_type for new id.
      (void)graph_vertex_.emplace_back(id, fmt_type[node_with_index]);
    }
    return id;
  }

  TransformOpPtr op_handle_;
  NodePtr basenode_;
  std::set<NodePtr> flexible_ops_;
  std::set<NodePtr> trans_ops_;
  std::set<NodePtr> visited_;

  std::map<NodeWithIndex, FormatType> fmt_type;
  std::map<NodeWithIndex, size_t> node_id_;
  std::vector<NodeWithIndex> ori_node_;  // node_id to NodePtr, this vector is indexed from 1
  std::vector<std::pair<size_t, FormatType>> graph_vertex_;
  std::vector<std::pair<size_t, size_t>> graph_edges_;
};

bool TransformOpOptimizer::Process(const LiteGraphPtr &litegraph, const TransformOpCreator &creator) const {
  auto &ops = litegraph->ops();
  bool changed = false;
  auto check_is_trans_op = [&creator](const NodePtr &node) { return creator.IsTransOp(node); };
  auto ori_trans_op_num = std::count_if(ops.begin(), ops.end(), check_is_trans_op);
  std::set<NodePtr> nodes_may_change;
  for (auto &op : ops) {
    if (check_is_trans_op(op) && !op->inputs().empty() && op->input(0)->format != op->format) {
      auto mutator = Mutator(op, creator.CreateHandle(op));
      auto ret = mutator.Run(&nodes_may_change);
      if (ret == Mutator::ResultStatus::kRollback) {
        return false;
      }
      changed = changed || (ret == Mutator::ResultStatus::kChanged);
    }
  }
  if (!changed) {
    return false;
  }
  auto &new_ops = litegraph->GetOrderedNodes();
  auto new_trans_op_num = std::count_if(new_ops.begin(), new_ops.end(), check_is_trans_op);
  if (new_trans_op_num >= ori_trans_op_num) {
    return false;
  }
  for (auto &op : new_ops) {
    if (nodes_may_change.count(op) != 0) {
      op->SetBaseInfo(op->As<PrimOp>()->Infer(op->inputs(), op->attrs()));
    }
  }
  return true;
}

void TransformOpOptimizer::Init() {
  (void)supported_ops_.emplace_back(TRANS_OP_CREATOR("Transpose", TransposeHandle));
  (void)supported_ops_.emplace_back(TRANS_OP_CREATOR("LayoutTransform", LayoutTransformHandle));
}

bool TransformOpOptimizer::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  for (auto node : todos) {
    if (!AnfUtils::IsGraphKernel(node)) {
      continue;
    }
    for (const auto &creator : supported_ops_) {
      auto sub_func_graph = GetCNodeFuncGraph(node);
      auto litegraph = GkUtils::AnfGraph2LiteGraph(sub_func_graph);
      if (Process(litegraph, creator)) {
        changed = true;
        auto new_funcgraph = GkUtils::LiteGraph2AnfGraph(litegraph, Callback::Instance());
        MS_EXCEPTION_IF_NULL(new_funcgraph);
        new_funcgraph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, sub_func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
        auto cnode = node->cast<CNodePtr>();
        AnfNodePtrList inputs(cnode->inputs().begin() + 1, cnode->inputs().end());
        auto new_node = CreateNewFuseCNode(func_graph, new_funcgraph, inputs);
        (void)mng->Replace(node, new_node);
        mng->AddFuncGraph(new_funcgraph);
      }
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
