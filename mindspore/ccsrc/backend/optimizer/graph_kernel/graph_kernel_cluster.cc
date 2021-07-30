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
#include "backend/optimizer/graph_kernel/graph_kernel_cluster.h"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <memory>
#include <utility>
#include <fstream>

#include "base/core_ops.h"
#include "ir/graph_utils.h"
#include "debug/common.h"
#include "utils/context/graph_kernel_flags.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/optimizer/pass/getitem_tuple.h"
#include "backend/optimizer/graph_kernel/update_state_formatter.h"

namespace mindspore {
namespace opt {
namespace {
std::vector<PrimitivePtr> GetClusterableOpList() {
  std::vector<PrimitivePtr> clusterable_ops = {
    prim::kPrimAbs,
    prim::kPrimAdd,
    prim::kPrimCast,
    prim::kPrimEqual,
    prim::kPrimExp,
    prim::kPrimInplaceAssign,
    prim::kPrimLog,
    prim::kPrimMaximum,
    prim::kPrimMinimum,
    prim::kPrimMul,
    prim::kPrimNeg,
    prim::kPrimPow,
    prim::kPrimRealDiv,
    prim::kPrimReciprocal,
    prim::kPrimReduceSum,
    prim::kPrimReshape,
    prim::kPrimRound,
    prim::kPrimRsqrt,
    prim::kPrimSqrt,
    prim::kPrimSub,
    prim::kPrimTanh,
    prim::kPrimTranspose,
#if ENABLE_D
    prim::kPrimMatMul,
    prim::KPrimTransData,
    prim::kPrimBatchMatMul,
#elif ENABLE_GPU
    prim::kPrimACos,
    prim::kPrimAcosh,
    prim::kPrimArgMax,
    prim::kPrimArgMin,
    prim::kPrimAsin,
    prim::kPrimAsinh,
    prim::kPrimAssign,
    prim::kPrimAtan,
    prim::kPrimAtan2,
    prim::kPrimCos,
    prim::kPrimDiv,
    prim::kPrimErf,
    prim::kPrimExpm1,
    prim::kPrimFloor,
    prim::kPrimFloorDiv,
    prim::kPrimFloorMod,
    prim::kPrimGreater,
    prim::kPrimGreaterEqual,
    prim::kPrimIsFinite,
    prim::kPrimIsInf,
    prim::kPrimIsNan,
    prim::kPrimLess,
    prim::kPrimLessEqual,
    prim::kPrimLogicalAnd,
    prim::kPrimLogicalOr,
    prim::kPrimLogicalNot,
    prim::kPrimMod,
    prim::kPrimNotEqual,
    prim::kPrimReduceMax,
    prim::kPrimReduceMin,
    prim::kPrimSelect,
    prim::kPrimSign,
    prim::kPrimSin,
#endif
  };
  const auto &flags = context::GraphKernelFlags::GetInstance();
  OpListFilter(&clusterable_ops, flags.enable_cluster_ops_only, flags.enable_cluster_ops, flags.disable_cluster_ops);
  return clusterable_ops;
}

size_t CountGraphKernelInnerNodes(const AnfNodePtr &node) {
  AnfNodePtrList node_list;
  kernel::GetValidKernelNodes(AnfAlgo::GetCNodeFuncGraphPtr(node), &node_list);
  return node_list.size();
}
}  // namespace

bool IsClusterableOp(const AnfNodePtr &node) {
  if (AnfAlgo::IsGraphKernel(node)) {
    return true;
  }
  if (IsKeepBasicNode(node)) {
    return false;
  }
  auto op_list = GetClusterableOpList();
  bool node_in_oplist = std::any_of(op_list.begin(), op_list.end(),
                                    [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  if (!node_in_oplist) {
    return false;
  }
#if ENABLE_D
  // For AICPU operators, only the Reshape can be clustered.
  if (AnfAlgo::GetProcessor(node) != kernel::Processor::AICORE && !IsPrimitiveCNode(node, prim::kPrimReshape)) {
    return false;
  }
#endif
  return true;
}

class Graph {
  struct Cluster {
    size_t cluster_id_;        // node_id of the representative.
    size_t cluster_size_{1};   // size of cluster, composite node is considered as one node.
    size_t basic_op_cnt_{1};   // basic node count, the inner nodes of composite node are counted.
    std::set<size_t> inputs_;  // inputs' cluster_id.
    size_t seed_{0};           // visited flag of dfs.
    size_t max_node_id_;       // largest node id of a cluster

    Cluster(size_t node_id, const AnfNodePtr &node, const std::unordered_map<AnfNodePtr, size_t> &node_idx_map)
        : cluster_id_(node_id), max_node_id_(node_id) {
      if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
        basic_op_cnt_ = 0;
      } else if (AnfAlgo::IsGraphKernel(node)) {
        // the basic_op_cnt_ is used to limit the composite op size
        basic_op_cnt_ = CountGraphKernelInnerNodes(node);
      }
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      for (const auto &inp : cnode->inputs()) {
        auto iter = node_idx_map.find(inp);
        if (iter != node_idx_map.end()) {
          // At the beginning, cluster_id is equal to node_id
          inputs_.insert(iter->second);
        }
      }
    }
    ~Cluster() = default;

    void Merge(Cluster *other_cluster) {
      other_cluster->cluster_id_ = cluster_id_;
      max_node_id_ = std::max(other_cluster->max_node_id_, max_node_id_);
      cluster_size_ += other_cluster->cluster_size_;
      basic_op_cnt_ += other_cluster->basic_op_cnt_;
      std::for_each(other_cluster->inputs_.begin(), other_cluster->inputs_.end(),
                    [this](size_t inp) { this->inputs_.insert(inp); });
      other_cluster->Clean();
    }

    // clean the info to free memory.
    void Clean() {
      inputs_.clear();
      cluster_size_ = 0;
      basic_op_cnt_ = 0;
      max_node_id_ = 0;
    }
  };  // struct Cluster

 public:
  // Init and build graph
  Graph(const AnfNodePtrList &nodes, const std::unordered_map<AnfNodePtr, size_t> &node_idx_map) {
    clusters_.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
      clusters_.emplace_back(i, nodes[i], node_idx_map);
    }
  }
  ~Graph() = default;

  // find the representative of the cluster
  int Find(size_t node_id) {
    size_t &pre_id = clusters_[node_id].cluster_id_;
    return (pre_id == clusters_[pre_id].cluster_id_) ? pre_id : (pre_id = Find(pre_id));
  }

  // merge clusters, the smallest cluster id will be the new cluster id.
  void Merge(const std::vector<size_t> &candidates) {
    size_t min_id = *std::min_element(candidates.begin(), candidates.end());
    for (auto id : candidates) {
      if (id == min_id) continue;
      clusters_[min_id].Merge(&clusters_[id]);
    }
  }

  // Collect nodes together that are in the same cluster.
  std::vector<std::vector<size_t>> CollectClusters() {
    std::vector<std::vector<size_t>> cluster_map(clusters_.size());
    for (size_t i = 0; i < clusters_.size(); i++) {
      cluster_map[Find(i)].push_back(i);
    }
    return cluster_map;
  }

  // Get cluster's max node id
  size_t GetClusterMaxNodeId(size_t cluster_id) { return clusters_[Find(cluster_id)].max_node_id_; }

  using VisitFunc = std::function<IncludeType(size_t)>;
  void Dfs(size_t node_id, VisitFunc visitor) {
    ++seen_;
    return DepthFirstSearch(Find(node_id), visitor);
  }

  // Get cluster size
  size_t GetSize(size_t cluster_id) { return clusters_[Find(cluster_id)].cluster_size_; }

  // Get cluster's basic op count
  size_t GetBasicNodeCount(size_t cluster_id) { return clusters_[Find(cluster_id)].basic_op_cnt_; }

  // Get cluster's inputs
  const std::set<size_t> &GetInputs(size_t cluster_id) {
    cluster_id = Find(cluster_id);
    RefreshInputs(cluster_id);
    return clusters_[cluster_id].inputs_;
  }

 private:
  void RefreshInputs(size_t i) {
    auto &inputs = clusters_[i].inputs_;
    for (auto iter = inputs.begin(); iter != inputs.end();) {
      size_t new_id = Find(*iter);
      if (new_id != *iter) {
        iter = inputs.erase(iter);
        inputs.insert(new_id);
      } else {
        ++iter;
      }
    }
    inputs.erase(i);
  }

  void DepthFirstSearch(size_t cluster_id, const VisitFunc &visitor) {
    if (clusters_[cluster_id].seed_ >= seen_) return;
    clusters_[cluster_id].seed_ = seen_;
    if (visitor(cluster_id) != FOLLOW) {
      return;
    }
    // traverse inputs in descending order.
    const auto &inputs = GetInputs(cluster_id);
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      DepthFirstSearch(*iter, visitor);
    }
  }

  std::vector<Cluster> clusters_;
  size_t seen_{0};
};  // class Graph

class CircleChecker {
 public:
  explicit CircleChecker(GraphPtr graph) : graph_(graph) {}
  ~CircleChecker() = default;

  void RemoveCircle(std::vector<size_t> *candidates) {
    if (candidates->size() <= 1) {
      return;
    }
    candidates_.clear();
    candidates_.insert(candidates->begin(), candidates->end());
    for (auto iter = candidates->begin(); iter != candidates->end(); ++iter) {
      if (!candidates_.count(*iter)) continue;
      circle_nodes_.clear();
      if (CheckCircle(*iter)) {
        RemoveCircleNodesFromCandidates();
      }
    }
    candidates->erase(std::remove_if(candidates->begin(), candidates->end(),
                                     [this](size_t c) { return this->candidates_.count(c) == 0; }),
                      candidates->end());
  }

 private:
  /**
   * Check circle. the candidate is collected into circle_nodes_ if it will form a circle.
   *
   * algorithm:
   * Search from the basenode's input that is NOT in candidates (the basenode is a candidate),
   * If it depends on a node that belongs to candidates, it will form a circle.
   *  e.g.     A -> x -> ... -> B
   *             -> y -> ... -> C
   * In this case, A, B and C are candidates while x and y are not.
   * Both x and y are inputs of A. assumes A is the basenode.
   * When searching from x, the B will be found and added into circle_nodes list,
   * and then when searching from y, the C will be found and added into circle_nodes list.
   */
  bool CheckCircle(size_t basenode) {
    const auto &inputs = graph_->GetInputs(basenode);
    std::set<size_t> visited_circle_nodes;
    for (auto x : inputs) {
      if (candidates_.count(x)) continue;
      bool has_circle = false;
      std::set<size_t> done;
      auto vis_func = [this, &has_circle, &done, &visited_circle_nodes](size_t node_id) {
        if (done.count(node_id) || acyclic_nodes_.count(node_id) || visited_circle_nodes.count(node_id)) {
          return EXCLUDE;
        }
        done.insert(node_id);
        if (candidates_.count(node_id)) {
          has_circle = true;
          circle_nodes_.push_back(node_id);
          return EXCLUDE;
        }
        // all nodes are indexed by topo order,
        // so if the current node's cluster's max node id is less than the minimal candidate, a circle cannot be formed
        // from this node.
        if (candidates_.empty() || graph_->GetClusterMaxNodeId(node_id) < *candidates_.begin()) {
          return EXCLUDE;
        }
        return FOLLOW;
      };
      graph_->Dfs(x, vis_func);
      if (has_circle) {
        visited_circle_nodes.insert(done.begin(), done.end());
      } else {
        acyclic_nodes_.insert(done.begin(), done.end());
      }
    }
    return !circle_nodes_.empty();
  }

  // remove all circle nodes from candidates
  void RemoveCircleNodesFromCandidates() {
    auto remove_from_candidates = [this](size_t node_id) {
      if (candidates_.count(node_id)) {
        candidates_.erase(node_id);
        return FOLLOW;
      }
      return EXCLUDE;
    };
    for (auto node : circle_nodes_) {
      graph_->Dfs(node, remove_from_candidates);
    }
  }

 private:
  GraphPtr graph_;               // bind the global graph
  std::set<size_t> candidates_;  // bind the input candidates
  std::vector<size_t> circle_nodes_;
  std::set<size_t> acyclic_nodes_;
};  // CircleChecker

std::vector<size_t> GraphKernelCluster::FindCandidates(size_t basenode_id) {
  std::vector<size_t> candidates;
  auto include = [this, &candidates, func_graph = nodes_[basenode_id]->func_graph()](size_t cluster_id) {
    const AnfNodePtr &node = this->nodes_[cluster_id];
    if (node->func_graph() != func_graph) {
      return EXCLUDE;
    }
    if (!IsClusterableOp(node) && !IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      return EXCLUDE;
    }
    candidates.push_back(cluster_id);
    // Do not search from clustered node again.
    if (this->graph_->GetSize(cluster_id) > 1) {
      return NOFOLLOW;
    }
    return FOLLOW;
  };
  graph_->Dfs(basenode_id, include);
  std::reverse(candidates.begin(), candidates.end());
  return candidates;
}

bool GraphKernelCluster::Process(const FuncGraphPtr &func_graph) {
  bool changed = false;
  for (int i = nodes_.size() - 1; i >= 0; i--) {
    // if the node has been clustered, it has tried to find its previous nodes, so it's unnecessary to try again.
    if (graph_->GetSize(i) > 1) {
      continue;
    }
    auto candidates = FindCandidates(i);
    CircleChecker(graph_).RemoveCircle(&candidates);
    RemoveWildGetitem(&candidates);
    if (candidates.empty()) continue;
    // merge candidates into one cluster
    graph_->Merge(candidates);
  }

  // Rebuild func_graphs
  auto clusters = graph_->CollectClusters();
  for (size_t i = 0; i < clusters.size(); i++) {
    auto node_without_getitem = std::count_if(clusters[i].begin(), clusters[i].end(), [this](size_t node_id) {
      return !IsPrimitiveCNode(this->nodes_[node_id], prim::kPrimTupleGetItem);
    });
    if (node_without_getitem == 0) continue;
    if (node_without_getitem == 1) {
      // Do not cluster a single GraphKernel again.
      // Do not cluster a single Assign.
      const auto &node = nodes_[clusters[i][0]];
      if (AnfAlgo::IsGraphKernel(node) || IsPrimitiveCNode(node, prim::kPrimAssign) || !IsClusterableOp(node)) {
        continue;
      }
    }
    CreateFuncGraph(func_graph, clusters[i]);
    changed = true;
  }
  return changed;
}

void GraphKernelCluster::CreateFuncGraph(const FuncGraphPtr &func_graph, const std::vector<size_t> &nodes_id) {
  AnfNodePtrList old_nodes;
  AnfNodePtr new_node;
  std::transform(nodes_id.begin(), nodes_id.end(), std::back_inserter(old_nodes),
                 [this](size_t id) { return this->nodes_[id]; });
  std::tie(new_node, std::ignore) = FuseNodesToSubGraph(old_nodes, func_graph, "fusion");
  std::shared_ptr<Pass> eliminate_getitem_pass = std::make_shared<opt::GetitemTuple>();
  eliminate_getitem_pass->Run(AnfAlgo::GetCNodeFuncGraphPtr(new_node));
  if (context::GraphKernelFlags::GetInstance().dump_as_text) {
    DumpClusterInfo(old_nodes, new_node);
  }
}

void GraphKernelCluster::DumpClusterInfo(const AnfNodePtrList &old_nodes, const AnfNodePtr &new_node) {
#ifdef ENABLE_DUMP_IR
  dump_buf_ << "Source nodes of " << new_node->fullname_with_scope() << " = " << new_node->DebugString() << std::endl;
  for (const auto &node : old_nodes) {
    dump_buf_ << "  " << node->fullname_with_scope() << " = " << node->DebugString() << std::endl;
  }
  dump_buf_ << "=======================" << std::endl;
#endif
}

void GraphKernelCluster::DumpToFile() {
#ifdef ENABLE_DUMP_IR
  auto pathname = std::string("./") + kGraphKernelDumpPath + "/graph_kernel_cluster.txt";
  auto realpath = Common::GetRealPath(pathname);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << pathname;
    return;
  }
  std::ofstream fout(realpath.value(), std::ios::app);
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << realpath.value() << "' failed!";
    return;
  }
  fout << dump_buf_.str() << std::endl;
  fout.close();
#endif
}

// The GetItem node should be clustered with its real input.
// If its real input is not in the candidates, the GetItem should be excluded.
void GraphKernelCluster::RemoveWildGetitem(std::vector<size_t> *candidates) {
  bool changed = false;
  std::set<size_t> candidates_set(candidates->begin(), candidates->end());

  for (auto iter = candidates_set.begin(); iter != candidates_set.end();) {
    size_t cluster_id = *iter;
    if (IsPrimitiveCNode(nodes_[cluster_id], prim::kPrimTupleGetItem)) {
      const auto &inputs = graph_->GetInputs(cluster_id);
      if (inputs.size() != 1) {
        MS_LOG(ERROR) << "Input size of GetItem(" << cluster_id << ") should be 1, but got " << inputs.size();
        candidates->clear();
        return;
      }
      auto prev_id = *(inputs.begin());
      if (!candidates_set.count(prev_id)) {
        iter = candidates_set.erase(iter);
        changed = true;
        continue;
      }
    }
    ++iter;
  }
  if (changed) {
    candidates->erase(std::remove_if(candidates->begin(), candidates->end(),
                                     [&candidates_set](size_t c) { return candidates_set.count(c) == 0; }),
                      candidates->end());
  }
}

void GraphKernelCluster::Init(const FuncGraphPtr &func_graph) {
  // process cnode only
  nodes_ = TopoSort(func_graph->get_return(), SuccIncoming,
                    [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  for (size_t i = 0; i < nodes_.size(); i++) {
    node_idx_map_[nodes_[i]] = i;
  }
  graph_ = std::make_shared<Graph>(nodes_, node_idx_map_);
  MS_EXCEPTION_IF_NULL(graph_);
}

bool GraphKernelCluster::Run(const FuncGraphPtr &func_graph) {
  (void)std::make_shared<ShrinkUpdateState>()->Run(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  Init(func_graph);
  bool changed = Process(func_graph);
  if (changed) {
    if (context::GraphKernelFlags::GetInstance().dump_as_text) {
      DumpToFile();
    }
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  Clean();
  (void)std::make_shared<SpreadUpdateState>()->Run(func_graph);
  return changed;
}
}  // namespace opt
}  // namespace mindspore
