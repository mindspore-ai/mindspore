/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/core/graph_kernel_cluster.h"

#include <algorithm>
#include <set>
#include <utility>
#include <fstream>
#include <string>

#include "utils/hash_map.h"
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "utils/file_utils.h"
#include "ops/sequence_ops.h"
#include "ops/nn_optimizer_ops.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
Graph::Cluster::Cluster(size_t node_id, const AnfNodePtr &node,
                        const mindspore::HashMap<AnfNodePtr, size_t> &node_idx_map)
    : cluster_id_(node_id) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (const auto &inp : cnode->inputs()) {
    auto iter = node_idx_map.find(inp);
    if (iter != node_idx_map.end()) {
      // At the beginning, cluster_id is equal to node_id
      (void)inputs_.insert(iter->second);
    }
  }
}

void Graph::Cluster::Merge(Cluster *other_cluster) {
  other_cluster->cluster_id_ = cluster_id_;
  cluster_size_ += other_cluster->cluster_size_;
  inputs_.insert(other_cluster->inputs_.cbegin(), other_cluster->inputs_.cend());
  other_cluster->Clean();
}

GraphPtr Graph::Build(const FuncGraphPtr &func_graph, AnfNodePtrList *nodes,
                      HashMap<AnfNodePtr, size_t> *node_idx_map) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto cnodes = TopoSort(func_graph->output(), SuccIncoming,
                         [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  HashMap<AnfNodePtr, size_t> tmp_node_idx_map;
  for (size_t i = 0; i < cnodes.size(); i++) {
    tmp_node_idx_map[cnodes[i]] = i;
  }
  auto graph_ptr = std::make_shared<Graph>(cnodes, tmp_node_idx_map);
  if (nodes != nullptr) {
    *nodes = std::move(cnodes);
  }
  if (node_idx_map != nullptr) {
    *node_idx_map = std::move(tmp_node_idx_map);
  }
  return graph_ptr;
}

Graph::Graph(const AnfNodePtrList &nodes, const HashMap<AnfNodePtr, size_t> &node_idx_map) {
  clusters_.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); i++) {
    (void)clusters_.emplace_back(i, nodes[i], node_idx_map);
  }
}

size_t Graph::Find(size_t node_id) {
  size_t &pre_id = clusters_[node_id].cluster_id_;
  return (pre_id == clusters_[pre_id].cluster_id_) ? pre_id : (pre_id = Find(pre_id));
}

void Graph::Merge(const std::vector<size_t> &candidates) {
  size_t min_id = *std::min_element(candidates.begin(), candidates.end());
  for (auto id : candidates) {
    if (id == min_id) {
      continue;
    }
    clusters_[min_id].Merge(&clusters_[id]);
  }
}

std::vector<std::vector<size_t>> Graph::CollectClusters() {
  std::vector<std::vector<size_t>> cluster_map(clusters_.size());
  for (size_t i = 0; i < clusters_.size(); i++) {
    cluster_map[Find(i)].push_back(i);
  }
  return cluster_map;
}

void Graph::Dfs(size_t node_id, const Graph::VisitFunc &visitor) {
  ++seen_;
  return DepthFirstSearch(Find(node_id), visitor);
}

const std::set<size_t> &Graph::GetInputs(size_t cluster_id) {
  cluster_id = Find(cluster_id);
  RefreshInputs(cluster_id);
  return clusters_[cluster_id].inputs_;
}

void Graph::RefreshInputs(size_t i) {
  auto &inputs = clusters_[i].inputs_;
  for (auto iter = inputs.cbegin(); iter != inputs.cend();) {
    size_t new_id = Find(*iter);
    if (new_id != *iter) {
      iter = inputs.erase(iter);
      (void)inputs.insert(new_id);
    } else {
      ++iter;
    }
  }
  (void)inputs.erase(i);
}

void Graph::DepthFirstSearch(size_t cluster_id, const VisitFunc &visitor) {
  if (clusters_[cluster_id].seed_ >= seen_) {
    return;
  }
  clusters_[cluster_id].seed_ = seen_;
  if (visitor(cluster_id) != FOLLOW) {
    return;
  }
  // traverse inputs in descending order.
  const auto &inputs = GetInputs(cluster_id);
  for (auto iter = inputs.crbegin(); iter != inputs.crend(); ++iter) {
    DepthFirstSearch(*iter, visitor);
  }
}

void CircleChecker::RemoveCircle(std::vector<size_t> *candidates) {
  if (candidates->size() <= 1) {
    return;
  }
  candidates_.clear();
  candidates_.insert(candidates->cbegin(), candidates->cend());
  for (auto iter = candidates->cbegin(); iter != candidates->cend(); ++iter) {
    if (candidates_.count(*iter) == 0) {
      continue;
    }
    circle_nodes_.clear();
    if (CheckCircle(*iter)) {
      RemoveCircleNodesFromCandidates();
    }
  }
  (void)candidates->erase(std::remove_if(candidates->begin(), candidates->end(),
                                         [this](size_t c) { return this->candidates_.count(c) == 0; }),
                          candidates->end());
}

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
bool CircleChecker::CheckCircle(size_t basenode) {
  const auto &inputs = graph_->GetInputs(basenode);
  std::set<size_t> visited_circle_nodes;
  for (auto x : inputs) {
    if (candidates_.count(x) > 0) {
      continue;
    }
    bool has_circle = false;
    std::set<size_t> done;
    auto vis_func = [this, &has_circle, &done, &visited_circle_nodes](size_t node_id) {
      if (done.count(node_id) > 0 || acyclic_nodes_.count(node_id) > 0 || visited_circle_nodes.count(node_id) > 0) {
        return EXCLUDE;
      }
      (void)done.insert(node_id);
      if (candidates_.count(node_id) > 0) {
        has_circle = true;
        circle_nodes_.push_back(node_id);
        return EXCLUDE;
      }
      return FOLLOW;
    };
    graph_->Dfs(x, vis_func);
    if (has_circle) {
      visited_circle_nodes.insert(done.cbegin(), done.cend());
    } else {
      acyclic_nodes_.insert(done.cbegin(), done.cend());
    }
  }
  return !circle_nodes_.empty();
}

void CircleChecker::RemoveCircleNodesFromCandidates() {
  auto remove_from_candidates = [this](size_t node_id) {
    if (candidates_.count(node_id) > 0) {
      (void)candidates_.erase(node_id);
      return FOLLOW;
    }
    return EXCLUDE;
  };
  for (auto node : circle_nodes_) {
    graph_->Dfs(node, remove_from_candidates);
  }
}

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
  for (int i = SizeToInt(nodes_.size()) - 1; i >= 0; i--) {
    // if the node has been clustered, it has tried to find its previous nodes, so it's unnecessary to try again.
    if (graph_->GetSize(IntToSize(i)) > 1) {
      continue;
    }
    auto candidates = FindCandidates(IntToSize(i));
    CircleChecker circle_checker(graph_);
    circle_checker.RemoveCircle(&candidates);
    RemoveWildGetitem(&candidates);
    if (candidates.empty()) {
      continue;
    }
    // merge candidates into one cluster
    graph_->Merge(candidates);
  }

  // Rebuild func_graphs
  auto clusters = graph_->CollectClusters();
  for (size_t i = 0; i < clusters.size(); i++) {
    auto node_without_getitem = std::count_if(clusters[i].begin(), clusters[i].end(), [this](size_t node_id) {
      return !IsPrimitiveCNode(this->nodes_[node_id], prim::kPrimTupleGetItem);
    });
    if (node_without_getitem == 0) {
      continue;
    }
    if (node_without_getitem == 1) {
      // Do not cluster a single GraphKernel again.
      // Do not cluster a single Assign.
      const auto &node = nodes_[clusters[i][0]];
      if (AnfUtils::IsGraphKernel(node) || IsPrimitiveCNode(node, prim::kPrimAssign) || !IsClusterableOp(node)) {
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
  (void)std::transform(nodes_id.begin(), nodes_id.end(), std::back_inserter(old_nodes),
                       [this](size_t id) { return this->nodes_[id]; });
  auto new_node = ReplaceNodesWithGraphKernelNode(old_nodes, func_graph, "fusion");
  if (GraphKernelFlags::GetInstance().dump_as_text) {
    DumpClusterInfo(old_nodes, new_node);
  }
}

void GraphKernelCluster::DumpClusterInfo(const AnfNodePtrList &old_nodes, const AnfNodePtr &new_node) {
  dump_buf_ << "Source nodes of " << new_node->fullname_with_scope() << " = " << new_node->DebugString() << std::endl;
  for (const auto &node : old_nodes) {
    dump_buf_ << "  " << node->fullname_with_scope() << " = " << node->DebugString() << std::endl;
  }
  dump_buf_ << "=======================" << std::endl;
}

void GraphKernelCluster::DumpToFile() {
  auto dir_path = FileUtils::CreateNotExistDirs(std::string("./") + kGraphKernelDumpPath);
  if (!dir_path.has_value()) {
    MS_LOG(ERROR) << "Failed to CreateNotExistDirs: ./" << kGraphKernelDumpPath;
    return;
  }
  std::string filepath = dir_path.value() + "/" + "graph_kernel_cluster.txt";
  ChangeFileMode(filepath, S_IWUSR);
  std::ofstream fout(filepath, std::ios::app);
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << filepath << "' failed!";
    ChangeFileMode(filepath, S_IRUSR);
    return;
  }
  fout << dump_buf_.str() << std::endl;
  fout.close();
  ChangeFileMode(filepath, S_IRUSR);
}

// The GetItem node should be clustered with its real input.
// If its real input is not in the candidates, the GetItem should be excluded.
void GraphKernelCluster::RemoveWildGetitem(std::vector<size_t> *candidates) {
  bool changed = false;
  std::set<size_t> candidates_set(candidates->begin(), candidates->end());

  for (auto iter = candidates_set.cbegin(); iter != candidates_set.cend();) {
    size_t cluster_id = *iter;
    if (IsPrimitiveCNode(nodes_[cluster_id], prim::kPrimTupleGetItem)) {
      const auto &inputs = graph_->GetInputs(cluster_id);
      if (inputs.size() != 1) {
        MS_LOG(ERROR) << "Input size of GetItem(" << cluster_id << ") should be 1, but got " << inputs.size();
        candidates->clear();
        return;
      }
      auto prev_id = *(inputs.cbegin());
      if (candidates_set.count(prev_id) == 0) {
        iter = candidates_set.erase(iter);
        changed = true;
        continue;
      }
    }
    ++iter;
  }
  if (changed) {
    (void)candidates->erase(std::remove_if(candidates->begin(), candidates->end(),
                                           [&candidates_set](size_t c) { return candidates_set.count(c) == 0; }),
                            candidates->end());
  }
}

void GraphKernelCluster::Init(const FuncGraphPtr &func_graph) {
  op_list_ = GetClusterableOpList();
  graph_ = Graph::Build(func_graph, &nodes_);
  MS_EXCEPTION_IF_NULL(graph_);
}

bool GraphKernelCluster::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  Init(func_graph);
  bool changed = Process(func_graph);
  if (changed) {
    if (GraphKernelFlags::GetInstance().dump_as_text) {
      DumpToFile();
    }
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  Clean();
  return changed;
}
}  // namespace mindspore::graphkernel
