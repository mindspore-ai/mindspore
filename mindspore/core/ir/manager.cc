/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "ir/manager.h"

#include <algorithm>
#include <list>

#include "ir/func_graph.h"
#include "utils/convert_utils_base.h"
#include "utils/counter.h"
#include "utils/trace_base.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace change {

struct Edge {
  CNodePtr cnode;
  int index;
  AnfNodePtr input;
  Edge(const CNodePtr &cnode, int index, const AnfNodePtr &input) : cnode(cnode), index(index), input(input) {}
  ~Edge() = default;
};

struct EdgeHash {
  std::size_t operator()(const Edge &e) const noexcept {
    const std::hash<AnfNodePtr> node_hash;
    return hash_combine({node_hash(e.cnode), IntToSize(e.index), node_hash(e.input)});
  }
};

struct EdgeEqual {
  bool operator()(const Edge &lhs, const Edge &rhs) const noexcept {
    return lhs.cnode == rhs.cnode && lhs.index == rhs.index && lhs.input == rhs.input;
  }
};

using EdgeCounter = Counter<Edge, EdgeHash, EdgeEqual>;
using NodeCounter = Counter<AnfNodePtr>;

struct ChangeCounter {
  EdgeCounter new_edges;
  EdgeCounter del_edges;
  NodeCounter new_nodes;
  NodeCounter del_nodes;

  template <typename Func>
  void ForEachAddedEdges(Func &&func) {
    new_edges.subtract_by(del_edges, std::forward<Func>(func));
  }

  template <typename Func>
  void ForEachRemovedEdges(Func &&func) {
    del_edges.subtract_by(new_edges, std::forward<Func>(func));
  }

  std::vector<AnfNodePtr> GetAddedNodes() { return new_nodes.subtract(del_nodes); }
  std::vector<AnfNodePtr> GetRemovedNodes() { return del_nodes.subtract(new_nodes); }
};

class SetEdge : public Change {
 public:
  SetEdge(const CNodePtr &cnode, int index, const AnfNodePtr &input) : edge_{cnode, index, input} {}
  ~SetEdge() override = default;

  void Apply(ChangeCounter *counter) override {
    auto &old_input = edge_.cnode->input(IntToSize(edge_.index));
    counter->del_nodes.add(old_input);
    counter->del_edges.add(edge_.cnode, edge_.index, old_input);
    edge_.cnode->set_input(IntToSize(edge_.index), edge_.input);
    counter->new_nodes.add(edge_.input);
    counter->new_edges.add(std::move(edge_));
  }

 private:
  Edge edge_;
};

class AddEdge : public Change {
 public:
  AddEdge(const CNodePtr &cnode, const AnfNodePtr &input) : cnode_{cnode}, input_{input} {}
  ~AddEdge() override = default;

  void Apply(ChangeCounter *counter) override {
    int index = static_cast<int>(cnode_->size());
    cnode_->add_input(input_);
    counter->new_nodes.add(input_);
    counter->new_edges.add(std::move(cnode_), index, std::move(input_));
  }

 private:
  CNodePtr cnode_;
  AnfNodePtr input_;
};

class SetParams : public Change {
 public:
  SetParams(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &params)
      : func_graph_{func_graph}, params_{params} {}
  ~SetParams() override = default;

  void Apply(ChangeCounter *counter) override {
    auto &old_params = func_graph_->parameters();
    for (auto &p : old_params) {
      counter->del_nodes.add(p);
    }
    func_graph_->set_parameters(params_);
    for (auto &p : params_) {
      counter->new_nodes.add(std::move(p));
    }
  }

 private:
  FuncGraphPtr func_graph_;
  std::vector<AnfNodePtr> params_;
};

class AddParam : public Change {
 public:
  AddParam(const FuncGraphPtr &func_graph, const ParameterPtr &param) : func_graph_{func_graph}, param_{param} {}
  ~AddParam() override = default;

  void Apply(ChangeCounter *counter) override {
    func_graph_->append_parameter(param_);
    counter->new_nodes.add(std::move(param_));
  }

 private:
  FuncGraphPtr func_graph_;
  ParameterPtr param_;
};

class InsertFrontParam : public Change {
 public:
  InsertFrontParam(const FuncGraphPtr &func_graph, const ParameterPtr &param)
      : func_graph_{func_graph}, param_{param} {}
  ~InsertFrontParam() override = default;

  void Apply(ChangeCounter *counter) override {
    func_graph_->PrependParameter(param_);
    counter->new_nodes.add(std::move(param_));
  }

 private:
  FuncGraphPtr func_graph_;
  ParameterPtr param_;
};

}  // namespace change

FuncGraphManagerPtr MakeManager(const std::vector<FuncGraphPtr> &func_graphs, bool manage) {
  auto m = std::make_shared<FuncGraphManager>(func_graphs, manage);
  m->Init();
  return m;
}

FuncGraphManagerPtr Manage(const std::vector<FuncGraphPtr> &func_graphs, bool manage) {
  FuncGraphManagerPtr m = nullptr;
  bool root = false;

  for (auto &fg : func_graphs) {
    if (fg == nullptr) {
      continue;
    }
    if (fg->manager() != nullptr) {
      m = fg->manager();
      break;
    }
  }

  if (m == nullptr) {
    std::vector<FuncGraphPtr> tmp;
    m = MakeManager(tmp, manage);
    root = true;
  }

  for (auto &fg : func_graphs) {
    if (fg == nullptr) {
      continue;
    }
    m->AddFuncGraph(fg, root);
  }
  return m;
}

FuncGraphManagerPtr Manage(FuncGraphPtr func_graph, bool manage) {
  std::vector<FuncGraphPtr> func_graphs = {func_graph};
  return Manage(func_graphs, manage);
}

FuncGraphManager::FuncGraphManager(const std::vector<FuncGraphPtr> &roots, bool manage)
    : roots_(roots), is_manage_(manage) {
  Reset();
}

void FuncGraphManager::Reset() {
  func_graphs_ = FuncGraphSet();
  func_graphs_index_ = FuncGraphIndexMap();
  all_nodes_ = AnfNodeSet();
  node_users_ = NodeUsersMap();
  signals_ = std::make_shared<Signals>();
  func_graph_parents_total_ = std::make_shared<FuncGraphParentsTotalComputer>(this);
  func_graph_parent_ = std::make_shared<ParentComputer>(this);
  children_ = std::make_shared<ChildrenComputer>(this);
  scopes_ = std::make_shared<ScopeComputer>(this);
  free_variables_total_ = std::make_shared<FVTotalComputer>(this);
  func_graphs_used_total_ = std::make_shared<FuncGraphsUsedTotalComputer>(this);
  recursive_ = std::make_shared<RecursiveComputer>(this);
  meta_fg_prim_total_ = std::make_shared<FuncGraphMetaFgPrimTotalComputer>(this);
}

void FuncGraphManager::Init() {
  auto roots = roots_;
  roots_ = FuncGraphSet();

  for (auto &fg : roots) {
    AddFuncGraph(fg, true);
  }
}

FuncGraphSet &FuncGraphManager::func_graph_parents_total(const FuncGraphPtr &fg) const {
  if (fg == nullptr) {
    MS_LOG(EXCEPTION) << "The parameter 'fg' should not be null.";
  }
  MS_LOG(DEBUG) << "Start func_graph_parents_total func graph " << fg->ToString();
  func_graph_parents_total_->Recompute(fg);
  MS_LOG(DEBUG) << "End func_graph_parents func graph " << fg->ToString();
  return func_graph_parents_total_->func_graph_parents_total_analysis()[fg];
}

FuncGraphPtr FuncGraphManager::parent(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(func_graph_parent_);
  MS_LOG(DEBUG) << "Start parents func graph " << fg->ToString();
  func_graph_parent_->Recompute(fg);
  if (func_graph_parent_->parent_analysis().count(fg) == 0) {
    MS_LOG(WARNING) << "This func graph is not in manager:" << fg->ToString();
    return nullptr;
  }
  MS_LOG(DEBUG) << "End parents func graph " << fg->ToString();
  return func_graph_parent_->parent_analysis()[fg];
}

FuncGraphSet &FuncGraphManager::children(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(children_);
  MS_LOG(DEBUG) << "Start child func graph " << fg->ToString();
  children_->Recompute(fg);
  return children_->children_analysis()[fg];
}

FuncGraphSet &FuncGraphManager::scopes(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(scopes_);
  MS_LOG(DEBUG) << "Start scopes func graph:" << fg->ToString();
  scopes_->Recompute(fg);
  MS_LOG(DEBUG) << "End scopes func graph:" << fg->ToString();
  return scopes_->scope_analysis()[fg];
}

FVTotalMap &FuncGraphManager::free_variables_total() const {
  MS_EXCEPTION_IF_NULL(free_variables_total_);
  free_variables_total_->Recompute();
  return free_variables_total_->fv_total_analysis();
}

FuncGraphSet &FuncGraphManager::func_graphs_used_total(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(func_graphs_used_total_);
  func_graphs_used_total_->Recompute(fg);
  return func_graphs_used_total_->func_graph_used_total_analysis()[fg];
}

const FuncGraphIndexPtr &FuncGraphManager::func_graph_index(const FuncGraphPtr &fg) const {
  auto iter = func_graphs_index_.find(fg);
  if (iter == func_graphs_index_.end()) {
    MS_LOG(EXCEPTION) << "Func graph: " << fg->ToString() << " is not add FuncGraphIndexMap.";
  }
  return func_graphs_index_.at(fg);
}

bool FuncGraphManager::recursive(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(recursive_);
  recursive_->Recompute(fg);
  if (recursive_->recursive_analysis().count(fg) == 0) {
    MS_LOG(WARNING) << "This func graph is not in manager: " << fg->ToString();
    return false;
  }
  return recursive_->recursive_analysis()[fg];
}

std::shared_ptr<std::list<FuncGraphPtr>> FuncGraphManager::recursive_graphs(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(recursive_);
  if (recursive(fg)) {
    if (recursive_->recursive_map().count(fg) == 0) {
      auto trace = std::list<FuncGraphPtr>();
      recursive_->CheckRecursiveGraphs(fg, &trace);
    }
    if (recursive_->recursive_map().count(fg) == 0) {
      MS_LOG(WARNING) << "This func graph is not in manager: " << fg->ToString();
      return nullptr;
    }
    return recursive_->recursive_map()[fg];
  } else {
    return nullptr;
  }
}

// Check if the function graph embed with `MetaFGPrim`, which currently covers kPrimJ and kPrimVmap and kPrimTaylor.
bool FuncGraphManager::func_graph_meta_fg_prim_total(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(meta_fg_prim_total_);
  MS_EXCEPTION_IF_NULL(fg);
  meta_fg_prim_total_->Recompute(fg);
  if (meta_fg_prim_total_->meta_fg_prim_total_analysis().count(fg) == 0) {
    MS_LOG(WARNING) << "This func graph is not in manager: " << fg->ToString();
    return false;
  }
  return meta_fg_prim_total_->meta_fg_prim_total_analysis()[fg];
}

// Add a func graph to this manager, optionally as a root func graph.
void FuncGraphManager::AddFuncGraph(const FuncGraphPtr &func_graph, bool is_root) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (is_root) {
    roots_.add(func_graph);
  }
  if (func_graphs_.contains(func_graph)) {
    return;
  }

  // Add func_graph as a managed graph.
  AddIntoManaged(func_graph);

  // New nodes to be acquired.
  std::vector<AnfNodePtr> new_nodes = func_graph->parameters();
  auto return_node = func_graph->get_return();
  if (return_node != nullptr) {
    (void)new_nodes.emplace_back(std::move(return_node));
  }

  (void)func_graphs_index_.emplace(func_graph, std::make_shared<FuncGraphPassIndex>());

  // Acquire all nodes from func_graph.
  AcquireNodes(std::move(new_nodes));
}

// Clear the all information in manager
void FuncGraphManager::Clear() noexcept {
  for (auto graph : func_graphs_) {
    graph->DecAttachedMngCnt();
    if (graph->attached_mng_cnt() == 0) {
      graph->ClearAllManagerInfo();
    } else if (graph->attached_mng_cnt() < 0) {
      MS_LOG(EXCEPTION) << "graph:" << graph->ToString() << " attached cnt not right:" << graph->attached_mng_cnt();
    }
  }

  func_graphs_.clear();
  func_graphs_index_.clear();
  all_nodes_.clear();
  node_users_.clear();
  roots_.clear();

  signals_->InvalidateComputer();
}

void FuncGraphManager::KeepRoots(const std::vector<FuncGraphPtr> &func_graphs) {
  MS_LOG(DEBUG) << "Start keep roots";
  bool root_exist = false;
  for (auto &item : func_graphs) {
    if (roots_.contains(item)) {
      root_exist = true;
      break;
    }
  }

  // if the new_root in roots_, we add new_root first, then calculate the func_graphs
  // relation to new_root, remove the func_graphs not relation to new_root
  // if the new_root not in roots_, we clear the all func_graphs in manager
  // then add the new_root
  if (root_exist || func_graphs.empty()) {
    FuncGraphSet roots(func_graphs);
    if (roots.empty()) {
      roots = roots_;
    } else {
      roots_.clear();
      for (auto &item : roots) {
        AddFuncGraph(item, true);
      }
    }

    FuncGraphSet keep;
    for (auto &item : roots) {
      MS_LOG(DEBUG) << "roots: " << item->ToString();
      keep.update(func_graphs_used_total(item));
#ifdef DEBUG
      for (auto &k : keep) {
        MS_LOG(DEBUG) << "keep: " << k->ToString();
      }
#endif
    }
    MaybeDropFuncGraphs(func_graphs_ - keep, true);
  } else {
    Clear();
    FuncGraphSet roots(func_graphs);
    for (auto &item : roots) {
      AddFuncGraph(item, true);
    }
  }
}

void FuncGraphManager::RemoveRoots() {
  MS_LOG(DEBUG) << "Start remove roots";
  roots_.clear();
  MaybeDropFuncGraphs(func_graphs_, true);
}

void FuncGraphManager::AddIntoManaged(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  if (is_manage_) {
    if (fg->manager() != nullptr && fg->manager().get() != this) {
      MS_LOG(INFO) << "A func graph can only have one manager.";
    }
    fg->set_manager(shared_from_this());
  }
  func_graphs_.add(fg);
  fg->IncAttachedMngCnt();
}

void FuncGraphManager::MaybeDropFuncGraphs(const FuncGraphSet &func_graphs, bool ignore_users) {
  std::list<FuncGraphPtr> todo(func_graphs.begin(), func_graphs.end());
  std::set<FuncGraphPtr> dropped;
  while (!todo.empty()) {
    FuncGraphPtr func_graph = std::move(todo.front());
    MS_EXCEPTION_IF_NULL(func_graph);
    todo.pop_front();
    MS_LOG(DEBUG) << "Maybe drop func graph " << func_graph->ToString();
    if (roots_.contains(func_graph)) {
      MS_LOG(DEBUG) << "Cannot drop as roots contains func graph: " << func_graph->ToString();
      continue;
    }
    auto &users_cnode_index = func_graph->func_graph_cnodes_index();
    if (!users_cnode_index.empty() && !ignore_users) {
      MS_LOG(DEBUG) << "Cannot drop as users not empty: " << func_graph->ToString();
      continue;
    }
    if (dropped.find(func_graph) != dropped.end()) {
      MS_LOG(DEBUG) << "Func graph had been dropped " << func_graph->ToString();
      continue;
    }
    (void)dropped.insert(func_graph);
    std::vector<AnfNodePtr> return_vec = {func_graph->get_return()};
    auto drop_graphs = MaybeDropNodes(std::move(return_vec));
    (void)todo.insert(todo.end(), drop_graphs.begin(), drop_graphs.end());
  }
  for (auto &fg : dropped) {
    MS_EXCEPTION_IF_NULL(fg);
    all_nodes_.difference_update(fg->parameters());
    EraseOneGraph(fg);
    if (fg->manager().get() == this) {
      fg->set_manager(nullptr);
    }
    MS_LOG(DEBUG) << "Func graph dropped " << fg->ToString();
  }
}

void FuncGraphManager::ProcessEdgeAdd(const AnfNodePtr &node, int index, const AnfNodePtr &input) {
  if (IsValueNode<FuncGraph>(input)) {
    AddFuncGraph(GetValueNode<FuncGraphPtr>(input));
  }
  auto &users_node = node_users_[input];
  users_node.add(std::make_pair(node, index));
  OnEdgeAdded(node, index, input);
}

void FuncGraphManager::ProcessEdgeRemove(const AnfNodePtr &node, int index, const AnfNodePtr &input) {
  auto iter = node_users_.find(input);
  if (iter == node_users_.end()) {
    return;
  }
  bool removed = iter->second.erase(std::make_pair(node, index));
  if (removed) {
    OnEdgeRemoved(node, index, input);
  }
}

void FuncGraphManager::ProcessInputsEdgeAdd(const CNodePtr &cnode) {
  const size_t count = cnode->size();
  for (size_t i = 0; i < count; ++i) {
    ProcessEdgeAdd(cnode, static_cast<int>(i), cnode->input(i));
  }
}

void FuncGraphManager::ProcessInputsEdgeRemove(const CNodePtr &cnode) {
  const size_t count = cnode->size();
  for (size_t i = 0; i < count; ++i) {
    ProcessEdgeRemove(cnode, static_cast<int>(i), cnode->input(i));
  }
}

static inline void FollowGraph(const FuncGraphPtr &fg, SeenNum seen, std::vector<AnfNodePtr> *nodes) {
  if (fg == nullptr) {
    return;
  }
  if (auto ret = fg->get_return(); ret != nullptr && ret->seen_ != seen) {
    (void)nodes->emplace_back(std::move(ret));
  }
}

void FuncGraphManager::AcquireNodes(std::vector<AnfNodePtr> &&nodes) {
  auto seen = NewSeenGeneration();
  while (!nodes.empty()) {
    // Take the last one.
    auto node = std::move(nodes.back());
    nodes.pop_back();
    MS_EXCEPTION_IF_NULL(node);
    // Skip visited nodes.
    if (node->seen_ == seen) {
      continue;
    }
    node->seen_ = seen;
    // Try add it to all_nodes_.
    auto insert_result = all_nodes_.insert(node);
    if (insert_result.second == false) {
      // Skip acquired nodes.
      continue;
    }
    // Add node to its func_graph.
    auto fg = node->func_graph();
    if (fg != nullptr) {
      fg->AddNode(node);
    }
    // Follow graph for value node.
    if (node->isa<ValueNode>()) {
      auto graph = GetValueNode<FuncGraphPtr>(node);
      FollowGraph(graph, seen, &nodes);
      continue;
    }
    // Follow graph for cnode or parameter.
    FollowGraph(fg, seen, &nodes);
    // Handle cnode.
    auto cnode = node->cast<CNodePtr>();
    if (cnode != nullptr) {
      // Handle input edges.
      ProcessInputsEdgeAdd(cnode);
      // Follow inputs.
      auto &inputs = cnode->inputs();
      (void)nodes.insert(nodes.end(), inputs.begin(), inputs.end());
    }
  }
}

FuncGraphSet FuncGraphManager::MaybeDropNodes(std::vector<AnfNodePtr> &&nodes) {
  FuncGraphSet drop_func_graphs;
  while (!nodes.empty()) {
    AnfNodePtr node = std::move(nodes.back());
    nodes.pop_back();
    if (node == nullptr) {
      // Here can not call 'MS_EXCEPTION_IF_NULL' to throw exception,
      // this method may be triggered by desctuctor.
      MS_LOG(WARNING) << "Node to be dropped is nullptr";
      continue;
    }
    if (!all_nodes_.contains(node)) {
      // Node not existed.
      continue;
    }
    auto &users = node_users_[node];
    if (!users.empty()) {
      // Node is in used.
      continue;
    }
    if (node->isa<Parameter>() && node->func_graph() != nullptr) {
      // Node is a used parameter.
      auto &parameters = node->func_graph()->parameters();
      if (std::find(parameters.begin(), parameters.end(), node) != parameters.end()) {
        continue;
      }
    }
    if (IsValueNode<FuncGraph>(node)) {
      // The FuncGraph may need to be dropped.
      auto fg = GetValueNode<FuncGraphPtr>(node);
      drop_func_graphs.add(fg);
    }
    // Handle cnode.
    if (auto cnode = node->cast<CNodePtr>(); cnode != nullptr) {
      // Remove inputs edges.
      ProcessInputsEdgeRemove(cnode);
      // Handle inputs nodes.
      auto &inputs = cnode->inputs();
      (void)nodes.insert(nodes.end(), inputs.begin(), inputs.end());
    }
    // Remove it from all_nodes_;
    (void)all_nodes_.erase(node);
    // Drop node from its func graph.
    if (auto fg = node->func_graph(); fg != nullptr) {
      fg->DropNode(node);
    }
    // Remove it from node_users.
    (void)node_users_.erase(node);
  }
  return drop_func_graphs;
}

void FuncGraphManager::SetParameters(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &parameters) {
  auto tr = Transact();
  tr.SetParameters(fg, parameters);
  tr.Commit();
}

void FuncGraphManager::AddParameter(const FuncGraphPtr &fg, const AnfNodePtr &parameter) {
  auto tr = Transact();
  tr.AddParameter(fg, parameter);
  tr.Commit();
}

void FuncGraphManager::InsertFrontParameter(const FuncGraphPtr &fg, const AnfNodePtr &parameter) {
  auto tr = Transact();
  tr.InsertFrontParameter(fg, parameter);
  tr.Commit();
}

bool FuncGraphManager::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(new_node);
  auto tr = Transact();
  bool success = tr.Replace(old_node, new_node);
  if (success) {
    tr.Commit();
  }
  return success;
}

bool FuncGraphManager::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node, const AnfNodePtr &mask_node) {
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(new_node);
  auto tr = Transact();
  bool success = tr.Replace(old_node, new_node, mask_node);
  if (success) {
    tr.Commit();
  }
  return success;
}

void FuncGraphManager::SetEdge(const AnfNodePtr &node, int index, const AnfNodePtr &value) {
  auto tr = Transact();
  tr.SetEdge(node, index, value);
  tr.Commit();
}

void FuncGraphManager::AddEdge(const AnfNodePtr &node, const AnfNodePtr &value) {
  auto tr = Transact();
  tr.AddEdge(node, value);
  tr.Commit();
}

void FuncGraphManager::MoveAllCNodeDropGraph(const FuncGraphPtr &source, const FuncGraphPtr &target,
                                             const AnfNodePtr &call_node, const ScopePtr &scope) {
  MS_EXCEPTION_IF_NULL(source);
  CNodePtr source_return = source->get_return();
  MS_EXCEPTION_IF_NULL(source_return);
  AnfNodePtr source_output = source->output();
  const auto &source_prim = source_return->input(0);

  int index = 0;
  (void)node_users_[source_prim].erase(make_pair(source_return, index));
  OnEdgeRemoved(source_return, index, source_prim);
  index = 1;
  (void)node_users_[source_output].erase(make_pair(source_return, index));
  OnEdgeRemoved(source_return, index, source_output);
  (void)all_nodes_.erase(source_return);
  (void)node_users_.erase(source_return);
  source->DropNode(source_return);
  for (auto &node : source->nodes()) {
    node->set_func_graph(target);
    if (node->scope() == kDefaultScope) {
      node->set_scope(scope);
    }
    if (node->isa<CNode>()) {
      MS_LOG(DEBUG) << "Start move inlined node:" << node->DebugString();
      auto new_debug_info = DebugInfo::UpdateInlineCNodeDebugInfo(call_node->debug_info(), node->debug_info());
      auto node_new_debug_info = std::dynamic_pointer_cast<NodeDebugInfo>(new_debug_info);
      node->set_debug_info(node_new_debug_info);
      node_new_debug_info->set_node(node);
    }
  }

  MoveAllNodes(source, target);
  all_nodes_.difference_update(source->parameters());
  EraseOneGraph(source);
  source->set_dropped(true);
  if (source->manager().get() == this) {
    source->set_manager(nullptr);
  }
}

void FuncGraphManager::OnEdgeAdded(const AnfNodePtr &node, int index, const AnfNodePtr &input) {
  auto fg = node->func_graph();
  if (input->isa<ValueNode>()) {
    fg->AddValueNode(input);
    if (IsValueNode<FuncGraph>(input)) {
      auto used = GetValueNode<FuncGraphPtr>(input);
      used->AddFuncGraphCNodeIndex(std::make_shared<CNodeIndexPair>(std::make_pair(node, index)));
      if (fg->AddFuncGraphUsed(used)) {
        signals_->InvalidateComputer();
      }
    }
    if (IsPrimitiveCNode(node, prim::kPrimJ) || IsPrimitiveCNode(node, prim::kPrimVmap) ||
        IsPrimitiveCNode(node, prim::kPrimTaylor) || IsPrimitiveCNode(node, prim::kPrimShard)) {
      fg->AddMetaFgPrimValueNode(input);
    }
  } else if (IsPrimitiveCNode(node, prim::kPrimVmap) && IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
    // To handle the model ensembling scenario in vmap, whose input is a celllist, taking an arbitrary function graph
    // is sufficient.
    constexpr int64_t kIndex1 = 1;
    auto func_union = dyn_cast<CNode>(input);
    if (IsValueNode<FuncGraph>(func_union->input(kIndex1))) {
      fg->AddMetaFgPrimValueNode(func_union->input(kIndex1));
    }
  } else if (fg != nullptr && fg != input->func_graph()) {
    if (fg->AddFreeVariable(input)) {
      signals_->InvalidateComputer();
    }
  }
}

void FuncGraphManager::OnEdgeRemoved(const AnfNodePtr &node, int index, const AnfNodePtr &input) {
  auto fg = node->func_graph();
  if (fg != nullptr && input->isa<ValueNode>()) {
    fg->DropValueNode(input);
    if (IsValueNode<FuncGraph>(input)) {
      auto used = GetValueNode<FuncGraphPtr>(input);
      used->DropFuncGraphCNodeIndex(std::make_shared<CNodeIndexPair>(std::make_pair(node, index)));
      if (fg->DropFuncGraphUsed(used)) {
        signals_->InvalidateComputer();
      }
    }
    if (IsPrimitiveCNode(node, prim::kPrimJ) || IsPrimitiveCNode(node, prim::kPrimVmap) ||
        IsPrimitiveCNode(node, prim::kPrimTaylor)) {
      fg->DropMetaFgPrimValueNode(input);
    }
  } else if (fg != nullptr && fg != input->func_graph()) {
    if (fg->DropFreeVariable(input)) {
      signals_->InvalidateComputer();
    }
  }
}

void FuncGraphManager::MoveAllNodes(const FuncGraphPtr &source, const FuncGraphPtr &target) {
  target->CopyNodes(source);
  target->CopyValueNodes(source);
  target->CopyFuncGraphCNodesIndex(source);
  target->CopyFreeVariables(source);
  target->CopyFuncGraphsUsed(source);
  target->CopyMetaFgPrimValueNodes(source);
  source->ClearAllManagerInfo();
  signals_->InvalidateComputer();
}

void FuncGraphManager::CommitChanges(std::vector<change::ChangePtr> &&changes) {
  // Apply changes.
  change::ChangeCounter counter;
  for (auto &change : changes) {
    change->Apply(&counter);
  }
  changes.clear();

  // Process added edges.
  counter.ForEachAddedEdges([this](const change::Edge &edge) {  //
    ProcessEdgeAdd(edge.cnode, edge.index, edge.input);
  });

  // Process added nodes.
  AcquireNodes(counter.GetAddedNodes());

  // Process removed edges.
  counter.ForEachRemovedEdges([this](const change::Edge &edge) {  //
    ProcessEdgeRemove(edge.cnode, edge.index, edge.input);
  });

  // Process removed nodes.
  auto drop_func_graphs = MaybeDropNodes(counter.GetRemovedNodes());
  if (!drop_func_graphs.empty()) {
    MaybeDropFuncGraphs(drop_func_graphs);
  }
}

void FuncGraphManager::EraseOneGraph(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  bool erase_ret = func_graphs_.erase(fg->shared_from_base<FuncGraph>());
  if (!erase_ret) {
    return;
  }
  fg->DecAttachedMngCnt();
  if (fg->attached_mng_cnt() == 0) {
    fg->ClearAllManagerInfo();
  }
}

void FuncGraphTransaction::SetParameters(FuncGraphPtr fg, const std::vector<AnfNodePtr> &params) {
  (void)changes_.emplace_back(std::make_unique<change::SetParams>(fg, params));
}

void FuncGraphTransaction::AddParameter(FuncGraphPtr fg, const AnfNodePtr &param) {
  (void)changes_.emplace_back(std::make_unique<change::AddParam>(fg, param->cast<ParameterPtr>()));
}

void FuncGraphTransaction::InsertFrontParameter(FuncGraphPtr fg, const AnfNodePtr &param) {
  (void)changes_.emplace_back(std::make_unique<change::InsertFrontParam>(fg, param->cast<ParameterPtr>()));
}

bool FuncGraphTransaction::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(new_node);
  FuncGraphPtr old_func_graph = old_node->func_graph();
  if (old_func_graph != nullptr && old_func_graph->get_return() == old_node) {
    MS_LOG(WARNING) << "Cannot replace the return node of a func graph " << old_func_graph->ToString();
    return false;
  }
  auto &users = manager_->node_users()[old_node];
  for (auto &node : users) {
    SetEdge(node.first, node.second, new_node);
  }
  return true;
}

bool FuncGraphTransaction::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node,
                                   const AnfNodePtr &mask_node) {
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(new_node);
  FuncGraphPtr old_func_graph = old_node->func_graph();
  if (old_func_graph != nullptr && old_func_graph->get_return() == old_node) {
    MS_LOG(WARNING) << "Cannot replace the return node of a func graph " << old_func_graph->ToString();
    return false;
  }
  auto &users = manager_->node_users()[old_node];
  for (auto &node : users) {
    if (node.first == mask_node) {
      SetEdge(node.first, node.second, new_node);
    }
  }
  return true;
}

void FuncGraphTransaction::SetEdge(const AnfNodePtr &src_node, int k, const AnfNodePtr &v) {
  if (k < 0) {
    MS_LOG(EXCEPTION) << "Invalid value k = " << k;
  }
  MS_EXCEPTION_IF_NULL(src_node);
  auto cnode = src_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(EXCEPTION) << "src_node should be a cnode, but cast failed.";
  }
  (void)changes_.emplace_back(std::make_unique<change::SetEdge>(cnode, k, v));
}

void FuncGraphTransaction::AddEdge(const AnfNodePtr &src_node, const AnfNodePtr &v) {
  MS_EXCEPTION_IF_NULL(src_node);
  auto cnode = src_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(EXCEPTION) << "src_node should be a cnode, but cast failed.";
  }
  (void)changes_.emplace_back(std::make_unique<change::AddEdge>(cnode, v));
}

void FuncGraphTransaction::Commit() { manager_->CommitChanges(std::move(changes_)); }

DepComputer::DepComputer(const FuncGraphManager *const manager) : manager_(manager), validate_(false) {
  MS_EXCEPTION_IF_NULL(manager_);
  manager_->signals()->InvalidateComputer.connect(this, &DepComputer::OnInvalidateComputer);
}

void DepComputer::Recompute() {
  if (!validate_) {
    RealRecompute();
    validate_ = true;
  }
}

void DepComputer::Recompute(const FuncGraphPtr &fg) {
  if (func_graphs_validate_.count(fg) == 0 || !func_graphs_validate_[fg]) {
    RealRecompute(fg);
    func_graphs_validate_[fg] = true;
  }
}

FuncGraphSetPtr FuncGraphParentsTotalComputer::SeekParents(
  const FuncGraphPtr &fg, mindspore::HashMap<FuncGraphPtr, FuncGraphSetPtr> *seen_fgs) {
  auto iter = seen_fgs->find(fg);
  if (iter != seen_fgs->end()) {
    return iter->second;
  }
  FuncGraphSetPtr parents = std::make_shared<FuncGraphSet>();

  // Append all the fvs in fg.
  auto &fvs = fg->free_variables();
  for (auto fv : fvs) {
    auto fv_node = fv.first;
    MS_EXCEPTION_IF_NULL(fv_node);
    auto fv_func_graph = fv_node->func_graph();
    if (fv_func_graph == nullptr) {
      MS_LOG(INFO) << "Meet a FV '" << fv_node->DebugString() << "' whose func graph is null, during seeking for "
                   << fg->ToString() << "\nFV: " << trace::GetDebugInfo(fv_node->debug_info());
      continue;
    }
    parents->add(fv_func_graph);
  }

  // Search the fv in fg's child func graph.
  auto &fgs = fg->func_graphs_used();
  for (auto &item : fgs) {
    auto gt = item.first;
    if (gt->seen_ == 1) {
      continue;
    }
    gt->seen_ = 1;
    parents->update(SeekParents(gt, seen_fgs));
    gt->seen_ = 0;
  }
  (void)parents->erase(fg);
  (*seen_fgs)[fg] = parents;
  return parents;
}

void FuncGraphParentsTotalComputer::RealRecompute(FuncGraphPtr fg) {
  MS_EXCEPTION_IF_NULL(fg);
  mindspore::HashMap<FuncGraphPtr, FuncGraphSetPtr> seen_fgs;
  fg->seen_ = 1;
  auto parents = SeekParents(fg, &seen_fgs);
  func_graph_parents_total_analysis_[fg].update(parents);
  fg->seen_ = 0;
}

bool set_len_compare(const FuncGraphSetPair &lhs, const FuncGraphSetPair &rhs) {
  auto l1 = lhs.second.size();
  auto l2 = rhs.second.size();
  return l1 < l2;
}

void ParentComputer::RealRecompute(FuncGraphPtr fg) {
  this->parent_analysis_[fg] = nullptr;
  // Note: must be a copy other than reference as it is modified thereafter.
  auto deps = this->manager_->func_graph_parents_total(fg);
  if (deps.empty()) {
    this->parent_analysis_[fg] = nullptr;
    return;
  } else if (deps.size() == 1) {
    this->parent_analysis_[fg] = deps.front();
    return;
  } else {
    // return nearest parent as parent
    FuncGraphSet deps_copy(deps);
    for (auto &dep : deps) {
      auto parent_deps = this->manager_->func_graph_parents_total(dep);
      for (auto &p_d : parent_deps) {
        if (deps_copy.count(p_d) > 0) {
          (void)deps_copy.erase(p_d);
        }
      }
      if (deps_copy.size() == 1) {
        this->parent_analysis_[fg] = deps_copy.front();
        return;
      }
    }
  }
}

void ChildrenComputer::RealRecompute(FuncGraphPtr fg) {
  MS_EXCEPTION_IF_NULL(manager_);
  auto used_fg_total = manager_->func_graphs_used_total(fg);
  for (auto &used_fg : used_fg_total) {
    if (manager_->parent(used_fg) == fg) {
      children_analysis_[fg].add(used_fg);
    }
  }
}

void ScopeComputer::RealRecompute(FuncGraphPtr fg) {
  MS_EXCEPTION_IF_NULL(manager_);
  auto &children = manager_->children(fg);

  scope_analysis_[fg] = FuncGraphSet();
  scope_analysis_[fg].add(fg);
  for (auto &child : children) {
    scope_analysis_[fg].add(child);
  }
}

void FVTotalComputer::RealRecompute() {
  auto manager = DepComputer::manager_;
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &fg : manager->func_graphs()) {
    fv_total_analysis_[fg] = OrderedMap<BaseRef, int, BaseRefHash>();
  }

  for (auto &fg : manager->func_graphs()) {
    // add all free variable nodes
    AnfNodeCounterMap items = fg->free_variables();
    for (auto &iter : items) {
      auto curr = fg;
      while (curr != nullptr) {
        fv_total_analysis_[curr][iter.first] = iter.second;
        curr = manager->parent(curr);
        if (curr != nullptr) {
          const AnfNodeSet &all_nodes = curr->nodes();
          if (all_nodes.contains(iter.first)) {
            break;
          }
        }
      }
    }

    // add all FGs of free variables
    auto &used = fg->func_graphs_used();
    for (auto &iter : used) {
      auto p = manager->parent(iter.first);
      if (p == nullptr) {
        continue;
      }
      auto curr = fg;
      while (curr != nullptr && curr != p) {
        fv_total_analysis_[curr][iter.first] = iter.second;
        curr = manager->parent(curr);
      }
    }
  }
}

void FuncGraphsUsedTotalComputer::RealRecompute(FuncGraphPtr fg) {
  MS_EXCEPTION_IF_NULL(manager_);
  std::vector<FuncGraphPtr> todo;
  std::vector<FuncGraphPtr> todo_new;

  todo.push_back(fg);
  while (!todo.empty()) {
    todo_new.clear();
    for (auto &gt : todo) {
      for (auto &item : gt->func_graphs_used()) {
        auto used_fg = item.first;
        if (used_fg == fg) {
          func_graph_used_total_analysis_[fg].add(used_fg);
          continue;
        }
        if (func_graph_used_total_analysis_[fg].count(used_fg) == 0) {
          todo_new.push_back(used_fg);
        }
        MS_LOG(DEBUG) << fg->ToString() << " add func graph " << used_fg->ToString();
        func_graph_used_total_analysis_[fg].add(used_fg);
      }
    }
    todo = todo_new;
  }
}

bool CheckRecursive(const FuncGraphManager *const manager, const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<FuncGraphPtr> todo;
  std::vector<FuncGraphPtr> todo_new;
  todo.push_back(fg);
  FuncGraphSet used_total;
  while (!todo.empty()) {
    todo_new.clear();
    for (auto &gt : todo) {
      for (auto &item : gt->func_graphs_used()) {
        auto used_g = item.first;
        if (used_g == fg) {
          return true;
        }
        if (used_total.count(used_g) == 0) {
          todo_new.push_back(used_g);
        }
        used_total.add(used_g);
      }
    }
    todo = todo_new;
  }
  return false;
}

void RecursiveComputer::RealRecompute(FuncGraphPtr fg) {
  this->recursive_analysis_[fg] = CheckRecursive(this->manager_, fg);
}

void RecursiveComputer::CheckRecursiveGraphs(const FuncGraphPtr &fg, std::list<FuncGraphPtr> *trace) {
  MS_EXCEPTION_IF_NULL(trace);
  auto res = std::find(trace->begin(), trace->end(), fg);
  // find recursive
  if (res != trace->end()) {
    auto recur_ptr = std::make_shared<std::list<FuncGraphPtr>>(res, trace->end());
    for (auto iter = res; iter != trace->end(); (void)iter++) {
      MS_LOG(DEBUG) << "Recursive graph " << (*iter)->ToString();
      recursive_map_[*iter] = recur_ptr;
    }
  } else {
    trace->push_back(fg);
    auto &items = fg->func_graphs_used();
    for (auto iter = items.begin(); iter != items.end(); (void)iter++) {
      CheckRecursiveGraphs(iter->first, trace);
    }
    trace->pop_back();
    if (recursive_map_.count(fg) == 0) {
      recursive_map_[fg] = nullptr;
    }
  }
}

bool FuncGraphMetaFgPrimTotalComputer::SeekMetaFgPrim(const FuncGraphPtr &fg, SeenNum seen_num) {
  MS_EXCEPTION_IF_NULL(fg);
  if (fg->seen_ == seen_num) {
    MS_LOG(DEBUG) << fg->ToString() << " had been checked";
    return false;
  }

  // Check MetaFgPrim (J/Vmap/Taylor) FuncGraph input.
  const auto &meta_fg_prim_values = fg->meta_fg_prim_value_nodes();
  if (!meta_fg_prim_values.empty()) {
    auto contains_meta_fg_prim =
      std::find_if(meta_fg_prim_values.begin(), meta_fg_prim_values.end(), [seen_num](const auto &iter) {
        // Check g1->MetaFgPrim(fg)->g2->g cycle.
        if (IsValueNode<FuncGraph>(iter.first)) {
          auto func_graph = GetValuePtr<FuncGraph>(iter.first);
          return func_graph->seen_ != seen_num;
        }
        if (IsValueNode<Primitive>(iter.first)) {
          // Exclude the primitive of MetaFgPrim (J/Vmap/Taylor) itself.
          auto prim = GetValueNode<PrimitivePtr>(iter.first);
          return (prim->name() != prim::kPrimJ->name() && prim->name() != prim::kPrimVmap->name() &&
                  prim->name() != prim::kPrimTaylor->name());
        }
        return false;
      });
    if (contains_meta_fg_prim != meta_fg_prim_values.end()) {
      MS_EXCEPTION_IF_NULL(contains_meta_fg_prim->first);
      MS_LOG(DEBUG) << fg->ToString() << " contains MetaFgPrim(" << contains_meta_fg_prim->first->DebugString() << ")";
      return true;
    }
  }

  // Check MetaFgPrim (J/Vmap/Taylor) CNode as FV.
  const auto &fv_nodes = fg->free_variables();
  if (!fv_nodes.empty()) {
    auto contains_meta_fg_prim_cnode = std::find_if(fv_nodes.begin(), fv_nodes.end(), [seen_num](const auto &iter) {
      // Check if the FV is a MetaFgPrim (J/Vmap/Taylor) call CNode.
      return IsPrimitiveCNode(iter.first, prim::kPrimJ) || IsPrimitiveCNode(iter.first, prim::kPrimVmap) ||
             IsPrimitiveCNode(iter.first, prim::kPrimTaylor);
    });
    if (contains_meta_fg_prim_cnode != fv_nodes.end()) {
      MS_EXCEPTION_IF_NULL(contains_meta_fg_prim_cnode->first);
      MS_LOG(DEBUG) << fg->ToString() << " contains FV MetaFgPrim (J/Vmap/Taylor) ("
                    << contains_meta_fg_prim_cnode->first->DebugString() << ")";
      return true;
    }
  }

  // Check if func graphs used contains J(func_graph), J(Primitive), Vmap(func_graph), Vmap(Primitive),
  // Taylor(func_graph) or Taylor(Primitive).
  fg->seen_ = seen_num;
  for (auto &item : fg->func_graphs_used()) {
    auto used_g = item.first;
    MS_EXCEPTION_IF_NULL(used_g);
    if (SeekMetaFgPrim(used_g, seen_num)) {
      MS_LOG(DEBUG) << fg->ToString() << " users func graph " << used_g->ToString()
                    << " which contains J(func_graph), J(Primitive), Vmap(func_graph), Vmap(Primitive), "
                    << "Taylor(func_graph) or Taylor(Primitive)";
      return true;
    }
  }
  MS_LOG(DEBUG) << fg->ToString() << " doesn't contain J(func_graph), J(Primitive), Vmap(func_graph), Vmap(Primitive), "
                << "Taylor(func_graph) or Taylor(Primitive)";
  return false;
}

void FuncGraphMetaFgPrimTotalComputer::RealRecompute(FuncGraphPtr fg) {
  this->meta_fg_prim_total_analysis_[fg] = SeekMetaFgPrim(fg, NewFgSeenGeneration());
}
}  // namespace mindspore
