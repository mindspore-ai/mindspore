/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <numeric>
#include <list>
#include "./common.h"
#include "utils/profile.h"
#include "operator/ops.h"
#include "debug/trace.h"

namespace mindspore {

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
  all_nodes_ = AnfNodeSet();
  node_users_ = NodeUsersMap();

  signals_ = std::make_shared<Signals>();
  nodes_ = std::make_shared<NodesCollector>(this);
  valuenodes_ = std::make_shared<ValueNodesCollector>(this);
  free_variables_direct_ = std::make_shared<FVDirectCollector>(this);
  func_graph_valuenodes_ = std::make_shared<FuncGraphValueNodesCollector>(this);
  func_graphs_used_ = std::make_shared<FuncGraphsUsedCollector>(this);
  func_graph_users_ = std::make_shared<FuncGraphUsersCollector>(this);
  func_graph_user_cnodes_ = std::make_shared<FuncGraphUserNodesCollector>(this);
  func_graph_child_direct_ = std::make_shared<FuncGraphChildDirect>(this);
  func_graph_parents_direct_ = std::make_shared<FuncGraphParentsDirectCollector>(this);
  func_graph_j_direct_ = std::make_shared<FuncGraphJDirectCollector>(this);

  func_graph_parents_total_ = std::make_shared<FuncGraphParentsTotalComputer>(this);
  func_graph_parent_ = std::make_shared<ParentComputer>(this);
  children_ = std::make_shared<ChildrenComputer>(this);
  scopes_ = std::make_shared<ScopeComputer>(this);
  free_variables_total_ = std::make_shared<FVTotalComputer>(this);
  func_graphs_used_total_ = std::make_shared<FuncGraphsUsedTotalComputer>(this);
  recursive_ = std::make_shared<RecursiveComputer>(this);
  j_total_ = std::make_shared<FuncGraphJTotalComputer>(this);
}

void FuncGraphManager::Init() {
  auto roots = roots_;
  roots_ = FuncGraphSet();

  for (auto &fg : roots) {
    AddFuncGraph(fg, true);
  }
}

FuncGraphSet &FuncGraphManager::func_graph_parents_total(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
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

bool FuncGraphManager::recursive(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  recursive_->Recompute(fg);
  if (recursive_->recursive_analysis().count(fg) == 0) {
    MS_LOG(WARNING) << "This func graph is not in manager: " << fg->ToString();
    return false;
  }
  return recursive_->recursive_analysis()[fg];
}

std::shared_ptr<std::list<FuncGraphPtr>> FuncGraphManager::recursive_graphs(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  if (recursive(fg)) {
    if (!recursive_->recursive_map().count(fg)) {
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

bool FuncGraphManager::func_graph_j_total(const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(j_total_);
  MS_EXCEPTION_IF_NULL(fg);
  j_total_->Recompute(fg);
  if (j_total_->j_total_analysis().count(fg) == 0) {
    MS_LOG(WARNING) << "This func graph is not in manager: " << fg->ToString();
    return false;
  }
  return j_total_->j_total_analysis()[fg];
}

// add a func graph to this manager, optionally as a root func graph.
void FuncGraphManager::AddFuncGraph(FuncGraphPtr func_graph, bool is_root) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (is_root) {
    roots_.add(func_graph);
  }
  if (func_graphs_.contains(func_graph)) {
    return;
  }
  AddIntoManaged(func_graph);
  MS_EXCEPTION_IF_NULL(signals_);
  signals_->AddFuncGraph(func_graph);
  std::vector<AnfNodePtr> para = func_graph->parameters();
  AcquireNodes(para);
  std::vector<AnfNodePtr> return_vec({func_graph->get_return()});
  AcquireNodes(return_vec);
}

// clear the all information in manager
void FuncGraphManager::Clear() {
  func_graphs_.clear();
  all_nodes_.clear();
  node_users_.clear();
  roots_.clear();

  signals_->InvalidateCollector();
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
    if (fg->manager() != nullptr && (&(*fg->manager()) != this)) {
      MS_LOG(WARNING) << "A func graph can only have one manager.";
    }
    FuncGraphManagerPtr this_manager = shared_from_this();
    fg->set_manager(this_manager);
  }
  func_graphs_.add(fg);
}

void FuncGraphManager::MaybeDropFuncGraphs(const FuncGraphSet &func_graphs, bool ignore_users) {
  FuncGraphSet todo(func_graphs);
  std::set<FuncGraphPtr> dropped;
  // int count = 0;
  while (!todo.empty()) {
    FuncGraphPtr func_graph = todo.pop();
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_LOG(DEBUG) << "Maybe drop func graph " << func_graph->ToString();
    if (roots_.contains(func_graph)) {
      MS_LOG(DEBUG) << "Cannot drop as roots contains func graph: " << func_graph->ToString();
      continue;
    }
    MS_EXCEPTION_IF_NULL(func_graph_users_);
    auto &users = func_graph_users_->count_func_graphs_map()[func_graph];
    if (!users.empty() && !ignore_users) {
      MS_LOG(DEBUG) << "Cannot drop as users not empty: " << func_graph->ToString();
      continue;
    }
    if (dropped.find(func_graph) != dropped.end()) {
      MS_LOG(DEBUG) << "Func graph had been dropped " << func_graph->ToString();
      continue;
    }
    (void)dropped.insert(func_graph);
    std::vector<AnfNodePtr> return_vec = {func_graph->get_return()};
    todo.update(MaybeDropNodes(return_vec));
  }
  MS_EXCEPTION_IF_NULL(signals_);
  for (auto &fg : dropped) {
    MS_EXCEPTION_IF_NULL(fg);
    signals_->DropFuncGraph(fg);
    all_nodes_.difference_update(fg->parameters());
    (void)func_graphs_.erase(fg);
    if (fg->manager().get() == this) {
      fg->set_manager(nullptr);
    }
    MS_LOG(DEBUG) << "Func graph dropped " << fg->ToString();
  }
}

void FuncGraphManager::ProcessEdge(AnfNodePtr node, int index, AnfNodePtr inp, EdgeProcessDirection direction) {
  MS_EXCEPTION_IF_NULL(inp);
  if (direction == kDecEdge) {
    MS_LOG(DEBUG) << "Remove node " << node->ToString() << " input[" << index << "] " << inp->ToString();
    auto &users_node = node_users_[inp];
    if (!users_node.contains(make_pair(node, index))) {
      return;
    }
    (void)users_node.erase(make_pair(node, index));
    signals_->DropEdge(node, index, inp);
  } else {
    MS_LOG(DEBUG) << "Add node " << node->ToString() << " input[" << index << "] " << inp->ToString();
    if (inp->func_graph() != nullptr) {
      AddFuncGraph(inp->func_graph());
    }
    if (IsValueNode<FuncGraph>(inp)) {
      MS_LOG(DEBUG) << "Input[" << index << "] is const graph " << inp->ToString();
      AddFuncGraph(GetValueNode<FuncGraphPtr>(inp));
    }
    auto &users_node = node_users_[inp];
    users_node.add(make_pair(node, index));
    MS_EXCEPTION_IF_NULL(signals_);
    signals_->AddEdge(node, index, inp);
  }
}

void FuncGraphManager::ProcessInputs(const AnfNodePtr &node, EdgeProcessDirection direction) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    int index = 0;
    for (auto &inp : cnode->inputs()) {
      ProcessEdge(cnode, index, inp, direction);
      ++index;
    }
  }
}

IncludeType FuncGraphManager::Limit(const AnfNodePtr &node) {
  if (all_nodes_.contains(node)) {
    return EXCLUDE;
  } else {
    return FOLLOW;
  }
}

void FuncGraphManager::AcquireNodes(const std::vector<AnfNodePtr> &nodes) {
  AnfNodeSet acq;
  for (auto &node : nodes) {
    std::function<IncludeType(AnfNodePtr)> limit = std::bind(&FuncGraphManager::Limit, this, std::placeholders::_1);

    AnfNodeSet new_nodes = AnfNodeSet(DeepScopedGraphSearch(node, limit));

    all_nodes_.update(new_nodes);
    acq.update(new_nodes);
  }

  for (auto &node : acq) {
    MS_EXCEPTION_IF_NULL(node);
    FuncGraphPtr fg = node->func_graph();
    if (fg != nullptr) {
      AddFuncGraph(fg);
    }
    signals_->AddNode(node);
    ProcessInputs(node, kIncEdge);
  }
}

FuncGraphSetPtr FuncGraphManager::MaybeDropNodes(const std::vector<AnfNodePtr> &nodes) {
  AnfNodeSet nodes_ordered(nodes);
  FuncGraphSetPtr func_graphs_to_check = std::make_shared<FuncGraphSet>();
  MS_EXCEPTION_IF_NULL(signals_);

  while (!nodes_ordered.empty()) {
    AnfNodePtr node = nodes_ordered.pop();
    MS_EXCEPTION_IF_NULL(node);
    if (!all_nodes_.contains(node)) {
      continue;
    }
    AnfNodeIndexSet &users = node_users_[node];

    std::vector<AnfNodePtr> parameters;
    if (!users.empty() ||
        (node->isa<Parameter>() && parameters.end() != std::find(parameters.begin(), parameters.end(), node))) {
      continue;
    }
    if (IsValueNode<FuncGraph>(node)) {
      auto fg = GetValueNode<FuncGraphPtr>(node);
      func_graphs_to_check->add(fg);
      MS_LOG(DEBUG) << "Set value of node " << node->DebugString() << " from func graph " << fg->ToString()
                    << " to null";
    }
    ProcessInputs(node, kDecEdge);
    (void)all_nodes_.erase(node);
    signals_->DropNode(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      nodes_ordered.update(cnode->inputs());
    }
    (void)node_users_.erase(node);
  }
  return func_graphs_to_check;
}

void FuncGraphManager::SetParameters(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &parameters) {
  auto tr = Transact();
  tr.SetParameters(fg, parameters);
  tr.Commit();
}

bool FuncGraphManager::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
  auto tr = Transact();
  bool success = tr.Replace(old_node, new_node);
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

void FuncGraphManager::MoveAllCNodeDropGraph(FuncGraphPtr source, FuncGraphPtr target, const ScopePtr &scope) {
  AnfNodePtr source_return = source->get_return();
  AnfNodePtr source_output = source->output();
  AnfNodePtr source_prim = source_return->cast<CNodePtr>()->input(0);

  int index = 0;
  (void)node_users_[source_prim].erase(make_pair(source_return, index));
  signals_->DropEdge(source_return, index, source_prim);
  index = 1;
  (void)node_users_[source_output].erase(make_pair(source_return, index));
  signals_->DropEdge(source_return, index, source_output);
  (void)all_nodes_.erase(source_return);
  (void)node_users_.erase(source_return);
  signals_->DropNode(source_return);
  for (auto &node : source->nodes()) {
    node->set_func_graph(target);
    if (node->scope() == kDefaultScope) {
      node->set_scope(scope);
    }
  }
  for (auto &used : source->func_graphs_used()) {
    (void)func_graph_users_->Inc(used.first, target, used.second);
    (void)this->func_graph_users()[used.first].erase(source);
  }
  for (auto &child : this->func_graph_child_direct()[source]) {
    (void)func_graph_parents_direct_->Inc(child.first, target, child.second);
    (void)this->func_graph_parents_direct()[child.first].erase(source);
  }
  for (auto &fv_count : this->free_variables_direct()[source]) {
    auto fv_g = fv_count.first->func_graph();
    auto &count_on_g = this->func_graph_child_direct()[fv_g];
    auto pair = count_on_g.find(source);
    if (fv_g != target && pair != count_on_g.end()) {
      (void)func_graph_child_direct_->Inc(fv_g, target, pair->second);
    }
    (void)count_on_g.erase(source);
  }
  signals_->MoveAllCNode(source, target);
  signals_->InvalidateComputer();
  signals_->DropFuncGraph(source);
  all_nodes_.difference_update(source->parameters());
  (void)func_graphs_.erase(source);
  if (source->manager().get() == this) {
    source->set_manager(nullptr);
  }
}

FuncGraphTransaction FuncGraphManager::Transact() {
  auto tr = FuncGraphTransaction(this);
  return tr;
}

void FuncGraphManager::ParseChanges(const std::vector<Change> &changes, EdgeTupleCounter *add_edges,
                                    EdgeTupleCounter *rm_edges, Counter<AnfNodePtr> *adds, Counter<AnfNodePtr> *rms) {
  for (auto &iter : changes) {
    auto operation = iter.op;
    auto args = iter.args;
    if (operation == Change::kTxSetEdge) {
      auto edge = args.cast<ArgsOfSetEdge>();
      auto old_node = edge.root_node->input(edge.index);
      (*rm_edges)[std::make_pair(edge.root_node, std::make_pair(edge.index, old_node))] += 1;
      (*add_edges)[std::make_pair(edge.root_node, std::make_pair(edge.index, edge.new_node))] += 1;
      (*rms)[old_node] += 1;
      (*adds)[edge.new_node] += 1;
      edge.root_node->set_input(edge.index, edge.new_node);
    } else if (operation == Change::kTxSetParams) {
      auto param = args.cast<ArgsOfSetParams>();
      MS_EXCEPTION_IF_NULL(param.func_graph);
      auto old_parameters = param.func_graph->parameters();
      for (auto &p : param.params) {
        (*adds)[p] += 1;
      }
      for (auto &p : old_parameters) {
        (*rms)[p] += 1;
      }
      param.func_graph->set_parameters(param.params);
    }
  }
}

void FuncGraphManager::CommitChanges(const std::vector<Change> &changes) {
  EdgeTupleCounter add_edges;
  EdgeTupleCounter rm_edges;
  Counter<AnfNodePtr> adds;
  Counter<AnfNodePtr> rms;
  ParseChanges(changes, &add_edges, &rm_edges, &adds, &rms);

  auto sub_edges = add_edges - rm_edges;
  for (auto &iter : sub_edges) {
    auto root_node = iter.first.first;
    int index = iter.first.second.first;
    auto new_node = iter.first.second.second;
    ProcessEdge(root_node, index, new_node, kIncEdge);
  }

  auto sub_nodes = adds - rms;
  std::vector<AnfNodePtr> nodes;
  (void)std::transform(sub_nodes.begin(), sub_nodes.end(), std::back_inserter(nodes),
                       [](const std::pair<const AnfNodePtr, int> &iter) -> AnfNodePtr { return iter.first; });

  AcquireNodes(nodes);

  auto sub_edges_reverse = rm_edges - add_edges;
  for (auto &iter : sub_edges_reverse) {
    auto root_node = iter.first.first;
    int index = iter.first.second.first;
    auto old_node = iter.first.second.second;
    ProcessEdge(root_node, index, old_node, kDecEdge);
  }

  auto sub_nodes_reverse = rms - adds;
  std::vector<AnfNodePtr> nodes_reverse;

  (void)std::transform(sub_nodes_reverse.begin(), sub_nodes_reverse.end(), std::back_inserter(nodes_reverse),
                       [](const std::pair<const AnfNodePtr, int> &iter) -> AnfNodePtr { return iter.first; });

  auto drop_func_graphs = MaybeDropNodes(nodes_reverse);
  MaybeDropFuncGraphs(*drop_func_graphs);
}

void FuncGraphTransaction::SetParameters(FuncGraphPtr fg, const std::vector<AnfNodePtr> &params) {
  changes_.emplace_back(Change::kTxSetParams, ArgsOfSetParams{fg, params});
}

bool FuncGraphTransaction::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(new_node);
  FuncGraphPtr old_func_graph = old_node->func_graph();
  if (old_func_graph != nullptr && old_func_graph->get_return() == old_node) {
    MS_LOG(WARNING) << "Cannot replace the return node of a func graph " << old_func_graph->ToString();
    return false;
  }
  auto users = manager_->node_users()[old_node];
  for (auto &node : users) {
    SetEdge(node.first, node.second, new_node);
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
  changes_.emplace_back(Change::kTxSetEdge, ArgsOfSetEdge{cnode, v, IntToSize(k)});
}

void FuncGraphTransaction::Commit() {
  std::vector<Change> changes;
  changes_.swap(changes);
  manager_->CommitChanges(changes);
}

FuncGraphAnalysis::FuncGraphAnalysis(const FuncGraphManager *const manager)
    : manager_(manager), include_func_graph_none_(false) {
  manager_->signals()->AddFuncGraph.connect(this, &FuncGraphAnalysis::OnAddFuncGraph);
  manager_->signals()->DropFuncGraph.connect(this, &FuncGraphAnalysis::OnDropFuncGraph);
  manager_->signals()->AddEdge.connect(this, &FuncGraphAnalysis::OnAddEdge);
  manager_->signals()->DropEdge.connect(this, &FuncGraphAnalysis::OnDropEdge);
  manager_->signals()->MoveAllCNode.connect(this, &FuncGraphAnalysis::OnMoveAllCNode);
}

NodesCollector::NodesCollector(const FuncGraphManager *const m) : DepCollector(m), nodes_analysis_() {
  include_func_graph_none_ = true;
  nodes_analysis_[nullptr] = AnfNodeSet();

  manager_->signals()->AddNode.connect(this, &NodesCollector::OnAddNode);
  manager_->signals()->DropNode.connect(this, &NodesCollector::OnDropNode);
}

void NodesCollector::OnAddNode(AnfNodePtr n) {
  if (nodes_analysis_.find(n->func_graph()) == nodes_analysis_.end()) {
    nodes_analysis_[n->func_graph()] = AnfNodeSet();
  }

  nodes_analysis_[n->func_graph()].add(n);
}

void NodesCollector::OnDropNode(AnfNodePtr n) {
  (void)nodes_analysis_[n->func_graph()].erase(n);
  auto graph = n->func_graph();
  // Remove the node from order list.
  if (graph) {
    graph->EraseUnusedNodeInOrder(n);
  }
}

void NodesCollector::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr dst) {
  // change the owner of node except for the src's return node
  for (auto &it : nodes_analysis_[src]) {
    nodes_analysis_[dst].add(it);
  }
  (void)nodes_analysis_.erase(src);
}

void DepCollector::OnAddEdge(AnfNodePtr node, int index, AnfNodePtr inp) { OnModEdge(node, index, inp, kIncEdge); }

DepCollector::DepCollector(const FuncGraphManager *const manager) : FuncGraphAnalysis(manager) {
  MS_EXCEPTION_IF_NULL(manager_);
  manager_->signals()->InvalidateCollector.connect(this, &DepCollector::OnInvalidateCollector);
}

void DepCollector::OnDropEdge(AnfNodePtr node, int index, AnfNodePtr inp) { OnModEdge(node, index, inp, kDecEdge); }

bool CounterAnfNodeCollector::Inc(const FuncGraphPtr &func_graph, const AnfNodePtr &key, int count = 1) {
  auto &d = count_nodes_map_[func_graph];
  if (d.count(key) == 0) {
    d[key] = count;
    return true;
  } else {
    d[key] += count;
  }
  return false;
}

bool CounterAnfNodeCollector::Dec(const FuncGraphPtr &func_graph, const AnfNodePtr &key, int count = 1) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto &d = count_nodes_map_[func_graph];
  if (d.count(key) != 0) {
    if (d[key] == count) {
      (void)d.erase(key);
      return true;
    } else {
      d[key] -= count;
      if (d[key] < 0) {
        MS_LOG(EXCEPTION) << "Count of key '" << key->ToString()
                          << "' dec from 0. NodeInfo: " << trace::GetDebugInfo(func_graph->debug_info());
      }
    }
  }
  return false;
}

bool CounterAnfNodeCollector::Mod(const FuncGraphPtr &func_graph, const AnfNodePtr &key, int count) {
  if (count > 0) {
    return Inc(func_graph, key, count);
  } else if (count < 0) {
    return Dec(func_graph, key, -count);
  } else {
    MS_LOG(EXCEPTION) << "Count of key '" << key->ToString()
                      << "' cannot be 0. NodeInfo: " << trace::GetDebugInfo(func_graph->debug_info());
  }
}

bool CounterFuncGraphCollector::Inc(const FuncGraphPtr &func_graph, const FuncGraphPtr &key, int count = 1) {
  auto &d = count_func_graphs_map_[func_graph];
  if (d.count(key) == 0) {
    d[key] = count;
    return true;
  } else {
    d[key] += count;
  }
  return false;
}

bool CounterFuncGraphCollector::Dec(const FuncGraphPtr &func_graph, const FuncGraphPtr &key, int count = 1) {
  auto &d = count_func_graphs_map_[func_graph];
  if (d.count(key) != 0) {
    if (d[key] == count) {
      (void)d.erase(key);
      return true;
    } else {
      d[key] -= count;
      if (d[key] < 0) {
        MS_LOG(EXCEPTION) << "Count of key '" << key->ToString()
                          << "' dec from 0. NodeInfo: " << trace::GetDebugInfo(func_graph->debug_info());
      }
    }
  }
  return false;
}

bool CounterFuncGraphCollector::Mod(const FuncGraphPtr &func_graph, const FuncGraphPtr &key, int count) {
  if (count > 0) {
    return Inc(func_graph, key, count);
  } else if (count < 0) {
    return Dec(func_graph, key, -count);
  } else {
    MS_LOG(EXCEPTION) << "Count of key '" << key->ToString()
                      << "' cannot be 0. NodeInfo: " << trace::GetDebugInfo(func_graph->debug_info());
  }
}

void ValueNodesCollector::OnModEdge(AnfNodePtr node, int, AnfNodePtr inp, EdgeProcessDirection direction) {
  MS_EXCEPTION_IF_NULL(node);
  if (inp->isa<ValueNode>()) {
    (void)Mod(node->func_graph(), inp, direction);
  }
}

void ValueNodesCollector::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr dst) {
  for (auto &it : count_nodes_map_[src]) {
    (void)Inc(dst, it.first, it.second);
  }
  (void)count_nodes_map_.erase(src);
}

// if inp is a graph ValueNode, this graph's FuncGraphValueNodesCollector's value is inp self
void FuncGraphValueNodesCollector::OnModEdge(AnfNodePtr, int, AnfNodePtr inp, EdgeProcessDirection direction) {
  if (IsValueNode<FuncGraph>(inp)) {
    (void)Mod(GetValueNode<FuncGraphPtr>(inp), inp, direction);
  }
}

void FuncGraphValueNodesCollector::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr dst) {
  for (auto &it : count_nodes_map_[src]) {
    (void)Inc(dst, it.first, it.second);
  }
  (void)count_nodes_map_.erase(src);
}

void FVDirectCollector::OnModEdge(AnfNodePtr node, int, AnfNodePtr inp, EdgeProcessDirection direction) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(inp);
  FuncGraphPtr fg1 = node->func_graph();
  FuncGraphPtr fg2 = inp->func_graph();
  if (nullptr != fg1 && nullptr != fg2 && fg1 != fg2) {
    (void)Mod(fg1, inp, direction);
  }
}

void FVDirectCollector::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr dst) {
  for (auto &it : count_nodes_map_[src]) {
    FuncGraphPtr fg2 = it.first->func_graph();
    if (fg2 != dst) {
      (void)Inc(dst, it.first, it.second);
    }
  }
  (void)count_nodes_map_.erase(src);
}

static FuncGraphPtr ParentProxy(const FuncGraphPtr &fg) {
  FuncGraphPtr gn = std::make_shared<FuncGraph>();
  (void)gn->transforms().insert(std::make_pair("proxy", FuncGraphTransform(fg)));
  return gn;
}

void FuncGraphChildDirect::OnModEdge(AnfNodePtr node, int, AnfNodePtr inp, EdgeProcessDirection direction) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(inp);
  FuncGraphPtr fg1 = node->func_graph();
  FuncGraphPtr fg2 = inp->func_graph();
  if (nullptr != fg1 && nullptr != fg2 && fg1 != fg2) {
    (void)Mod(fg2, fg1, direction);
  }
}

void FuncGraphChildDirect::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr dst) {
  for (auto &it : count_func_graphs_map_[src]) {
    FuncGraphPtr fg = it.first;
    if (fg != dst) {
      (void)Inc(dst, fg, it.second);
    }
  }
  (void)count_func_graphs_map_.erase(src);
}

void FuncGraphParentsDirectCollector::OnModEdge(AnfNodePtr node, int, AnfNodePtr inp, EdgeProcessDirection direction) {
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr fg1 = node->func_graph();
  // possible child parent
  if (IsValueNode<FuncGraph>(inp)) {
    FuncGraphPtr fg2 = GetValueNode<FuncGraphPtr>(inp);
    if (Mod(fg1, ParentProxy(fg2), direction)) {
      manager_->signals()->InvalidateComputer();
    }
  }
  // from fv
  FuncGraphPtr fg2 = inp->func_graph();
  if (nullptr != fg1 && nullptr != fg2 && fg1 != fg2) {
    // node use fv will in here, fg1's node use fg2's node, so fg1 is child and fg2 is parent
    if (Mod(fg1, fg2, direction)) {
      manager_->signals()->InvalidateComputer();
    }
  }
}

void FuncGraphParentsDirectCollector::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr dst) {
  for (auto &it : count_func_graphs_map_[src]) {
    if (it.first != dst) {
      (void)Inc(dst, it.first, it.second);
    }
  }
  (void)count_func_graphs_map_.erase(src);
}

void FuncGraphsUsedCollector::OnModEdge(AnfNodePtr node, int, AnfNodePtr inp, EdgeProcessDirection direction) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsValueNode<FuncGraph>(inp)) {
    (void)Mod(node->func_graph(), GetValueNode<FuncGraphPtr>(inp), direction);
  }
}

void FuncGraphsUsedCollector::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr dst) {
  // all graph use in src need to change to dst, so meger the to dst use
  for (auto &it : count_func_graphs_map_[src]) {
    (void)Inc(dst, it.first, it.second);
  }
  (void)count_func_graphs_map_[dst].erase(src);
  (void)count_func_graphs_map_.erase(src);
}

void FuncGraphUsersCollector::OnModEdge(AnfNodePtr node, int, AnfNodePtr inp, EdgeProcessDirection direction) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsValueNode<FuncGraph>(inp)) {
    (void)Mod(GetValueNode<FuncGraphPtr>(inp), node->func_graph(), direction);
  }
}

void FuncGraphUsersCollector::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr) {
  // all graph use in src need to change to dst, so add dst user
  (void)count_func_graphs_map_.erase(src);
}

void FuncGraphUserNodesCollector::OnModEdge(AnfNodePtr node, int, AnfNodePtr inp, EdgeProcessDirection direction) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsValueNode<FuncGraph>(inp)) {
    (void)Mod(GetValueNode<FuncGraphPtr>(inp), node, direction);
  }
}

void FuncGraphUserNodesCollector::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr dst) {
  for (auto &it : count_nodes_map_[src]) {
    (void)Inc(dst, it.first, it.second);
  }
  (void)count_nodes_map_.erase(src);
}

void FuncGraphJDirectCollector::OnModEdge(AnfNodePtr node, int, AnfNodePtr inp, EdgeProcessDirection direction) {
  if (IsValueNode<FuncGraph>(inp) && IsPrimitiveCNode(node, prim::kPrimJ)) {
    (void)Mod(node->func_graph(), GetValueNode<FuncGraphPtr>(inp), direction);
    MS_LOG(DEBUG) << "" << node->func_graph()->ToString() << " users func graph "
                  << GetValueNode<FuncGraphPtr>(inp)->ToString() << " which contains J(func_graph), dir: " << direction;
  }
}

void FuncGraphJDirectCollector::OnMoveAllCNode(FuncGraphPtr src, FuncGraphPtr dst) {
  // all graph use in src need to change to dst, so meger the to dst use
  for (auto &it : count_func_graphs_map_[src]) {
    (void)Inc(dst, it.first, it.second);
  }
  (void)count_func_graphs_map_.erase(src);
}

DepComputer::DepComputer(const FuncGraphManager *const manager) : FuncGraphAnalysis(manager) {
  MS_EXCEPTION_IF_NULL(manager_);
  manager_->signals()->InvalidateComputer.connect(this, &DepComputer::OnInvalidateComputer);
  validate_ = false;
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

FuncGraphSetPtr FuncGraphParentsTotalComputer::SeekParents(const FuncGraphPtr &fg, const FuncGraphSetPtr &path) {
  if (path == nullptr || path->contains(fg)) {
    return std::make_shared<FuncGraphSet>();
  }
  FuncGraphSetPtr parents = std::make_shared<FuncGraphSet>();
  FuncGraphToFuncGraphCounterMap &deps = *all_parents_direct_;
  for (auto &dep : deps[fg]) {
    MS_EXCEPTION_IF_NULL(dep.first);
    auto proxy = dep.first->transforms().find("proxy");
    if (proxy != dep.first->transforms().end()) {
      path->add(fg);
      auto gt = proxy->second.func_graph();
      parents->update(SeekParents(gt, path));
    } else {
      parents->add(dep.first);
    }
  }
  (void)parents->erase(fg);
  return parents;
}

void FuncGraphParentsTotalComputer::RealRecompute(FuncGraphPtr fg) {
  MS_EXCEPTION_IF_NULL(fg);
  all_parents_direct_ = &(manager_->func_graph_parents_direct());
  MS_LOG(DEBUG) << "" << fg->ToString() << " total func graph dep size:" << (*all_parents_direct_)[fg].size();
  func_graph_parents_total_analysis_[fg].update(SeekParents(fg));
  MS_LOG(DEBUG) << "FuncGraphParentsTotalComputer end: " << func_graph_parents_total_analysis_[fg].size();
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
    this->parent_analysis_[fg] = deps.pop();
    return;
  } else {
    // return nearest parent as parent
    FuncGraphSet deps_copy(deps);
    for (auto &dep : deps) {
      auto parent_deps = this->manager_->func_graph_parents_total(dep);
      for (auto &p_d : parent_deps) {
        if (deps_copy.count(p_d)) {
          (void)deps_copy.erase(p_d);
        }
      }
      if (deps_copy.size() == 1) {
        this->parent_analysis_[fg] = deps_copy.pop();
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
    count_nodes_map_[fg] = OrderedMap<AnfNodePtr, int>();
    count_func_graphs_map_[fg] = OrderedMap<FuncGraphPtr, int>();
  }

  for (auto &fg : manager->func_graphs()) {
    AnfNodeCounterMap items = manager->free_variables_direct()[fg];
    for (auto &iter : items) {
      auto curr = fg;
      while (curr) {
        (void)CounterAnfNodeCollector::Mod(curr, iter.first, iter.second);
        curr = manager->parent(curr);
        const AnfNodeSet &nodes = manager->nodes()[curr];
        if (nodes.contains(iter.first)) {
          break;
        }
      }
    }

    auto items_fg = manager->func_graphs_used()[fg];
    for (auto &iter : items_fg) {
      auto p = manager->parent(iter.first);
      if (p == nullptr) {
        continue;
      }
      auto curr = fg;
      while (curr != p) {
        (void)CounterFuncGraphCollector::Mod(curr, iter.first, iter.second);
        curr = manager->parent(curr);
      }
    }
  }
  for (auto &fg : manager->func_graphs()) {
    auto &fvp = count_nodes_map_[fg];
    auto &fvg = count_func_graphs_map_[fg];
    for (auto &item : fvp) {
      fv_total_analysis_[fg][item.first] = item.second;
    }
    for (auto &item : fvg) {
      fv_total_analysis_[fg][item.first] = item.second;
    }
  }
}

void FuncGraphsUsedTotalComputer::RealRecompute(FuncGraphPtr fg) {
  MS_EXCEPTION_IF_NULL(manager_);
  auto &used = this->manager_->func_graphs_used();
  std::vector<FuncGraphPtr> todo;
  std::vector<FuncGraphPtr> todo_new;

  todo.push_back(fg);
  while (!todo.empty()) {
    todo_new.clear();
    for (auto &gt : todo) {
      for (auto &item : used[gt]) {
        auto used_fg = item.first;
        if (used_fg == fg) {
          func_graph_used_total_analysis_[fg].add(used_fg);
          continue;
        }
        if (func_graph_used_total_analysis_[fg].count(used_fg) == 0) {
          todo_new.push_back(used_fg);
        }
        MS_LOG(DEBUG) << "" << fg->ToString() << " add func graph " << used_fg->ToString();
        func_graph_used_total_analysis_[fg].add(used_fg);
      }
    }
    todo = todo_new;
  }
}

bool CheckRecursive(const FuncGraphManager *const manager, const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(manager);
  auto &used = manager->func_graphs_used();
  std::vector<FuncGraphPtr> todo;
  std::vector<FuncGraphPtr> todo_new;
  todo.push_back(fg);
  FuncGraphSet used_total;
  while (!todo.empty()) {
    todo_new.clear();
    for (auto &gt : todo) {
      for (auto &item : used[gt]) {
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
    auto &used_fgs = manager_->func_graphs_used()[fg];
    for (auto iter = used_fgs.begin(); iter != used_fgs.end(); (void)iter++) {
      CheckRecursiveGraphs(iter->first, trace);
    }
    trace->pop_back();
    if (!recursive_map_.count(fg)) {
      recursive_map_[fg] = nullptr;
    }
  }
}

bool FuncGraphJTotalComputer::SeekJ(const FuncGraphPtr &fg, const FuncGraphSetPtr &path) {
  MS_EXCEPTION_IF_NULL(path);
  if (path->contains(fg)) {
    MS_LOG(DEBUG) << "" << fg->ToString() << " had been checked";
    return false;
  }
  MS_EXCEPTION_IF_NULL(manager_);
  auto &func_graph_counter_map = manager_->func_graph_j_direct();
  if (!func_graph_counter_map[fg].empty()) {
    // check g1->J(fg)->g2->g cycle;
    auto contains_j =
      std::find_if(func_graph_counter_map[fg].begin(), func_graph_counter_map[fg].end(),
                   [path](const std::pair<FuncGraphPtr, int> iter) { return !path->contains(iter.first); });
    if (contains_j != func_graph_counter_map[fg].end()) {
      MS_LOG(DEBUG) << "" << fg->ToString() << " contains J(" << contains_j->first->ToString() << ")";
      return true;
    }
  }
  path->add(fg);

  // check if func graphs used contains J(func_graph);
  auto &used = this->manager_->func_graphs_used();
  for (auto &item : used[fg]) {
    auto used_g = item.first;
    if (SeekJ(used_g, path)) {
      MS_LOG(DEBUG) << "" << fg->ToString() << " users func graph " << used_g->ToString()
                    << " which contains J(func_graph)";
      return true;
    }
  }
  MS_LOG(DEBUG) << "" << fg->ToString() << " doesn't contain J(func_graph)";
  return false;
}

void FuncGraphJTotalComputer::RealRecompute(FuncGraphPtr fg) {
  std::shared_ptr<FuncGraphSet> path = std::make_shared<FuncGraphSet>();
  this->j_total_analysis_[fg] = SeekJ(fg, path);
}
}  // namespace mindspore
