/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "base/core_ops.h"

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

  func_graph_parents_total_ = std::make_shared<FuncGraphParentsTotalComputer>(this);
  func_graph_parent_ = std::make_shared<ParentComputer>(this);
  children_ = std::make_shared<ChildrenComputer>(this);
  scopes_ = std::make_shared<ScopeComputer>(this);
  free_variables_total_ = std::make_shared<FVTotalComputer>(this);
  func_graphs_used_total_ = std::make_shared<FuncGraphsUsedTotalComputer>(this);
  recursive_ = std::make_shared<RecursiveComputer>(this);
  j_total_ = std::make_shared<FuncGraphJTotalComputer>(this);

  limit_ = std::bind(&FuncGraphManager::Limit, this, std::placeholders::_1);
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

// Add a func graph to this manager, optionally as a root func graph.
void FuncGraphManager::AddFuncGraph(FuncGraphPtr func_graph, bool is_root) {
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
  new_nodes.emplace_back(func_graph->get_return());

  // Acquire all nodes from func_graph.
  AcquireNodes(new_nodes);
}

// Clear the all information in manager
void FuncGraphManager::Clear() {
  for (auto graph : func_graphs_) {
    graph->DecAttachedMngCnt();
    if (graph->attached_mng_cnt() == 0) {
      graph->ClearAllManagerInfo();
    } else if (graph->attached_mng_cnt() < 0) {
      MS_LOG(EXCEPTION) << "graph:" << graph->ToString() << " attached cnt not right:" << graph->attached_mng_cnt();
    }
  }

  func_graphs_.clear();
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
    if (fg->manager() != nullptr && (&(*fg->manager()) != this)) {
      MS_LOG(INFO) << "A func graph can only have one manager.";
    }
    FuncGraphManagerPtr this_manager = shared_from_this();
    fg->set_manager(this_manager);
  }
  func_graphs_.add(fg);
  fg->IncAttachedMngCnt();
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
    todo.update(MaybeDropNodes(return_vec));
  }
  for (auto &fg : dropped) {
    MS_EXCEPTION_IF_NULL(fg);
    all_nodes_.difference_update(fg->parameters());
    EraseOneGraph(fg.get());
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
    DropEdge(node, index, inp);
  } else {
    MS_LOG(DEBUG) << "Add node " << node->ToString() << " input[" << index << "] " << inp->ToString();
    if (IsValueNode<FuncGraph>(inp)) {
      MS_LOG(DEBUG) << "Input[" << index << "] is const graph " << inp->ToString();
      AddFuncGraph(GetValueNode<FuncGraphPtr>(inp));
    }
    auto &users_node = node_users_[inp];
    users_node.add(make_pair(node, index));
    AddEdge(node, index, inp);
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
    AnfNodeSet new_nodes = AnfNodeSet(DeepScopedGraphSearch(node, limit_));

    all_nodes_.update(new_nodes);
    acq.update(new_nodes);
  }

  for (auto &node : acq) {
    MS_EXCEPTION_IF_NULL(node);
    auto fg = node->func_graph();
    if (fg != nullptr) {
      fg->AddNode(node);
    }
    ProcessInputs(node, kIncEdge);
  }
}

FuncGraphSetPtr FuncGraphManager::MaybeDropNodes(const std::vector<AnfNodePtr> &nodes) {
  AnfNodeSet nodes_ordered(nodes);
  FuncGraphSetPtr func_graphs_to_check = std::make_shared<FuncGraphSet>();
  while (!nodes_ordered.empty()) {
    AnfNodePtr node = nodes_ordered.pop();
    if (node == nullptr) {
      // Here can not call 'MS_EXCEPTION_IF_NULL' to throw exception, this method may be triggered by desctuctor
      MS_LOG(WARNING) << "Node to be dropped is nullptr";
      continue;
    }
    if (!all_nodes_.contains(node)) {
      continue;
    }
    AnfNodeIndexSet &users = node_users_[node];
    if (!users.empty()) {
      continue;
    }

    if (node->isa<Parameter>() && node->func_graph() != nullptr) {
      auto &parameters = node->func_graph()->parameters();
      if (std::find(parameters.begin(), parameters.end(), node) != parameters.end()) {
        continue;
      }
    }

    if (IsValueNode<FuncGraph>(node)) {
      auto fg = GetValueNode<FuncGraphPtr>(node);
      func_graphs_to_check->add(fg);
      MS_LOG(DEBUG) << "Set value of node " << node->DebugString() << " from func graph " << fg->ToString()
                    << " to null";
    }
    ProcessInputs(node, kDecEdge);
    (void)all_nodes_.erase(node);
    if (node->func_graph() != nullptr) {
      node->func_graph()->DropNode(node);
    }

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

void FuncGraphManager::AddParameter(const FuncGraphPtr &fg, const AnfNodePtr &parameter) {
  auto tr = Transact();
  tr.AddParameter(fg, parameter);
  tr.Commit();
}

bool FuncGraphManager::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
  auto func_graph = old_node->func_graph();
  auto tr = Transact();
  bool success = tr.Replace(old_node, new_node);
  if (success) {
    tr.Commit();
    if (func_graph != nullptr) {
      func_graph->ReplaceInOrder(old_node, new_node);
    }
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

void FuncGraphManager::MoveAllCNodeDropGraph(FuncGraphPtr source, FuncGraphPtr target, const ScopePtr &scope) {
  AnfNodePtr source_return = source->get_return();
  AnfNodePtr source_output = source->output();
  AnfNodePtr source_prim = source_return->cast<CNodePtr>()->input(0);

  int index = 0;
  (void)node_users_[source_prim].erase(make_pair(source_return, index));
  DropEdge(source_return, index, source_prim);
  index = 1;
  (void)node_users_[source_output].erase(make_pair(source_return, index));
  DropEdge(source_return, index, source_output);
  (void)all_nodes_.erase(source_return);
  (void)node_users_.erase(source_return);
  source->DropNode(source_return);
  for (auto &node : source->nodes()) {
    node->set_func_graph(target);
    if (node->scope() == kDefaultScope) {
      node->set_scope(scope);
    }
  }

  MoveAllNodes(source, target);
  all_nodes_.difference_update(source->parameters());
  EraseOneGraph(source.get());
  source->set_dropped(true);
  if (source->manager().get() == this) {
    source->set_manager(nullptr);
  }
}

void FuncGraphManager::AddEdge(AnfNodePtr node, int index, AnfNodePtr input) {
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
    if (IsPrimitiveCNode(node, prim::kPrimJ)) {
      fg->AddJValueNode(input);
    }
  } else if (fg != nullptr && fg != input->func_graph()) {
    if (fg->AddFreeVariable(input)) {
      signals_->InvalidateComputer();
    }
  }
}

void FuncGraphManager::DropEdge(AnfNodePtr node, int index, AnfNodePtr input) {
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
    if (IsPrimitiveCNode(node, prim::kPrimJ)) {
      fg->DropJValueNode(input);
    }
  } else if (fg != nullptr && fg != input->func_graph()) {
    if (fg->DropFreeVariable(input)) {
      signals_->InvalidateComputer();
    }
  }
}

void FuncGraphManager::MoveAllNodes(FuncGraphPtr source, FuncGraphPtr target) {
  target->CopyNodes(source);
  target->CopyValueNodes(source);
  target->CopyFuncGraphCNodesIndex(source);
  target->CopyFreeVariables(source);
  target->CopyFuncGraphsUsed(source);
  target->CopyJValueNodes(source);
  source->ClearAllManagerInfo();
  signals_->InvalidateComputer();
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
    switch (operation) {
      case Change::kTxSetEdge: {
        auto edge = args.cast<ArgsOfSetEdge>();
        auto old_node = edge.root_node->input(edge.index);
        (*rm_edges)[std::make_pair(edge.root_node, std::make_pair(edge.index, old_node))] += 1;
        (*add_edges)[std::make_pair(edge.root_node, std::make_pair(edge.index, edge.new_node))] += 1;
        (*rms)[old_node] += 1;
        (*adds)[edge.new_node] += 1;
        edge.root_node->set_input(edge.index, edge.new_node);
      } break;
      case Change::kTxAddEdge: {
        auto edge = args.cast<ArgsOfAddEdge>();
        auto index = edge.root_node->inputs().size();
        (*add_edges)[std::make_pair(edge.root_node, std::make_pair(index, edge.new_node))] += 1;
        (*adds)[edge.new_node] += 1;
        edge.root_node->add_input(edge.new_node);
      } break;
      case Change::kTxSetParams: {
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
      } break;
      case Change::kTxAddParam: {
        auto param = args.cast<ArgsOfAddParam>();
        MS_EXCEPTION_IF_NULL(param.func_graph);
        (*adds)[param.param] += 1;
        auto param_node = param.param->cast<ParameterPtr>();
        param.func_graph->append_parameter(param_node);
      } break;
      default:
        break;
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

void FuncGraphManager::EraseOneGraph(FuncGraph *fg) {
  MS_EXCEPTION_IF_NULL(fg);
  size_t erase_cnt = func_graphs_.erase(fg->shared_from_base<FuncGraph>());
  if (!erase_cnt) {
    return;
  }
  fg->DecAttachedMngCnt();
  if (fg->attached_mng_cnt() == 0) {
    fg->ClearAllManagerInfo();
  }
}

void FuncGraphTransaction::SetParameters(FuncGraphPtr fg, const std::vector<AnfNodePtr> &params) {
  changes_.emplace_back(Change::kTxSetParams, ArgsOfSetParams{fg, params});
}

void FuncGraphTransaction::AddParameter(FuncGraphPtr fg, const AnfNodePtr &param) {
  changes_.emplace_back(Change::kTxAddParam, ArgsOfAddParam{fg, param});
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

void FuncGraphTransaction::AddEdge(const AnfNodePtr &src_node, const AnfNodePtr &v) {
  MS_EXCEPTION_IF_NULL(src_node);
  auto cnode = src_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(EXCEPTION) << "src_node should be a cnode, but cast failed.";
  }
  changes_.emplace_back(Change::kTxAddEdge, ArgsOfAddEdge{cnode, v});
}

void FuncGraphTransaction::Commit() {
  std::vector<Change> changes;
  changes_.swap(changes);
  manager_->CommitChanges(changes);
}

DepComputer::DepComputer(const FuncGraphManager *const manager) : manager_(manager) {
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

FuncGraphSetPtr FuncGraphParentsTotalComputer::SeekParents(const FuncGraphPtr &fg, size_t seen_num) {
  if (fg->seen_ == seen_num) {
    return std::make_shared<FuncGraphSet>();
  }
  FuncGraphSetPtr parents = std::make_shared<FuncGraphSet>();

  // Append all the fvs in fg.
  auto &fvs = fg->free_variables();
  for (auto fv : fvs) {
    parents->add(fv.first->func_graph());
  }

  // Search the fv in fg's child func graph.
  auto &fgs = fg->func_graphs_used();
  for (auto &item : fgs) {
    fg->seen_ = seen_num;
    auto gt = item.first;
    parents->update(SeekParents(gt, seen_num));
  }
  (void)parents->erase(fg);
  return parents;
}

void FuncGraphParentsTotalComputer::RealRecompute(FuncGraphPtr fg) {
  MS_EXCEPTION_IF_NULL(fg);
  func_graph_parents_total_analysis_[fg].update(SeekParents(fg, NewFgSeenGeneration()));
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
      while (curr != p) {
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
    if (!recursive_map_.count(fg)) {
      recursive_map_[fg] = nullptr;
    }
  }
}

bool FuncGraphJTotalComputer::SeekJ(const FuncGraphPtr &fg, size_t seen_num) {
  MS_EXCEPTION_IF_NULL(fg);
  if (fg->seen_ == seen_num) {
    MS_LOG(DEBUG) << fg->ToString() << " had been checked";
    return false;
  }
  const auto &j_values = fg->j_value_nodes();
  if (!j_values.empty()) {
    auto contains_j =
      std::find_if(j_values.begin(), j_values.end(), [seen_num](const std::pair<AnfNodePtr, int> &iter) {
        // check g1->J(fg)->g2->g cycle.
        if (IsValueNode<FuncGraph>(iter.first)) {
          auto func_graph = GetValueNode<FuncGraphPtr>(iter.first);
          return func_graph->seen_ != seen_num;
        }
        if (IsValueNode<Primitive>(iter.first)) {
          // exclude the primitive of J itself.
          auto prim = GetValueNode<PrimitivePtr>(iter.first);
          return prim->name() != prim::kPrimJ->name();
        }
        return false;
      });
    if (contains_j != j_values.end()) {
      MS_LOG(DEBUG) << fg->ToString() << " contains J(" << contains_j->first->DebugString() << ")";
      return true;
    }
  }
  fg->seen_ = seen_num;

  // check if func graphs used contains J(func_graph) or J(Primitive)
  for (auto &item : fg->func_graphs_used()) {
    auto used_g = item.first;
    if (SeekJ(used_g, seen_num)) {
      MS_LOG(DEBUG) << fg->ToString() << " users func graph " << used_g->ToString()
                    << " which contains J(func_graph) or J(Primitive)";
      return true;
    }
  }
  MS_LOG(DEBUG) << fg->ToString() << " doesn't contain J(func_graph) or J(Primitive)";
  return false;
}

void FuncGraphJTotalComputer::RealRecompute(FuncGraphPtr fg) {
  this->j_total_analysis_[fg] = SeekJ(fg, NewFgSeenGeneration());
}
}  // namespace mindspore
