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

#include "frontend/optimizer/dead_node_eliminate.h"

#include <memory>
#include <vector>
#include <deque>
#include <set>
#include <utility>
#include "utils/utils.h"
#include "base/core_ops.h"
#include "utils/func_graph_analyzer.h"
namespace mindspore {
namespace opt {
namespace {
bool IsFuncGraphCallNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto input0 = node->cast<CNodePtr>()->input(0);
  if (IsValueNode<Primitive>(input0)) {
    return false;
  }
  return true;
}
}  // namespace

class VisitContext {
 public:
  VisitContext() = default;
  explicit VisitContext(const std::vector<int64_t> &index_stack) { (void)index_stacks_.insert(index_stack); }
  ~VisitContext() = default;

  bool Add(const std::vector<int64_t> &index_stack) {
    if (index_stacks_.find(index_stack) != index_stacks_.end()) {
      return false;
    }
    (void)index_stacks_.insert(index_stack);
    return true;
  }

  bool IndexVisited(int64_t index) const {
    return std::any_of(index_stacks_.begin(), index_stacks_.end(), [&index](const std::vector<int64_t> &index_stack) {
      return !index_stack.empty() && index_stack.back() == index;
    });
  }

  std::set<std::vector<int64_t>> index_stacks_;
};
using VisitContextPtr = std::shared_ptr<VisitContext>;

class ContextManager {
 public:
  ContextManager() = default;
  ~ContextManager() = default;
  HashMap<AnfNodePtr, VisitContextPtr> contexts_;

  bool AddContext(const AnfNodePtr &node, const std::vector<int64_t> &index_stack) {
    auto it = contexts_.find(node);
    if (it == contexts_.end()) {
      MS_LOG(DEBUG) << "Add node: " << node->DebugString();
      contexts_[node] = std::make_shared<VisitContext>(index_stack);
      return true;
    }
    return it->second->Add(index_stack);
  }

  bool IndexVisited(const CNodePtr &node, int64_t index) const {
    auto it = contexts_.find(node);
    if (it == contexts_.end()) {
      return false;
    }
    return it->second->IndexVisited(index);
  }
};

void VisitNode(const AnfNodePtr &node, const FuncGraphAnalyzer &analyzer, std::vector<int64_t> index_stack, size_t seen,
               ContextManager *context_manager) {
  if (IS_OUTPUT_ON(DEBUG)) {
    MS_LOG(DEBUG) << "Visit node: " << node->DebugString();
    for (size_t i = 0; i < index_stack.size(); i++) {
      MS_LOG(DEBUG) << "index_stack[" << i << "]: " << index_stack[i];
    }
  }
  // If context exist, node need visit again to avoid repeatedly visiting.
  if (!context_manager->AddContext(node, index_stack)) {
    return;
  }
  node->seen_ = seen;
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto tuple_getitem = node->cast<CNodePtr>();
    // Get cur index
    auto output_index_value_node = tuple_getitem->input(kInputNodeOutputIndexInTupleGetItem);
    MS_EXCEPTION_IF_NULL(output_index_value_node);
    auto value_node = output_index_value_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto output_idx = LongToSize(GetValue<int64_t>(value_node->value()));
    index_stack.push_back(output_idx);
    auto real_input = tuple_getitem->input(kRealInputNodeIndexInTupleGetItem);
    VisitNode(real_input, analyzer, index_stack, seen, context_manager);
    return;
  }
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    // If make_tuple in make_tuple, visit may start with inner tuple_getitem.
    if (index_stack.empty()) {
      return;
    }
    auto make_tuple = node->cast<CNodePtr>();
    auto output_idx = index_stack.back();
    index_stack.pop_back();
    VisitNode(make_tuple->input(1 + output_idx), analyzer, index_stack, seen, context_manager);
    return;
  }
  if (IsFuncGraphCallNode(node)) {
    const auto &caller_func_graphs = analyzer.GetCallerFuncGraphs(node);
    for (const auto &fg : caller_func_graphs) {
      auto new_index_stack = std::vector<int64_t>(index_stack);
      VisitNode(fg->output(), analyzer, new_index_stack, seen, context_manager);
    }
    return;
  }
  if (node->isa<Parameter>()) {
    const auto &func_callers = analyzer.GetFuncGraphCallers(node->func_graph());
    for (auto &caller : func_callers) {
      const auto &args = analyzer.GetArg(node, caller);
      auto new_index_stack = std::vector<int64_t>(index_stack);
      for (const auto &arg : args) {
        VisitNode(arg, analyzer, new_index_stack, seen, context_manager);
      }
    }
    return;
  }
  if (node->isa<ValueTuple>()) {
    // TupleGetItem's input may not be a MakeTuple but a ValueTuple.
    return;
  }
  MS_LOG(DEBUG) << "Reach the end node: " << node->DebugString() << ", but index stack is not empty.";
}

std::vector<AnfNodePtr> GenerateOutputTempGetItems(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> output_tmp_getitems;
  std::deque<AnfNodePtr> todo = {func_graph->output()};
  while (!todo.empty()) {
    const auto node = todo.back();
    todo.pop_back();
    MS_EXCEPTION_IF_NULL(node->abstract());
    if (!node->abstract()->isa<abstract::AbstractTuple>()) {
      if (node != func_graph->output()) {
        (void)output_tmp_getitems.emplace_back(node);
      }
      continue;
    }
    auto abstract_tuple = node->abstract()->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abstract_tuple);
    int64_t index = 0;
    for (const auto &elm : abstract_tuple->elements()) {
      auto new_tuple_getitem =
        func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), node, NewValueNode(MakeValue(index))});
      new_tuple_getitem->set_abstract(elm);
      MS_LOG(INFO) << "New tuple getitem: " << new_tuple_getitem->DebugString() << ", index: " << index;
      todo.push_front(new_tuple_getitem);
      index++;
    }
  }
  return output_tmp_getitems;
}

bool IsScalarValueNode(const AnfNodePtr &node) {
  if (!IsValueNode<Scalar>(node)) {
    return false;
  }
  if (node->abstract() == nullptr) {
    return false;
  }
  return node->abstract()->isa<abstract::AbstractScalar>();
}

AnfNodePtr MakeScalarZero() {
  auto zero = NewValueNode(MakeValue(0));
  auto abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(0));
  zero->set_abstract(abs);
  return zero;
}

bool EraseNode(const CNodePtr &cnode, size_t input_idx, const FuncGraphManagerPtr &manager) {
  // Scalar(int) no need convert to Scalar(0), and Scalar(0) cannot be erased once again.
  auto dead_node = cnode->input(input_idx);
  if (IsScalarValueNode(dead_node)) {
    return false;
  }
  MS_LOG(WARNING) << "Erase dead node: " << dead_node->DebugString() << ", user: " << cnode->DebugString();
  // Can't use `Replace`, must use `SetEdge`.
  manager->SetEdge(cnode, SizeToInt(input_idx), MakeScalarZero());
  return true;
}

bool EraseMakeTupleInput(const std::vector<AnfNodePtr> &make_tuples, const FuncGraphPtr &func_graph,
                         const ContextManager &context_manager, size_t seen) {
  bool change = false;
  for (const auto &make_tuple : make_tuples) {
    MS_LOG(DEBUG) << "Check make_tuple:" << make_tuple->DebugString();
    auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
    for (size_t i = 1; i < make_tuple_cnode->size(); i++) {
      // If make_tuple was not visited ,it may be a make tuple of swith_layer or addn and some other ops.
      auto input_edge_visited = context_manager.IndexVisited(make_tuple_cnode, i - 1);
      // Can use `context_manager.contexts_.find(make_tuple_cnode) != context_manager.contexts_.end()`.
      auto make_tuple_visited = make_tuple_cnode->seen_ == seen;
      if (!input_edge_visited && make_tuple_visited) {
        change = EraseNode(make_tuple_cnode, i, func_graph->manager()) || change;
      }
    }
  }
  return change;
}

void VisitValue(const ValuePtr &value, std::vector<int64_t> indexes,
                HashMap<ValuePtr, HashSet<int64_t>> *visited_values) {
  MS_EXCEPTION_IF_NULL(value);
  MS_LOG(DEBUG) << "Visit value:" << value->ToString();
  if (indexes.empty()) {
    MS_LOG(DEBUG) << "Indexes empty";
    return;
  }
  const auto visit_index = indexes.back();
  (*visited_values)[value].insert(visit_index);
  auto value_tuple = value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  if (LongToSize(visit_index) >= value_tuple->size()) {
    MS_LOG(EXCEPTION) << "Index: " << visit_index << " out of range: " << value_tuple->size();
  }
  indexes.pop_back();
  MS_LOG(DEBUG) << "Visit index: " << visit_index;
  VisitValue(value_tuple->value()[LongToSize(visit_index)], indexes, visited_values);
}

std::pair<ValuePtr, abstract::AbstractBasePtr> EraseValue(const ValuePtr &value, const abstract::AbstractBasePtr &abs,
                                                          const HashMap<ValuePtr, HashSet<int64_t>> &visited_values,
                                                          bool need_erase) {
  if (need_erase) {
    auto new_value = MakeValue(0);
    auto new_abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(0));
    new_abs->set_value(new_value);
    MS_LOG(WARNING) << "Erase value:" << value->ToString();
    return {new_value, new_abs};
  }
  auto it = visited_values.find(value);
  if (it == visited_values.end()) {
    return {value, abs};
  }
  const auto &all_visit_index = it->second;

  auto value_tuple = value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abs_tuple);
  auto new_elements = std::vector<ValuePtr>(value_tuple->value());
  auto new_abstracts = std::vector<abstract::AbstractBasePtr>(abs_tuple->elements());
  if (new_elements.size() != new_abstracts.size()) {
    MS_LOG(EXCEPTION) << "Value size: " << new_elements.size()
                      << " is not equal to abstract size: " << new_abstracts.size();
  }

  bool change = false;
  for (size_t i = 0; i < value_tuple->value().size(); i++) {
    auto value_i = new_elements[i];
    auto abs_i = new_abstracts[i];
    // Avoid repeatedly erase.
    MS_LOG(DEBUG) << "value_i[" << i << "]: " << value_i->ToString();
    if (value_i->isa<Scalar>()) {
      continue;
    }
    bool need_erase_i = all_visit_index.find(SizeToLong(i)) == all_visit_index.end();
    auto [ret_value, ret_abs] = EraseValue(value_i, abs_i, visited_values, need_erase_i);
    if (ret_value != value_i) {
      new_elements[i] = ret_value;
      new_abstracts[i] = ret_abs;
      change = true;
    }
  }
  if (change) {
    value_tuple = std::make_shared<ValueTuple>(new_elements);
    abs_tuple = std::make_shared<abstract::AbstractTuple>(new_abstracts);
    abs_tuple->set_value(value_tuple);
  }
  return {value_tuple, abs_tuple};
}

bool EraseDeadValues(const std::vector<AnfNodePtr> &value_tuple_nodes, const ContextManager &context_manager) {
  bool change = false;
  for (const auto &value_tuple : value_tuple_nodes) {
    auto it = context_manager.contexts_.find(value_tuple);
    if (it == context_manager.contexts_.end()) {
      continue;
    }
    HashMap<ValuePtr, HashSet<int64_t>> visited_values;
    const auto value = GetValueNode(value_tuple);
    for (const auto &context : it->second->index_stacks_) {
      VisitValue(value, context, &visited_values);
    }
    // Erase the unvisited values.
    auto [new_value, new_abs] = EraseValue(value, value_tuple->abstract(), visited_values, false);
    if (new_value != value) {
      value_tuple->cast<ValueNodePtr>()->set_value(new_value);
      value_tuple->set_abstract(new_abs);
      MS_LOG(DEBUG) << "Set new value of node: " << value_tuple->DebugString();
      change = true;
    }
  }
  return change;
}

std::shared_ptr<HashSet<size_t>> GetUsedParameters(const FuncGraphPtr &func_graph) {
  auto used_parameter_indexes = std::make_shared<HashSet<size_t>>();
  if (func_graph->manager() == nullptr) {
    return used_parameter_indexes;
  }
  const auto &manager_node_users = func_graph->manager()->node_users();
  const auto &parameters = func_graph->parameters();
  // Traverse to find all unused parameters.
  size_t index = 0;
  for (const auto &parameter : parameters) {
    const auto &node_users_it = manager_node_users.find(parameter);
    if (node_users_it != manager_node_users.end() && !node_users_it->second.empty()) {
      (void)used_parameter_indexes->insert(index);
    }
    index++;
  }
  return used_parameter_indexes;
}

bool EraseArg(size_t user_index, const CNodePtr &arg_user, const FuncGraphManagerPtr &manager) {
  size_t arg_start_idx = 0;
  const size_t kFuncGraphCallArgStartIdx = 2;
  const size_t kPartialArgStartIdx = 2;
  if (IsFuncGraphCallNode(arg_user)) {
    arg_start_idx = kFuncGraphCallArgStartIdx;
  } else if (IsPrimitiveCNode(arg_user, prim::kPrimPartial)) {
    arg_start_idx = kPartialArgStartIdx;
  } else {
    MS_LOG(EXCEPTION) << "Unexpected arg user: " << arg_user->DebugString();
  }
  if (user_index < arg_start_idx) {
    return false;
  }
  return EraseNode(arg_user, user_index, manager);
}

void VisitClosureArg(const FuncClosurePtr &closure, const CNodePtr &call, const HashSet<size_t> &used_indexes,
                     OrderedMap<CNodePtr, HashSet<size_t>> *visited_args) {
  auto arg_indexes = closure->arg_indexes_;
  // Add call node args to all args.
  auto arg_users = closure->arg_users_;
  for (size_t i = 1; i < call->inputs().size(); i++) {
    arg_indexes.push_back(i);
    arg_users.push_back(call);
  }
  const auto &fg = closure->func_graph_;
  if (arg_indexes.size() != fg->parameters().size()) {
    MS_LOG(EXCEPTION) << "Args size: " << arg_indexes.size()
                      << " is not equal to parameters size: " << fg->parameters().size()
                      << ". call: " << call->DebugString() << ", fg: " << fg->ToString();
  }
  for (size_t i = 0; i < arg_users.size(); i++) {
    // Insert a empty set to keep arg user record in map.
    if (visited_args->find(arg_users[i]) == visited_args->end()) {
      (*visited_args)[arg_users[i]] = HashSet<size_t>();
    }
    if (used_indexes.find(i) != used_indexes.end()) {
      MS_LOG(DEBUG) << "Visit arg user: " << arg_users[i]->DebugString() << ", idx: " << arg_indexes[i];
      (*visited_args)[arg_users[i]].insert(arg_indexes[i]);
    }
  }
}

// If the parameter is a function parameter, the arg will be converted to a DeadNod after renormalize, so the arg need
// to be erased.
bool EraseUnusedArgs(const std::vector<AnfNodePtr> &all_calls, const FuncGraphAnalyzer &analyzer,
                     const FuncGraphPtr &root_graph) {
  bool change = false;
  //  OrderedMap<AnfNodePtr, OrderedSet<size_t>> call_unused_indexes;
  HashMap<FuncGraphPtr, std::shared_ptr<HashSet<size_t>>> func_graphs_used_indexes;
  OrderedMap<CNodePtr, HashSet<size_t>> visited_args;
  // Visit all args of all calls.
  for (const auto &call : all_calls) {
    // Get unused indexes of call node.
    auto closures = analyzer.GetCallClosures(call);
    for (const auto &closure : closures) {
      std::shared_ptr<HashSet<size_t>> cur_fg_used_indexes;
      auto it = func_graphs_used_indexes.find(closure->func_graph_);
      if (it != func_graphs_used_indexes.end()) {
        cur_fg_used_indexes = it->second;
      } else {
        // Get unused parameter indexes of graph.
        cur_fg_used_indexes = GetUsedParameters(closure->func_graph_);
        func_graphs_used_indexes[closure->func_graph_] = cur_fg_used_indexes;
      }
      VisitClosureArg(closure, call->cast<CNodePtr>(), *cur_fg_used_indexes, &visited_args);
    }
  }
  // Erase unvisited args.
  for (const auto &[arg_user, visit_indexes] : visited_args) {
    for (size_t i = 0; i < arg_user->inputs().size(); i++) {
      if (visit_indexes.find(i) == visit_indexes.end()) {
        change = EraseArg(i, arg_user, root_graph->manager()) || change;
      }
    }
  }
  return change;
}

// Visit graphs by DFS.
void VisitGraph(const FuncGraphPtr &func_graph,
                const OrderedMap<FuncGraphPtr, OrderedSet<FuncGraphPtr>> &graph_relations,
                HashSet<FuncGraphPtr> *visited_graphs) {
  (void)visited_graphs->insert(func_graph);
  auto it = graph_relations.find(func_graph);
  if (it == graph_relations.end()) {
    return;
  }
  const auto &sub_graphs = it->second;
  for (const auto &sub_graph : sub_graphs) {
    if (visited_graphs->find(sub_graph) != visited_graphs->end()) {
      continue;
    }
    MS_LOG(DEBUG) << "Visit from graph: " << func_graph->ToString() << " to graph: " << sub_graph->ToString();
    VisitGraph(sub_graph, graph_relations, visited_graphs);
  }
}

bool EraseGraphCaller(const FuncGraphPtr &func_graph, const FuncGraphAnalyzer &analyzer,
                      const FuncGraphManagerPtr &manager) {
  const auto &calls = analyzer.GetFuncGraphCallers(func_graph);
  bool change = false;
  for (const auto &call : calls) {
    auto call_closures = analyzer.GetCallClosures(call);
    // In fact, we can remove the arg user here, but in order to keep dead node eliminating strategy common, we consider
    // dead node only come from make tuple's input and caller's arg, so we erase the arg(which is input of arg user)
    // instead of arg user here.
    for (const auto &closure : call_closures) {
      for (size_t i = 0; i < closure->arg_users_.size(); i++) {
        (void)EraseArg(closure->arg_indexes_[i], closure->arg_users_[i], manager);
      }
    }
    change = true;
  }
  return change;
}

std::shared_ptr<OrderedMap<FuncGraphPtr, OrderedSet<FuncGraphPtr>>> GetGraphRelations(
  const OrderedSet<FuncGraphPtr> &all_graphs, const FuncGraphAnalyzer &analyzer, const FuncGraphManagerPtr &manager) {
  auto graph_relations = std::make_shared<OrderedMap<FuncGraphPtr, OrderedSet<FuncGraphPtr>>>();
  for (const auto &func_graph : all_graphs) {
    const auto &graph_callers = analyzer.GetFuncGraphCallers(func_graph);
    for (const auto &caller : graph_callers) {
      // If call exist in graph.
      if (manager->all_nodes().find(caller) != manager->all_nodes().end()) {
        (*graph_relations)[caller->func_graph()].insert(func_graph);
      }
    }
  }
  return graph_relations;
}

bool EraseCircleGraphs(const FuncGraphPtr &root_graph, const FuncGraphAnalyzer &analyzer,
                       const OrderedMap<FuncGraphPtr, OrderedSet<FuncGraphPtr>> &graph_relations) {
  HashSet<FuncGraphPtr> visited_graphs;
  VisitGraph(root_graph, graph_relations, &visited_graphs);
  bool change = false;
  // Eliminate unvisited graph's caller
  for (const auto &it : graph_relations) {
    const auto graph = it.first;
    if (graph->manager() == nullptr) {
      continue;
    }
    if (visited_graphs.find(graph) == visited_graphs.end()) {
      MS_LOG(WARNING) << "Erase unvisited graph: " << graph->ToString();
      change = EraseGraphCaller(graph, analyzer, graph->manager()) || change;
    }
  }
  return change;
}

std::shared_ptr<HashMap<std::string, std::vector<AnfNodePtr>>> SearchVisitNodes(const FuncGraphPtr &func_graph) {
  auto ret = std::make_shared<HashMap<std::string, std::vector<AnfNodePtr>>>();
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      (*ret)["tuple_getitem"].emplace_back(node);
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      (*ret)["make_tuple"].emplace_back(node);
    } else if (IsValueNode<ValueTuple>(node)) {
      (*ret)["value_tuple"].emplace_back(node);
    } else if (IsValueNode<FuncGraph>(node)) {
      (*ret)["graph_value_node"].emplace_back(node);
    } else if (IsFuncGraphCallNode(node)) {
      (*ret)["func_graph_call"].emplace_back(node);
    }
  }
  return ret;
}

std::shared_ptr<OrderedSet<FuncGraphPtr>> GetAllFuncGraphs(const std::vector<AnfNodePtr> &value_nodes) {
  auto func_graphs = std::make_shared<OrderedSet<FuncGraphPtr>>();
  (void)std::for_each(value_nodes.begin(), value_nodes.end(), [&func_graphs](const AnfNodePtr &node) {
    func_graphs->insert(GetValueNode<FuncGraphPtr>(node));
  });
  return func_graphs;
}

bool EliminateDeadNode(const FuncGraphPtr &func_graph) {
  // Travers all tuple getitem nodes to visit.
  FuncGraphAnalyzer analyzer(func_graph);
  analyzer.Run();
  // Don't handle no-incorporate-call situation to improve performance.
  if (!analyzer.HasIncorporateCall()) {
    return false;
  }

  bool change = false;
  bool cycle_change = true;
  while (cycle_change) {
    cycle_change = false;
    ContextManager context_manager;
    auto visited_nodes = SearchVisitNodes(func_graph);
    auto seen = NewSeenGeneration();
    std::vector<int64_t> index_stack;
    // Visit from all tuple_getitem.
    for (const auto &tuple_getitem : (*visited_nodes)["tuple_getitem"]) {
      VisitNode(tuple_getitem, analyzer, index_stack, seen, &context_manager);
    }
    // Visit from root graph output.
    const auto &output_getitems = GenerateOutputTempGetItems(func_graph);
    for (const auto &tuple_getitem : output_getitems) {
      VisitNode(tuple_getitem, analyzer, index_stack, seen, &context_manager);
    }
    // 1. Erase all make tuple's unused input.
    cycle_change =
      EraseMakeTupleInput((*visited_nodes)["make_tuple"], func_graph, context_manager, seen) || cycle_change;
    // 2. Erase all value tuple's dead values.
    cycle_change = EraseDeadValues((*visited_nodes)["value_tuple"], context_manager) || cycle_change;
    // 3. Erase unused parameter's arg.
    const auto &all_func_graph_calls = (*visited_nodes)["func_graph_call"];
    cycle_change = EraseUnusedArgs(all_func_graph_calls, analyzer, func_graph) || cycle_change;
    // 4. Erase circle closures's all caller arg.
    // Erase circle graphs: caller[fg1] = fg2, caller[fg2] = fg1, fg1 and fg2 are redundant.
    auto all_graphs = GetAllFuncGraphs((*visited_nodes)["graph_value_node"]);
    auto graph_relations = GetGraphRelations(*all_graphs, analyzer, func_graph->manager());
    cycle_change = EraseCircleGraphs(func_graph, analyzer, *graph_relations) || cycle_change;

    change = change || cycle_change;
  }
  return change;
}
}  // namespace opt
}  // namespace mindspore
