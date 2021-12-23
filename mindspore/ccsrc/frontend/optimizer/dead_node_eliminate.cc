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

  bool IndexVisited(int64_t index) {
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

  bool IndexVisited(const CNodePtr &node, int64_t index) {
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
    MS_LOG(WARNING) << "Visit node:" << node->DebugString();
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
        output_tmp_getitems.emplace_back(node);
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

bool EraseMakeTupleInput(const FuncGraphPtr &func_graph, const CNodePtr &make_tuple, size_t input_idx) {
  // Scalar(int) no need convert to Scalar(0), and Scalar(0) cannot be erased once again.
  auto node = make_tuple->input(input_idx);
  if (IsScalarValueNode(node)) {
    return false;
  }
  MS_LOG(WARNING) << "Erase dead node: " << node->DebugString() << ", user make_tuple: " << make_tuple->DebugString();
  auto new_tensor = NewValueNode(MakeValue(0));
  auto abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(0));
  new_tensor->set_abstract(abs);
  // Can't use `Replace`, must user `SetEdge`.
  func_graph->manager()->SetEdge(make_tuple, input_idx, new_tensor);
  return true;
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
    MS_LOG(WARNING) << "value_i:[" << i << "]: " << value_i->ToString();
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

bool EraseValueTuple(const AnfNodePtr &node, const std::set<std::vector<int64_t>> &contexts) {
  HashMap<ValuePtr, HashSet<int64_t>> visited_values;
  const auto value = GetValueNode(node);
  for (const auto &context : contexts) {
    VisitValue(value, context, &visited_values);
  }
  // Erase the unvisited values.
  auto [new_value, new_abs] = EraseValue(value, node->abstract(), visited_values, false);
  if (new_value != value) {
    node->cast<ValueNodePtr>()->set_value(new_value);
    node->set_abstract(new_abs);
    MS_LOG(DEBUG) << "Set new value of node: " << node->DebugString();
    return true;
  }
  return false;
}

bool EliminateDeadNode(const FuncGraphPtr &func_graph) {
  // Travers all tuple getitem nodes to visit.
  FuncGraphAnalyzer analyzer(func_graph);
  analyzer.Run();
  // Don't handle no-incorporate-call situation to improve performance.
  if (!analyzer.HasIncorporateCall()) {
    return false;
  }

  auto seen = NewSeenGeneration();
  std::vector<int64_t> index_stack;
  bool change = false;
  bool cycle_change = true;
  while (cycle_change) {
    ContextManager context_manager;
    std::vector<AnfNodePtr> tuple_getitem_nodes;
    std::vector<AnfNodePtr> make_tuple_nodes;
    std::vector<AnfNodePtr> value_tuples;
    const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
    for (const auto &node : all_nodes) {
      if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
        tuple_getitem_nodes.emplace_back(node);
      } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
        make_tuple_nodes.emplace_back(node);
      } else if (IsValueNode<ValueTuple>(node)) {
        value_tuples.emplace_back(node);
      }
    }
    // Visit from all tuple_getitem.
    for (const auto &tuple_getitem : tuple_getitem_nodes) {
      VisitNode(tuple_getitem, analyzer, index_stack, seen, &context_manager);
    }
    // Visit from root graph output.
    const auto &output_getitems = GenerateOutputTempGetItems(func_graph);
    for (const auto &tuple_getitem : output_getitems) {
      VisitNode(tuple_getitem, analyzer, index_stack, seen, &context_manager);
    }
    // Check all make tuple's input
    cycle_change = false;
    for (const auto &make_tuple : make_tuple_nodes) {
      MS_LOG(WARNING) << "Check make_tuple:" << make_tuple->DebugString();
      auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
      for (size_t i = 1; i < make_tuple_cnode->size(); i++) {
        // If make_tuple was not visited ,it may be a make tuple of swith_layer or addn and some other ops.
        auto input_edge_visited = context_manager.IndexVisited(make_tuple_cnode, i - 1);
        // Can use `context_manager.contexts_.find(make_tuple_cnode) != context_manager.contexts_.end()`.
        auto make_tuple_visited = make_tuple_cnode->seen_ == seen;
        MS_LOG(WARNING) << "Check [" << i - 1 << "]:"
                        << ", input_edge_visited: " << input_edge_visited
                        << ", make_tuple_visited: " << make_tuple_visited;

        if (!input_edge_visited && make_tuple_visited) {
          cycle_change = EraseMakeTupleInput(func_graph, make_tuple_cnode, i) || cycle_change;
        }
      }
    }
    // Check all value tuple
    for (const auto &value_tuple : value_tuples) {
      auto it = context_manager.contexts_.find(value_tuple);
      if (it == context_manager.contexts_.end()) {
        continue;
      }
      cycle_change = EraseValueTuple(value_tuple, it->second->index_stacks_) || cycle_change;
    }
    change = change || cycle_change;
  }
  return change;
}
}  // namespace opt
}  // namespace mindspore
