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
      contexts_[node] = std::make_shared<VisitContext>(index_stack);
      return true;
    }
    return it->second->Add(index_stack);
  }
};

void VisitNode(const AnfNodePtr &node, const FuncGraphAnalyzer &analyzer, std::vector<int64_t> index_stack, size_t seen,
               ContextManager *context_manager) {
  if (IS_OUTPUT_ON(DEBUG)) {
    MS_LOG(DEBUG) << "Visit node:" << node->DebugString();
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
  } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    // If make_tuple in make_tuple, visit may start with inner tuple_getitem.
    if (index_stack.empty()) {
      return;
    }
    auto make_tuple = node->cast<CNodePtr>();
    auto output_idx = index_stack.back();
    index_stack.pop_back();
    VisitNode(make_tuple->input(1 + output_idx), analyzer, index_stack, seen, context_manager);
  } else if (IsFuncGraphCallNode(node)) {
    const auto &caller_func_graphs = analyzer.GetCallerFuncGraphs(node);
    for (const auto &fg : caller_func_graphs) {
      auto new_index_stack = std::vector<int64_t>(index_stack);
      VisitNode(fg->output(), analyzer, new_index_stack, seen, context_manager);
    }
  } else if (node->isa<Parameter>()) {
    const auto &func_callers = analyzer.GetFuncGraphCallers(node->func_graph());
    for (auto &caller : func_callers) {
      const auto &args = analyzer.GetArg(node, caller);
      auto new_index_stack = std::vector<int64_t>(index_stack);
      for (const auto &arg : args) {
        VisitNode(arg, analyzer, new_index_stack, seen, context_manager);
      }
    }
  } else {
    if (!index_stack.empty()) {
      // TupleGetItem's input may not be a MakeTuple but a ValueTuple.
      MS_LOG(DEBUG) << "Reach the end node: " << node->DebugString() << ", but index stack is not empty.";
    }
    return;
  }
}

void EraseMakeTupleInput(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  // Don't eliminate the parameter of graph
  if (node->isa<Parameter>()) {
    MS_LOG(WARNING) << "Parameter:" << node->DebugString() << " is dead node and can't be erased.";
    return;
  }
  auto new_tensor = NewValueNode(MakeValue(0));
  auto abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(0));
  new_tensor->set_abstract(abs);
  func_graph->manager()->Replace(node, new_tensor);
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

bool EliminateDeadNode(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> tuple_getitem_nodes;
  std::vector<AnfNodePtr> make_tuple_nodes;
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      tuple_getitem_nodes.emplace_back(node);
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      make_tuple_nodes.emplace_back(node);
    }
  }
  // Travers all tuple getitem nodes to visit.
  FuncGraphAnalyzer analyzer(func_graph);
  analyzer.Run();
  // Don't handle no-incorporate-call situation to improve performance.
  if (!analyzer.HasIncorporateCall()) {
    return false;
  }
  auto seen = NewSeenGeneration();
  std::vector<int64_t> index_stack;
  ContextManager context_manager;
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
  bool change = false;
  for (const auto &make_tuple : make_tuple_nodes) {
    auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
    for (size_t i = 1; i < make_tuple_cnode->size(); i++) {
      const auto &input = make_tuple_cnode->input(i);
      // If make_tuple was not visited ,it may be a make tuple of swith_layer or addn and some other ops.
      if (input->seen_ != seen && make_tuple_cnode->seen_ == seen && !IsScalarValueNode(input)) {
        MS_LOG(INFO) << "Find dead node: " << input->DebugString();
        change = true;
        EraseMakeTupleInput(func_graph, input);
      }
    }
  }
  return change;
}
}  // namespace opt
}  // namespace mindspore
