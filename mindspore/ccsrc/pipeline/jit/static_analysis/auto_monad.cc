/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/static_analysis/auto_monad.h"
#include <list>
#include <vector>
#include <stack>
#include <string>
#include <utility>
#include <memory>
#include <algorithm>
#include "ir/anf.h"
#include "pipeline/jit/parse/resolve.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/multitype_funcgraph.h"
#include "utils/flags.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/log_adapter.h"
#include "utils/ordered_map.h"
#include "utils/ordered_set.h"
#include "base/effect_info.h"
#include "ops/core_ops.h"
#include "abstract/abstract_value.h"
#include "pipeline/jit/debug/trace.h"

namespace mindspore {
namespace pipeline {
namespace {  // namespace anonymous
using ClassTypePtr = std::shared_ptr<parse::ClassType>;
using RefInputs = OrderedMap<AnfNodePtr, std::vector<size_t>>;

// Add or get a monad parameter.
AnfNodePtr AddMonadParameter(const FuncGraphPtr &func_graph, const std::string &name,
                             const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  size_t params_size = func_graph->parameters().size();
  size_t io_monad_location = params_size;
  // Search for existed parameters, return it if found.
  for (size_t i = 0; i < params_size; i++) {
    auto &node = func_graph->parameters()[i];
    auto para = dyn_cast<Parameter>(node);
    if (para == nullptr) {
      continue;
    }
    auto para_abs = para->abstract();
    if (para_abs && *para_abs == *abs) {
      return para;
    }
    if (HasAbstractIOMonad(para)) {
      io_monad_location = i;
    }
  }
  // Create a new parameter if not existed.
  auto para = std::make_shared<Parameter>(func_graph);
  para->set_name(name);
  para->debug_info()->set_name(name);
  para->set_abstract(abs);
  // If io monad parameter added before u monad parameter, should insert u monad before io monad in parameters
  if (io_monad_location != params_size && abs->isa<abstract::AbstractUMonad>()) {
    std::vector<AnfNodePtr> params = func_graph->parameters();
    (void)params.insert(params.begin() + SizeToInt(io_monad_location), para);
    func_graph->set_parameters(params);
  } else {
    func_graph->add_parameter(para);
  }
  return para;
}

// Gets side effect propagate attribute value from a ClassType object.
int GetSideEffectPropagate(const ClassTypePtr &class_type) {
  if (class_type) {
    auto obj = class_type->obj();
    if (py::hasattr(obj, GRAPH_FLAG_SIDE_EFFECT_PROPAGATE)) {
      auto value = py::getattr(obj, GRAPH_FLAG_SIDE_EFFECT_PROPAGATE);
      return value.cast<int>();
    }
  }
  return 0;
}

// Gets 'side_effect_propagate' attribute value from a primitive.
int GetSideEffectPropagate(const PrimitivePtr &prim) {
  if (prim) {
    auto attr = prim->GetAttr(GRAPH_FLAG_SIDE_EFFECT_PROPAGATE);
    if (attr && attr->isa<Int64Imm>()) {
      return static_cast<int>(attr->cast<Int64ImmPtr>()->value());
    }
  }
  return 0;
}

// Gets ref inputs and its indexes from a cnode.
RefInputs GetRefInputs(const CNodePtr &cnode) {
  RefInputs ref_inputs;
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto &input = cnode->inputs().at(i);
    if (common::AnfAlgo::HasAbstractRef(input)) {
      ref_inputs[input].push_back(i);
    }
  }
  return ref_inputs;
}

// Return true if cnode has ref input.
bool HasRefInput(const CNodePtr &cnode) {
  if (cnode == nullptr || cnode->inputs().empty()) {
    return false;
  }
  auto &inputs = cnode->inputs();
  // Return true if any of arguments is ref.
  return std::any_of(inputs.begin() + 1, inputs.end(),
                     [](const auto &input) { return common::AnfAlgo::HasAbstractRef(input); });
}

// Return true if cnode has tuple(ref) or list(ref).
bool HasRefSequenceInput(const CNodePtr &cnode) {
  if (cnode == nullptr || cnode->inputs().empty()) {
    return false;
  }
  auto &inputs = cnode->inputs();
  for (size_t index = 1; index < inputs.size(); ++index) {
    const auto &input = cnode->input(index);
    MS_EXCEPTION_IF_NULL(input);
    if (common::AnfAlgo::SequenceHasAbstractRef(input)) {
      return true;
    }
  }
  return false;
}

// Return true if we don't need Load for the given primitive.
// i.e. keep Ref as Ref for some primitives.
bool IsKeepRef(const PrimitivePtr &prim) {
  return (GetSideEffectPropagate(prim) != 0) || IsPrimitiveEquals(prim, prim::kPrimRefToEmbed) ||
         IsPrimitiveEquals(prim, prim::kPrimPull) || IsPrimitiveEquals(prim, prim::kPrimMakeTuple) ||
         IsPrimitiveEquals(prim, prim::kPrimMakeList);
}

// Gets func_graph from the given cnode, return nullptr if it is not a func graph call.
FuncGraphPtr GetFuncGraph(const CNodePtr &cnode) {
  if (cnode != nullptr && !cnode->inputs().empty()) {
    return GetValueNode<FuncGraphPtr>(cnode->input(0));
  }
  return nullptr;
}

// Gets first input as cnode from the given cnode,
// return null if input[0] is not a cnode.
CNodePtr GetFuncCNode(const CNodePtr &cnode) {
  if (cnode != nullptr && !cnode->inputs().empty()) {
    return dyn_cast<CNode>(cnode->input(0));
  }
  return nullptr;
}

// Gets first input as function parameter from the given cnode,
// return null if input[0] is not a parameter.
ParameterPtr GetFuncParameter(const CNodePtr &cnode) {
  if (cnode != nullptr && !cnode->inputs().empty()) {
    return dyn_cast<Parameter>(cnode->input(0));
  }
  return nullptr;
}

// Gets first input as MultitypeFuncGraph from the given cnode,
// return null if input[0] is not a MultitypeFuncGraph.
prim::MultitypeFuncGraphPtr GetFuncMultitypeFuncGraph(const CNodePtr &cnode) {
  if (cnode != nullptr && !cnode->inputs().empty()) {
    return GetValueNode<prim::MultitypeFuncGraphPtr>(cnode->input(0));
  }
  return nullptr;
}

// The cnode is non-effect-node, and the cnode is real node, and the inputs of cnode is dynamic.
bool IsNonEffectRealNodeAndInputIsDynamic(const CNodePtr &cnode) {
  static const PrimitiveSet dynamic_input_node_prims = {
    prim::kPrimStack,        prim::kPrimConcat,   prim::kPrimAddN,         prim::kPrimIdentityN,
    prim::kPrimSparseConcat, prim::kPrimMeshgrid, prim::kPrimDynamicStitch};
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    return false;
  }
  return dynamic_input_node_prims.find(prim) != dynamic_input_node_prims.end();
}

// --------------------------------------------------------------------
// SCC (Strongly Connected Components) related types.
// --------------------------------------------------------------------
using SccVector = mindspore::HashSet<FuncGraphPtr>;
using SccPtr = std::shared_ptr<SccVector>;
using SccMap = mindspore::HashMap<FuncGraphPtr, SccPtr>;

// ---------------------------------------------------------------------
// SccFinder find SCCs using Tarjan's algorithm.
// ---------------------------------------------------------------------
class SccFinder {
 public:
  explicit SccFinder(const FuncGraphPtr &root) : root_(root) {}
  ~SccFinder() = default;
  void Run() { (void)Search(root_); }
  SccMap scc_map() { return std::move(scc_map_); }

 private:
  // Save state of a func graph.
  struct State {
    size_t index = 0;
    size_t lowlink = 0;
    bool in_stack = false;
    explicit State(size_t index) : index(index), lowlink(index), in_stack(false) {}
    ~State() = default;
  };

  // Search SCCs from the given graph.
  State &Search(FuncGraphPtr graph) {
    // Create graph state, set it as visited.
    MS_EXCEPTION_IF_NULL(graph);
    auto [inserted, ok] = visited_.emplace(graph, std::make_unique<State>(index_++));
    if (!ok) {
      MS_LOG(EXCEPTION) << "Already visited: " << graph->ToString();
    }
    auto &state = *(inserted->second);
    // Push visited graph to stack.
    stack_.push(graph);
    state.in_stack = true;
    // Search successor graphs.
    for (auto &used : graph->func_graphs_used()) {
      auto &sg = used.first;
      auto iter = visited_.find(sg);
      if (iter == visited_.end()) {
        // Successor graph has not yet been visited, recurse on it.
        auto &sg_state = Search(sg);
        state.lowlink = std::min(state.lowlink, sg_state.lowlink);
      } else if (iter->second->in_stack) {
        // Successor graph is in stack and hence in the current SCC.
        state.lowlink = std::min(state.lowlink, iter->second->index);
      }
    }
    // If index == lowlink, this means it is the root of SCC.
    if (state.index == state.lowlink) {
      // Pop members of the SCC from stack, they are on top of its root.
      auto scc = std::make_shared<SccVector>();
      while (!stack_.empty()) {
        auto g = stack_.top();
        stack_.pop();
        auto found = visited_.find(g);
        if (found == visited_.end()) {
          MS_LOG(EXCEPTION) << "Unexpected graph: " << g->ToString();
        }
        found->second->in_stack = false;
        // Add graph to SCC, and create the map from graph to SCC.
        scc->insert(g);
        scc_map_.emplace(g, scc);
        if (g == graph) {
          break;
        }
      }
      // SCC should not be empty.
      if (scc->empty()) {
        MS_LOG(EXCEPTION) << "Invalid SCC for: " << graph->ToString();
      }
    }
    return state;
  }

  // The root graph.
  FuncGraphPtr root_;

  // Current index by DFS order.
  size_t index_ = 1;

  // Visited graphs and their states.
  mindspore::HashMap<FuncGraphPtr, std::unique_ptr<State>> visited_;

  // The stack for Tarjan algorithm.
  std::stack<FuncGraphPtr> stack_;

  // The result SCC map, from graph to its SCC.
  SccMap scc_map_;
};

struct SwitchLayerCall {
  CNodePtr caller;
  EffectInfo effect_info;
  std::vector<FuncGraphPtr> branches;
};

class NodeStackGuard {
 public:
  NodeStackGuard(OrderedSet<AnfNodePtr> *stack, const AnfNodePtr &node) : stack_(stack) { stack_->push_front(node); }
  ~NodeStackGuard() {
    try {
      (void)stack_->pop();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Exception when pop. Error info " << e.what();
    }

    stack_ = nullptr;
  }

 private:
  OrderedSet<AnfNodePtr> *stack_;
};

// -------------------------------------------------------------------------------
// SideEffectFinder search and mark side effects for graph and its sub-graphs.
// -------------------------------------------------------------------------------
class SideEffectFinder {
 public:
  static void Search(const FuncGraphPtr &root) {
    SideEffectFinder finder(root);
    finder.Run();
  }

 private:
  explicit SideEffectFinder(const FuncGraphPtr &root) : root_(root) {}
  ~SideEffectFinder() = default;

  void Run() {
    // To handle recursive calls, we generate SCC map before search.
    GenerateSccMap();
    // Update order list to include outer cnodes.
    UpdateOrderLists();
    // Find side effects by DFS from the top graph.
    (void)GetEffectInfo(root_);
    // Check switch calls, add monad arguments if need.
    HandleSwitchCalls();
    // Check switch layer calls, add monad arguments if need.
    HandleSwitchLayerCalls();
  }

  void UpdateOrderLists() const {
    // Some cnodes used in current func graph but belong to other func graph, we have to
    // insert them into order list so that we can handle side effects for them.
    UpdateOrderList(root_);
    for (auto &fg : root_->func_graphs_used_total()) {
      UpdateOrderList(fg);
    }
  }

  static void UpdateOrderList(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    OrderedSet<CNodePtr> new_order_list;
    const auto &order_list = func_graph->order_list();
    for (auto &cnode : order_list) {
      PushToOrderList(func_graph, cnode, &new_order_list);
    }
    func_graph->set_order_list(std::move(new_order_list));
  }

  static void PushToOrderList(const FuncGraphPtr &fg, const CNodePtr &cnode, OrderedSet<CNodePtr> *new_order_list) {
    MS_EXCEPTION_IF_NULL(cnode);
    MS_EXCEPTION_IF_NULL(new_order_list);
    if (new_order_list->contains(cnode)) {
      return;
    }
    for (auto &input : cnode->inputs()) {
      auto input_cnode = dyn_cast<CNode>(input);
      if (input_cnode != nullptr && input_cnode->func_graph() != fg) {
        PushToOrderList(fg, input_cnode, new_order_list);
      }
    }
    new_order_list->push_back(cnode);
  }

  // Generate SCC map by SccFinder.
  void GenerateSccMap() {
    SccFinder scc_finder(root_);
    scc_finder.Run();
    scc_map_ = std::move(scc_finder.scc_map());
  }

  // Gets branch graph from a switch cnode at given input index.
  FuncGraphPtr GetSwitchBranch(const CNodePtr &cnode, size_t index) const {
    MS_EXCEPTION_IF_NULL(cnode);
    return GetValueNode<FuncGraphPtr>(cnode->inputs().at(index));
  }

  // Gets branch graphs from a switch cnode.
  std::vector<FuncGraphPtr> GetSwitchBranches(const CNodePtr &cnode) const {
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr size_t switch_cnode_size = 4;
    constexpr size_t true_index = 2;
    constexpr size_t false_index = 3;
    // Check size.
    if (cnode->size() != switch_cnode_size) {
      MS_LOG(EXCEPTION) << "Invalid switch: " << cnode->DebugString();
    }
    // Add both branches, in some case, only one branch is set.
    std::vector<FuncGraphPtr> branches;
    auto true_branch = GetSwitchBranch(cnode, true_index);
    if (true_branch != nullptr) {
      (void)branches.emplace_back(true_branch);
    }
    auto false_branch = GetSwitchBranch(cnode, false_index);
    if (false_branch != nullptr) {
      (void)branches.emplace_back(false_branch);
    }
    if (branches.empty()) {
      MS_LOG(EXCEPTION) << "Invalid switch: " << cnode->DebugString();
    }
    return branches;
  }

  // Add monad parameter to switch branch graphs.
  void AddMonadParameters(const std::vector<FuncGraphPtr> &branches, const std::string &name,
                          const AbstractBasePtr &abs) const {
    for (auto &branch : branches) {
      (void)AddMonadParameter(branch, name, abs);
    }
  }

  // Trace effect info for Switch cnode.
  EffectInfo TraceSwitchEffectInfo(const CNodePtr &cnode) {
    // Find branches from switch cnode.
    auto branches = GetSwitchBranches(cnode);
    // Save branch caller, so that we can update arguments for the caller.
    SaveBranchCaller(cnode, branches);
    // For some case, only one branch is set.
    if (branches.size() == 1) {
      auto &branch = branches.front();
      return GetEffectInfo(branch);
    }
    // When both branches are set, merge their effect infos.
    EffectInfo info = MergeEffectInfo(branches);
    if (info.state == EffectInfo::kDetected) {
      // Setup both branches according the merged effect info.
      SetupEffectBranches(info, branches);
    }
    return info;
  }

  // Trace effect info for SwitchLayer cnode.
  EffectInfo TraceSwitchLayerEffectInfo(const CNodePtr &cnode) {
    // Find branches from switch_layer cnode.
    auto branches = GetSwitchLayerBranches(cnode);
    // Merge effect info from all branches.
    EffectInfo info = MergeEffectInfo(branches);
    if (info.state == EffectInfo::kDetected) {
      // Setup branches according the merged effect info.
      SetupEffectBranches(info, branches);
      // Save the switch_layer call, so that we can add monad argument for it if need.
      auto &call = switch_layer_calls_.emplace_back();
      call.caller = caller_;
      call.effect_info = info;
      call.branches = move(branches);
    }
    return info;
  }

  void HandleSwitchCalls() {
    for (auto &call : switch_calls_) {
      const auto &caller = call.first;
      const auto &branches = call.second;
      CheckAndFixSwitchCall(caller, branches);
    }
  }

  void CheckAndFixSwitchCall(const CNodePtr &caller, const FuncGraphVector &branches) const {
    MS_EXCEPTION_IF_NULL(caller);
    const auto caller_input_size = caller->inputs().size();
    for (auto &branch : branches) {
      MS_EXCEPTION_IF_NULL(branch);
      if (caller_input_size != branch->parameters().size() + 1) {
        // Fix branch if number of parameter mismatch.
        FixSwitchBranch(caller, branch);
        // The number of parameter should matched after fix.
        if (caller_input_size != branch->parameters().size() + 1) {
          MS_LOG(EXCEPTION) << "Fix switch branch parameters failed! " << caller->DebugString();
        }
      }
    }
  }

  void FixSwitchBranch(const CNodePtr &caller, const FuncGraphPtr &branch) const {
    MS_EXCEPTION_IF_NULL(branch);
    for (size_t i = caller->size() - 1; i > 0; --i) {
      auto &input = caller->input(i);
      if (HasAbstractUMonad(input)) {
        (void)AddMonadParameter(branch, "u", input->abstract());
      } else if (HasAbstractIOMonad(input)) {
        (void)AddMonadParameter(branch, "io", input->abstract());
      }
    }
  }

  void HandleSwitchLayerCalls() {
    for (auto &call : switch_layer_calls_) {
      const auto &info = call.effect_info;
      const auto &branches = call.branches;
      auto new_info = MergeEffectInfo(branches);
      // Reset branches if effect info changed.
      if (new_info.memory != info.memory || new_info.load != info.load || new_info.io != info.io) {
        AddMonadForCaller(call.caller, new_info);
        SetupEffectBranches(new_info, branches);
      }
    }
  }

  // Gets branch graphs from a switch_layer cnode.
  std::vector<FuncGraphPtr> GetSwitchLayerBranches(const CNodePtr &cnode) {
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr size_t func_tuple_index = 2;
    constexpr int recursive_level = 2;
    if (cnode->size() <= func_tuple_index) {
      MS_LOG(EXCEPTION) << "Invalid switch_layer: " << cnode->DebugString(recursive_level);
    }
    auto func_tuple = cnode->inputs().at(func_tuple_index);
    return GetGraphsFromTuple(func_tuple);
  }

  FuncGraphPtr GetGraphFromSwitchWithDeadNode(const CNodePtr &cnode) {
    auto node = cnode->inputs()[0];
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimSwitch)) {
      return nullptr;
    }
    const auto &inputs = node->cast<CNodePtr>()->inputs();
    auto cond_node = inputs[kSwitchCondIndex];
    auto cond_abs = cond_node->abstract();
    MS_EXCEPTION_IF_NULL(cond_abs);
    auto cond_abs_val = cond_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(cond_abs_val);
    if (cond_abs_val == kValueAny) {
      return nullptr;
    }
    auto cond_abs_bool_val = dyn_cast<BoolImm>(cond_abs_val);
    MS_EXCEPTION_IF_NULL(cond_abs_bool_val);
    auto branch = cond_abs_bool_val->value() ? inputs[kSwitchTrueBranchIndex] : inputs[kSwitchFalseBranchIndex];
    return GetValueNode<FuncGraphPtr>(branch);
  }

  // Get and trace graphs from a tuple of func node for switch_layer.
  std::vector<FuncGraphPtr> GetGraphsFromTuple(const AnfNodePtr &func_tuple) {
    // The func tuple maker.
    if (IsPrimitiveCNode(func_tuple, prim::kPrimMakeTuple)) {
      return GetGraphsFromMakeTuple(func_tuple->cast<CNodePtr>());
    }
    // Trace tuple from parameter.
    auto para = dyn_cast<Parameter>(func_tuple);
    if (para != nullptr) {
      std::vector<FuncGraphPtr> graphs;
      ForEachRealArguments(para,
                           [this, &graphs](const AnfNodePtr &arg) { graphs = std::move(GetGraphsFromTuple(arg)); });
      return graphs;
    }
    // Trace tuple returned from func graph call.
    auto cnode = dyn_cast<CNode>(func_tuple);
    auto func_graph = GetFuncGraph(cnode);
    if (func_graph != nullptr) {
      return GetGraphsFromTuple(func_graph->output());
    }
    // Trace tuple returned from func graph call including switch with dead node.
    func_graph = GetGraphFromSwitchWithDeadNode(cnode);
    if (func_graph != nullptr) {
      return GetGraphsFromTuple(func_graph->output());
    }
    MS_LOG(EXCEPTION) << "Invalid input for switch_layer: func_graph is nullptr.";
  }

  // Get graphs from a tuple of funcs make node for switch_layer.
  std::vector<FuncGraphPtr> GetGraphsFromMakeTuple(const CNodePtr &make_tuple) const {
    MS_EXCEPTION_IF_NULL(make_tuple);
    auto &inputs = make_tuple->inputs();
    constexpr int recursive_level = 2;
    if (inputs.size() <= 1) {
      MS_LOG(EXCEPTION) << "Invalid make_tuple for switch_layer: " << make_tuple->DebugString(recursive_level);
    }
    std::vector<FuncGraphPtr> graphs;
    graphs.reserve(inputs.size() - 1);
    for (size_t i = 1; i < inputs.size(); ++i) {
      auto func_graph = GetValueNode<FuncGraphPtr>(inputs.at(i));
      if (func_graph == nullptr) {
        MS_LOG(WARNING) << "Non-graph found in switch_layer input: " << make_tuple->DebugString(recursive_level)
                        << " index=" << i;
        continue;
      }
      graphs.push_back(func_graph);
    }
    return graphs;
  }

  // Trace effect info from tuple_getitem cnode.
  EffectInfo TraceTupleGetItemEffectInfo(const CNodePtr &cnode, std::stack<int64_t> *tuple_indexes) {
    constexpr size_t tuple_input = 1;
    constexpr size_t index_input = 2;
    constexpr size_t cnode_size = 3;
    if (cnode->size() != cnode_size) {
      MS_LOG(EXCEPTION) << "Invalid tuple_getitem: " << cnode->DebugString();
    }
    // Get item index.
    auto &index_node = cnode->inputs().at(index_input);
    auto index_value = GetValueNode<Int64ImmPtr>(index_node);
    if (index_value == nullptr) {
      MS_LOG(EXCEPTION) << "Tuple_getitem with non-const index " << cnode->DebugString();
    }
    int64_t index = index_value->value();

    // Get tuple value.
    const auto &tuple_node = cnode->inputs().at(tuple_input);
    // Push tuple index.
    tuple_indexes->push(index);
    return TraceTupleEffectInfo(tuple_node, tuple_indexes);
  }

  EffectInfo TraceTupleEffectInfo(const AnfNodePtr &tuple_node, std::stack<int64_t> *tuple_indexes) {
    MS_EXCEPTION_IF_NULL(tuple_indexes);
    auto para = dyn_cast<Parameter>(tuple_node);
    if (para != nullptr) {
      return TraceTupleParaEffectInfo(para, *tuple_indexes);
    }
    auto tuple_cnode = dyn_cast<CNode>(tuple_node);
    if (tuple_cnode != nullptr) {
      return TraceTupleCNodeEffectInfo(tuple_cnode, tuple_indexes);
    }
    // Should not reach here.
    MS_LOG(EXCEPTION) << "Side effects untraceable: tuple_cnode is nullptr.";
  }

  EffectInfo TraceTupleParaEffectInfo(const ParameterPtr &para, const std::stack<int64_t> &tuple_indexes) {
    EffectInfo info{EffectInfo::kDetected, false, false, false, false};
    ForEachRealArguments(para, [this, &info, tuple_indexes](const AnfNodePtr &arg) {
      // Merge real argument effect info.
      auto tuple_indexes_copy = tuple_indexes;
      auto arg_info = TraceTupleEffectInfo(arg, &tuple_indexes_copy);
      info.Merge(arg_info);
    });
    return info;
  }

  EffectInfo TraceTupleCNodeEffectInfo(const CNodePtr &cnode, std::stack<int64_t> *tuple_indexes) {
    MS_EXCEPTION_IF_NULL(tuple_indexes);
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitiveWithoutDoSignature(cnode);
    constexpr int recursive_level = 2;
    // Trace MakeTuple.
    if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple)) {
      if (tuple_indexes->empty()) {
        MS_LOG(EXCEPTION) << "Unexpected make_tuple: " << cnode->DebugString(recursive_level);
      }
      // Pop out tuple index.
      auto top_index = tuple_indexes->top();
      tuple_indexes->pop();
      size_t input_index = 0;
      // Support tuple index is negative
      if (top_index < 0) {
        if (SizeToLong(cnode->size()) + top_index < 0) {
          MS_LOG(EXCEPTION) << "Invalid make_tuple: " << cnode->DebugString() << " index=" << top_index;
        }
        input_index = static_cast<size_t>(cnode->size() + top_index);
      } else {
        // Follow the tuple item according the index.
        input_index = static_cast<size_t>(top_index) + 1;
      }
      if (input_index >= cnode->size()) {
        MS_LOG(EXCEPTION) << "Invalid make_tuple: " << cnode->DebugString() << " index=" << top_index;
      }
      if (tuple_indexes->empty()) {
        // Trace non-tuple.
        return TraceEffectInfo(cnode->inputs().at(input_index));
      }
      // This is the tuple of tuple case.
      return TraceTupleEffectInfo(cnode->inputs().at(input_index), tuple_indexes);
    }
    // Trace TupleGetItem (tuple of tuple).
    if (IsPrimitiveEquals(prim, prim::kPrimTupleGetItem)) {
      return TraceTupleGetItemEffectInfo(cnode, tuple_indexes);
    }
    // Trace primitive propagating side effect from its input, such as Depend, etc.
    int input_index = GetSideEffectPropagate(prim);
    if (input_index > 0 && input_index < static_cast<int>(cnode->size())) {
      return TraceTupleEffectInfo(cnode->input(static_cast<size_t>(input_index)), tuple_indexes);
    }
    // Tuple returned from func graph call.
    auto func_graph = GetFuncGraph(cnode);
    if (func_graph != nullptr) {
      return TraceTupleEffectInfo(func_graph->output(), tuple_indexes);
    }
    // Tuple returned from a Switch call.
    if (cnode->size() == 1 && IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch)) {
      return TraceTupleFromSwitch(cnode->input(0)->cast<CNodePtr>(), *tuple_indexes);
    }
    // Tuple is returned from J().
    //   %1 = J(primal)
    //   tuple = %1(args)
    if (cnode->size() > 0 && IsPrimitiveCNode(cnode->input(0), prim::kPrimJ)) {
      MS_LOG(DEBUG) << "Tuple from J: " << cnode->DebugString(recursive_level);
      constexpr size_t func_index = 1;
      auto j_conde = cnode->input(0)->cast<CNodePtr>();
      auto j_func = j_conde->input(func_index);
      auto func_info = TraceEffectInfo(j_func);
      // In order to add the Umonad arg to the bprop_top_cell in advance,
      // so that the side effects in the bprop graph are sorted earlier than the side effects of the optimizer.
      return {EffectInfo::kDetected, false, false, false, func_info.back_mem};
    }
    // Rare case.
    MS_LOG(WARNING) << "Tuple untraceable from: " << cnode->DebugString(recursive_level);
    return {EffectInfo::kDetected, false, false, false};
  }

  // Trace effect info from a Switch node that output is a tuple.
  EffectInfo TraceTupleFromSwitch(const CNodePtr &switch_cnode, const std::stack<int64_t> &tuple_indexes) {
    auto branches = GetSwitchBranches(switch_cnode);
    EffectInfo info = {EffectInfo::kDetected, false, false, false, false};
    for (auto &branch : branches) {
      auto tuple_indexes_copy = tuple_indexes;
      EffectInfo branch_info = TraceTupleEffectInfo(branch->output(), &tuple_indexes_copy);
      info.Merge(branch_info);
    }
    return info;
  }

  // Setup all branches according the effect info.
  void SetupEffectBranches(const EffectInfo &info, const std::vector<FuncGraphPtr> &branches) {
    // Setup monad parameters for all branches according the effect info.
    if (info.memory || info.load) {
      AddMonadParameters(branches, "u", kUMonad->ToAbstract());
    }
    if (info.io) {
      AddMonadParameters(branches, "io", kIOMonad->ToAbstract());
    }
    // Set merged effect info to both branches.
    for (auto &branch : branches) {
      MS_EXCEPTION_IF_NULL(branch);
      branch->SetEffectInfo(info);
      // Update caller if it is existed.
      UpdateBranchCaller(branch);
    }
  }

  // Merge effect info for switch or switch_layer branch graphs.
  EffectInfo MergeEffectInfo(const std::vector<FuncGraphPtr> &branches) {
    EffectInfo info = {EffectInfo::kDetected, false, false, false, false};
    for (auto &branch : branches) {
      MS_EXCEPTION_IF_NULL(branch);
      EffectInfo branch_info = GetEffectInfo(branch);
      info.Merge(branch_info);
    }
    return info;
  }

  // Trace a cnode for effect info.
  EffectInfo TraceEffectInfo(const CNodePtr &cnode) {
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitiveWithoutDoSignature(cnode);
    if (IsPrimitiveEquals(prim, prim::kPrimSwitch)) {
      // Special handling for Switch primitive.
      return TraceSwitchEffectInfo(cnode);
    }

    if (IsPrimitiveEquals(prim, prim::kPrimSwitchLayer)) {
      // Special handling for SwitchLayer primitive.
      return TraceSwitchLayerEffectInfo(cnode);
    }

    if (IsPrimitiveEquals(prim, prim::kPrimTupleGetItem)) {
      // Trace tuple_getitem.
      std::stack<int64_t> tuple_indexes;
      return TraceTupleGetItemEffectInfo(cnode, &tuple_indexes);
    }

    if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple)) {
      // Trace make_tuple.
      const auto &inputs = cnode->inputs();
      EffectInfo info{EffectInfo::kDetected, false, false, false, false};
      for (size_t i = 1; i < inputs.size(); ++i) {
        auto input_info = TraceEffectInfo(inputs[i]);
        info.Merge(input_info);
      }
      return info;
    }

    // For high-order pritimive such as Partial,
    // we trace effect info from its argument.
    int index_prim = GetSideEffectPropagate(prim);
    if (index_prim > 0 && index_prim < static_cast<int>(cnode->size())) {
      return TraceEffectInfo(cnode->input(static_cast<size_t>(index_prim)));
    }

    // For func graph calls, we trace effect info from graph output.
    auto called_graph = GetFuncGraph(cnode);
    if (called_graph != nullptr) {
      // Save the caller of the graph, so that we can update
      // monad parameters for it when requires.
      (void)graph_callers_[called_graph].emplace(cnode);
      return TraceEffectInfo(called_graph->output());
    }

    auto func_cnode = GetFuncCNode(cnode);
    if (func_cnode != nullptr) {
      //
      // For ClassType as the input[0], if it is a primitive class
      // with 'side_effect_propagate' attribute, we trace side effect
      // from its argument indxed by the attribute value.
      //
      // e.g.:
      //     setpara = P.Partial()(P.Assign, self.para)
      //     setpara(x)
      //
      auto class_type = GetValueNode<ClassTypePtr>(func_cnode->input(0));
      if (class_type != nullptr) {
        int index = GetSideEffectPropagate(class_type);
        if (index > 0 && index < static_cast<int>(cnode->size())) {
          return TraceEffectInfo(cnode->input(static_cast<size_t>(index)));
        }
      }

      // For high order cnode, trace effect info from the output of the input cnode.
      return TraceOutputEffectInfo(func_cnode);
    }

    // Otherwise, assume no side effect and stop trace.
    MS_LOG(INFO) << "CNode side effect unknown: " << cnode->DebugString();
    return {EffectInfo::kDetected, false, false, false, false};
  }

  // Trace effect info from output of the cnode.
  EffectInfo TraceOutputEffectInfo(const CNodePtr &cnode) {
    MS_EXCEPTION_IF_NULL(cnode);
    std::vector<ValuePtr> values;
    GetOutputValues(cnode, &values);
    if (values.size() == 1) {
      return GetEffectInfo(values.front());
    }
    EffectInfo info{EffectInfo::kDetected, false, false, false, false};
    for (auto &value : values) {
      info.Merge(GetEffectInfo(value));
    }
    return info;
  }

  EffectInfo GetEffectInfo(const ValuePtr &value) {
    MS_EXCEPTION_IF_NULL(value);
    // FuncGraph.
    auto graph = dyn_cast<FuncGraph>(value);
    if (graph != nullptr) {
      return GetEffectInfo(graph);
    }
    // Primitive.
    auto prim = dyn_cast<Primitive>(value);
    if (prim != nullptr) {
      return GetPrimEffectInfo(prim);
    }
    MS_LOG(INFO) << "Value side effect unknown: " << value->ToString();
    return {EffectInfo::kDetected, false, false, false, false};
  }

  void GetOutputValues(const CNodePtr &cnode, std::vector<ValuePtr> *values) {
    MS_EXCEPTION_IF_NULL(cnode);
    // CNode is a func graph call.
    auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    if (graph != nullptr) {
      GetOutputValues(graph, values);
      return;
    }
    // CNode is applying another cnode.
    auto func_cnode = dyn_cast<CNode>(cnode->input(0));
    if (func_cnode != nullptr) {
      GetOutputValues(func_cnode, values);
      return;
    }
    // Primitive cnode.
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (IsPrimitiveEquals(prim, prim::kPrimSwitch)) {
      // Switch.
      auto branches = GetSwitchBranches(cnode);
      GetOutputValues(branches, values);
      return;
    }
    if (IsPrimitiveEquals(prim, prim::kPrimSwitchLayer)) {
      // Switch layer.
      auto branches = GetSwitchLayerBranches(cnode);
      GetOutputValues(branches, values);
      return;
    }
    if (IsPrimitiveEquals(prim, prim::kPrimPartial)) {
      // Partial.
      auto fg = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (fg != nullptr) {
        GetOutputValues(fg, values);
        return;
      }
    }
    // Other cases not supported yet.
    MS_LOG(INFO) << "Output unknown: " << cnode->DebugString();
  }

  void GetOutputValues(const FuncGraphPtr &graph, std::vector<ValuePtr> *values) {
    MS_EXCEPTION_IF_NULL(graph);
    MS_EXCEPTION_IF_NULL(values);
    auto output = graph->output();
    // Output is a value node.
    auto value = GetValueNode(output);
    if (value != nullptr) {
      (void)values->emplace_back(value);
      return;
    }

    // Output is a cnode.
    auto cnode = dyn_cast<CNode>(output);
    if (cnode != nullptr) {
      GetOutputValues(cnode, values);
      return;
    }

    MS_LOG(INFO) << "Unexpected output: " << output->DebugString();
  }

  void GetOutputValues(const std::vector<FuncGraphPtr> &graphs, std::vector<ValuePtr> *values) {
    for (auto &graph : graphs) {
      GetOutputValues(graph, values);
    }
  }

  // Trace an AnfNode for effect info.
  EffectInfo TraceEffectInfo(const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    // Trace cnode.
    auto cnode = node->cast<CNodePtr>();
    if (cnode != nullptr) {
      return TraceEffectInfo(cnode);
    }

    // Trace parameter.
    auto para = node->cast<ParameterPtr>();
    if (para != nullptr) {
      return TraceEffectInfo(para);
    }

    // Trace primitive.
    auto prim = GetPrimitiveWithoutDoSignature(node);
    if (prim != nullptr) {
      return GetPrimEffectInfo(prim);
    }

    // Trace func graph.
    auto graph = GetValueNode<FuncGraphPtr>(node);
    if (graph != nullptr) {
      return GetEffectInfo(graph);
    }

    // Other ValueNode has no side effects. For example: ValueNode<ClassType> node.
    //  node1 = ValueNode<ClassType> class 'mindspore.ops.operations.debug_ops.Print'
    //  node2 = _get_cache_prim(node1) // the node has side effects.
    if (node->isa<ValueNode>()) {
      MS_LOG(DEBUG) << "The ValueNode has no side effect: " << node->DebugString();
      return {EffectInfo::kDetected, false, false, false, false};
    }
    // Something is wrong if we reached here.
    MS_LOG(WARNING) << "The effect info of the node is untraceable: " << node->DebugString()
                    << ".\nLine:" << trace::GetDebugInfo(node->debug_info());
    return {EffectInfo::kDetected, false, false, false, false};
  }

  int GetParameterIndex(const FuncGraphPtr &func_graph, const ParameterPtr &para) const {
    int parameter_index = 0;
    for (auto &parameter : func_graph->parameters()) {
      if (para == parameter) {
        return parameter_index;
      }
      ++parameter_index;
    }
    MS_LOG(EXCEPTION) << "Parameter not found: " << (para ? para->DebugString() : "<null>");
  }

  // Trace effect info from function parameter.
  EffectInfo TraceEffectInfo(const ParameterPtr &para) {
    EffectInfo info{EffectInfo::kDetected, false, false, false, false};
    ForEachRealArguments(para, [this, &info](const AnfNodePtr &arg) {
      // Merge caller input effect info.
      auto input_info = TraceEffectInfo(arg);
      info.Merge(input_info);
    });
    return info;
  }

  void ForEachRealArguments(const ParameterPtr &para, const std::function<void(const AnfNodePtr &)> &handler) {
    MS_EXCEPTION_IF_NULL(para);
    auto func_graph = para->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    // Find index of the parameter, starts from 0.
    const int para_index = GetParameterIndex(func_graph, para);
    const size_t input_index = static_cast<size_t>(para_index) + 1;
    // Search user cnodes of the func graph.
    auto &users = func_graph->func_graph_cnodes_index();
    if (users.empty()) {
      MS_LOG(WARNING) << "Unused graph for parameter " << para->DebugString();
    }
    // Push the parameter to a stack so that we can check cycle binding.
    NodeStackGuard param_stack_guard(&formal_param_stack_, para);
    for (auto &user : users) {
      auto use_index = user.first->second;
      if (use_index != 0) {
        // Skip non-caller usage.
        continue;
      }
      // Caller cnode.
      auto cnode = dyn_cast<CNode>(user.first->first);
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode != nullptr && input_index < cnode->size()) {
        auto &input = cnode->input(input_index);
        if (formal_param_stack_.contains(input)) {
          // Skip if the input is a parameter that we are finding its real argument.
          continue;
        }
        handler(input);
      }
    }
  }

  // For call node, returns effect info of the callee graph.
  EffectInfo GetCallEffectInfo(const CNodePtr &cnode) {
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr size_t min_call_node_size = 2;
    if (cnode->size() < min_call_node_size) {
      MS_LOG(EXCEPTION) << "Invalid call node: " << cnode->DebugString();
    }
    auto func_graph = GetValueNode<FuncGraphPtr>(cnode->inputs().at(1));
    if (func_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid call node: " << cnode->DebugString();
    }
    return GetEffectInfo(func_graph);
  }

  // Detect effect info by depth first search.
  EffectInfo DetectEffectInfo(const CNodePtr &cnode) {
    // For primitive, get effect info from its attributes and inputs.
    auto prim = GetCNodePrimitiveWithoutDoSignature(cnode);
    if (prim != nullptr) {
      // Skip 'return' cnode.
      if (IsPrimitiveEquals(prim, prim::kPrimReturn)) {
        return {EffectInfo::kDetected, false, false, false, false};
      }
      // Special handling for 'call' cnode.
      if (IsPrimitiveEquals(prim, prim::kPrimCall)) {
        return GetCallEffectInfo(cnode);
      }
      auto info = GetPrimEffectInfo(prim);
      if (!info.memory && !IsKeepRef(prim)) {
        // For primitive calls, if no memory effects but
        // Ref parameter used, we will insert 'load' before them.
        // Except for primitives like J(f) or Partial(f, x) which propagate side effect,
        // load is inserted inside the func_graph f.
        info.load = HasRefInput(cnode);
      }
      if (!info.memory && IsNonEffectRealNodeAndInputIsDynamic(cnode)) {
        info.load = HasRefSequenceInput(cnode);
      }
      return info;
    }

    // For func graph, detect effect info by its children cnodes.
    auto func_graph = GetFuncGraph(cnode);
    if (func_graph != nullptr) {
      // Save the caller of the graph, so that we can update
      // monad parameters for it when requires.
      (void)graph_callers_[func_graph].emplace(cnode);
      return GetEffectInfo(func_graph);
    }

    // When input[0] is a cnode, it is a function returned from
    // a high-order function call, we trace it by return value.
    auto func_cnode = GetFuncCNode(cnode);
    if (func_cnode != nullptr) {
      caller_ = cnode;
      return TraceEffectInfo(func_cnode);
    }

    // When input[0] is a parameter, it is a function parameter for
    // the high-order function, we trace it by caller.
    auto func_para = GetFuncParameter(cnode);
    if (func_para != nullptr) {
      return TraceEffectInfo(func_para);
    }

    // When input[0] is a MultitypeFuncGraph, it's not specialized
    // as one of its parameters is AbstractUndertermined,
    // This MultitypeFuncGraph may be specialized at next Renormalize
    // process, but we have to keep the order by insert UMonad now,
    // otherwise order will be lost in next Renormalize.
    // So assume it has memory side effect conservatively.
    auto func_multitype = GetFuncMultitypeFuncGraph(cnode);
    if (func_multitype != nullptr) {
      MS_LOG(DEBUG) << "Assume memory side effect for: " << cnode->DebugString();
      return {EffectInfo::kDetected, true, false, false, false};
    }

    // For other cnodes, we assume that they have no side effects.
    MS_LOG(DEBUG) << "Assume no side effect for: " << cnode->DebugString();
    return {EffectInfo::kDetected, false, false, false, false};
  }

  // Gets EffectInfo for CNode.
  EffectInfo GetEffectInfo(const CNodePtr &cnode) {
    const auto &effect_info = cnode->GetEffectInfo();
    if (effect_info.state == EffectInfo::kDetected) {
      // Effect info already detected, return it.
      return effect_info;
    }

    // Detect effect info for the cnode.
    EffectInfo info = DetectEffectInfo(cnode);
    if (info.state == EffectInfo::kDetected) {
      // Save detected info into cnode.
      cnode->SetEffectInfo(info);
    }
    return info;
  }

  // Gets SCC that the given graph belongs to.
  SccPtr GetScc(const FuncGraphPtr &func_graph) const {
    auto found = scc_map_.find(func_graph);
    if (found == scc_map_.end()) {
      MS_LOG(EXCEPTION) << "SCC not found for " << (func_graph ? func_graph->ToString() : "FG(null)");
    }
    return found->second;
  }

  // Set effect info for all member graphs in the SCC.
  void SetSccEffectInfo(const SccPtr &scc, const EffectInfo &info) const {
    MS_EXCEPTION_IF_NULL(scc);
    for (auto &g : *scc) {
      MS_EXCEPTION_IF_NULL(g);
      g->SetEffectInfo(info);
    }
  }

  // Gets EffectInfo for func graph.
  EffectInfo GetEffectInfo(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &effect_info = func_graph->GetEffectInfo();
    if (effect_info.state != EffectInfo::kUnknown) {
      // Effect info already set, return it.
      return effect_info;
    }
    // Get SCC that this graph belongs to.
    auto scc = GetScc(func_graph);
    MS_EXCEPTION_IF_NULL(scc);
    // To prevent SCC members be visited again, we set effect info
    // to 'kDetecting' state before start to check cnodes.
    EffectInfo info{EffectInfo::kDetecting, false, false, false, false};
    SetSccEffectInfo(scc, info);
    // Check side effects for all cnodes in the SCC.
    std::vector<CNodePtr> undetected;
    for (auto &g : *scc) {
      MS_EXCEPTION_IF_NULL(g);
      for (auto &cnode : g->order_list()) {
        auto cnode_effect = GetEffectInfo(cnode);
        if (cnode_effect.state != EffectInfo::kDetected) {
          // For side effect undetected node, it could be a call to the SCC member graph,
          // we will try to check side effect again after SCC side effect detected.
          undetected.push_back(cnode);
        }
        // Merge effect info from the node.
        info.Merge(cnode_effect);
      }
      // Make sure all sub-graphs is checked. since some sub-graphs may not directly called,
      // for example: return ValueNode(sub_graph).
      for (auto &sg : g->func_graphs_used()) {
        (void)GetEffectInfo(sg.first);
      }
    }
    // Update effect into for all members of the SCC.
    info.state = EffectInfo::kDetected;
    SetSccEffectInfo(scc, info);
    // Check undetected cnodes again after side effect of the SCC is detected.
    for (auto &cnode : undetected) {
      MS_EXCEPTION_IF_NULL(cnode);
      auto cnode_effect = GetEffectInfo(cnode);
      // Side effect should be detected now, except free variable nodes that not belong to current SCC.
      if (cnode_effect.state != EffectInfo::kDetected && scc->find(cnode->func_graph()) != scc->end()) {
        MS_LOG(EXCEPTION) << "Side effect is undectable: " << cnode->DebugString();
      }
    }
    // graph which need PipelineSplit doesn't have effect.
    if (func_graph->stage() != -1) {
      info.memory = false;
      info.load = false;
      info.io = false;
    }
    return info;
  }

  // The caller of switch node is also a caller of the branches, we save them
  // so that we can update monad parameters for the caller when it requires.
  void SaveBranchCaller(const CNodePtr &switch_node, const FuncGraphVector &branches) {
    MS_EXCEPTION_IF_NULL(switch_node);
    auto fg = switch_node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto manager = fg->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &node_users = manager->node_users();
    auto found = node_users.find(switch_node);
    if (found == node_users.end()) {
      MS_LOG(WARNING) << "Caller not found for " << switch_node->DebugString();
      return;
    }
    bool is_multi_branches = (branches.size() > 1);
    for (auto &user : found->second) {
      auto cnode = dyn_cast<CNode>(user.first);
      if (cnode == nullptr || user.second != 0) {
        continue;
      }
      // The cnode is the switch caller.
      if (is_multi_branches) {
        // Caller to branches.
        (void)switch_calls_.emplace(cnode, branches);
      }
      for (auto &branch : branches) {
        // Branch to caller.
        (void)graph_callers_[branch].emplace(cnode);
      }
    }
  }

  void UpdateBranchCaller(const FuncGraphPtr &branch) {
    MS_EXCEPTION_IF_NULL(branch);
    auto iter = graph_callers_.find(branch);
    if (iter == graph_callers_.end()) {
      return;
    }
    const auto &info = branch->GetEffectInfo();
    for (auto &caller : iter->second) {
      AddMonadForCaller(caller, info);
    }
  }

  void AddMonadForCaller(const CNodePtr &caller, const EffectInfo &info) const {
    if (info.memory || info.load) {
      // Add u monad argument to caller if need.
      AddMonadArgument(caller, kUMonad);
    }
    if (info.io) {
      // Add io monad argument to caller if need.
      AddMonadArgument(caller, kIOMonad);
    }
  }

  void AddMonadArgument(const CNodePtr &cnode, const ValuePtr &monad) const {
    MS_EXCEPTION_IF_NULL(cnode);
    MS_EXCEPTION_IF_NULL(monad);
    auto monad_abs = monad->ToAbstract();
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto abs = cnode->inputs().at(i)->abstract();
      if (abs != nullptr && *abs == *monad_abs) {
        // Skip if monad argument already existed.
        return;
      }
    }
    // Add monad argument if not yet.
    auto monad_input = NewValueNode(monad);
    monad_input->set_abstract(monad_abs);
    if ((monad == kUMonad) && cnode->size() > 1 && HasAbstractIOMonad(cnode->inputs().back())) {
      // Insert u monad before io monad.
      size_t last_index = cnode->size() - 1;
      cnode->add_input(cnode->input(last_index));
      cnode->set_input(last_index, monad_input);
    } else {
      // Add monad as the last input.
      cnode->add_input(monad_input);
    }
  }

  // The root graph.
  FuncGraphPtr root_;

  // SCC map.
  SccMap scc_map_;

  // Map graph to its caller cnodes, so that we can add monad inputs to the
  // caller cnode when we late found that the graph added monad parameters.
  mindspore::HashMap<FuncGraphPtr, mindspore::HashSet<CNodePtr>> graph_callers_;

  // Current high order func caller cnode.
  CNodePtr caller_ = nullptr;

  // Save switch caller cnodes and their branches, so that we can check and
  // update monad parameters for branches according the caller inputs.
  mindspore::HashMap<CNodePtr, FuncGraphVector> switch_calls_;

  // switch_layer_calls save all switch_layer calls, so that
  // we can check whether monad argument should be added for them.
  std::vector<SwitchLayerCall> switch_layer_calls_;

  // Save traced formal parameters so that we can check cycle parameter binding.
  OrderedSet<AnfNodePtr> formal_param_stack_;
};  // class SideEffectFinder

// --------------------------------------------------------------------
// AutoMonadConverter converts side-effect cnodes into monad form.
// --------------------------------------------------------------------
class AutoMonadConverter {
 public:
  static bool Handle(const FuncGraphPtr &func_graph, bool top) {
    AutoMonadConverter converter(func_graph, top);
    return converter.Run();
  }

 private:
  AutoMonadConverter(const FuncGraphPtr &func_graph, bool top)
      : func_graph_(func_graph), manager_(func_graph->manager()), top_(top) {}

  ~AutoMonadConverter() = default;

  bool Run() {
    // Handle cnodes for side effects.
    const auto &info = func_graph_->GetEffectInfo();
    if (info.state == EffectInfo::kDetected) {
      HandleCNodes();
    }

    // Safe to clear isolated nodes after handled side effect nodes.
    ClearIsolatedNodes();

    // Clean up after conversion finished.
    func_graph_->ClearOrderList();
    return has_effect_cnodes_;
  }

  // Check if there are side effects from effect info.
  static bool HasSideEffects(const EffectInfo &info) { return (info.memory || info.io || info.load || info.back_mem); }

  // Gets effect info for a cnode.
  const EffectInfo &GetEffectInfo(const CNodePtr &cnode) const {
    MS_EXCEPTION_IF_NULL(cnode);
    auto &effect_info = cnode->GetEffectInfo();
    if (effect_info.state != EffectInfo::kDetected) {
      // Effect info should have been set by SideEffectFinder.
      MS_LOG(EXCEPTION) << "Side effects not detected: " << cnode->DebugString();
    }
    return effect_info;
  }

  // Handle CNodes for side effects.
  void HandleCNodes() {
    // Check whether UpdateState and Depend are required.
    bool update_state = NeedUpdateState();

    // Check all cnodes in order list.
    for (auto &cnode : func_graph_->order_list()) {
      auto &info = GetEffectInfo(cnode);
      has_effect_cnodes_ = (has_effect_cnodes_ || HasSideEffects(info));
      if (cnode->func_graph() != func_graph_) {
        // Handle outer cnode.
        HandleOuterNode(cnode, info);
      } else {
        // Handle cnode with memory side effects.
        if (info.memory) {
          HandleMemoryEffects(cnode, update_state);
        } else if (info.load) {
          // If no memory side effects, handle load if need.
          HandleLoad(cnode, update_state);
        }
        // Handle cnode with IO side effects.
        if (info.io) {
          HandleIoEffects(cnode, update_state);
        }
        // If the node has no side effects but 'no_eliminate' flag is set,
        // we save it to no_eliminate_nodes and handle them late.
        if (!info.memory && !info.io && IsNoEliminateNode(cnode)) {
          (void)no_eliminate_nodes_.emplace_back(cnode);
        }
      }
      cnode->SetEffectHandled(true);
    }
    // Attach no eliminate nodes to output.
    HandleNoEliminateNodes();
    // Attach monad to output if required.
    if (update_state) {
      AttachMonadToOutput();
    }
  }

  // Return true if the given cnode is primitive cnode with 'no_eliminate' flag.
  bool IsNoEliminateNode(const CNodePtr &cnode) const {
    if (cnode == nullptr || cnode->size() == 0) {
      return false;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      return false;
    }
    return GetPrimitiveFlag(prim, ATTR_NO_ELIMINATE);
  }

  // Attach no eliminate nodes to output.
  void HandleNoEliminateNodes() {
    if (no_eliminate_nodes_.empty()) {
      // Skip if no nodes to be handled.
      return;
    }
    // If only one node, attach it to output directly.
    if (no_eliminate_nodes_.size() == 1) {
      AttachToOutput(no_eliminate_nodes_.front());
      return;
    }
    // For multiple nodes, attach them to output by a tuple.
    std::vector<AnfNodePtr> tuple_inputs;
    AbstractBasePtrList element_abstracts;
    tuple_inputs.reserve(no_eliminate_nodes_.size() + 1);
    element_abstracts.reserve(no_eliminate_nodes_.size());
    (void)tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (auto &node : no_eliminate_nodes_) {
      (void)tuple_inputs.emplace_back(node);
      (void)element_abstracts.emplace_back(node->abstract());
    }
    auto make_tuple_node = func_graph_->NewCNode(tuple_inputs);
    make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(element_abstracts));
    AttachToOutput(make_tuple_node);
  }

  // Clean no side effect dependency nodes.
  //   From:  output = Depend(output, StopGrad)
  //          return output
  //
  //   To:    return output
  void ClearIsolatedNodes() const {
    auto output = GetGraphOutput();
    constexpr size_t attach_index = 2;
    if (IsPrimitiveCNode(output, prim::kPrimDepend) &&
        IsPrimitiveCNode(output->cast<CNodePtr>()->input(attach_index), prim::kPrimStopGradient)) {
      // Replace Depend(orig_output, StopGrad) node with orig_output.
      // After that, nodes may be eliminated if have no side effects.
      auto &orig_output = output->cast<CNodePtr>()->input(1);
      func_graph_->set_output(orig_output);
    }
  }

  void HandleOuterNode(const CNodePtr &cnode, const EffectInfo &info) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (info.memory || info.load) {
      (void)GetUniverse();
      bool load_with_primitive = (info.load && IsPrimitiveCNode(cnode));
      if (!cnode->IsEffectHandled() && !load_with_primitive) {
        auto u_node = NewValueNode(kUMonad);
        u_node->set_abstract(kUMonad->ToAbstract());
        cnode->add_input(u_node);
      }
    }
    if (info.io) {
      (void)GetIoState();
      if (!cnode->IsEffectHandled()) {
        auto io = NewValueNode(kIOMonad);
        io->set_abstract(kIOMonad->ToAbstract());
        cnode->add_input(io);
      }
    }
  }

  //
  // Convert cnode with memory side effect to monad form,
  // from:
  //    output = func(input)
  // to:
  //    output = func(input, u)
  //    u = UpdateState(u, output) # if update_state is true
  //
  void HandleMemoryEffects(const CNodePtr &cnode, bool update_state) {
    const auto &u = GetUniverse();
    AddMonadInput(cnode, u);
    if (update_state) {
      u_ = UpdateState(u, cnode);
    }
  }

  //
  // Convert cnode with io side effect to monad form,
  // from:
  //    output = func(input)
  // to:
  //    output = func(input, io)
  //    io = UpdateState(io, output) # if update_state is true
  //
  void HandleIoEffects(const CNodePtr &cnode, bool update_state) {
    const auto &io = GetIoState();
    AddMonadInput(cnode, io);
    if (update_state) {
      io_ = UpdateState(io, cnode);
    }
  }

  void HandleLoad(const CNodePtr &cnode, bool update_state) {
    MS_EXCEPTION_IF_NULL(cnode);
    // Check if a sequence which has ref exists in the inputs of the cnode, and the cnode is a real node.
    if (IsNonEffectRealNodeAndInputIsDynamic(cnode)) {
      return InsertLoadForSequenceRef(cnode, update_state);
    }
    if (IsValueNode<Primitive>(cnode->input(0))) {
      // For primitive calls that use Ref as input, insert Loads before them.
      InsertLoads(cnode, update_state);
    } else {
      // For non-primitive calls, load is used inside the callee,
      // We do not insert load for it but handle it as a side
      // effects cnode.
      HandleMemoryEffects(cnode, update_state);
    }
  }

  AnfNodePtr NewItemNode(const AnfNodePtr &node, const AbstractBasePtr &seq_abs, const AbstractBasePtr &item_abs,
                         size_t index) {
    std::vector<AnfNodePtr> item_inputs;
    if (seq_abs->isa<abstract::AbstractTuple>()) {
      (void)item_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
    } else if (seq_abs->isa<abstract::AbstractList>()) {
      (void)item_inputs.emplace_back(NewValueNode(prim::kPrimListGetItem));
    }
    (void)item_inputs.emplace_back(node);
    (void)item_inputs.emplace_back(NewValueNode(SizeToLong(index)));
    auto new_item = func_graph_->NewCNode(std::move(item_inputs));
    new_item->set_abstract(item_abs);
    if (item_abs->isa<abstract::AbstractRefTensor>()) {
      // Current u monad.
      auto current_u = GetUniverse();
      // Make a Load for item node.
      new_item = MakeLoad(node, new_item, current_u);
    }
    return new_item;
  }

  // params = (param1, param2, ..., value)
  // addn(params, xxx)  non-effect-node need insert load for params.
  void InsertLoadForSequenceRef(const CNodePtr &cnode, bool update_state) {
    const auto &inputs = cnode->inputs();
    abstract::AbstractBasePtrList new_seq_abstracts;
    for (size_t index = 1; index < inputs.size(); ++index) {
      const auto &input = inputs[index];
      const auto &input_abs = input->abstract();
      MS_EXCEPTION_IF_NULL(input_abs);
      if (!input_abs->isa<abstract::AbstractTuple>() && !input_abs->isa<abstract::AbstractList>()) {
        (void)new_seq_abstracts.emplace_back(input_abs);
        continue;
      }
      // Handle the input which is sequence.
      std::vector<AnfNodePtr> new_sequence_inputs;
      if (input_abs->isa<abstract::AbstractTuple>()) {
        (void)new_sequence_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
      } else if (input_abs->isa<abstract::AbstractList>()) {
        (void)new_sequence_inputs.emplace_back(NewValueNode(prim::kPrimMakeList));
      }
      auto seq_abs = input_abs->cast_ptr<abstract::AbstractSequence>();
      MS_EXCEPTION_IF_NULL(seq_abs);
      const auto &elements = seq_abs->elements();
      for (size_t item_index = 0; item_index < elements.size(); ++item_index) {
        const auto &item_abs = elements[item_index];
        auto item = NewItemNode(input, input_abs, item_abs, item_index);
        (void)new_sequence_inputs.emplace_back(item);
        (void)new_seq_abstracts.emplace_back(item->abstract());
      }
      auto new_seq = func_graph_->NewCNode(std::move(new_sequence_inputs));
      MS_LOG(DEBUG) << "Replace the input of non-effect-node:" << cnode->DebugString()
                    << " with:" << new_seq->DebugString();
      if (input_abs->isa<abstract::AbstractTuple>()) {
        new_seq->set_abstract(std::make_shared<abstract::AbstractTuple>(new_seq_abstracts));
      } else if (input_abs->isa<abstract::AbstractList>()) {
        new_seq->set_abstract(std::make_shared<abstract::AbstractList>(new_seq_abstracts));
      }
      manager_->SetEdge(cnode, SizeToInt(index), new_seq);
      if (update_state) {
        auto current_u = GetUniverse();
        // In the order_enforce phase, the cnode will be added to the updatestate to ensure the order,
        // and the input of the updatestate is maintained here to 2.
        // to ensure the verification of the updatestate in the relevant pass.
        u_ = UpdateState(current_u, new_seq);
      }
    }
  }

  //
  // Insert Loads for a primitive cnode that use Ref as input.
  // for example, from:
  //    out = Prim(self.para1, self.para2, other_args)
  // to:
  //    p1 = Load(self.para1, u)
  //    p2 = Load(self.para2, u)
  //    t = make_tuple(p1, p2) # if update_state
  //    u1 = UpdateState(u, t)   # is required
  //    out = Prim(p1, p2, other_args)
  //
  void InsertLoads(const CNodePtr &cnode, bool update_state) {
    // Find ref inputs.
    auto ref_inputs = GetRefInputs(cnode);
    if (ref_inputs.empty()) {
      MS_LOG(WARNING) << "Ref input not found for load insertion: " << cnode->DebugString();
      return;
    }
    // Current u monad.
    auto current_u = GetUniverse();
    // Create Load cnodes.
    auto loads = MakeLoads(cnode, ref_inputs, current_u);
    if (loads.empty() || !update_state) {
      // Skip UpdateState insertion.
      return;
    }
    // Insert UpdateState if required.
    if (loads.size() == 1) {
      // One Load, no make_tuple needed.
      u_ = UpdateState(current_u, loads.front());
      return;
    }
    // Multiple Loads, Create a MakeTuple before UpdateState.
    abstract::AbstractBasePtrList load_abstracts;
    std::transform(loads.begin(), loads.end(), std::back_inserter(load_abstracts),
                   [](const AnfNodePtr &load) { return load->abstract(); });
    loads.insert(loads.begin(), NewValueNode(prim::kPrimMakeTuple));
    auto make_tuple = func_graph_->NewCNode(loads);
    make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(load_abstracts));
    u_ = UpdateState(current_u, make_tuple);
  }

  std::vector<AnfNodePtr> MakeLoads(const CNodePtr &cnode, const RefInputs &ref_inputs, const AnfNodePtr &u) {
    std::vector<AnfNodePtr> loads;
    for (auto &ref_input : ref_inputs) {
      // Make a Load cnode for ref input.
      auto &ref = ref_input.first;
      auto load = MakeLoad(cnode, ref, u);
      // Replace input with the load cnode.
      for (size_t index : ref_input.second) {
        manager_->SetEdge(cnode, SizeToInt(index), load);
      }
      (void)loads.emplace_back(std::move(load));
    }
    return loads;
  }

  CNodePtr MakeLoad(const AnfNodePtr &node, const AnfNodePtr &ref, const AnfNodePtr &u) {
    static const std::string primitive_target = "primitive_target";
    // Create Load cnode.
    auto load_prim = NewValueNode(prim::kPrimLoad);
    auto load_cnode = func_graph_->NewCNode({load_prim, ref, u});
    // Set device target for Load CNode.
    std::string target = GetCNodeTarget(node);
    load_cnode->set_user_data(primitive_target, std::make_shared<std::string>(target));
    // Set load_cnode abstract to Tensor according the input Ref[Tensor].
    auto ref_abs = dyn_cast<abstract::AbstractRefTensor>(ref->abstract());
    MS_EXCEPTION_IF_NULL(ref_abs);
    load_cnode->set_abstract(ref_abs->CloneAsTensor());
    return load_cnode;
  }

  // Add or replace monad input.
  void AddMonadInput(const CNodePtr &cnode, const AnfNodePtr &monad) {
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr size_t max_monad_inputs = 2;
    auto monad_abs = monad->abstract();
    auto &inputs = cnode->inputs();
    int last = static_cast<int>(inputs.size()) - 1;
    int stop = last - max_monad_inputs;
    // Search monad in inputs, replace it if found.
    for (int i = last; i > 0 && i > stop; --i) {
      size_t index = static_cast<size_t>(i);
      auto input_abs = inputs[index]->abstract();
      if (input_abs && *input_abs == *monad_abs) {
        manager_->SetEdge(cnode, i, monad);
        return;
      }
    }
    // If monad not found in inputs, add a monad input.
    manager_->AddEdge(cnode, monad);
  }

  void AttachMonadToOutput() const {
    if (u_) {
      AttachToOutput(u_);
    }
    if (io_) {
      AttachToOutput(io_);
    }
  }

  void AttachToOutput(const AnfNodePtr &node) const {
    auto output = GetGraphOutput();
    TraceGuard guard(std::make_shared<TraceCopy>(output->debug_info()));
    auto depend = NewValueNode(prim::kPrimDepend);
    // If isolated nodes dependencies exist.
    if (IsPrimitiveCNode(output, prim::kPrimDepend) &&
        IsPrimitiveCNode(output->cast<CNodePtr>()->input(kDependAttachNodeIndex), prim::kPrimStopGradient)) {
      // Insert new Depend node before isolated Depend node.
      auto isolated_depend = output->cast<CNodePtr>();
      auto &orig_output = isolated_depend->input(1);
      auto state_depend = func_graph_->NewCNode({depend, orig_output, node});
      state_depend->set_abstract(orig_output->abstract());
      manager_->SetEdge(isolated_depend, 1, state_depend);
      return;
    }
    // Insert Depend node and set it as output, if no isolated nodes.
    auto depend_cnode = func_graph_->NewCNode({depend, output, node});
    depend_cnode->set_abstract(output->abstract());
    func_graph_->set_output(depend_cnode);
  }

  AnfNodePtr GetGraphOutput() const {
    auto output = func_graph_->output();
    if (output != nullptr) {
      return output;
    }
    return NewValueNode(kNone);
  }

  AnfNodePtr UpdateState(const AnfNodePtr &state, const AnfNodePtr &attach) {
    MS_EXCEPTION_IF_NULL(attach);
    // Not attach UpdateState if set kAttrIgnoreSideEffect.
    auto attr_ignore_side_effect = attach->cast<CNodePtr>()->GetAttr(kAttrIgnoreSideEffect);
    auto ignore_side_effect = attr_ignore_side_effect != nullptr && attr_ignore_side_effect->isa<BoolImm>() &&
                              GetValue<bool>(attr_ignore_side_effect);
    if (ignore_side_effect) {
      return state;
    }

    auto update_state = NewValueNode(prim::kPrimUpdateState);
    auto update_state_cnode = func_graph_->NewCNode({update_state, state, attach});
    update_state_cnode->set_abstract(state->abstract());
    return update_state_cnode;
  }

  AnfNodePtr &GetUniverse() {
    if (u_ == nullptr) {
      if (top_) {
        u_ = NewValueNode(kUMonad);
        u_->set_abstract(kUMonad->ToAbstract());
      } else {
        u_ = AddMonadParameter(func_graph_, "u", kUMonad->ToAbstract());
      }
    }
    return u_;
  }

  AnfNodePtr &GetIoState() {
    if (io_ == nullptr) {
      if (top_) {
        io_ = NewValueNode(kIOMonad);
        io_->set_abstract(kIOMonad->ToAbstract());
      } else {
        io_ = AddMonadParameter(func_graph_, "io", kIOMonad->ToAbstract());
      }
    }
    return io_;
  }

  // Return true if update_state should be used in this func graph.
  // In some case, update_state can be omitted, such as:
  //   def side_effect_tail_call(args):
  //       a = pure_func(args)
  //       return side_effect_call(a)
  bool NeedUpdateState() const {
    // Search for the only one side effect cnode.
    CNodePtr side_effect_cnode = nullptr;
    for (auto &cnode : func_graph_->order_list()) {
      if (HasSideEffect(cnode)) {
        if (side_effect_cnode != nullptr) {
          // There are multiple side effect cnodes, update state is required.
          return true;
        }
        side_effect_cnode = cnode;
      }
    }
    if (side_effect_cnode == nullptr) {
      // No side effect cnode, no update state.
      return false;
    }
    if (IsPrimitiveCNode(side_effect_cnode)) {
      // Always add update_state for primitive cnode.
      return true;
    }
    // If the only side effect cnode is not the tail call, update_state is required.
    return func_graph_->output() != side_effect_cnode;
  }

  bool HasSideEffect(const CNodePtr &cnode) const {
    const auto &cnode_info = GetEffectInfo(cnode);
    return (cnode_info.memory || cnode_info.load || cnode_info.io);
  }

  // The func graph to be converted.
  const FuncGraphPtr &func_graph_;

  // The func graph manager, used for graph edge update.
  FuncGraphManagerPtr manager_;

  // True if converting top graph.
  const bool top_;

  // True if there are side effect cnodes within this func graph.
  bool has_effect_cnodes_ = false;

  // CNodes that should not be eliminated even it is isolated node.
  std::vector<CNodePtr> no_eliminate_nodes_;

  // Current memory state node, null if no memory side effects.
  AnfNodePtr u_;

  // Current IO state node, null if no IO side effects.
  AnfNodePtr io_;
};  // class AutoMonadConverter
}  // namespace

// Entry point of the auto-monad phase,
// the func_graph should be resolved and infer is done.
// return true if side effect nodes found in func_graph.
bool AutoMonad(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->manager());

  // Search and mark side effects for the graph and sub-graphs.
  // this should be called before auto-monad starts.
  SideEffectFinder::Search(func_graph);

  // Execute auto-monad conversion on top graph.
  bool has_effects = AutoMonadConverter::Handle(func_graph, true);
  // Convert used sub-graphs.
  auto fg_used_total = func_graph->func_graphs_used_total();
  for (auto &fg : fg_used_total) {
    auto top_flag = fg->has_flag(mindspore::kFuncGraphFlagBackPropEntry);
    if (fg->stage() != -1) {
      top_flag = true;
    }
    bool fg_has_effects = AutoMonadConverter::Handle(fg, top_flag);
    has_effects = has_effects || fg_has_effects;
  }
  return has_effects;
}

bool ReAutoMonad(const FuncGraphPtr &func_graph) {
  // AutoMonad for bprop network, only Monad for func graphs which back propogators have side effects.
  // Or AutoMonad for MultitypeFuncGraph which specialized in Renormalize other than the first Specialize pass.
  MS_EXCEPTION_IF_NULL(func_graph);
  bool need_auto_monad = false;
  std::vector<FuncGraphPtr> auto_monaded_fg;
  func_graph->EraseUnusedNodeInOrder();
  for (auto &fg : func_graph->func_graphs_used_total()) {
    if (fg->has_flag(mindspore::kFuncGraphFlagReAutoMonad)) {
      auto_monaded_fg.push_back(fg);
      for (auto &used_fg : fg->func_graphs_used_total()) {
        used_fg->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
        auto_monaded_fg.push_back(used_fg);
      }
      need_auto_monad = true;
      MS_LOG(DEBUG) << "AutoMonad Grad for func graph: " << fg->ToString();
    }
    fg->EraseUnusedNodeInOrder();
  }
  bool changed = false;
  if (need_auto_monad) {
    for (auto &fg : func_graph->func_graphs_used_total()) {
      if (!fg->has_flag(mindspore::kFuncGraphFlagReAutoMonad)) {
        fg->ClearOrderList();
      }
    }
    changed = AutoMonad(func_graph);
    for (auto &fg : auto_monaded_fg) {
      fg->erase_flag(mindspore::kFuncGraphFlagReAutoMonad);
    }
    // After auto monad, Order List and Isolate nodes in graph and manager will be cleared.
  } else {
    func_graph->ClearOrderList();
    for (auto &fg : func_graph->func_graphs_used_total()) {
      fg->ClearOrderList();
    }
  }
  return changed;
}
}  // namespace pipeline
}  // namespace mindspore
