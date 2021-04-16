/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <set>
#include <map>
#include <list>
#include <unordered_map>
#include <vector>
#include <stack>
#include <utility>
#include <algorithm>
#include "pipeline/jit/parse/resolve.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/multitype_funcgraph.h"
#include "utils/flags.h"
#include "utils/utils.h"
#include "utils/ordered_map.h"
#include "base/core_ops.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace pipeline {
namespace {  // namespace anonymous

using ClassTypePtr = std::shared_ptr<parse::ClassType>;
using RefInputs = OrderedMap<AnfNodePtr, std::vector<size_t>>;

// Add or get a monad parameter.
AnfNodePtr AddMonadParameter(const FuncGraphPtr &func_graph, const std::string &name,
                             const abstract::AbstractBasePtr &abs) {
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
    params.insert(params.begin() + io_monad_location, para);
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

// Return true if the node has Ref abstract.
bool HasAbstractRef(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto &abs = node->abstract();
  return (abs != nullptr) && abs->isa<abstract::AbstractRef>();
}

// Gets ref inputs and its indexes from a cnode.
RefInputs GetRefInputs(const CNodePtr &cnode) {
  RefInputs ref_inputs;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto &input = cnode->inputs().at(i);
    if (HasAbstractRef(input)) {
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
  return std::any_of(inputs.begin() + 1, inputs.end(), [](const auto &input) { return HasAbstractRef(input); });
}

// Return true if we don't need Load for the given primitive.
// i.e. keep Ref as Ref for some primitives.
bool IsKeepRef(const PrimitivePtr &prim) {
  return (GetSideEffectPropagate(prim) != 0) || IsPrimitiveEquals(prim, prim::kPrimRefToEmbed) ||
         IsPrimitiveEquals(prim, prim::kPrimPull);
}

// Gets primitive if the node is a primitive value node.
PrimitivePtr GetPrimitive(const AnfNodePtr &node) {
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
  auto do_sig = dyn_cast<mindspore::prim::DoSignaturePrimitive>(prim);
  if (do_sig) {
    auto val = do_sig->function();
    return dyn_cast<Primitive>(val);
  }
  return prim;
}

// Gets primitive from the given cnode, return nullptr if cnode.inputs[0] is not a primitive.
PrimitivePtr GetPrimitive(const CNodePtr &cnode) {
  if (cnode == nullptr || cnode->inputs().empty()) {
    return nullptr;
  }
  return GetPrimitive(cnode->input(0));
}

// Gets func_graph from the given cnode, return nullptr if it is not a func graph call.
FuncGraphPtr GetFuncGraph(const CNodePtr &cnode) {
  if (cnode != nullptr && !cnode->inputs().empty()) {
    return GetValueNode<FuncGraphPtr>(cnode->input(0));
  }
  return nullptr;
}

// Gets class_type from the given cnode->inputs[0].
ClassTypePtr GetClassType(const CNodePtr &cnode) {
  if (cnode && !cnode->inputs().empty()) {
    auto apply = cnode->input(0);
    auto apply_cnode = dyn_cast<CNode>(apply);
    if (apply_cnode && !apply_cnode->inputs().empty()) {
      return GetValueNode<ClassTypePtr>(apply_cnode->input(0));
    }
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

// --------------------------------------------------------------------
// SCC (Strongly Connected Components) related types.
// --------------------------------------------------------------------
using SccVector = std::set<FuncGraphPtr>;
using SccPtr = std::shared_ptr<SccVector>;
using SccMap = std::unordered_map<FuncGraphPtr, SccPtr>;

// ---------------------------------------------------------------------
// SccFinder find SCCs using Tarjan's algorithm.
// ---------------------------------------------------------------------
class SccFinder {
 public:
  explicit SccFinder(FuncGraphPtr root) : root_(root) {}
  ~SccFinder() = default;
  void Run() { (void)Search(root_); }
  const SccMap &scc_map() { return scc_map_; }

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
  const State &Search(FuncGraphPtr graph) {
    // Create graph state, set it as visited.
    auto [inserted, ok] = visited_.emplace(graph, State(index_++));
    if (!ok) {
      MS_LOG(EXCEPTION) << "Already visited: " << graph->ToString();
    }
    auto &state = inserted->second;
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
      } else if (iter->second.in_stack) {
        // Successor graph is in stack and hence in the current SCC.
        state.lowlink = std::min(state.lowlink, iter->second.index);
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
        found->second.in_stack = false;
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

 private:
  // The root graph.
  FuncGraphPtr root_;

  // Current index by DFS order.
  size_t index_ = 1;

  // Visited graphs and their states.
  std::unordered_map<FuncGraphPtr, State> visited_;

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
    // Check switch layer calls, add monad arguments if need.
    HandleSwitchLayerCalls();
  }

  void UpdateOrderLists() {
    // Some cnodes used in current func graph but belong to other func graph, we have to
    // insert them into order list so that we can handle side effects for them.
    UpdateOrderList(root_);
    for (auto &fg : root_->func_graphs_used_total()) {
      UpdateOrderList(fg);
    }
  }

  static void UpdateOrderList(const FuncGraphPtr &func_graph) {
    OrderedSet<CNodePtr> new_order_list;
    const auto &order_list = func_graph->order_list();
    for (auto &cnode : order_list) {
      PushToOrderList(func_graph, cnode, &new_order_list);
    }
    func_graph->set_order_list(std::move(new_order_list));
  }

  static void PushToOrderList(const FuncGraphPtr &fg, const CNodePtr &cnode, OrderedSet<CNodePtr> *new_order_list) {
    MS_EXCEPTION_IF_NULL(cnode);
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
  FuncGraphPtr GetSwitchBranch(const CNodePtr &cnode, size_t index) {
    return GetValueNode<FuncGraphPtr>(cnode->inputs().at(index));
  }

  // Gets branch graphs from a switch cnode.
  std::vector<FuncGraphPtr> GetSwitchBranches(const CNodePtr &cnode) {
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
      branches.emplace_back(true_branch);
    }
    auto false_branch = GetSwitchBranch(cnode, false_index);
    if (false_branch != nullptr) {
      branches.emplace_back(false_branch);
    }
    if (branches.empty()) {
      MS_LOG(EXCEPTION) << "Invalid switch: " << cnode->DebugString();
    }
    return branches;
  }

  // Add monad parameter to switch branch graphs.
  void AddMonadParameters(const std::vector<FuncGraphPtr> &branches, const std::string &name,
                          const AbstractBasePtr &abs) {
    for (auto &branch : branches) {
      (void)AddMonadParameter(branch, name, abs);
    }
  }

  // Trace effect info for Switch cnode.
  EffectInfo TraceSwitchEffectInfo(const CNodePtr &cnode) {
    // Find branches from switch cnode.
    auto branches = GetSwitchBranches(cnode);
    // For some case, only one branch is set.
    if (branches.size() == 1) {
      auto &branch = branches.front();
      // Save branch caller, so that we can update arguments for the caller.
      SaveBranchCaller(cnode, branch);
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
      auto &call = switch_layer_calls.emplace_back();
      call.caller = caller_;
      call.effect_info = info;
      call.branches = move(branches);
    }
    return info;
  }

  void HandleSwitchLayerCalls() {
    for (auto &call : switch_layer_calls) {
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
    constexpr size_t func_tuple_index = 2;
    if (cnode->size() <= func_tuple_index) {
      MS_LOG(EXCEPTION) << "Invalid switch_layer: " << cnode->DebugString(2);
    }
    auto func_tuple = cnode->inputs().at(func_tuple_index);
    return GetGraphsFromTuple(func_tuple);
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
    MS_LOG(EXCEPTION) << "Invalid input for switch_layer: " << func_tuple->DebugString(2);
  }

  // Get graphs from a tuple of funcs make node for switch_layer.
  std::vector<FuncGraphPtr> GetGraphsFromMakeTuple(const CNodePtr &make_tuple) {
    auto &inputs = make_tuple->inputs();
    if (inputs.size() <= 1) {
      MS_LOG(EXCEPTION) << "Invalid make_tuple for switch_layer: " << make_tuple->DebugString(2);
    }
    std::vector<FuncGraphPtr> graphs;
    graphs.reserve(inputs.size() - 1);
    for (size_t i = 1; i < inputs.size(); ++i) {
      auto func_graph = GetValueNode<FuncGraphPtr>(inputs.at(i));
      if (func_graph == nullptr) {
        MS_LOG(WARNING) << "Non-graph found in switch_layer input: " << make_tuple->DebugString(2) << " index=" << i;
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
    auto para = dyn_cast<Parameter>(tuple_node);
    if (para != nullptr) {
      return TraceTupleParaEffectInfo(para, *tuple_indexes);
    }
    auto tuple_cnode = dyn_cast<CNode>(tuple_node);
    if (tuple_cnode != nullptr) {
      return TraceTupleCNodeEffectInfo(tuple_cnode, tuple_indexes);
    }
    // Should not reach here.
    MS_LOG(EXCEPTION) << "Side effects untraceable: " << tuple_node->DebugString();
  }

  EffectInfo TraceTupleParaEffectInfo(const ParameterPtr &para, const std::stack<int64_t> &tuple_indexes) {
    EffectInfo info{EffectInfo::kDetected, false, false, false};
    ForEachRealArguments(para, [this, &info, tuple_indexes](const AnfNodePtr &arg) {
      // Merge real argument effect info.
      auto tuple_indexes_copy = tuple_indexes;
      auto arg_info = TraceTupleEffectInfo(arg, &tuple_indexes_copy);
      info.Merge(arg_info);
    });
    return info;
  }

  EffectInfo TraceTupleCNodeEffectInfo(const CNodePtr &cnode, std::stack<int64_t> *tuple_indexes) {
    auto prim = GetPrimitive(cnode);
    // Trace MakeTuple.
    if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple)) {
      if (tuple_indexes->empty()) {
        MS_LOG(EXCEPTION) << "Unexpected make_tuple: " << cnode->DebugString(2);
        return {EffectInfo::kDetected, false, false, false};
      }
      // Pop out tuple index.
      auto index = tuple_indexes->top();
      tuple_indexes->pop();
      // Follow the tuple item according the index.
      size_t input_index = static_cast<size_t>(index) + 1;
      if (input_index >= cnode->size()) {
        MS_LOG(EXCEPTION) << "Invalid make_tuple: " << cnode->DebugString() << " index=" << index;
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
    // Trace primitive propagating side effect from its input, such as Depend, Identity, etc.
    int input_index = GetSideEffectPropagate(prim);
    if (input_index > 0 && input_index < static_cast<int>(cnode->size())) {
      return TraceTupleEffectInfo(cnode->input(static_cast<size_t>(input_index)), tuple_indexes);
    }
    // Tuple returned from func graph call.
    auto func_graph = GetFuncGraph(cnode);
    if (func_graph != nullptr) {
      return TraceTupleEffectInfo(func_graph->output(), tuple_indexes);
    }
    // Tuple is returned from J().
    //   %1 = J(primal)
    //   tuple = %1(args)
    if (cnode->size() > 0 && IsPrimitiveCNode(cnode->input(0), prim::kPrimJ)) {
      MS_LOG(DEBUG) << "Tuple from J: " << cnode->DebugString(2);
      return {EffectInfo::kDetected, false, false, false};
    }
    // Rare case.
    MS_LOG(WARNING) << "Tuple untraceable from: " << cnode->DebugString(2);
    return {EffectInfo::kDetected, false, false, false};
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
      branch->SetEffectInfo(info);
      // Update caller if it is existed.
      UpdateBranchCaller(branch);
    }
  }

  // Merge effect info for switch or switch_layer branch graphs.
  EffectInfo MergeEffectInfo(const std::vector<FuncGraphPtr> &branches) {
    EffectInfo info = {EffectInfo::kDetected, false, false, false};
    for (auto &branch : branches) {
      EffectInfo branch_info = GetEffectInfo(branch);
      info.Merge(branch_info);
    }
    return info;
  }

  // Trace a cnode for effect info.
  EffectInfo TraceEffectInfo(const CNodePtr &cnode) {
    auto prim = GetPrimitive(cnode);
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

    // For high-order pritimive such as Partial,
    // we trace effect info from its argument.
    int index_prim = GetSideEffectPropagate(prim);
    if (index_prim > 0 && index_prim < static_cast<int>(cnode->size())) {
      return TraceEffectInfo(cnode->input(static_cast<size_t>(index_prim)));
    }

    // For func graph calls, we trace effect info from graph output.
    auto called_graph = GetFuncGraph(cnode);
    if (called_graph) {
      return TraceEffectInfo(called_graph->output());
    }

    //
    // For ClassType as the input[0], if it is a primitive class
    // with 'side_effect_propagate' attribute, we trace side effect
    // from its argument indxed by the attribute value.
    //
    // e.g.:
    //     setpara = P.Partial()(P.Assign, self.para)
    //     setpara(x)
    //
    auto class_type = GetClassType(cnode);
    if (class_type) {
      int index = GetSideEffectPropagate(class_type);
      if (index > 0 && index < static_cast<int>(cnode->size())) {
        return TraceEffectInfo(cnode->input(static_cast<size_t>(index)));
      }
    }

    // Otherwise, no side effect found and stop trace.
    return {EffectInfo::kDetected, false, false, false};
  }

  // Trace an ANFNode for effect info.
  EffectInfo TraceEffectInfo(const AnfNodePtr &node) {
    if (node) {
      // Trace cnode.
      auto cnode = node->cast<CNodePtr>();
      if (cnode) {
        return TraceEffectInfo(cnode);
      }

      // Trace parameter.
      auto para = node->cast<ParameterPtr>();
      if (para) {
        return TraceEffectInfo(para);
      }

      // Trace primitive.
      auto prim = GetPrimitive(node);
      if (prim) {
        return GetPrimEffectInfo(prim);
      }

      // Trace func graph.
      auto value_node = node->cast<ValueNodePtr>();
      if (value_node && value_node->value()) {
        auto graph = value_node->value()->cast<FuncGraphPtr>();
        if (graph) {
          return GetEffectInfo(graph);
        }
      }
    }
    // Something is wrong if we reached here.
    MS_LOG(WARNING) << "EffectInfo untraceable: " << node->DebugString(2);
    return {EffectInfo::kDetected, false, false, false};
  }

  int GetParameterIndex(const FuncGraphPtr &func_graph, const ParameterPtr &para) {
    int index = 0;
    for (auto &parameter : func_graph->parameters()) {
      if (para == parameter) {
        return index;
      }
      ++index;
    }
    MS_LOG(EXCEPTION) << "Parameter not found: " << (para ? para->DebugString() : "<null>");
  }

  // Trace effect info from function parameter.
  EffectInfo TraceEffectInfo(const ParameterPtr &para) {
    EffectInfo info{EffectInfo::kDetected, false, false, false};
    ForEachRealArguments(para, [this, &info](const AnfNodePtr &arg) {
      // Merge caller input effect info.
      auto input_info = TraceEffectInfo(arg);
      info.Merge(input_info);
    });
    return info;
  }

  void ForEachRealArguments(const ParameterPtr &para, std::function<void(const AnfNodePtr &)> handler) {
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
    for (auto &user : users) {
      auto use_index = user.first->second;
      if (use_index != 0) {
        // Skip non-caller usage.
        continue;
      }
      // Caller cnode.
      auto cnode = dyn_cast<CNode>(user.first->first);
      if (cnode && input_index < cnode->size()) {
        handler(cnode->input(input_index));
      }
    }
  }

  // For call node, returns effect info of the callee graph.
  EffectInfo GetCallEffectInfo(const CNodePtr &cnode) {
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
    auto prim = GetPrimitive(cnode);
    if (prim) {
      // Skip 'return' cnode.
      if (IsPrimitiveEquals(prim, prim::kPrimReturn)) {
        return {EffectInfo::kDetected, false, false, false};
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
      return info;
    }

    // For func graph, detect effect info by its children cnodes.
    auto func_graph = GetFuncGraph(cnode);
    if (func_graph) {
      return GetEffectInfo(func_graph);
    }

    // When input[0] is a cnode, it is a function returned from
    // a high-order function call, we trace it by return value.
    auto func_cnode = GetFuncCNode(cnode);
    if (func_cnode) {
      caller_ = cnode;
      return TraceEffectInfo(func_cnode);
    }

    // When input[0] is a parameter, it is a function parameter for
    // the high-order function, we trace it by caller.
    auto func_para = GetFuncParameter(cnode);
    if (func_para) {
      return TraceEffectInfo(func_para);
    }

    // When input[0] is a MultitypeFuncGraph, it's not specialized
    // as one of its parameters is AbstractUndertermined,
    // This MultitypeFuncGraph may be specialized at next Renormalize
    // process, but we have to keep the order by insert UMonad now,
    // otherwise order will be lost in next Renormalize.
    // So assume it has memory side effect conservatively.
    auto func_multitype = GetFuncMultitypeFuncGraph(cnode);
    if (func_multitype) {
      MS_LOG(DEBUG) << "Assume memory side effect for: " << cnode->DebugString();
      return {EffectInfo::kDetected, true, false, false};
    }

    MS_LOG(WARNING) << "Side effect undetectable: " << cnode->DebugString(2);
    return {EffectInfo::kDetected, false, false, false};
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
  const SccPtr &GetScc(const FuncGraphPtr &func_graph) const {
    auto found = scc_map_.find(func_graph);
    if (found == scc_map_.end()) {
      MS_LOG(EXCEPTION) << "SCC not found for " << func_graph->ToString() << "." << func_graph->debug_info()->get_id();
    }
    return found->second;
  }

  // Set effect info for all member graphs in the SCC.
  void SetSccEffectInfo(const SccPtr &scc, const EffectInfo &info) {
    for (auto &g : *scc) {
      g->SetEffectInfo(info);
    }
  }

  // Gets EffectInfo for func graph.
  EffectInfo GetEffectInfo(const FuncGraphPtr &func_graph) {
    const auto &effect_info = func_graph->GetEffectInfo();
    if (effect_info.state != EffectInfo::kUnknown) {
      // Effect info already set, return it.
      return effect_info;
    }
    // Get SCC that this graph belongs to.
    auto &scc = GetScc(func_graph);
    // To prevent SCC members be visited again, we set effect info
    // to 'kDetecting' state before start to check cnodes.
    EffectInfo info{EffectInfo::kDetecting, false, false, false};
    SetSccEffectInfo(scc, info);
    // Check side effects for all cnodes in the SCC.
    std::vector<CNodePtr> undetected;
    for (auto &g : *scc) {
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
      auto cnode_effect = GetEffectInfo(cnode);
      // Side effect should be detected now.
      if (cnode_effect.state != EffectInfo::kDetected) {
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

  void SaveBranchCaller(const CNodePtr &switch_node, const FuncGraphPtr &branch) {
    auto manager = branch->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &node_users = manager->node_users();
    auto found = node_users.find(switch_node);
    if (found == node_users.end()) {
      MS_LOG(WARNING) << "Caller not found for " << switch_node->DebugString();
      return;
    }
    if (found->second.size() != 1) {
      MS_LOG(WARNING) << "Wrong callers " << found->second.size() << " for " << switch_node->DebugString();
      return;
    }
    auto &user = *found->second.begin();
    auto cnode = dyn_cast<CNode>(user.first);
    if (cnode != nullptr || user.second == 0) {
      branch_caller_map.emplace(branch, cnode);
    }
  }

  void UpdateBranchCaller(const FuncGraphPtr &branch) {
    auto iter = branch_caller_map.find(branch);
    if (iter == branch_caller_map.end()) {
      return;
    }
    const auto &caller = iter->second;
    const auto &info = branch->GetEffectInfo();
    AddMonadForCaller(caller, info);
  }

  void AddMonadForCaller(const CNodePtr &caller, const EffectInfo &info) {
    if (info.memory || info.load) {
      // Add u monad argument to caller if need.
      AddMonadArgument(caller, kUMonad);
    }
    if (info.io) {
      // Add io monad argument to caller if need.
      AddMonadArgument(caller, kIOMonad);
    }
  }

  void AddMonadArgument(const CNodePtr &cnode, const ValuePtr &monad) {
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

 private:
  // The root graph.
  FuncGraphPtr root_;

  // SCC map.
  SccMap scc_map_;

  // Single branch (in switch) and its caller cnode.
  std::map<FuncGraphPtr, CNodePtr> branch_caller_map;

  // Current high order func caller cnode.
  CNodePtr caller_ = nullptr;

  // switch_layer_calls save all switch_layer calls, so that
  // we can check whether monad argument should be added for them.
  std::vector<SwitchLayerCall> switch_layer_calls;
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
    // Handle cnodes if graph has side effects.
    if (HasSideEffects()) {
      HandleCNodes();
    }

    // Safe to clear isolated nodes after handled side effect nodes.
    ClearIsolatedNodes();

    // Clean up after conversion finished.
    func_graph_->ClearOrderList();
    return has_effect_cnodes_;
  }

  // Check if there are side effects from effect info.
  static bool HasSideEffects(const EffectInfo &info) { return (info.memory || info.io || info.load); }

  // Check if current graph has side effects.
  bool HasSideEffects() const {
    const auto &info = func_graph_->GetEffectInfo();
    if (info.state != EffectInfo::kDetected) {
      // Effect info should have been set by SideEffectFinder, except unused graphs.
      MS_LOG(INFO) << "No effect info for unused graph: " << func_graph_->ToString();
      return false;
    }
    return HasSideEffects(info);
  }

  // Gets effect info for a cnode.
  const EffectInfo &GetEffectInfo(const CNodePtr &cnode) {
    auto &effect_info = cnode->GetEffectInfo();
    if (effect_info.state != EffectInfo::kDetected) {
      // Effect info should have been set by SideEffectFinder.
      MS_LOG(EXCEPTION) << "Side effects not detected: " << cnode->DebugString();
    }
    return effect_info;
  }

  //
  // Handle CNodes for side effects.
  //
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
      }
      cnode->SetEffectHandled(true);
    }
    // Insert Depend nodes for states if required.
    if (update_state) {
      InsertStateDepends();
    }
  }

  // Clean no side effect dependency nodes.
  //   From:  output = Depend(output, StopGrad)
  //          return output
  //
  //   To:    return output
  void ClearIsolatedNodes() {
    auto output = GetGraphOutput();
    if (IsPrimitiveCNode(output, prim::kPrimDepend) &&
        IsPrimitiveCNode(output->cast<CNodePtr>()->input(2), prim::kPrimStopGradient)) {
      // Replace Depend(orig_output, StopGrad) node with orig_output.
      // After that, nodes may be eliminated if have no side effects.
      auto &orig_output = output->cast<CNodePtr>()->input(1);
      func_graph_->set_output(orig_output);
    }
  }

  void HandleOuterNode(const CNodePtr &cnode, const EffectInfo &info) {
    if (info.memory || info.load) {
      (void)GetUniverse();
      bool load_with_primitive = (info.load && IsPrimitiveCNode(cnode));
      if (!cnode->IsEffectHandled() && !load_with_primitive) {
        auto u = NewValueNode(kUMonad);
        u->set_abstract(kUMonad->ToAbstract());
        cnode->add_input(u);
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
    auto value = GetValueNode(cnode->input(0));
    if (value && value->isa<Primitive>()) {
      // For primitive calls that use Ref as input, insert Loads before them.
      InsertLoads(cnode, update_state);
    } else {
      // For non-primitive calls, load is used inside the callee,
      // We do not insert load for it but handle it as a side
      // effects cnode.
      HandleMemoryEffects(cnode, update_state);
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
    auto u = GetUniverse();
    // Create Load cnodes.
    auto loads = MakeLoads(cnode, ref_inputs, u);
    if (loads.empty() || !update_state) {
      // Skip UpdateState insertion.
      return;
    }
    // Insert UpdateState if required.
    if (loads.size() == 1) {
      // One Load, no make_tuple needed.
      u_ = UpdateState(u, loads.front());
      return;
    }
    // Multiple Loads, Create a MakeTuple before UpdateState.
    abstract::AbstractBasePtrList load_abstracts;
    std::transform(loads.begin(), loads.end(), std::back_inserter(load_abstracts),
                   [](const AnfNodePtr &load) { return load->abstract(); });
    loads.insert(loads.begin(), NewValueNode(prim::kPrimMakeTuple));
    auto make_tuple = func_graph_->NewCNode(loads);
    make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(load_abstracts));
    u_ = UpdateState(u, make_tuple);
  }

  std::vector<AnfNodePtr> MakeLoads(const CNodePtr &cnode, const RefInputs &ref_inputs, const AnfNodePtr &u) {
    std::vector<AnfNodePtr> loads;
    for (auto &ref_input : ref_inputs) {
      // Make a Load cnode for ref input.
      auto &ref = ref_input.first;
      auto load = MakeLoad(cnode, ref, u);
      // Replace input with the load cnode.
      for (size_t index : ref_input.second) {
        manager_->SetEdge(cnode, index, load);
      }
      loads.emplace_back(std::move(load));
    }
    return loads;
  }

  CNodePtr MakeLoad(const CNodePtr &cnode, const AnfNodePtr &ref, const AnfNodePtr &u) {
    static const std::string primitive_target = "primitive_target";
    // Create Load cnode.
    auto load_prim = NewValueNode(prim::kPrimLoad);
    auto load_cnode = func_graph_->NewCNode({load_prim, ref, u});
    // Set device target for Load CNode.
    std::string target = GetCNodeTarget(cnode);
    load_cnode->set_user_data(primitive_target, std::make_shared<std::string>(target));
    // Set load_cnode abstract to Tensor according the input Ref[Tensor].
    auto ref_abs = dyn_cast<abstract::AbstractRef>(ref->abstract());
    MS_EXCEPTION_IF_NULL(ref_abs);
    load_cnode->set_abstract(ref_abs->CloneAsTensor());
    return load_cnode;
  }

  // Add or replace monad input.
  void AddMonadInput(const CNodePtr &cnode, const AnfNodePtr &monad) {
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

  void InsertStateDepends() {
    if (u_) {
      // Insert Depend node for UMonad,
      // Gradient is required for memory side effects.
      InsertStateDepend(u_);
    }
    if (io_) {
      // No gradient required for IO operations.
      InsertStateDepend(io_);
    }
  }

  void InsertStateDepend(const AnfNodePtr &state) {
    auto output = GetGraphOutput();
    auto depend = NewValueNode(prim::kPrimDepend);
    // If isolated nodes dependencies exist.
    if (IsPrimitiveCNode(output, prim::kPrimDepend) &&
        IsPrimitiveCNode(output->cast<CNodePtr>()->input(2), prim::kPrimStopGradient)) {
      // Insert state Depend node into isolated Depend node.
      auto isolated_depend = output->cast<CNodePtr>();
      auto &orig_output = isolated_depend->input(1);
      auto state_depend = func_graph_->NewCNode({depend, orig_output, state});
      state_depend->set_abstract(orig_output->abstract());
      manager_->SetEdge(isolated_depend, 1, state_depend);
      return;
    }
    // Insert Depend node and set it as output, if no isolated nodes.
    auto depend_cnode = func_graph_->NewCNode({depend, output, state});
    depend_cnode->set_abstract(output->abstract());
    func_graph_->set_output(depend_cnode);
  }

  AnfNodePtr GetGraphOutput() {
    auto output = func_graph_->output();
    if (output != nullptr) {
      return output;
    }
    return NewValueNode(kNone);
  }

  AnfNodePtr UpdateState(const AnfNodePtr &state, const AnfNodePtr &attach) {
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
  bool NeedUpdateState() {
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

  bool HasSideEffect(const CNodePtr &cnode) {
    const auto &info = GetEffectInfo(cnode);
    return (info.memory || info.load || info.io);
  }

 private:
  // The func graph to be converted.
  const FuncGraphPtr &func_graph_;

  // The func graph manager, used for graph edge update.
  FuncGraphManagerPtr manager_;

  // True if converting top graph.
  const bool top_;

  // True if there are side effect cnodes within this func graph.
  bool has_effect_cnodes_ = false;

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
