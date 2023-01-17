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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARTIAL_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARTIAL_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <set>

#include "utils/hash_map.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
const auto kMinInputSizeOfCallWithArgs = 2;
// {{prim::kPrimPartial, X, Xs}, Ys} -> {X, Xs, Ys} or {X, Ys, Xs}
class PartialEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }
    X_ = nullptr;
    Xs_.clear();
    auto &inputs = node->cast<CNodePtr>()->inputs();
    Visit(inputs[0]);

    if (Xs_.size() == 0) {
      return nullptr;
    }

    // {X, Xs, Ys}
    std::vector<AnfNodePtr> args{};
    const auto xs_size = Xs_.size();
    // Xs_ don't have monad or Ys_ is 0.
    if (!HasAbstractMonad(Xs_.back()) || inputs.empty()) {
      args.push_back(X_);
      (void)std::copy(Xs_.begin(), Xs_.end(), std::back_inserter(args));
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args));
      TraceGuard guard(std::make_shared<TracePartialTransform>(node->debug_info()));
      auto new_node = node->func_graph()->NewCNode(args);
      new_node->set_abstract(node->abstract());
      return new_node;
    }
    // {X, Ys, Xs} if Xs has monad
    if (!IsValueNode<FuncGraph>(X_)) {
      constexpr auto recursive_level = 2;
      MS_LOG(EXCEPTION) << "not support yet as X_ is not a funcgraph. node: " << node->DebugString(recursive_level);
    }
    auto fg = GetValueNode<FuncGraphPtr>(X_);
    MS_EXCEPTION_IF_NULL(fg);
    if (fg->func_graph_cnodes_index().size() != 1) {
      // If a graph is used by 2 or more partial nodes at the same time, clone the graph.
      auto new_fg = BasicClone(fg);
      auto new_fg_node = NewValueNode(new_fg);
      fg->manager()->Replace(X_, new_fg_node);
      fg = new_fg;
      X_ = new_fg_node;
    }
    args.push_back(X_);
    // Ys first;
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args));
    (void)std::copy(Xs_.begin(), Xs_.end(), std::back_inserter(args));
    TraceGuard guard(std::make_shared<TracePartialTransform>(node->debug_info()));
    auto new_node = node->func_graph()->NewCNode(args);
    new_node->set_abstract(node->abstract());

    // reorder the formal parameter of fg.
    AnfNodePtrList new_params;
    (void)std::copy(fg->parameters().cbegin() + SizeToLong(xs_size), fg->parameters().cend(),
                    std::back_inserter(new_params));
    (void)std::copy(fg->parameters().cbegin(), fg->parameters().cbegin() + SizeToLong(xs_size),
                    std::back_inserter(new_params));
    fg->manager()->SetParameters(fg, new_params);
    return new_node;
  }

  void Visit(const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPartial)) {
      return;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    // {prim::kPrimPartial, X, Xs}
    if (inputs.size() <= 1) {
      return;
    }

    X_ = inputs[1];
    // fill Xs
    // {Partial, Function, Args....}
    constexpr auto args_index = 2;
    (void)std::copy(inputs.begin() + args_index, inputs.end(), std::back_inserter(Xs_));
  }

 private:
  AnfNodePtr X_{nullptr};
  std::vector<AnfNodePtr> Xs_{};
};

class ChoicePartialEliminater : public AnfVisitor {
 public:
  virtual ~ChoicePartialEliminater() = default;

  void Visit(const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPartial)) {
      if (IsValueNode<FuncGraph>(node)) {
        fg_list_.push_back(node);
        (void)args_list_.emplace_back(AnfNodePtrList{});
      }
      return;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    // {prim::kPrimPartial, G}
    if (inputs.size() < kPartialMinInputSize) {
      MS_LOG(EXCEPTION) << "Node should be Partial CNode, but: " << node->DebugString();
    }
    if (IsValueNode<FuncGraph>(inputs[1])) {
      fg_list_.push_back(inputs[1]);
      AnfNodePtrList args;
      // {Partial, Function, Args....}
      constexpr auto args_index = 2;
      (void)std::copy(inputs.begin() + args_index, inputs.end(), std::back_inserter(args));
      args_list_.push_back(args);
    }
    return;
  }

 protected:
  AnfNodePtrList fg_list_{};
  std::vector<AnfNodePtrList> args_list_{};

  // return value: true -- continue replace; false -- return nullptr;
  bool CheckFuncGraphAndArgs() {
    // Either one should be {Partial, G, X}
    auto has_partial_args =
      std::any_of(args_list_.cbegin(), args_list_.cend(), [](auto &args) { return args.size() != 0; });
    if (!has_partial_args) {
      return false;
    }

    // check funcgraph should be used once only.
    for (size_t i = 0; i < fg_list_.size(); i++) {
      auto fg_node = fg_list_[i];
      auto fg = GetValueNode<FuncGraphPtr>(fg_node);
      MS_EXCEPTION_IF_NULL(fg);
      if (fg->func_graph_cnodes_index().size() != 1) {
        // If a graph is used by 2 or more partial nodes at the same time, clone the graph.
        // BasicClone should be replaced by TransformableClone to avoid recursive.
        auto new_fg = TransformableClone(fg);
        auto manager = fg->manager();
        MS_EXCEPTION_IF_NULL(manager);
        manager->AddFuncGraph(new_fg);
        fg_list_[i] = NewValueNode(new_fg);
      }
    }
    return true;
  }

  // Merge partial's args and call's args
  // branch1: {{primPartial, Xs}, Zs} -> {{primPartial, Xs, Zs}}
  // branch2: {{primPartial, Ys}, Zs} -> {{primPartial, Ys, Zs}}
  void MergeArgs(const CNodePtr &call_node) {
    for (auto &args : args_list_) {
      (void)args.insert(args.end(), call_node->inputs().begin() + 1, call_node->inputs().end());
    }
  }

  // f(x1, x2, x3, z1, z2 ,monad1)
  // g(x4, x2, z1, z2, monad2)
  // h(x5, x2, x7, x8, z1, z2, monad3)
  // --> union_args = (x1, x2, x3, z1, z2, x4, x5, x7 ,x8, monad1, monad2, monad3)
  // h(x1, x2, x3, z1, z2, x4, x5, x7 ,x8, monad1, monad2, monad3)
  // f(x1, x2, x3, z1, z2, x4, x5, x7 ,x8, monad1, monad2, monad3)
  // g(x1, x2, x3, z1, z2, x4, x5, x7 ,x8, monad1, monad2, monad3)
  static AnfNodePtrList UnifyParameters(const AnfNodePtrList &fg_list, const std::vector<AnfNodePtrList> args_list) {
    if (fg_list.empty()) {
      return {};
    }
    auto first_func_graph = GetValueNode<FuncGraphPtr>(fg_list[0]);
    MS_EXCEPTION_IF_NULL(first_func_graph);
    const auto manager = first_func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto txn = manager->Transact();
    // Get all new args, new args is the union set of old args.
    auto new_args = ArgsUnion(args_list);
    auto old_args_index_map = GenOldArgsIndexes(fg_list, args_list);
    for (size_t branch_index = 0; branch_index < fg_list.size(); ++branch_index) {
      auto func_graph = GetValueNode<FuncGraphPtr>(fg_list[branch_index]);
      MS_EXCEPTION_IF_NULL(func_graph);
      auto new_parameters = GetFuncGraphNewParameters(func_graph, new_args, old_args_index_map);
      txn.SetParameters(func_graph, new_parameters);
    }
    txn.Commit();
    return new_args;
  }

 private:
  static std::vector<AnfNodePtr> ArgsUnion(const std::vector<AnfNodePtrList> args_list) {
    std::set<AnfNodePtr> no_monad_args;
    std::set<AnfNodePtr> monad_args;
    for (const auto &args : args_list) {
      for (const auto &arg : args) {
        if (HasAbstractMonad(arg)) {
          (void)monad_args.insert(arg);
          continue;
        }
        (void)no_monad_args.insert(arg);
      }
    }
    // Keep monad args after no monad args.
    std::vector<AnfNodePtr> union_args(no_monad_args.cbegin(), no_monad_args.cend());
    (void)union_args.insert(union_args.cend(), monad_args.cbegin(), monad_args.cend());
    return union_args;
  }

  static HashMap<FuncGraphPtr, HashMap<AnfNodePtr, size_t>> GenOldArgsIndexes(
    const AnfNodePtrList &fg_list, const std::vector<AnfNodePtrList> &args_list) {
    HashMap<FuncGraphPtr, HashMap<AnfNodePtr, size_t>> old_args_indexes;
    for (size_t i = 0; i < fg_list.size(); ++i) {
      const auto func_graph = GetValueNode<FuncGraphPtr>(fg_list[i]);
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &args = args_list[i];
      HashMap<AnfNodePtr, size_t> args_indexes;
      size_t arg_index = 0;
      for (const auto &arg : args) {
        (void)args_indexes.emplace(arg, arg_index++);
      }
      old_args_indexes[func_graph] = args_indexes;
    }
    return old_args_indexes;
  }

  static AnfNodePtr GetParameterByArg(const HashMap<FuncGraphPtr, HashMap<AnfNodePtr, size_t>> &all_old_args_index_map,
                                      const AnfNodePtr &arg) {
    MS_LOG(DEBUG) << "Get parameter by arg:" << arg->DebugString();
    for (const auto &[fg, old_args_index] : all_old_args_index_map) {
      auto it = old_args_index.find(arg);
      if (it == old_args_index.end()) {
        continue;
      }
      size_t arg_index = it->second;
      if (arg_index >= fg->parameters().size()) {
        MS_LOG(EXCEPTION) << "Index:" << arg_index << " out of range:" << fg->parameters().size();
      }
      return fg->parameters()[arg_index];
    }
    MS_LOG(EXCEPTION) << "Can't find parameter of arg:" << arg->DebugString();
  }

  static std::vector<AnfNodePtr> GetFuncGraphNewParameters(
    const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &new_args,
    const HashMap<FuncGraphPtr, HashMap<AnfNodePtr, size_t>> &all_old_args_index_map) {
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &old_parameters = func_graph->parameters();
    std::vector<AnfNodePtr> new_parameters(new_args.size());
    const auto &old_args_index_map = all_old_args_index_map.find(func_graph)->second;
    for (size_t new_arg_index = 0; new_arg_index < new_args.size(); ++new_arg_index) {
      const auto &new_arg = new_args[new_arg_index];
      auto arg_old_index_it = old_args_index_map.find(new_arg);
      // The new_arg is the arg of current func graph.
      if (arg_old_index_it != old_args_index_map.end()) {
        auto arg_old_index = arg_old_index_it->second;
        new_parameters[new_arg_index] = old_parameters[arg_old_index];
        MS_LOG(DEBUG) << "Find exist parameter:" << new_parameters[new_arg_index]->DebugString()
                      << ", arg_old_index:" << arg_old_index;
        continue;
      }
      // The new_arg is the arg of other func graph.
      const auto other_fg_parameter = GetParameterByArg(all_old_args_index_map, new_arg);
      MS_LOG(DEBUG) << "Get other fg's parameter:" << other_fg_parameter->DebugString();
      TraceGuard guard(std::make_shared<TraceCopy>(other_fg_parameter->debug_info()));
      ParameterPtr param = std::make_shared<Parameter>(func_graph);
      param->set_abstract(other_fg_parameter->abstract());
      new_parameters[new_arg_index] = param;
    }
    return new_parameters;
  }
};

// {{prim::kPrimSwitch, cond, {prim::kPrimPartial, G1, Xs}, {prim::kPrimPartial, G2, Ys}}, Zs} ->
// {{prim::kPrimSwitch, cond, G1, G2}, Xs Union Ys Union Zs}
// {{prim::kPrimSwitch, cond, {G1}, {prim::kPrimPartial, G2, Ys}}, Zs} -> {{prim::kPrimSwitch, cond, G1, G2}, Ys Union
// Zs}
// {{prim::kPrimSwitch, cond, {prim::kPrimPartial, G1, Xs}, {G2}}, Zs} -> {{prim::kPrimSwitch, cond, G1, G2}, Xs Union
// Zs}
class SwitchPartialEliminater : public ChoicePartialEliminater {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }
    auto switch_call = node->cast<CNodePtr>();
    if (!IsPrimitiveCNode(switch_call->input(0), prim::kPrimSwitch)) {
      return nullptr;
    }
    auto switch_node = switch_call->input(0)->cast<CNodePtr>();
    if (switch_node->size() != kSwitchInputSize) {
      return nullptr;
    }
    fg_list_.clear();
    args_list_.clear();
    const auto maybe_partial_1 = switch_node->input(kSwitchTrueBranchIndex);
    Visit(maybe_partial_1);
    const auto maybe_partial_2 = switch_node->input(kSwitchFalseBranchIndex);
    Visit(maybe_partial_2);

    // Either one should be {Partial, G, X}
    if (fg_list_.size() != kSwitchBranchesNum && args_list_.size() != kSwitchBranchesNum) {
      return nullptr;
    }
    // Should not continue;
    if (!CheckFuncGraphAndArgs()) {
      return nullptr;
    }
    MergeArgs(switch_call);
    if (args_list_[0] == args_list_[1]) {
      return BuildNewSwitchNode(switch_call, args_list_[0]);
    } else {
      const auto new_args = UnifyParameters(fg_list_, args_list_);
      return BuildNewSwitchNode(switch_call, new_args);
    }
  }

 private:
  AnfNodePtr BuildNewSwitchNode(const CNodePtr &switch_call, const std::vector<AnfNodePtr> &new_args) {
    const auto input0 = switch_call->input(0);
    MS_EXCEPTION_IF_NULL(input0);
    const auto switch_node = input0->cast<CNodePtr>();
    TraceGuard guard1(std::make_shared<TraceCopy>(switch_node->debug_info()));
    // {Switch, cond, G1, G2}
    std::vector<AnfNodePtr> switch_inputs = {switch_node->input(0), switch_node->input(1)};
    (void)switch_inputs.insert(switch_inputs.end(), fg_list_.begin(), fg_list_.end());
    const auto new_switch_cnode = switch_call->func_graph()->NewCNode(std::move(switch_inputs));
    new_switch_cnode->set_abstract(switch_node->abstract());
    // Create switch call.
    TraceGuard guard2(std::make_shared<TraceCopy>(switch_call->debug_info()));
    AnfNodePtrList switch_call_inputs{new_switch_cnode};
    (void)switch_call_inputs.insert(switch_call_inputs.end(), new_args.begin(), new_args.end());
    const auto new_call_node = switch_call->func_graph()->NewCNode(std::move(switch_call_inputs));
    new_call_node->set_abstract(switch_call->abstract());
    return new_call_node;
  }
};

// {{prim::kPrimSwitchLayer, cond, prim::MakeTuple{{prim::kPrimPartial, G1, Xs}, {prim::kPrimPartial, G2, Ys}}}, Zs} ->
// {{prim::kPrimSwitchLayer, cond, prim::MakeTuple{G1, G2}, Xs Union Ys Union Zs}
// {{prim::kPrimSwitchLayer, cond, prim::MakeTuple{{G1}, {prim::kPrimPartial, G2, Ys}}}, Zs} ->
// {{prim::kPrimSwitchLayer, cond, prim::MakeTuple{G1, G2}}, Ys Union Zs}
// {{prim::kPrimSwitchLayer, cond, prim::MakeTuple{{prim::kPrimPartial, G1, Xs}, {G2}}{}, Zs} ->
// {{prim::kPrimSwitchLayer, cond, prim::MakeTuple{G1, G2}}, Xs Union Zs}
class SwitchLayerPartialEliminater : public ChoicePartialEliminater {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }
    auto switch_layer_call = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_layer_call);
    // {SwitchLayer{}, Zs}
    if (!IsPrimitiveCNode(switch_layer_call->input(0), prim::kPrimSwitchLayer)) {
      return nullptr;
    }
    auto switch_layer_cnode = switch_layer_call->input(0)->cast<CNodePtr>();
    // {SwitchLayer, cond, MakeTuple{}}
    if (switch_layer_cnode->size() != kSwitchLayerInputSize) {
      return nullptr;
    }
    if (!IsPrimitiveCNode(switch_layer_cnode->input(kSwitchLayerBranchesIndex), prim::kPrimMakeTuple)) {
      return nullptr;
    }
    auto make_tuple_cnode = switch_layer_cnode->input(kSwitchLayerBranchesIndex)->cast<CNodePtr>();
    if (make_tuple_cnode->size() <= 1) {
      return nullptr;
    }

    fg_list_.clear();
    args_list_.clear();
    // Build funcgraph list and args list;
    for (size_t i = 1; i < make_tuple_cnode->size(); ++i) {
      Visit(make_tuple_cnode->input(i));
    }

    if (!CheckFuncGraphAndArgs()) {
      return nullptr;
    }
    MergeArgs(switch_layer_call);
    // All have the same args;
    auto args_equal =
      std::all_of(args_list_.cbegin() + 1, args_list_.cend(), [this](auto &args) { return args == args_list_[0]; });
    if (args_equal) {
      return BuildNewSwitchLayerNode(switch_layer_call, args_list_[0]);
    } else {
      const auto new_args = UnifyParameters(fg_list_, args_list_);
      return BuildNewSwitchLayerNode(switch_layer_call, new_args);
    }
  }

 private:
  AnfNodePtr BuildNewSwitchLayerNode(const CNodePtr &switch_layer_call_node, const AnfNodePtrList &new_args) {
    const auto switch_layer = switch_layer_call_node->input(0)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_layer);
    auto make_tuple_cnode = switch_layer->input(kSwitchLayerBranchesIndex)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple_cnode);
    // {primMakeTuple, G1, G2, ...}
    AnfNodePtrList make_tuple_args{make_tuple_cnode->input(0)};
    (void)make_tuple_args.insert(make_tuple_args.end(), fg_list_.begin(), fg_list_.end());
    TraceGuard guard1(std::make_shared<TraceCopy>(make_tuple_cnode->debug_info()));
    auto new_make_tuple_cnode = make_tuple_cnode->func_graph()->NewCNode(std::move(make_tuple_args));
    // {primSwitchLayer, cond, MakeTuple{}}
    TraceGuard guard2(std::make_shared<TraceCopy>(switch_layer->debug_info()));
    auto new_switch_layer =
      switch_layer->func_graph()->NewCNode({switch_layer->input(0), switch_layer->input(1), new_make_tuple_cnode});
    // Create new switch_layer call node.
    TraceGuard guard3(std::make_shared<TraceCopy>(switch_layer_call_node->debug_info()));
    AnfNodePtrList switch_layer_call_inputs{new_switch_layer};
    (void)switch_layer_call_inputs.insert(switch_layer_call_inputs.cend(), new_args.cbegin(), new_args.cend());
    auto new_node = switch_layer_call_node->func_graph()->NewCNode(std::move(switch_layer_call_inputs));
    new_node->set_abstract(switch_layer_call_node->abstract());
    return new_node;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARTIAL_ELIMINATE_H_
