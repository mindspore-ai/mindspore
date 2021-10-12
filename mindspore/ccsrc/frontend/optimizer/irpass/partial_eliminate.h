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
#include <unordered_map>
#include <utility>
#include <vector>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
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
    const auto &xs_size = Xs_.size();
    // Xs_ don't have monad or Ys_ is 0.
    if (!HasAbstractMonad(Xs_.back()) || inputs.empty()) {
      args.push_back(X_);
      (void)std::copy(Xs_.begin(), Xs_.end(), std::back_inserter(args));
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args));
      TraceGuard guard(std::make_shared<TracePartialTransform>(node->debug_info()));
      auto new_node = node->func_graph()->NewCNode(args);
      return new_node;
    }
    // {X, Ys, Xs} if Xs has monad
    if (!IsValueNode<FuncGraph>(X_)) {
      MS_LOG(EXCEPTION) << "not support yet as X_ is not a funcgraph. node: " << node->DebugString(2);
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
    std::copy(fg->parameters().cbegin() + xs_size, fg->parameters().cend(), std::back_inserter(new_params));
    std::copy(fg->parameters().cbegin(), fg->parameters().cbegin() + xs_size, std::back_inserter(new_params));
    fg->manager()->SetParameters(fg, new_params);
    return new_node;
  }

  void Visit(const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPartial)) {
      return;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    // {prim::kPrimPartial, X, Xs}
    if (inputs.size() < 2) {
      return;
    }

    X_ = inputs[1];
    // fill Xs
    (void)std::copy(inputs.begin() + 2, inputs.end(), std::back_inserter(Xs_));
  }

 private:
  AnfNodePtr X_{nullptr};
  std::vector<AnfNodePtr> Xs_{};
};

class ChoicePartialEliminater : public AnfVisitor {
 public:
  virtual ~ChoicePartialEliminater() = default;

 protected:
  AnfNodePtrList fg_list_{};
  std::vector<AnfNodePtrList> args_list_{};

  void Visit(const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPartial)) {
      if (IsValueNode<FuncGraph>(node)) {
        fg_list_.push_back(node);
        args_list_.push_back(AnfNodePtrList{});
      }
      return;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    // {prim::kPrimPartial, G, Xs}
    if (inputs.size() < 3) {
      MS_LOG(EXCEPTION) << "Node should be Partial CNode, but: " << node->DebugString();
      return;
    }
    if (IsValueNode<FuncGraph>(inputs[1])) {
      fg_list_.push_back(inputs[1]);
      AnfNodePtrList args;
      (void)std::copy(inputs.begin() + 2, inputs.end(), std::back_inserter(args));
      args_list_.push_back(args);
    }
    return;
  }

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
        auto new_fg = BasicClone(fg);
        auto manager = fg->manager();
        MS_EXCEPTION_IF_NULL(manager);
        manager->AddFuncGraph(new_fg);
        fg_node->cast<ValueNodePtr>()->set_value(new_fg);
      }
    }
    return true;
  }

  // f(x1, x2, x3, z1, z2)
  // g(x4, x2, z1, z2)
  // h(x5, x2, x7, x8, z1, z2)
  // --> anchor_fg = h
  // h(x5, x2, x7, x8, x1, x3, x4, z1, z2)
  // f(x5, x2, x7, x8, x1, x3, x4, z1, z2)
  // g(x5, x2, x7, x8, x1, x3, x4, z1, z2)
  // as z1, z2 maybe U or IO monad.
  AnfNodePtrList UnifyParameters(const size_t &anchor_index, const AnfNodePtrList &fg_list,
                                 const std::vector<AnfNodePtrList> args_list) {
    std::vector<size_t> inputs_index_list[args_list.size()];
    size_t extra_input_counter = 0;
    AnfNodePtrList extra_inputs;
    const auto &anchor_args = args_list[anchor_index];
    size_t anchor_args_size = anchor_args.size();
    auto anchor_fg = GetValueNode<FuncGraphPtr>(fg_list[anchor_index]);
    MS_EXCEPTION_IF_NULL(anchor_fg);
    // Find the new location of the old_inputs except Zs;
    for (size_t i = 0; i < args_list.size(); ++i) {
      if (i == anchor_index) {
        continue;
      }
      const auto &another_args = args_list[i];
      auto &curr_inputs_index = inputs_index_list[i];
      for (size_t j = 0; j < another_args.size(); ++j) {
        size_t k;
        for (k = 0; k < anchor_args_size; ++k) {
          if (another_args[j] == anchor_args[k]) {
            curr_inputs_index.push_back(k);
            break;
          }
        }
        if (k == anchor_args_size) {
          // check if used by another func_graph;
          for (k = 0; k < extra_input_counter; ++k) {
            if (another_args[j] == extra_inputs[k]) {
              curr_inputs_index.push_back(anchor_args_size + k);
              break;
            }
          }
          if (k == extra_input_counter) {
            extra_inputs.push_back(another_args[j]);
            curr_inputs_index.push_back(anchor_args_size + extra_input_counter);
            extra_input_counter++;
          }
        }
      }
    }

    auto manager = anchor_fg->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto txn = manager->Transact();

    size_t anchor_params_size = anchor_fg->parameters().size();
    const auto &anchor_fg_params = anchor_fg->parameters();
    for (size_t i = 0; i < args_list.size(); ++i) {
      if (i == anchor_index) {
        continue;
      }
      AnfNodePtrList new_params;
      new_params.resize(anchor_params_size + extra_input_counter);

      const auto &curr_inputs_index = inputs_index_list[i];
      auto another_fg = GetValueNode<FuncGraphPtr>(fg_list[i]);
      MS_EXCEPTION_IF_NULL(another_fg);
      const auto &old_params = another_fg->parameters();
      const auto &old_args = args_list[i];
      for (size_t j = 0; j < old_args.size(); j++) {
        new_params[curr_inputs_index[j]] = old_params[j];
      }
      // Zs_
      for (size_t j = old_args.size(), k = 0; j < old_params.size(); ++j, ++k) {
        new_params[anchor_args_size + extra_input_counter + k] = old_params[j];
      }
      // unused inputs
      for (size_t j = 0; j < anchor_args_size; ++j) {
        if (new_params[j] == nullptr) {
          TraceGuard guard(std::make_shared<TraceCopy>(anchor_fg_params[j]->debug_info()));
          ParameterPtr param = std::make_shared<Parameter>(another_fg);
          new_params[j] = param;
        }
      }
      // extra inputs used by another func_graph;
      for (size_t j = 0; j < extra_inputs.size(); ++j) {
        if (new_params[anchor_args_size + j] == nullptr) {
          TraceGuard guard(std::make_shared<TraceCopy>(extra_inputs[j]->debug_info()));
          ParameterPtr param = std::make_shared<Parameter>(another_fg);
          new_params[anchor_args_size + j] = param;
        }
      }
      // set the parameter for another_fg and replace it's parameters;
      txn.SetParameters(another_fg, new_params);
    }
    // Reorder Zs_ and add extra parameters for anchor_fg;
    // add extra parameter for anchor_fg;
    AnfNodePtrList new_params;
    new_params.reserve(anchor_params_size + extra_input_counter);
    // reuse parameters for anchor_args;
    std::copy(anchor_fg_params.cbegin(), anchor_fg_params.cbegin() + anchor_args_size, std::back_inserter(new_params));
    // Extra parameters;
    for (size_t i = 0; i < extra_inputs.size(); ++i) {
      TraceGuard guard(std::make_shared<TraceCopy>(extra_inputs[i]->debug_info()));
      ParameterPtr param = std::make_shared<Parameter>(anchor_fg);
      new_params.push_back(param);
    }
    // Reorder Zs_ to last;
    for (size_t i = anchor_args_size; i < anchor_params_size; ++i) {
      new_params.push_back(anchor_fg_params[i]);
    }
    txn.SetParameters(anchor_fg, new_params);
    txn.Commit();

    return extra_inputs;
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
    auto cnode = node->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch)) {
      return nullptr;
    }
    auto input0_cnode = cnode->input(0)->cast<CNodePtr>();
    if (input0_cnode->size() != 4) {
      return nullptr;
    }

    fg_list_.clear();
    args_list_.clear();
    auto &maybe_partial_1 = input0_cnode->input(2);
    Visit(maybe_partial_1);
    auto &maybe_partial_2 = input0_cnode->input(3);
    Visit(maybe_partial_2);

    // Either one should be {Partial, G, X}
    if (fg_list_.size() != 2 && args_list_.size() != 2) {
      return nullptr;
    }
    // Should not continue;
    if (!CheckFuncGraphAndArgs()) {
      return nullptr;
    }

    if (args_list_[0] == args_list_[1]) {
      auto new_node =
        BuildNewSwitchNode(cnode, input0_cnode, fg_list_[0], fg_list_[1], args_list_[0], AnfNodePtrList{});
      return new_node;
    } else {
      // find partial funcgraph with the longest args as anchor;
      size_t max_args_pos = 0;
      if (args_list_[0].size() > args_list_[1].size()) {
        max_args_pos = 0;
      } else {
        max_args_pos = 1;
      }

      auto extra_inputs = UnifyParameters(max_args_pos, fg_list_, args_list_);
      auto new_node =
        BuildNewSwitchNode(cnode, input0_cnode, fg_list_[0], fg_list_[1], args_list_[max_args_pos], extra_inputs);
      return new_node;
    }
  }

 private:
  AnfNodePtr BuildNewSwitchNode(const CNodePtr &old_cnode, const CNodePtr input0_cnode, const AnfNodePtr &G1,
                                const AnfNodePtr &G2, const AnfNodePtrList &partial_args,
                                const AnfNodePtrList &extra_args) {
    TraceGuard guard1(std::make_shared<TraceCopy>(input0_cnode->debug_info()));
    // {Switch, cond, G1, G2}
    auto switch_cnode = old_cnode->func_graph()->NewCNode({input0_cnode->input(0), input0_cnode->input(1), G1, G2});
    AnfNodePtrList args{switch_cnode};
    (void)std::copy(partial_args.begin(), partial_args.end(), std::back_inserter(args));
    (void)std::copy(extra_args.begin(), extra_args.end(), std::back_inserter(args));
    // Zs
    if (old_cnode->size() >= 2) {
      (void)std::copy(old_cnode->inputs().begin() + 1, old_cnode->inputs().end(), std::back_inserter(args));
    }
    TraceGuard guard2(std::make_shared<TraceCopy>(old_cnode->debug_info()));
    auto new_node = old_cnode->func_graph()->NewCNode(args);
    new_node->set_abstract(old_cnode->abstract());
    return new_node;
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
    auto cnode = node->cast<CNodePtr>();
    // {SwitchLayer{}, Zs}
    if (!IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitchLayer)) {
      return nullptr;
    }
    auto switch_layer_cnode = cnode->input(0)->cast<CNodePtr>();
    // {SwitchLayer, cond, MakeTuple{}}
    if (switch_layer_cnode->size() != 3) {
      return nullptr;
    }
    if (!IsPrimitiveCNode(switch_layer_cnode->input(2), prim::kPrimMakeTuple)) {
      return nullptr;
    }
    auto make_tuple_cnode = switch_layer_cnode->input(2)->cast<CNodePtr>();
    if (make_tuple_cnode->size() < 2) {
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
    // All have the same args;
    auto args_equal =
      std::all_of(args_list_.cbegin() + 1, args_list_.cend(), [this](auto &args) { return args == args_list_[0]; });
    if (args_equal) {
      auto new_node = BuildNewSwitchLayerNode(cnode, switch_layer_cnode, args_list_[0], AnfNodePtrList{});
      return new_node;
    } else {
      // find partial funcgraph with the longest args as anchor;
      size_t max_args_pos = 0, max_args_len = 0;
      for (size_t i = 0; i < args_list_.size(); ++i) {
        if (max_args_len < args_list_[i].size()) {
          max_args_len = args_list_[i].size();
          max_args_pos = i;
        }
      }
      auto extra_inputs = UnifyParameters(max_args_pos, fg_list_, args_list_);
      auto new_node = BuildNewSwitchLayerNode(cnode, switch_layer_cnode, args_list_[max_args_pos], extra_inputs);
      return new_node;
    }
  }

 private:
  AnfNodePtr BuildNewSwitchLayerNode(const CNodePtr &old_cnode, const CNodePtr switch_layer_cnode,
                                     const AnfNodePtrList &anchor_partial_args, const AnfNodePtrList &extra_args) {
    auto make_tuple_cnode = switch_layer_cnode->input(2)->cast<CNodePtr>();
    AnfNodePtrList make_tuple_args{make_tuple_cnode->input(0)};
    make_tuple_args.insert(make_tuple_args.end(), fg_list_.begin(), fg_list_.end());
    TraceGuard guard1(std::make_shared<TraceCopy>(make_tuple_cnode->debug_info()));
    // {MakeTuple, G1, G2, ...}
    auto new_make_tuple_cnode = old_cnode->func_graph()->NewCNode(make_tuple_args);

    TraceGuard guard2(std::make_shared<TraceCopy>(switch_layer_cnode->debug_info()));
    // {SwitchLayer, cond, MakeTuple{}}
    auto new_switch_layer_cnode = old_cnode->func_graph()->NewCNode(
      {switch_layer_cnode->input(0), switch_layer_cnode->input(1), new_make_tuple_cnode});
    AnfNodePtrList args{new_switch_layer_cnode};
    (void)std::copy(anchor_partial_args.begin(), anchor_partial_args.end(), std::back_inserter(args));
    (void)std::copy(extra_args.begin(), extra_args.end(), std::back_inserter(args));
    // Zs
    if (old_cnode->size() >= 2) {
      (void)std::copy(old_cnode->inputs().begin() + 1, old_cnode->inputs().end(), std::back_inserter(args));
    }
    TraceGuard guard3(std::make_shared<TraceCopy>(old_cnode->debug_info()));
    auto new_node = old_cnode->func_graph()->NewCNode(args);
    new_node->set_abstract(old_cnode->abstract());
    return new_node;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARTIAL_ELIMINATE_H_
