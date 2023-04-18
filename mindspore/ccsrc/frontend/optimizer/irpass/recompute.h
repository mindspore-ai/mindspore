/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RECOMPUTE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RECOMPUTE_H_

#include <vector>
#include <algorithm>
#include <utility>
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "include/common/utils/anfalgo.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
namespace irpass {
constexpr auto kAddedRecomputeDependAttr = "added_recompute_depend";
constexpr auto kHandledNotRecomputeNodeFlag = "handled_not_recompute_node";

bool EnableGraphReuse() {
  static const auto graph_reuse_env = common::GetEnv("MS_DEV_GRAPH_REUSE");
  static const auto graph_reuse_enable = graph_reuse_env == "1" || graph_reuse_env == "2";
  return graph_reuse_enable;
}

bool HasBpropGetter(const OptimizerPtr &opt, const AnfNodePtr &k_fg_caller) {
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_users = manager->node_users();
  auto iter = node_users.find(k_fg_caller);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "The node " << k_fg_caller->DebugString() << " should have users.";
  }

  return std::any_of(iter->second.begin(), iter->second.end(), [](const std::pair<AnfNodePtr, int> &node_and_idx) {
    auto user = node_and_idx.first;
    return IsPrimitiveCNode(user, prim::kPrimTupleGetItem) &&
           common::AnfAlgo::GetTupleGetItemOutIndex(user->cast<CNodePtr>()) == 1;
  });
}

class AddRecomputePrimal : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &opt, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);

    if (!is_match_) {
      return nullptr;
    }

    if (!HasBpropGetter(opt, k_fg_caller_)) {
      return nullptr;
    }

    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    // Replace the forward result of k graph with the original primal graph.
    std::vector<AnfNodePtr> inputs{NewValueNode(primal_fg_)};
    (void)inputs.insert(inputs.cend(), k_fg_caller_->inputs().begin() + 1, k_fg_caller_->inputs().end());
    auto new_primal_fg_caller = fg->NewCNodeInOrder(inputs);
    k_fg_caller_->set_user_data("primal_fg_caller", new_primal_fg_caller);
    return new_primal_fg_caller;
  }

  void Visit(const CNodePtr &cnode) override {
    if (!EnableGraphReuse()) {
      return;
    }
    auto called_fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
    if (called_fg == nullptr || !called_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
      return;
    }
    auto iter = called_fg->transforms().find("primal");
    if (iter == called_fg->transforms().end()) {
      return;
    }
    auto primal_fg = iter->second.func_graph();
    if (primal_fg != nullptr && primal_fg->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      primal_fg_ = primal_fg;
      k_fg_caller_ = cnode;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    auto idx = GetValueNode<Int64ImmPtr>(vnode);
    // The k_fg_caller return a tuple of forward result and bprop.
    if (idx != nullptr && k_fg_caller_ != nullptr && idx->value() == 0) {
      is_match_ = true;
    }
  }

  void Reset() {
    primal_fg_ = nullptr;
    k_fg_caller_ = nullptr;
    is_match_ = false;
  }

 private:
  FuncGraphPtr primal_fg_{nullptr};
  CNodePtr k_fg_caller_{nullptr};
  bool is_match_{false};
};

class RemoveNotRecomputeNode : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &opt, const AnfNodePtr &node) override {
    if (!EnableGraphReuse()) {
      return nullptr;
    }
    Reset();
    auto k_fg_caller = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(k_fg_caller);
    if (!IsMatch(k_fg_caller)) {
      return nullptr;
    }

    MS_EXCEPTION_IF_NULL(opt);
    auto manager = opt->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);

    bool has_bprop_getter = HasBpropGetter(opt, k_fg_caller);
    // If the k graph has been handled, the call nodes of k and primal graph should be handled.
    if (k_fg_->has_flag(kHandledNotRecomputeNodeFlag)) {
      return CreateNewCallerForHandledKGraph(manager, fg, k_fg_caller, has_bprop_getter);
    }

    // The k graph only contains primal should not be recomputed.
    if (!has_bprop_getter) {
      return nullptr;
    }

    k_fg_->set_flag(kHandledNotRecomputeNodeFlag, true);
    std::vector<AnfNodePtr> new_primal_fg_outputs{NewValueNode(prim::kPrimMakeTuple), primal_fg_->output()};
    std::vector<AnfNodePtr> k_fg_nodes = TopoSort(k_fg_->get_return(), SuccDeeperSimple);
    int64_t not_recompute_count = 0;
    for (const auto &node_in_k_fg : k_fg_nodes) {
      auto [cnode_k_fg, primal_cnode] = GetNotRecomputeKGraphAndPrimalCNode(node_in_k_fg);
      if (cnode_k_fg == nullptr || primal_cnode == nullptr) {
        continue;
      }
      ++not_recompute_count;
      // Erase the flag to do inline later.
      cnode_k_fg->erase_flag(FUNC_GRAPH_NOT_RECOMPUTE_K_GRAPH);
      // Replace the primal node in k graph with the node in primal graph.
      (void)new_primal_fg_outputs.emplace_back(primal_cnode);
      auto para = k_fg_->add_parameter();
      auto cnode_k_fg_output = cnode_k_fg->output();
      if (!IsPrimitiveCNode(cnode_k_fg_output, prim::kPrimMakeTuple)) {
        MS_LOG(EXCEPTION) << "The output of k graph should be make_tuple, but got " << cnode_k_fg_output->DebugString();
      }
      (void)manager->Replace(cnode_k_fg_output->cast<CNodePtr>()->input(1), para);
    }
    if (not_recompute_count == 0) {
      return nullptr;
    }

    primal_fg_->set_output(primal_fg_->NewCNode(new_primal_fg_outputs));
    auto primal_fg_caller = k_fg_caller->user_data<CNode>("primal_fg_caller");
    UpdateForwardResult(manager, fg, primal_fg_caller);
    // Add new arguments to k graph caller.
    return CreateNewKGraphCaller(fg, k_fg_caller, primal_fg_caller, not_recompute_count);
  }

  AnfNodePtr CreateNewCallerForHandledKGraph(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg,
                                             const CNodePtr &k_fg_caller, bool has_bprop_getter) {
    auto not_recompute_count = SizeToLong(k_fg_->parameters().size() - (k_fg_caller->size() - 1));
    if (not_recompute_count == 0) {
      return nullptr;
    }
    if (!has_bprop_getter) {
      std::vector<AnfNodePtr> new_primal_caller_inputs{NewValueNode(primal_fg_)};
      (void)new_primal_caller_inputs.insert(new_primal_caller_inputs.cend(), k_fg_caller->inputs().begin() + 1,
                                            k_fg_caller->inputs().end());
      auto new_primal_caller = fg->NewCNodeInOrder(new_primal_caller_inputs);
      return new_primal_caller;
    }

    auto primal_fg_caller = k_fg_caller->user_data<CNode>("primal_fg_caller");
    UpdateForwardResult(manager, fg, primal_fg_caller);
    return CreateNewKGraphCaller(fg, k_fg_caller, primal_fg_caller, not_recompute_count);
  }

  static AnfNodePtr CreateNewKGraphCaller(const FuncGraphPtr &fg, const CNodePtr &k_fg_caller,
                                          const CNodePtr &primal_fg_caller, int64_t not_recompute_count) {
    std::vector<AnfNodePtr> new_k_fg_caller_inputs;
    (void)new_k_fg_caller_inputs.insert(new_k_fg_caller_inputs.cend(), k_fg_caller->inputs().begin(),
                                        k_fg_caller->inputs().end());
    for (int64_t i = 1; i <= not_recompute_count; ++i) {
      auto extra_forward_result =
        fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), primal_fg_caller, NewValueNode(i)});
      (void)new_k_fg_caller_inputs.emplace_back(extra_forward_result);
    }
    return fg->NewCNodeInOrder(new_k_fg_caller_inputs);
  }

  static void UpdateForwardResult(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg,
                                  const AnfNodePtr &primal_fg_caller) {
    MS_EXCEPTION_IF_NULL(primal_fg_caller);
    auto forward_result = fg->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), primal_fg_caller, NewValueNode(static_cast<int64_t>(0))});
    (void)manager->Replace(primal_fg_caller, forward_result);
  }

  static std::pair<FuncGraphPtr, AnfNodePtr> GetNotRecomputeKGraphAndPrimalCNode(const AnfNodePtr &node) {
    auto cnode = dyn_cast<CNode>(node);
    if (cnode == nullptr) {
      return std::make_pair(nullptr, nullptr);
    }
    // call (k_fg, ...)
    auto cnode_k_fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
    if (cnode_k_fg == nullptr || !cnode_k_fg->has_flag(FUNC_GRAPH_NOT_RECOMPUTE_K_GRAPH)) {
      return std::make_pair(nullptr, nullptr);
    }
    // k_fg -> primal
    auto primal_cnode = GetPrimalCNode(cnode_k_fg);
    if (primal_cnode == nullptr) {
      MS_LOG(DEBUG) << "The cnode k_fg " << cnode_k_fg->ToString() << " should have corresponding primal_cnode.";
      return std::make_pair(nullptr, nullptr);
    }
    MS_LOG(DEBUG) << "primal_cnode: " << primal_cnode->DebugString();
    return std::make_pair(cnode_k_fg, primal_cnode);
  }

  static AnfNodePtr GetPrimalCNode(const FuncGraphPtr &cnode_k_fg) {
    auto primal_cnode_iter = cnode_k_fg->transforms().find("primal_cnode");
    if (primal_cnode_iter == cnode_k_fg->transforms().end()) {
      MS_LOG(DEBUG) << "Not found the primal cnode of k graph " << cnode_k_fg->ToString();
      return nullptr;
    }
    auto primal_cnode = primal_cnode_iter->second.primal_cnode();
    return primal_cnode;
  }

  bool IsMatch(const CNodePtr &k_fg_caller) {
    auto k_fg = GetValueNode<FuncGraphPtr>(k_fg_caller->input(0));
    if (k_fg == nullptr || !k_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
      return false;
    }

    auto primal_iter = k_fg->transforms().find("primal");
    if (primal_iter == k_fg->transforms().end()) {
      return false;
    }
    auto primal_fg = primal_iter->second.func_graph();
    if (primal_fg == nullptr || !primal_fg->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      return false;
    }

    k_fg_ = k_fg;
    primal_fg_ = primal_fg;
    return true;
  }

  void Reset() {
    k_fg_ = nullptr;
    primal_fg_ = nullptr;
  }

 private:
  FuncGraphPtr k_fg_{nullptr};
  FuncGraphPtr primal_fg_{nullptr};
};

class AddRecomputeDepend : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &opt, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);

    if (!is_match_) {
      return nullptr;
    }
    MS_EXCEPTION_IF_NULL(opt);
    auto manager = opt->manager();
    MS_EXCEPTION_IF_NULL(manager);

    auto bprop_caller = dyn_cast<CNode>(GetBpropCaller(manager, node));
    if (bprop_caller == nullptr) {
      return nullptr;
    }
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    // Add depend to make sure the forward nodes of k graph are executed after the gradient from last bprop is ready.
    auto bprop_caller_fg = bprop_caller->func_graph();
    std::vector<AnfNodePtr> new_k_fg_caller_inputs{k_fg_caller_->input(0)};
    auto dout = bprop_caller->input(1);
    (void)std::transform(k_fg_caller_->inputs().begin() + 1, k_fg_caller_->inputs().end(),
                         std::back_inserter(new_k_fg_caller_inputs),
                         [&fg, &bprop_caller_fg, &dout](const AnfNodePtr &node) -> AnfNodePtr {
                           auto ret_node = node;
                           if (bprop_caller_fg != fg) {
                             if (HasAbstractUMonad(node)) {
                               ret_node = NewValueNode(kUMonad);
                             } else if (HasAbstractIOMonad(node)) {
                               ret_node = NewValueNode(kIOMonad);
                             }
                           }
                           return bprop_caller_fg->NewCNodeInOrder({NewValueNode(prim::kPrimDepend), ret_node, dout});
                         });
    auto new_k_fg_caller = bprop_caller_fg->NewCNodeInOrder(new_k_fg_caller_inputs);
    auto new_tuple_getitem = bprop_caller_fg->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), new_k_fg_caller, NewValueNode(static_cast<int64_t>(1))});
    // Add attr in case of repeatedly handling.
    new_k_fg_caller->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    return new_tuple_getitem;
  }

  static AnfNodePtr GetBpropCaller(const FuncGraphManagerPtr &manager, const AnfNodePtr &bprop_getter) {
    const auto &node_users = manager->node_users();
    auto iter = node_users.find(bprop_getter);
    if (iter == node_users.end()) {
      return nullptr;
    }
    if (iter->second.size() != 1) {
      MS_LOG(EXCEPTION) << "The number of bprop caller should be 1, but got " << iter->second.size()
                        << ", bprop_getter: " << bprop_getter->DebugString();
    }
    auto user_node_idx = iter->second.begin();
    if (user_node_idx->second != 0) {
      MS_LOG(EXCEPTION) << "The bprop_getter should be used in input 0, but got " << user_node_idx->second;
    }
    return user_node_idx->first;
  }

  void Visit(const CNodePtr &cnode) override {
    if (!EnableGraphReuse()) {
      return;
    }
    auto call_fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
    if (call_fg == nullptr || !call_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH) ||
        cnode->HasAttr(kAddedRecomputeDependAttr)) {
      return;
    }
    k_fg_caller_ = cnode;
  }

  void Visit(const ValueNodePtr &vnode) override {
    auto idx = GetValueNode<Int64ImmPtr>(vnode);
    if (idx != nullptr && k_fg_caller_ != nullptr && idx->value() == 1) {
      is_match_ = true;
    }
  }

  void Reset() {
    k_fg_caller_ = nullptr;
    is_match_ = false;
  }

 private:
  CNodePtr k_fg_caller_{nullptr};
  bool is_match_{false};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RECOMPUTE_H_
