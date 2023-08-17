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
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "include/common/utils/anfalgo.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
namespace irpass {
constexpr auto kAddedRecomputeDependAttr = "added_recompute_depend";
constexpr auto kHandledNotRecomputeNodeFlag = "handled_not_recompute_node";
constexpr auto kPrimalFgCallerUserDataKey = "primal_fg_caller";

bool EnableGraphReuse();

bool HasBpropGetter(const OptimizerPtr &opt, const AnfNodePtr &k_fg_caller);

AnfNodePtr GetBpropCaller(const FuncGraphManagerPtr &manager, const AnfNodePtr &bprop_getter);

bool AddRecomputeNodes(const FuncGraphPtr &root, const opt::OptimizerPtr &opt);

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
        MS_LOG(INTERNAL_EXCEPTION) << "The output of k graph should be make_tuple, but got "
                                   << cnode_k_fg_output->DebugString();
      }
      (void)manager->Replace(cnode_k_fg_output->cast<CNodePtr>()->input(1), para);
    }
    if (not_recompute_count == 0) {
      return nullptr;
    }

    primal_fg_->set_output(primal_fg_->NewCNode(new_primal_fg_outputs));
    auto primal_fg_caller = k_fg_caller->user_data<CNode>(kPrimalFgCallerUserDataKey);
    UpdateForwardResult(manager, primal_fg_caller);
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

    auto primal_fg_caller = k_fg_caller->user_data<CNode>(kPrimalFgCallerUserDataKey);
    UpdateForwardResult(manager, primal_fg_caller);
    return CreateNewKGraphCaller(fg, k_fg_caller, primal_fg_caller, not_recompute_count);
  }

  static AnfNodePtr CreateNewKGraphCaller(const FuncGraphPtr &fg, const CNodePtr &k_fg_caller,
                                          const CNodePtr &primal_fg_caller, int64_t not_recompute_count) {
    std::vector<AnfNodePtr> new_k_fg_caller_inputs;
    (void)new_k_fg_caller_inputs.insert(new_k_fg_caller_inputs.cend(), k_fg_caller->inputs().begin(),
                                        k_fg_caller->inputs().end());
    auto primal_fg_caller_fg = primal_fg_caller->func_graph();
    for (int64_t i = 1; i <= not_recompute_count; ++i) {
      auto extra_forward_result = primal_fg_caller_fg->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), primal_fg_caller, NewValueNode(i)});
      (void)new_k_fg_caller_inputs.emplace_back(extra_forward_result);
    }
    auto new_k_fg_caller = fg->NewCNodeInOrder(new_k_fg_caller_inputs);
    if (k_fg_caller->HasAttr(kAddedRecomputeDependAttr)) {
      new_k_fg_caller->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    }
    return new_k_fg_caller;
  }

  static void UpdateForwardResult(const FuncGraphManagerPtr &manager, const AnfNodePtr &primal_fg_caller) {
    MS_EXCEPTION_IF_NULL(primal_fg_caller);
    auto fg = primal_fg_caller->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
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
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RECOMPUTE_H_
