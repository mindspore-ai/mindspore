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

#include "frontend/optimizer/irpass/recompute.h"

namespace mindspore {
namespace opt {
namespace irpass {
bool EnableGraphReuse() {
  static const auto cell_reuse_env = common::GetEnv("MS_DEV_CELL_REUSE");
  static const auto cell_reuse_enable = cell_reuse_env == "1" || cell_reuse_env == "2";
  return cell_reuse_enable;
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

AnfNodePtr GetBpropCaller(const FuncGraphManagerPtr &manager, const AnfNodePtr &bprop_getter) {
  MS_EXCEPTION_IF_NULL(manager);
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

namespace {
constexpr auto kGradientsFlag = "Gradients";

bool IsRecomputeKGraphCaller(const AnfNodePtr &node) {
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr) {
    return false;
  }
  auto call_fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
  if (call_fg != nullptr && call_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
    return true;
  }
  return false;
}

bool IsGradNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->fullname_with_scope().compare(0, strlen(kGradientsFlag), kGradientsFlag) == 0;
}

bool AddNewPrimalNode(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const AnfNodePtr &origin_primal,
                      const AnfNodePtr &new_primal) {
  bool changed = false;
  auto node_users = manager->node_users()[origin_primal];
  for (auto &node_and_idx : node_users) {
    auto user = node_and_idx.first;
    MS_EXCEPTION_IF_NULL(user);
    // The forward part may have multiple outputs.
    if (IsPrimitiveCNode(user, prim::kPrimTupleGetItem)) {
      // Make new tuple_getitem to get corresponding output.
      auto new_primal_getitem = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), new_primal,
                                              user->cast_ptr<CNode>()->input(kInputNodeOutputIndexInTupleGetItem)});
      changed = AddNewPrimalNode(manager, fg, user, new_primal_getitem) || changed;
      continue;
    }
    // Set edge to not recomputed primal nodes.
    if (!IsRecomputeKGraphCaller(user) && !IsGradNode(user)) {
      MS_LOG(DEBUG) << "Set edge to user: " << user->DebugString() << ", new primal: " << new_primal->DebugString();
      manager->SetEdge(user, node_and_idx.second, new_primal);
      changed = true;
    }
  }
  return changed;
}

bool HasRecomputedInput(const CNodePtr &k_fg_caller_cnode) {
  for (auto &input : k_fg_caller_cnode->inputs()) {
    // The recomputed input should be a tuple_getitem to get the forward part of recomputed k graph.
    if (!IsPrimitiveCNode(input, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto tmp = input->cast<CNodePtr>()->input(1);
    auto input_k_fg_caller = tmp;
    // The forward part may have multiple outputs.
    if (IsPrimitiveCNode(tmp, prim::kPrimTupleGetItem)) {
      input_k_fg_caller = tmp->cast<CNodePtr>()->input(1);
    }
    if (IsRecomputeKGraphCaller(input_k_fg_caller)) {
      return true;
    }
  }
  return false;
}

AnfNodePtr GetForwardGetter(const FuncGraphManagerPtr &manager, const CNodePtr &node) {
  const auto &user_nodes = manager->node_users()[node];
  for (const auto &iter : user_nodes) {
    if (IsPrimitiveCNode(iter.first, prim::kPrimTupleGetItem)) {
      auto idx = GetValueNode<Int64ImmPtr>(iter.first->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
      if (idx != nullptr && idx->value() == 0) {
        return iter.first;
      }
    }
  }
  return nullptr;
}

AnfNodePtr GetBpropGetter(const FuncGraphManagerPtr &manager, const CNodePtr &node) {
  const auto &user_nodes = manager->node_users()[node];
  for (const auto &iter : user_nodes) {
    if (IsPrimitiveCNode(iter.first, prim::kPrimTupleGetItem)) {
      auto idx = GetValueNode<Int64ImmPtr>(iter.first->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
      if (idx != nullptr && idx->value() == 1) {
        return iter.first;
      }
    }
  }
  return nullptr;
}

bool HasRecomputedOutput(const FuncGraphManagerPtr &manager, const AnfNodePtr &node) {
  // The forward part may have multiple outputs.
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    const auto &user_nodes = manager->node_users()[node];
    return std::any_of(user_nodes.begin(), user_nodes.end(),
                       [&manager](const auto &iter) { return HasRecomputedOutput(manager, iter.first); });
  }
  return IsRecomputeKGraphCaller(node);
}

void GetGradUsers(const FuncGraphManagerPtr &manager, const CNodePtr &node, std::vector<AnfNodePtr> *grad_users) {
  // The forward part may have multiple outputs.
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    const auto &user_nodes = manager->node_users()[node];
    for (const auto &iter : user_nodes) {
      GetGradUsers(manager, iter.first->cast<CNodePtr>(), grad_users);
    }
    return;
  }
  if (IsGradNode(node)) {
    const auto &inputs = node->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (IsGradNode(inputs[i])) {
        (void)grad_users->emplace_back(inputs[i]);
      }
    }
  }
}

void GetDependencies(const FuncGraphManagerPtr &manager, const CNodePtr &k_fg_caller,
                     std::vector<std::pair<CNodePtr, AnfNodePtr>> *depends) {
  bool is_recompute_k_fg_caller = IsRecomputeKGraphCaller(k_fg_caller);
  // We only handle the recomputed k graph caller.
  if (!is_recompute_k_fg_caller && !IsPrimitiveCNode(k_fg_caller, prim::kPrimTupleGetItem)) {
    return;
  }
  if (is_recompute_k_fg_caller) {
    auto forward_getter = GetForwardGetter(manager, k_fg_caller);
    // If the k graph caller has no forward getter, it should not output to any other recomputed nodes.
    if (forward_getter == nullptr) {
      auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, k_fg_caller));
      // Add the dout input of its bprop function to the dependencies.
      (void)depends->emplace_back(std::make_pair(k_fg_caller, bprop_caller->cast<CNodePtr>()->input(1)));
      return;
    }
    if (!HasRecomputedOutput(manager, forward_getter)) {
      std::vector<AnfNodePtr> grad_users;
      // Add the other inputs of the grad node to the dependencies.
      GetGradUsers(manager, forward_getter->cast<CNodePtr>(), &grad_users);
      if (!grad_users.empty()) {
        for (auto &user : grad_users) {
          (void)depends->emplace_back(std::make_pair(k_fg_caller, user));
        }
        return;
      }
      // Add the dout input of its bprop function to the dependencies.
      auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, k_fg_caller));
      (void)depends->emplace_back(std::make_pair(k_fg_caller, bprop_caller->cast<CNodePtr>()->input(1)));
      return;
    }
  }

  const auto &user_nodes = manager->node_users()[k_fg_caller];
  for (const auto &iter : user_nodes) {
    if (IsPrimitiveCNode(iter.first, prim::kPrimTupleGetItem)) {
      auto idx = GetValueNode<Int64ImmPtr>(iter.first->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
      // Skip bprop getter.
      if (idx != nullptr && idx->value() == 1 && is_recompute_k_fg_caller) {
        continue;
      }
    }
    GetDependencies(manager, iter.first->cast<CNodePtr>(), depends);
  }
}

AnfNodePtr MoveKCallerToBprop(const FuncGraphManagerPtr &manager, const FuncGraphPtr &bprop_fg, const CNodePtr &node,
                              const std::vector<AnfNodePtr> &depend_inputs) {
  std::vector<AnfNodePtr> new_inputs;
  if (IsRecomputeKGraphCaller(node)) {
    if (!HasRecomputedInput(node)) {
      (void)std::copy(node->inputs().begin(), node->inputs().end(), std::back_inserter(new_inputs));
      new_inputs[1] =
        bprop_fg->NewCNode({NewValueNode(prim::kPrimDepend), new_inputs[1], bprop_fg->NewCNode(depend_inputs)});
    } else {
      for (auto &input : node->inputs()) {
        if (!input->isa<CNode>()) {
          (void)new_inputs.emplace_back(input);
          continue;
        }
        (void)new_inputs.emplace_back(MoveKCallerToBprop(manager, bprop_fg, input->cast<CNodePtr>(), depend_inputs));
      }
    }
    auto new_k_fg_caller = bprop_fg->NewCNode(new_inputs);
    new_k_fg_caller->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    // Replace the bprop getter with the new k graph caller in bprop graph.
    auto origin_bprop_getter = GetBpropGetter(manager, node);
    if (origin_bprop_getter != nullptr) {
      auto new_bprop_getter = bprop_fg->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), new_k_fg_caller, NewValueNode(static_cast<int64_t>(1))});
      manager->Replace(origin_bprop_getter, new_bprop_getter);
    }
    return new_k_fg_caller;
  }
  // If it is not tuple_getitem, it should be node which is not set recomputed.
  if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return node;
  }
  for (auto &input : node->inputs()) {
    if (!input->isa<CNode>()) {
      (void)new_inputs.emplace_back(input);
      continue;
    }
    (void)new_inputs.emplace_back(MoveKCallerToBprop(manager, bprop_fg, input->cast<CNodePtr>(), depend_inputs));
  }
  return bprop_fg->NewCNode(new_inputs);
}

CNodePtr GetKGraphCallerFromTupleGetitem(const AnfNodePtr &node) {
  auto idx = GetValueNode<Int64ImmPtr>(node->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
  // The k_fg_caller return a tuple of forward result and bprop.
  if (idx == nullptr || idx->value() != 0) {
    return nullptr;
  }
  auto k_fg_caller = node->cast<CNodePtr>()->input(1);
  MS_EXCEPTION_IF_NULL(k_fg_caller);
  return k_fg_caller->cast<CNodePtr>();
}

FuncGraphPtr GetRecomputeGradGraph(const FuncGraphPtr &k_fg) {
  constexpr auto fprop_output_size = 3;
  auto output = k_fg->output();
  MS_EXCEPTION_IF_NULL(output);
  if (!IsPrimitiveCNode(output, prim::kPrimMakeTuple) || output->cast_ptr<CNode>()->size() != fprop_output_size) {
    return nullptr;
  }
  auto grad_fg = GetValueNode<FuncGraphPtr>(output->cast_ptr<CNode>()->input(fprop_output_size - 1));
  if (grad_fg == nullptr) {
    return nullptr;
  }
  if (!grad_fg->has_flag(FUNC_GRAPH_RECOMPUTE_GRAD_GRAPH)) {
    return nullptr;
  }
  return grad_fg;
}

void AddDependNodes(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const CNodePtr &k_fg_caller_cnode) {
  // Get the nodes which the recomputed part should depend on;
  std::vector<std::pair<CNodePtr, AnfNodePtr>> dependencies;
  GetDependencies(manager, k_fg_caller_cnode, &dependencies);
  if (dependencies.empty()) {
    return;
  }
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimMakeTuple)};
  (void)std::transform(dependencies.begin(), dependencies.end(), std::back_inserter(depend_inputs),
                       [](const auto &pair) { return pair.second; });
  // Add the dependency nodes to the first recomputed nodes.
  auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, k_fg_caller_cnode));
  auto bprop_fg = bprop_caller->func_graph();
  MS_EXCEPTION_IF_NULL(bprop_fg);
  if (bprop_fg == fg) {
    auto depend =
      fg->NewCNode({NewValueNode(prim::kPrimDepend), k_fg_caller_cnode->input(1), fg->NewCNode(depend_inputs)});
    depend->AddAttr("recompute_insert", MakeValue(true));
    manager->SetEdge(k_fg_caller_cnode, 1, depend);
    return;
  }
  // If the graph of the bprop caller is not the same as the graph of k graph caller, we should move the k graph
  // caller to the graph of the bprop.
  for (size_t i = 0; i < dependencies.size(); ++i) {
    if (i > 0 && dependencies[i].first == dependencies[i - 1].first) {
      continue;
    }
    auto new_k_fg_caller = MoveKCallerToBprop(manager, bprop_fg, dependencies[i].first, depend_inputs);
    auto forward_getter = GetForwardGetter(manager, dependencies[i].first);
    if (forward_getter == nullptr) {
      manager->Replace(dependencies[i].first, new_k_fg_caller);
    } else {
      auto new_forward_getter = bprop_fg->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), new_k_fg_caller, NewValueNode(static_cast<int64_t>(0))});
      manager->Replace(forward_getter, new_forward_getter);
    }
  }
}
}  // namespace

bool AddRecomputePrimitive(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  if (!EnableGraphReuse()) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  bool changed = false;
  auto all_node = TopoSort(root->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (auto iter = all_node.crbegin(); iter != all_node.crend(); iter++) {
    auto node = *iter;
    if (!IsPrimitiveCNode(*iter, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto k_fg_caller_cnode = GetKGraphCallerFromTupleGetitem(node);
    if (k_fg_caller_cnode == nullptr) {
      continue;
    }
    auto k_fg = GetValueNode<FuncGraphPtr>(k_fg_caller_cnode->input(0));
    if (k_fg == nullptr || !k_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
      continue;
    }
    auto primal_iter = k_fg->transforms().find("primal");
    if (primal_iter == k_fg->transforms().end()) {
      continue;
    }
    auto primal_primitive = primal_iter->second.primitive();
    if (primal_primitive == nullptr) {
      continue;
    }
    auto grad_fg = GetRecomputeGradGraph(k_fg);
    if (grad_fg == nullptr) {
      continue;
    }
    // Erase the flag in case of reprocessing.
    grad_fg->erase_flag(FUNC_GRAPH_RECOMPUTE_GRAD_GRAPH);
    // Replace the forward getter with the origin primal.
    constexpr auto recursive_level = 2;
    MS_LOG(DEBUG) << "Handle recompute k graph forward getter: " << node->DebugString(recursive_level);
    std::vector<AnfNodePtr> inputs{NewValueNode(primal_primitive)};
    (void)inputs.insert(inputs.cend(), k_fg_caller_cnode->inputs().begin() + 1, k_fg_caller_cnode->inputs().end());
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto new_primal = fg->NewCNodeInOrder(inputs);
    changed = AddNewPrimalNode(manager, fg, node, new_primal) || changed;
    k_fg_caller_cnode->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    if (HasRecomputedInput(k_fg_caller_cnode)) {
      continue;
    }

    MS_LOG(DEBUG) << "Not has recomputed input k_fg_caller_cnode: " << k_fg_caller_cnode->DebugString();
    AddDependNodes(manager, fg, k_fg_caller_cnode);
  }
  return changed;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
