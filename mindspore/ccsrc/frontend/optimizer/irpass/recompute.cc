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
#include <set>
#include <unordered_map>
#include "ops/array_ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
bool EnableCellReuse() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  return cell_reuse;
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
constexpr auto kAttrReplacedWithPrimal = "replaced_with_primal";
constexpr auto kAttrRecomputeMakeTuple = "recompute_make_tuple";

bool WithRecomputedScope(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &full_name_with_scope = node->fullname_with_scope();
  return full_name_with_scope.compare(0, strlen(kAttrRecompute), kAttrRecompute) == 0;
}

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

bool WithGradientScope(const AnfNodePtr &node) {
  return node->fullname_with_scope().compare(0, strlen(kGradientsFlag), kGradientsFlag) == 0;
}

bool IsFromBpropOutput(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return false;
  }
  auto cur_node = node;
  while (IsPrimitiveCNode(cur_node, prim::kPrimTupleGetItem)) {
    cur_node = cur_node->cast<CNodePtr>()->input(kRealInputNodeIndexInTupleGetItem);
  }
  if (WithGradientScope(cur_node)) {
    return true;
  }
  auto cur_cnode = cur_node->cast<CNodePtr>();
  if (cur_cnode == nullptr) {
    return false;
  }
  auto func_abs = dyn_cast<abstract::FuncGraphAbstractClosure>(cur_cnode->input(0)->abstract());
  if (func_abs == nullptr) {
    return false;
  }
  auto fg = func_abs->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  return fg->has_flag(FUNC_GRAPH_RECOMPUTE_GRAD_GRAPH);
}

bool IsGradNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return WithGradientScope(node) || IsFromBpropOutput(node);
}

bool IsFpropReturn(const AnfNodePtr &make_tuple) {
  auto cnode = make_tuple->cast<CNodePtr>();
  constexpr size_t fprop_output_size = 2;
  if (cnode->size() != fprop_output_size + 1) {
    return false;
  }
  return IsValueNode<FuncGraph>(cnode->input(fprop_output_size));
}

AnfNodePtr GetPrimalFromFprop(const FuncGraphPtr &k_fg) {
  if (!IsPrimitiveCNode(k_fg->output(), prim::kPrimMakeTuple)) {
    return nullptr;
  }
  auto k_fg_outputs = k_fg->output()->cast<CNodePtr>()->inputs();
  if (k_fg_outputs.size() != 3) {
    return nullptr;
  }
  return k_fg_outputs[kIndex1];
}

bool ShouldAddNewPrimalOutput(const AnfNodePtr &node, bool recompute_cell) {
  return !IsGradNode(node) || recompute_cell;
}

bool IsForwardDepend(const AnfNodePtr &node) {
  return IsPrimitiveCNode(node, prim::kPrimDepend) && !node->cast_ptr<CNode>()->HasAttr(kRecomputeInsert);
}

bool AddNewPrimalNode(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const AnfNodePtr &origin_primal,
                      const AnfNodePtr &new_primal, bool recompute_cell,
                      std::unordered_map<AnfNodePtr, AnfNodePtr> *origin_to_new_primal) {
  bool changed = false;
  auto node_users = manager->node_users()[origin_primal];
  for (auto &node_and_idx : node_users) {
    auto user = node_and_idx.first;
    MS_EXCEPTION_IF_NULL(user);
    // The forward part may have multiple outputs.
    if (IsPrimitiveCNode(user, prim::kPrimTupleGetItem) && ShouldAddNewPrimalOutput(user, recompute_cell)) {
      // Make new tuple_getitem to get corresponding output.
      auto new_primal_getitem = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), new_primal,
                                              user->cast_ptr<CNode>()->input(kInputNodeOutputIndexInTupleGetItem)});
      changed =
        AddNewPrimalNode(manager, fg, user, new_primal_getitem, recompute_cell, origin_to_new_primal) || changed;
      continue;
    }
    if (IsForwardDepend(user) && ShouldAddNewPrimalOutput(user, recompute_cell)) {
      // Make new depend node in forward to get corresponding output.
      auto new_depend = fg->NewCNode(user->cast_ptr<CNode>()->inputs());
      new_depend->set_input(IntToSize(node_and_idx.second), new_primal);
      changed = AddNewPrimalNode(manager, fg, user, new_depend, recompute_cell, origin_to_new_primal) || changed;
      continue;
    }
    // The op like concat will have a make_tuple input.
    if (IsPrimitiveCNode(user, prim::kPrimMakeTuple) && !IsFpropReturn(user) &&
        ShouldAddNewPrimalOutput(user, recompute_cell)) {
      auto user_cnode = user->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(user_cnode);
      if (user_cnode->HasAttr(kAttrRecomputeMakeTuple)) {
        manager->SetEdge(user_cnode, node_and_idx.second, new_primal);
        continue;
      }
      auto iter = origin_to_new_primal->find(user);
      if (iter != origin_to_new_primal->end()) {
        // The new make_tuple has been created, just set its inputs.
        manager->SetEdge(iter->second, node_and_idx.second, new_primal);
        continue;
      }
      // Create a new primal make_tuple.
      std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
      for (size_t i = 1; i < user_cnode->size(); ++i) {
        (void)make_tuple_inputs.emplace_back(user_cnode->input(i));
      }
      auto new_primal_make_tuple = fg->NewCNode(make_tuple_inputs);
      new_primal_make_tuple->set_input(node_and_idx.second, new_primal);
      new_primal_make_tuple->AddAttr(kAttrRecomputeMakeTuple, MakeValue(true));
      (void)origin_to_new_primal->emplace(user, new_primal_make_tuple);
      changed =
        AddNewPrimalNode(manager, fg, user, new_primal_make_tuple, recompute_cell, origin_to_new_primal) || changed;
      continue;
    }

    // Set edge to not recomputed primal nodes.
    if (recompute_cell || (!IsRecomputeKGraphCaller(user) && !IsGradNode(user))) {
      MS_LOG(DEBUG) << "Set edge to user: " << user->DebugString() << ", new primal: " << new_primal->DebugString();
      manager->SetEdge(user, node_and_idx.second, new_primal);
      changed = true;
    }
  }
  return changed;
}

bool IsRecomputeCell(const FuncGraphPtr &k_fg) {
  auto primal_iter = k_fg->transforms().find("primal");
  if (primal_iter == k_fg->transforms().end()) {
    MS_LOG(EXCEPTION) << "The k_fg: " << k_fg << " should have primal part.";
  }
  return primal_iter->second.func_graph() != nullptr;
}

bool HasRecomputedInput(const CNodePtr &k_fg_caller_cnode) {
  for (auto &input : k_fg_caller_cnode->inputs()) {
    if (IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
      return HasRecomputedInput(input->cast<CNodePtr>());
    }
    if (IsPrimitiveCNode(input, prim::kPrimDepend) && HasRecomputedInput(input->cast<CNodePtr>())) {
      return true;
    }
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

    auto cnode = dyn_cast_ptr<CNode>(input_k_fg_caller);
    if (cnode == nullptr) {
      continue;
    }
    auto call_fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
    // The output of recomputed cell would not be recomputed.
    if (call_fg != nullptr && call_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH) && !IsRecomputeCell(call_fg)) {
      return true;
    }
  }
  return false;
}

bool IsForwardGetterTupleGetItem(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return false;
  }
  auto idx = GetValueNode<Int64ImmPtr>(node->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
  if (idx != nullptr && idx->value() == 0) {
    return true;
  }
  return false;
}

AnfNodePtr GetForwardGetter(const FuncGraphManagerPtr &manager, const CNodePtr &node) {
  const auto &user_nodes = manager->node_users()[node];
  auto iter = std::find_if(user_nodes.begin(), user_nodes.end(), [](const auto &node_and_idx) -> bool {
    return IsForwardGetterTupleGetItem(node_and_idx.first);
  });
  if (iter != user_nodes.end()) {
    return iter->first;
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
  if (IsOneOfPrimitiveCNode(node, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple, prim::kPrimDepend})) {
    const auto &user_nodes = manager->node_users()[node];
    return std::any_of(user_nodes.begin(), user_nodes.end(),
                       [&manager](const auto &iter) { return HasRecomputedOutput(manager, iter.first); });
  }
  return IsRecomputeKGraphCaller(node);
}

void GetGradUsers(const FuncGraphManagerPtr &manager, const CNodePtr &node, const CNodePtr &pre_node,
                  std::vector<AnfNodePtr> *grad_users) {
  // The forward part may have multiple outputs.
  if (IsOneOfPrimitiveCNode(node, {prim::kPrimTupleGetItem, prim::kPrimDepend})) {
    const auto &user_nodes = manager->node_users()[node];
    for (const auto &iter : user_nodes) {
      GetGradUsers(manager, iter.first->cast<CNodePtr>(), node, grad_users);
    }
    return;
  }
  if (IsGradNode(node)) {
    const auto &inputs = node->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (inputs[i] != pre_node && !inputs[i]->isa<ValueNode>() && IsGradNode(inputs[i])) {
        (void)grad_users->emplace_back(inputs[i]);
      }
    }
  }
}

bool IsFromForwardGetter(const AnfNodePtr &forward_getter, const AnfNodePtr &depend_node) {
  if (forward_getter == depend_node) {
    return true;
  }
  if (!IsOneOfPrimitiveCNode(depend_node, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple, prim::kPrimZerosLike})) {
    return false;
  }
  const auto &depend_node_inputs = depend_node->cast<CNodePtr>()->inputs();
  return std::any_of(depend_node_inputs.begin(), depend_node_inputs.end(),
                     [&forward_getter](const auto &input) { return IsFromForwardGetter(forward_getter, input); });
}

void GetDependencies(const FuncGraphManagerPtr &manager, const CNodePtr &k_fg_caller,
                     mindspore::CompactSet<CNodePtr> *final_nodes, mindspore::CompactSet<AnfNodePtr> *dependencies) {
  if (final_nodes->find(k_fg_caller) != final_nodes->end()) {
    return;
  }
  bool is_recompute_k_fg_caller = IsRecomputeKGraphCaller(k_fg_caller);
  // We only handle the recomputed k graph caller.
  if (!is_recompute_k_fg_caller &&
      !IsOneOfPrimitiveCNode(k_fg_caller, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple, prim::kPrimDepend})) {
    return;
  }
  if (is_recompute_k_fg_caller) {
    auto forward_getter = GetForwardGetter(manager, k_fg_caller);
    // If the k graph caller has no forward getter, it should not output to any other recomputed nodes.
    if (forward_getter == nullptr) {
      auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, k_fg_caller));
      // Add the dout input of its bprop function to the dependencies.
      if (bprop_caller == nullptr) {
        return;
      }
      (void)final_nodes->insert(k_fg_caller);
      (void)dependencies->insert(bprop_caller->cast<CNodePtr>()->input(1));
      return;
    }
    if (!HasRecomputedOutput(manager, forward_getter)) {
      std::vector<AnfNodePtr> grad_users;
      // Add the other inputs of the grad node to the dependencies.
      GetGradUsers(manager, forward_getter->cast<CNodePtr>(), k_fg_caller, &grad_users);
      if (!grad_users.empty()) {
        for (auto &user : grad_users) {
          (void)final_nodes->insert(k_fg_caller);
          (void)dependencies->insert(user);
        }
        return;
      }
      // Add the dout input of its bprop function to the dependencies.
      auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, k_fg_caller));
      if (bprop_caller == nullptr) {
        return;
      }
      (void)final_nodes->insert(k_fg_caller);
      auto dout = bprop_caller->cast<CNodePtr>()->input(1);
      if (IsPrimitiveCNode(dout, prim::kPrimMakeTuple) && IsFromForwardGetter(forward_getter, dout)) {
        return;
      }
      (void)dependencies->insert(dout);
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
    GetDependencies(manager, iter.first->cast<CNodePtr>(), final_nodes, dependencies);
  }
}

void CopyOriginalInputs(const FuncGraphPtr &bprop_fg, const CNodePtr &node, const AnfNodePtr &depend_nodes,
                        std::vector<AnfNodePtr> *new_inputs) {
  (void)std::transform(
    node->inputs().begin(), node->inputs().end(), std::back_inserter(*new_inputs),
    [&bprop_fg](const AnfNodePtr &input) -> AnfNodePtr {
      // Make sure there is only one u monad fv.
      if (HasAbstractUMonad(input) && input->func_graph() != nullptr && input->func_graph() != bprop_fg) {
        return NewValueNode(kUMonad);
      }
      return input;
    });
  // The recomputed cell should insert depend node at all inputs.
  if (!IsRecomputeCell(GetValueNode<FuncGraphPtr>(node->input(0)))) {
    auto depend = bprop_fg->NewCNode({NewValueNode(prim::kPrimDepend), (*new_inputs)[1], depend_nodes});
    depend->AddAttr(kRecomputeInsert, MakeValue(true));
    (*new_inputs)[1] = depend;
  }
}

CNodePtr MoveKCallerToBprop(const FuncGraphManagerPtr &manager, const FuncGraphPtr &bprop_fg, const CNodePtr &node,
                            const AnfNodePtr &depend_nodes,
                            std::unordered_map<CNodePtr, CNodePtr> *origin_to_new_nodes) {
  auto iter = origin_to_new_nodes->find(node);
  if (iter != origin_to_new_nodes->end()) {
    return iter->second;
  }
  std::vector<AnfNodePtr> new_inputs;
  if (IsRecomputeKGraphCaller(node)) {
    if (!node->HasAttr(kAttrReplacedWithPrimal)) {
      return node;
    }
    if (!HasRecomputedInput(node)) {
      CopyOriginalInputs(bprop_fg, node, depend_nodes, &new_inputs);
    } else {
      for (auto &input : node->inputs()) {
        if (!input->isa<CNode>()) {
          (void)new_inputs.emplace_back(input);
          continue;
        }
        (void)new_inputs.emplace_back(
          MoveKCallerToBprop(manager, bprop_fg, input->cast<CNodePtr>(), depend_nodes, origin_to_new_nodes));
      }
    }
    if (IsRecomputeCell(GetValueNode<FuncGraphPtr>(node->input(0)))) {
      // Add the dout input of its bprop function to the dependencies.
      auto new_depend_nodes = depend_nodes;
      auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, node));
      if (bprop_caller != nullptr) {
        std::vector<AnfNodePtr> new_depend_nodes_inputs;
        (void)std::copy(depend_nodes->cast<CNodePtr>()->inputs().begin(),
                        depend_nodes->cast<CNodePtr>()->inputs().end(), std::back_inserter(new_depend_nodes_inputs));
        (void)new_depend_nodes_inputs.emplace_back(bprop_caller->cast<CNodePtr>()->input(1));
        new_depend_nodes = bprop_fg->NewCNode(new_depend_nodes_inputs);
      }
      for (size_t i = 1; i < new_inputs.size(); ++i) {
        auto depend = bprop_fg->NewCNode({NewValueNode(prim::kPrimDepend), new_inputs[i], new_depend_nodes});
        depend->AddAttr(kRecomputeInsert, MakeValue(true));
        new_inputs[i] = depend;
      }
    }
    auto new_k_fg_caller = bprop_fg->NewCNode(new_inputs);
    new_k_fg_caller->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    new_k_fg_caller->AddAttr(kAttrReplacedWithPrimal, MakeValue(true));
    auto primal_fg_caller = node->user_data<CNode>(kPrimalFgCallerUserDataKey);
    if (primal_fg_caller != nullptr) {
      new_k_fg_caller->set_user_data(kPrimalFgCallerUserDataKey, primal_fg_caller);
    }
    // Replace the bprop getter with the new k graph caller in bprop graph.
    auto origin_bprop_getter = GetBpropGetter(manager, node);
    if (origin_bprop_getter != nullptr) {
      auto new_bprop_getter = bprop_fg->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), new_k_fg_caller, NewValueNode(static_cast<int64_t>(1))});
      new_bprop_getter->set_abstract(origin_bprop_getter->abstract());
      (void)manager->Replace(origin_bprop_getter, new_bprop_getter);
    }
    (void)origin_to_new_nodes->emplace(node, new_k_fg_caller);
    return new_k_fg_caller;
  }
  // If it is not tuple_getitem, it should be node which is not set recomputed.
  if (!IsOneOfPrimitiveCNode(
        node, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple, prim::kPrimDepend, prim::kPrimUpdateState})) {
    return node;
  }
  // If the other branch has not been handle, it should not create new forward getter.
  if (IsForwardGetterTupleGetItem(node)) {
    auto real_node = node->cast<CNodePtr>()->input(1);
    if (IsRecomputeKGraphCaller(real_node) && !real_node->cast<CNodePtr>()->HasAttr(kAttrReplacedWithPrimal)) {
      return node;
    }
  }
  for (auto &input : node->inputs()) {
    if (!input->isa<CNode>()) {
      (void)new_inputs.emplace_back(input);
      continue;
    }
    (void)new_inputs.emplace_back(
      MoveKCallerToBprop(manager, bprop_fg, input->cast<CNodePtr>(), depend_nodes, origin_to_new_nodes));
  }
  auto new_node = bprop_fg->NewCNode(new_inputs);
  (void)origin_to_new_nodes->emplace(node, new_node);
  return new_node;
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

void ReplaceFinalForwardGetter(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg,
                               const AnfNodePtr &origin_forward_getter, const AnfNodePtr &new_forward_getter) {
  auto node_users = manager->node_users()[origin_forward_getter];
  for (auto &node_and_idx : node_users) {
    auto user = node_and_idx.first;
    MS_EXCEPTION_IF_NULL(user);
    MS_LOG(DEBUG) << "User: " << user->DebugString();
    // The forward part may have multiple outputs.
    if (IsPrimitiveCNode(user, prim::kPrimTupleGetItem)) {
      // Make new tuple_getitem to get corresponding output.
      auto new_getitem = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), new_forward_getter,
                                       user->cast_ptr<CNode>()->input(kInputNodeOutputIndexInTupleGetItem)});
      ReplaceFinalForwardGetter(manager, fg, user, new_getitem);
      continue;
    }
    if (IsPrimitiveCNode(user, prim::kPrimDepend)) {
      // Make new depend to get corresponding output.
      auto new_depend = fg->NewCNode(user->cast_ptr<CNode>()->inputs());
      new_depend->set_input(IntToSize(node_and_idx.second), new_forward_getter);
      ReplaceFinalForwardGetter(manager, fg, user, new_depend);
      continue;
    }
    MS_LOG(DEBUG) << "Set edge for user: " << user->DebugString();
    manager->SetEdge(user, node_and_idx.second, new_forward_getter);
  }
}

void GetAllRecomputeKFgCallers(const CNodePtr &final_node, mindspore::HashSet<CNodePtr> *recompute_k_fg_callers) {
  for (const auto &input : final_node->inputs()) {
    if (!input->isa<CNode>()) {
      continue;
    }
    auto input_cnode = input->cast<CNodePtr>();
    if (IsPrimitiveCNode(input_cnode, prim::kPrimTupleGetItem)) {
      GetAllRecomputeKFgCallers(input_cnode, recompute_k_fg_callers);
      continue;
    }
    // Only get the nodes visited in this round.
    if (!input_cnode->HasAttr(kAttrReplacedWithPrimal) || !IsRecomputeKGraphCaller(input) ||
        recompute_k_fg_callers->find(input_cnode) != recompute_k_fg_callers->end()) {
      continue;
    }
    (void)recompute_k_fg_callers->insert(input_cnode);
    GetAllRecomputeKFgCallers(input_cnode, recompute_k_fg_callers);
  }
}

bool IsFromRecomputeKFgCaller(const FuncGraphPtr &bprop_fg, const mindspore::HashSet<CNodePtr> &recompute_k_fg_callers,
                              const CNodePtr &node, mindspore::HashMap<CNodePtr, bool> *is_from_recompute_k_fg_caller) {
  auto iter = is_from_recompute_k_fg_caller->find(node);
  if (iter != is_from_recompute_k_fg_caller->end()) {
    return iter->second;
  }
  if (recompute_k_fg_callers.find(node) != recompute_k_fg_callers.end()) {
    (void)is_from_recompute_k_fg_caller->emplace(node, true);
    return true;
  }

  for (const auto &input : node->inputs()) {
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>()) {
      continue;
    }
    auto input_cnode = input->cast<CNodePtr>();
    if (input_cnode->func_graph() != bprop_fg) {
      AnfNodePtr cur_node = input_cnode;
      while (IsPrimitiveCNode(cur_node, prim::kPrimTupleGetItem)) {
        cur_node = cur_node->cast<CNodePtr>()->input(1);
      }
      if (cur_node->isa<CNode>() &&
          recompute_k_fg_callers.find(cur_node->cast<CNodePtr>()) != recompute_k_fg_callers.end()) {
        (void)is_from_recompute_k_fg_caller->emplace(node, true);
        return true;
      }
      continue;
    }
    if (IsFromRecomputeKFgCaller(bprop_fg, recompute_k_fg_callers, input_cnode, is_from_recompute_k_fg_caller)) {
      (void)is_from_recompute_k_fg_caller->emplace(node, true);
      return true;
    }
  }
  (void)is_from_recompute_k_fg_caller->emplace(node, false);
  return false;
}

void AddDependNodes(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const CNodePtr &k_fg_caller_cnode) {
  // Get the nodes which the recomputed part should depend on;
  mindspore::CompactSet<CNodePtr> final_nodes;
  mindspore::CompactSet<AnfNodePtr> dependencies;
  GetDependencies(manager, k_fg_caller_cnode, &final_nodes, &dependencies);
  if (dependencies.empty()) {
    return;
  }
  FuncGraphPtr bprop_fg;
  auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, k_fg_caller_cnode));
  if (bprop_caller == nullptr) {
    bprop_fg = (*dependencies.begin())->func_graph();
  } else {
    bprop_fg = bprop_caller->func_graph();
  }
  MS_EXCEPTION_IF_NULL(bprop_fg);
  // Filter the dependent nodes in case of producing loops.
  mindspore::HashSet<CNodePtr> recompute_k_fg_callers;
  for (const auto &final_node : final_nodes) {
    (void)recompute_k_fg_callers.insert(final_node);
    GetAllRecomputeKFgCallers(final_node, &recompute_k_fg_callers);
  }
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimMakeTuple)};
  mindspore::HashMap<CNodePtr, bool> is_from_recompute_k_fg_caller;
  (void)std::copy_if(dependencies.begin(), dependencies.end(), std::back_inserter(depend_inputs),
                     [bprop_fg, &recompute_k_fg_callers, &is_from_recompute_k_fg_caller](const AnfNodePtr &dependency) {
                       if (!dependency->isa<CNode>()) {
                         return true;
                       }
                       return !IsFromRecomputeKFgCaller(bprop_fg, recompute_k_fg_callers, dependency->cast<CNodePtr>(),
                                                        &is_from_recompute_k_fg_caller);
                     });
  // Add the dependency nodes to the first recomputed nodes.
  auto depend_nodes = bprop_fg->NewCNode(depend_inputs);
  if (bprop_fg == fg) {
    if (!IsRecomputeCell(GetValueNode<FuncGraphPtr>(k_fg_caller_cnode->input(0)))) {
      auto depend = fg->NewCNode({NewValueNode(prim::kPrimDepend), k_fg_caller_cnode->input(1), depend_nodes});
      depend->AddAttr(kRecomputeInsert, MakeValue(true));
      manager->SetEdge(k_fg_caller_cnode, 1, depend);
      k_fg_caller_cnode->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    } else {
      std::vector<AnfNodePtr> new_k_fg_caller_inputs{k_fg_caller_cnode->input(0)};
      (void)std::transform(k_fg_caller_cnode->inputs().begin() + 1, k_fg_caller_cnode->inputs().end(),
                           std::back_inserter(new_k_fg_caller_inputs),
                           [&fg, &depend_nodes](const AnfNodePtr &input) -> AnfNodePtr {
                             auto depend = fg->NewCNodeInOrder({NewValueNode(prim::kPrimDepend), input, depend_nodes});
                             depend->AddAttr(kRecomputeInsert, MakeValue(true));
                             return depend;
                           });
      auto new_k_fg_caller = fg->NewCNodeInOrder(new_k_fg_caller_inputs);
      auto primal_fg_caller = k_fg_caller_cnode->user_data<CNode>(kPrimalFgCallerUserDataKey);
      if (primal_fg_caller != nullptr) {
        new_k_fg_caller->set_user_data(kPrimalFgCallerUserDataKey, primal_fg_caller);
      }
      (void)manager->Replace(k_fg_caller_cnode, new_k_fg_caller);
      new_k_fg_caller->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
      new_k_fg_caller->AddAttr(kAttrReplacedWithPrimal, MakeValue(true));
    }
    return;
  }
  // If the graph of the bprop caller is not the same as the graph of k graph caller, we should move the k graph
  // caller to the graph of the bprop.
  std::unordered_map<CNodePtr, CNodePtr> origin_to_new_nodes;
  for (const auto &final_node : final_nodes) {
    auto new_k_fg_caller = MoveKCallerToBprop(manager, bprop_fg, final_node, depend_nodes, &origin_to_new_nodes);
    new_k_fg_caller->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
  }
  for (auto &iter : origin_to_new_nodes) {
    if (!IsRecomputeKGraphCaller(iter.first)) {
      continue;
    }
    auto forward_getter = GetForwardGetter(manager, iter.first);
    if (forward_getter == nullptr) {
      (void)manager->Replace(iter.first, iter.second);
    } else {
      auto new_forward_getter =
        bprop_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), iter.second, NewValueNode(static_cast<int64_t>(0))});
      ReplaceFinalForwardGetter(manager, bprop_fg, forward_getter, new_forward_getter);
    }
  }
}

void AddDuplicatedAttr(const FuncGraphPtr &k_fg) {
  for (const auto &node : k_fg->nodes()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    node->cast_ptr<CNode>()->AddAttr(kAttrDuplicated, MakeValue(true));
  }
}

void AddCseAttr(const FuncGraphPtr &root, bool changed) {
  if (!changed) {
    return;
  }
  auto all_node = TopoSort(root->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_node) {
    if (WithRecomputedScope(node)) {
      node->cast<CNodePtr>()->AddAttr(kAttrNeedCseAfterRecompute, MakeValue(true));
    }
  }
}

AnfNodePtr GetPrimal(const FuncGraphPtr &k_fg, bool *recompute_cell) {
  auto primal_iter = k_fg->transforms().find("primal");
  if (primal_iter == k_fg->transforms().end()) {
    return nullptr;
  }
  AnfNodePtr primal = nullptr;
  auto primal_fg = primal_iter->second.func_graph();
  if (primal_fg != nullptr) {
    primal = NewValueNode(primal_fg);
    *recompute_cell = true;
  } else {
    auto primal_primitive = primal_iter->second.primitive();
    if (primal_primitive != nullptr) {
      primal = NewValueNode(primal_primitive);
    }
  }
  return primal;
}

bool IsNestedRecomputed(const AnfNodePtr &node) {
  auto fg = node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  return fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH);
}

void SetPrimalAttrs(const CNodePtr &new_primal, const FuncGraphPtr &k_fg) {
  auto forward_in_k_fg = GetPrimalFromFprop(k_fg);
  auto forward_cnode_in_k_fg = dyn_cast<CNode>(forward_in_k_fg);
  if (forward_cnode_in_k_fg != nullptr) {
    new_primal->set_primal_attrs(forward_cnode_in_k_fg->primal_attrs());
  }
}
}  // namespace

bool AddRecomputeNodes(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  if (!EnableCellReuse()) {
    return false;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_save_graphs = context->CanDump(kIntroductory);
  if (enable_save_graphs) {
    DumpIR("before_recompute_root.ir", root);
  }
#endif
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  bool changed = false;
  auto all_node = TopoSort(root->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (auto iter = all_node.crbegin(); iter != all_node.crend(); (void)iter++) {
    const auto &node = *iter;
    if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto k_fg_caller_cnode = GetKGraphCallerFromTupleGetitem(node);
    if (k_fg_caller_cnode == nullptr || k_fg_caller_cnode->HasAttr(kAddedRecomputeDependAttr)) {
      continue;
    }
    auto k_fg = GetValueNode<FuncGraphPtr>(k_fg_caller_cnode->input(0));
    if (k_fg == nullptr || !k_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
      continue;
    }
    if (IsNestedRecomputed(k_fg_caller_cnode)) {
      MS_LOG(WARNING)
        << "The node and its graph both have been set recomputed, the node would not be handled. The node: "
        << k_fg_caller_cnode->DebugString();
      continue;
    }
    bool recompute_cell = false;
    auto primal = GetPrimal(k_fg, &recompute_cell);
    if (primal == nullptr) {
      continue;
    }
    // Replace the forward getter with the origin primal.
    constexpr auto recursive_level = 2;
    MS_LOG(DEBUG) << "Handle recompute k graph forward getter: " << node->DebugString(recursive_level);
    std::vector<AnfNodePtr> inputs{primal};
    (void)inputs.insert(inputs.cend(), k_fg_caller_cnode->inputs().begin() + 1, k_fg_caller_cnode->inputs().end());
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto new_primal = fg->NewCNodeInOrder(inputs);
    if (IsValueNode<Primitive>(primal)) {
      SetPrimalAttrs(new_primal, k_fg);
    }
    std::unordered_map<AnfNodePtr, AnfNodePtr> origin_to_new_primal;
    bool change = AddNewPrimalNode(manager, fg, node, new_primal, recompute_cell, &origin_to_new_primal);
    changed = change || changed;
    if (change && recompute_cell) {
      k_fg_caller_cnode->set_user_data(kPrimalFgCallerUserDataKey, new_primal);
    }
    k_fg_caller_cnode->AddAttr(kAttrReplacedWithPrimal, MakeValue(true));
    // Add duplicated attr to help debugging.
    AddDuplicatedAttr(k_fg);
    if (HasRecomputedInput(k_fg_caller_cnode)) {
      continue;
    }

    MS_LOG(DEBUG) << "Not has recomputed input k_fg_caller_cnode: " << k_fg_caller_cnode->DebugString();
    AddDependNodes(manager, fg, k_fg_caller_cnode);
  }
  AddCseAttr(root, changed);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_recompute_root.ir", root);
  }
#endif
  return changed;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
