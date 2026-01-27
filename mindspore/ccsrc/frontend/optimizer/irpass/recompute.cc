/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include <utility>
#include <vector>
#include <algorithm>
#include <string>
#include "primitive/array_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"
#include "primitive/auto_generate/gen_ops_primitive_z.h"

/**
 * @file recompute.cc
 * @brief Recomputation optimization pass implementation
 *
 * Recomputation is a memory optimization technique to reduce memory usage during training.
 * The core idea is: recompute the forward intermediate results during backward propagation,
 * instead of saving these results during forward propagation.
 *
 * Key concepts:
 * - K-Graph: A composite graph containing forward computation and backward propagation function,
 *            with output structure (forward_result, bprop_fn)
 * - Primal: The original forward computation function or operator
 * - Forward Getter: TupleGetItem node (index=0) that retrieves forward result from K-graph
 * - Bprop Getter: TupleGetItem node (index=1) that retrieves backward function from K-graph
 *
 * Workflow:
 * 1. Traverse all K-graph caller nodes marked for recomputation
 * 2. Replace forward result usage with original Primal to avoid saving intermediate results
 * 3. Re-execute K-graph call during backward propagation to achieve recomputation
 * 4. Add appropriate dependency edges to ensure recomputation executes at the correct timing
 */
namespace mindspore {
namespace opt {
namespace irpass {

/**
 * @brief Check if a K-graph caller node has a bprop getter
 * @param opt Optimizer pointer for accessing FuncGraphManager
 * @param k_fg_caller K-graph caller node
 * @return True if there exists a TupleGetItem (index=1) that retrieves the backward function
 *
 * K-graph output structure is (forward_result, bprop_fn), index=1 corresponds to bprop function
 */
bool HasBpropGetter(const OptimizerPtr &opt, const AnfNodePtr &k_fg_caller) {
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_users = manager->node_users();
  auto iter = node_users.find(k_fg_caller);
  if (iter == node_users.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, k_fg_caller) << "The node " << k_fg_caller->DebugString() << " should have users.";
  }

  return std::any_of(iter->second.begin(), iter->second.end(), [](const std::pair<AnfNodePtr, int> &node_and_idx) {
    auto user = node_and_idx.first;
    return IsPrimitiveCNode(user, prim::kPrimTupleGetItem) &&
           common::AnfAlgo::GetTupleGetItemOutIndex(user->cast<CNodePtr>()) == 1;
  });
}

namespace {
// Scope prefix for gradient computation nodes
constexpr auto kGradientsFlag = "Gradients";
// Attribute marking that node has been replaced with Primal
constexpr auto kAttrReplacedWithPrimal = "replaced_with_primal";
// Attribute marking MakeTuple nodes created during recomputation
constexpr auto kAttrRecomputeMakeTuple = "recompute_make_tuple";

void DumpRecomputeIR([[maybe_unused]] const std::string &filename, [[maybe_unused]] const FuncGraphPtr &func_graph) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR(filename, func_graph);
  }
#endif
}

/**
 * @brief Check if node is within Cell-level recomputation scope
 * @param node Node to check
 * @return True if node's fullname_with_scope starts with "recompute"
 */
bool WithRecomputedScope(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &full_name_with_scope = node->fullname_with_scope();
  return full_name_with_scope.compare(0, strlen(kAttrRecompute), kAttrRecompute) == 0;
}

/**
 * @brief Check if node is a recompute K-graph caller
 * @param node Node to check
 * @return True if the called FuncGraph has FUNC_GRAPH_RECOMPUTE_K_GRAPH flag
 */
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

/**
 * @brief Check if node is within gradient computation scope
 * @param node Node to check
 * @return True if node's fullname_with_scope starts with "Gradients"
 */
bool WithGradientScope(const AnfNodePtr &node) {
  return node->fullname_with_scope().compare(0, strlen(kGradientsFlag), kGradientsFlag) == 0;
}

/**
 * @brief Check if node originates from backward propagation output
 * @param node Node to check
 * @return True if node can be traced back to bprop function output
 *
 * Traces through TupleGetItem chain to check if the origin is a gradient graph
 */
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

/**
 * @brief Check if node is a gradient computation node
 * @param node Node to check
 * @return True if node is within gradient scope or originates from bprop output
 */
bool IsGradNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return WithGradientScope(node) || IsFromBpropOutput(node);
}

/**
 * @brief Check if MakeTuple node is a K-graph return value
 * @param make_tuple MakeTuple node
 * @return True if it matches K-graph return structure (forward_result, bprop_fn)
 *
 * K-graph output is a tuple containing 2 elements: forward result and bprop function
 */
bool IsFpropReturn(const AnfNodePtr &make_tuple) {
  auto cnode = make_tuple->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  constexpr size_t fprop_output_size = 2;
  if (cnode->size() != fprop_output_size + 1) {
    return false;
  }
  return IsValueNode<FuncGraph>(cnode->input(fprop_output_size));
}

/**
 * @brief Get Primal (forward computation result) from K-graph
 * @param k_fg K-graph
 * @return Forward computation result node, or nullptr if not exists
 *
 * K-graph output structure: MakeTuple(primal_result, bprop_fn)
 */
AnfNodePtr GetPrimalFromFprop(const FuncGraphPtr &k_fg) {
  if (!IsPrimitiveCNode(k_fg->output(), prim::kPrimMakeTuple)) {
    return nullptr;
  }
  auto k_fg_cnode = k_fg->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(k_fg_cnode);
  auto k_fg_outputs = k_fg_cnode->inputs();
  if (k_fg_outputs.size() != 3) {
    return nullptr;
  }
  return k_fg_outputs[kIndex1];
}

/**
 * @brief Determine if a new Primal output should be added for the node
 * @param node Node to check
 * @return True if node is not a gradient node
 *
 * Only nodes in forward propagation need to be replaced with new Primal output
 */
bool ShouldAddNewPrimalOutput(const AnfNodePtr &node) { return !IsGradNode(node); }

/**
 * @brief Check if node is a forward propagation Depend node
 * @param node Node to check
 * @return True if it's Depend without kRecomputeInsert attribute
 *
 * Depend nodes inserted by recomputation have kRecomputeInsert attribute
 */
bool IsForwardDepend(const AnfNodePtr &node) {
  return IsPrimitiveCNode(node, prim::kPrimDepend) && !node->cast_ptr<CNode>()->HasAttr(kRecomputeInsert);
}

/**
 * @brief Add new Primal nodes for users of the original Primal node
 * @param manager FuncGraph manager
 * @param fg Current FuncGraph
 * @param origin_primal Original Primal node (usually TupleGetItem)
 * @param new_primal Newly created Primal node
 * @param not_recompute_cell_output Whether to not recompute Cell output
 * @param origin_to_new_primal Mapping from original nodes to new nodes
 * @return True if any edge was modified
 *
 * This function recursively processes all users of the original Primal node:
 * - TupleGetItem: Create new TupleGetItem to get corresponding output
 * - Depend: Create new Depend node
 * - MakeTuple: Create new MakeTuple node
 * - Others: Replace edge to new Primal only if original Primal is Cell with non-recompute output,
 *           or if it's not a recompute K-graph caller and not a gradient node
 */
bool AddNewPrimalNode(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const AnfNodePtr &origin_primal,
                      const AnfNodePtr &new_primal, bool not_recompute_cell_output,
                      std::unordered_map<AnfNodePtr, AnfNodePtr> *origin_to_new_primal) {
  bool changed = false;
  auto node_users = manager->node_users()[origin_primal];
  for (auto &node_and_idx : node_users) {
    auto user = node_and_idx.first;
    MS_EXCEPTION_IF_NULL(user);
    // The forward part may have multiple outputs.
    if (IsPrimitiveCNode(user, prim::kPrimTupleGetItem) && ShouldAddNewPrimalOutput(user)) {
      // Make new tuple_getitem to get corresponding output.
      auto new_primal_getitem = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), new_primal,
                                              user->cast_ptr<CNode>()->input(kInputNodeOutputIndexInTupleGetItem)});
      changed =
        AddNewPrimalNode(manager, fg, user, new_primal_getitem, not_recompute_cell_output, origin_to_new_primal) ||
        changed;
      continue;
    }
    if (IsForwardDepend(user) && ShouldAddNewPrimalOutput(user)) {
      // Make new depend node in forward to get corresponding output.
      auto new_depend = fg->NewCNode(user->cast_ptr<CNode>()->inputs());
      new_depend->set_input(IntToSize(node_and_idx.second), new_primal);
      changed =
        AddNewPrimalNode(manager, fg, user, new_depend, not_recompute_cell_output, origin_to_new_primal) || changed;
      continue;
    }
    // The op like concat will have a make_tuple input.
    if (IsPrimitiveCNode(user, prim::kPrimMakeTuple) && !IsFpropReturn(user) && ShouldAddNewPrimalOutput(user)) {
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
        AddNewPrimalNode(manager, fg, user, new_primal_make_tuple, not_recompute_cell_output, origin_to_new_primal) ||
        changed;
      continue;
    }

    // Set edge to not recomputed primal nodes.
    if (not_recompute_cell_output || (!IsRecomputeKGraphCaller(user) && !IsGradNode(user))) {
      MS_LOG(DEBUG) << "Set edge to user: " << user->DebugString() << ", new primal: " << new_primal->DebugString();
      manager->SetEdge(user, node_and_idx.second, new_primal);
      changed = true;
    }
  }
  return changed;
}

/**
 * @brief Check if K-graph is a recompute Cell
 * @param k_fg K-graph
 * @param recompute_cell_output [out] Output parameter indicating whether to recompute Cell output
 * @return True if K-graph's Primal is a FuncGraph (not Primitive)
 *
 * Recompute Cell is an entire Cell marked by user via cell.recompute(),
 * as opposed to single operator recomputation
 */
bool IsRecomputeCell(const FuncGraphPtr &k_fg, bool *recompute_cell_output = nullptr) {
  if (k_fg == nullptr) {
    return false;
  }
  auto primal_iter = k_fg->transforms().find("primal");
  if (primal_iter == k_fg->transforms().end()) {
    MS_LOG_WITH_NODE(EXCEPTION, k_fg->return_node()) << "The k_fg: " << k_fg << " should have primal part.";
  }
  auto primal_fg = primal_iter->second.func_graph();
  if (primal_fg == nullptr) {
    return false;
  }
  if (!primal_fg->has_attr(FUNC_GRAPH_FLAG_OUTPUT_RECOMPUTE)) {
    MS_LOG(ERROR) << "The primal graph " << primal_fg->ToString() << " should have output recompute attribute.";
    return false;
  }
  if (recompute_cell_output != nullptr) {
    *recompute_cell_output = GetValue<bool>(primal_fg->get_attr(FUNC_GRAPH_FLAG_OUTPUT_RECOMPUTE));
  }
  return true;
}

/**
 * @brief Check if K-graph caller node has inputs from other recomputed nodes
 * @param k_fg_caller_cnode K-graph caller node
 * @return True if any input comes from a K-graph marked for recomputation
 *
 * Used to determine if current recompute node depends on outputs from other recompute nodes,
 * which affects the dependency edge insertion strategy
 */
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
    if (call_fg == nullptr) {
      continue;
    }
    // Check the input is set recomputed, include recomputed cell with output recomputed.
    auto recompute_cell_output = false;
    auto recompute_cell = IsRecomputeCell(call_fg, &recompute_cell_output);
    if (call_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH) && (!recompute_cell || recompute_cell_output)) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Check if node is a TupleGetItem that retrieves forward result (index=0)
 * @param node Node to check
 * @return True if it's TupleGetItem with index=0
 *
 * K-graph output is (forward_result, bprop_fn), index=0 retrieves forward result
 */
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

/**
 * @brief Get the forward result getter of a K-graph caller node
 * @param manager FuncGraph manager
 * @param node K-graph caller node
 * @return Forward result getter (TupleGetItem, index=0), or nullptr if not exists
 */
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

/**
 * @brief Check if node's output is used by other recomputed nodes
 * @param manager FuncGraph manager
 * @param node Node to check
 * @return True if node's users include recompute K-graph callers
 *
 * Recursively checks through TupleGetItem, MakeTuple, Depend intermediate nodes
 */
bool HasRecomputedOutput(const FuncGraphManagerPtr &manager, const AnfNodePtr &node) {
  // The forward part may have multiple outputs.
  if (IsOneOfPrimitiveCNode(node, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple, prim::kPrimDepend})) {
    const auto &user_nodes = manager->node_users()[node];
    return std::any_of(user_nodes.begin(), user_nodes.end(),
                       [&manager](const auto &iter) { return HasRecomputedOutput(manager, iter.first); });
  }
  return IsRecomputeKGraphCaller(node);
}

/**
 * @brief Get users of the node in gradient computation
 * @param manager FuncGraph manager
 * @param node Current node
 * @param pre_node Previous node (for exclusion)
 * @param grad_users [out] Output parameter storing list of gradient users
 *
 * Collects users of current node in gradient computation for determining recomputation dependencies
 */
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

/**
 * @brief Check if dependency node originates from forward result getter
 * @param forward_getter Forward result getter node
 * @param depend_node Dependency node to check
 * @return True if depend_node directly or indirectly originates from forward_getter
 *
 * Used to avoid creating cycles in dependency chain
 */
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

/**
 * @brief Collect dependencies for recomputation nodes
 * @param manager FuncGraph manager
 * @param k_fg_caller K-graph caller node
 * @param final_nodes [out] Set of final nodes to process
 * @param dependencies [out] Set of dependency nodes
 *
 * Analyzes which nodes the recomputation node needs to depend on for correct execution:
 * - If no forward output is used by other recompute nodes, depends on dout (input gradient of bprop)
 * - If there are gradient users, depends on other inputs of those gradient nodes
 * - Recursively processes TupleGetItem, MakeTuple, Depend intermediate nodes
 */
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

/**
 * @brief Get K-graph caller from TupleGetItem node
 * @param node TupleGetItem node
 * @return The CNode if TupleGetItem's index=0 and input is CNode; otherwise nullptr
 *
 * Used to trace back from forward result getter to K-graph caller node
 */
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

/**
 * @brief Recursively collect all recompute K-graph callers
 * @param final_node Starting node
 * @param recompute_k_fg_callers [out] Output parameter storing all recompute K-graph callers
 *
 * Starting from given node, recursively traverses inputs to collect all K-graph caller nodes
 * marked for recomputation
 */
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

/**
 * @brief Check if node originates from recompute K-graph caller
 * @param bprop_fg Backward propagation FuncGraph
 * @param recompute_k_fg_callers Set of recompute K-graph callers
 * @param node Node to check
 * @param is_from_recompute_k_fg_caller [in/out] Cache map for results
 * @return True if node directly or indirectly originates from recompute K-graph caller
 *
 * Used to filter dependency nodes to avoid creating cyclic dependencies
 */
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

/**
 * @brief Add kAttrDuplicated attribute to all nodes in K-graph
 * @param k_fg K-graph
 *
 * Marks these nodes as duplicated nodes after recomputation, for debugging purposes
 */
void AddDuplicatedAttr(const FuncGraphPtr &k_fg) {
  for (const auto &node : k_fg->nodes()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    node->cast_ptr<CNode>()->AddAttr(kAttrDuplicated, MakeValue(true));
  }
}

/**
 * @brief Add CSE attribute to nodes within recomputation scope
 * @param root Root FuncGraph
 * @param changed Whether any changes occurred
 *
 * Marks nodes that need Common Subexpression Elimination (CSE) after recomputation
 */
void AddCseAttr(const FuncGraphPtr &root, bool changed) {
  if (!changed) {
    return;
  }
  auto all_node = TopoSort(root->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_node) {
    if (WithRecomputedScope(node)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      cnode->AddAttr(kAttrNeedCseAfterRecompute, MakeValue(true));
    }
  }
}

/**
 * @brief Get Primal (original forward computation function or operator) from K-graph
 * @param k_fg K-graph
 * @param recompute_cell [out] Output parameter indicating whether it's a recompute Cell
 * @param recompute_cell_output [out] Output parameter indicating whether to recompute Cell output
 * @return Primal node (ValueNode of FuncGraph or Primitive)
 *
 * K-graph's transforms contains "primal" information:
 * - FuncGraph: Forward subgraph
 * - Primitive: Forward operator
 */
AnfNodePtr GetPrimal(const FuncGraphPtr &k_fg, bool *recompute_cell, bool *recompute_cell_output) {
  auto primal_iter = k_fg->transforms().find("primal");
  if (primal_iter == k_fg->transforms().end()) {
    return nullptr;
  }
  AnfNodePtr primal = nullptr;
  auto primal_fg = primal_iter->second.func_graph();
  if (primal_fg != nullptr) {
    primal = NewValueNode(primal_fg);
    *recompute_cell = true;
    if (primal_fg->has_attr(FUNC_GRAPH_FLAG_OUTPUT_RECOMPUTE)) {
      *recompute_cell_output = GetValue<bool>(primal_fg->get_attr(FUNC_GRAPH_FLAG_OUTPUT_RECOMPUTE));
    } else {
      MS_LOG(ERROR) << "The primal graph " << primal_fg->ToString() << " should have output recompute attribute.";
    }
  } else {
    auto primal_primitive = primal_iter->second.primitive();
    if (primal_primitive != nullptr) {
      primal = NewValueNode(primal_primitive);
    }
  }
  return primal;
}

/**
 * @brief Check if node is nested recomputation
 * @param node Node to check
 * @return True if node's FuncGraph is also marked for recomputation
 *
 * Nested recomputation means the node itself is marked for recomputation,
 * and its containing graph is also marked for recomputation.
 * This case requires special handling
 */
bool IsNestedRecomputed(const AnfNodePtr &node) {
  auto fg = node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  return fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH);
}

/**
 * @brief Copy forward computation attributes from K-graph to new Primal node
 * @param new_primal Newly created Primal CNode
 * @param k_fg K-graph
 *
 * Ensures new Primal node preserves original computation's attribute information
 */
void SetPrimalAttrs(const CNodePtr &new_primal, const FuncGraphPtr &k_fg) {
  auto forward_in_k_fg = GetPrimalFromFprop(k_fg);
  auto forward_cnode_in_k_fg = dyn_cast<CNode>(forward_in_k_fg);
  if (forward_cnode_in_k_fg != nullptr) {
    new_primal->set_primal_attrs(forward_cnode_in_k_fg->primal_attrs());
  }
}

/**
 * @brief Find or register Monad free variable for backward graph
 * @param bprop_fg Backward propagation FuncGraph
 * @param monad Monad node (UMonad or IOMonad)
 * @param bprop_to_monad_fv [in/out] Mapping from bprop graph to Monad free variable
 * @return Corresponding Monad free variable
 *
 * Ensures each backward graph has only one Monad free variable, avoiding duplicate creation
 */
AnfNodePtr FindMonadFv(const FuncGraphPtr &bprop_fg, const AnfNodePtr &monad,
                       mindspore::HashMap<FuncGraphPtr, AnfNodePtr> *bprop_to_monad_fv) {
  auto iter = bprop_to_monad_fv->find(bprop_fg);
  if (iter != bprop_to_monad_fv->end()) {
    return iter->second;
  }
  bprop_to_monad_fv->emplace(bprop_fg, monad);
  return monad;
}
}  // namespace

/**
 * @brief Copy original node's inputs and handle Monad free variables
 * @param bprop_fg Backward propagation FuncGraph
 * @param node Original node
 * @param depend_nodes Dependency nodes (for inserting Depend)
 * @param new_inputs [out] Output parameter storing new input list
 *
 * Copies original node's inputs to new input list, while:
 * - Handling UMonad and IOMonad free variables, ensuring each bprop graph has only one
 * - Inserting Depend node at first input to establish recomputation dependency
 */
void Recomputation::CopyOriginalInputs(const FuncGraphPtr &bprop_fg, const CNodePtr &node,
                                       const AnfNodePtr &depend_nodes, std::vector<AnfNodePtr> *new_inputs) {
  (void)std::transform(node->inputs().begin(), node->inputs().end(), std::back_inserter(*new_inputs),
                       [this, &bprop_fg](const AnfNodePtr &input) -> AnfNodePtr {
                         // Make sure there is only one u monad fv.
                         if (input->func_graph() == nullptr || input->func_graph() == bprop_fg) {
                           return input;
                         }
                         if (HasAbstractUMonad(input)) {
                           return FindMonadFv(bprop_fg, input, &bprop_to_umonad_fv_);
                         } else if (HasAbstractIOMonad(input)) {
                           return FindMonadFv(bprop_fg, input, &bprop_to_iomonad_fv_);
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

/**
 * @brief Move K-graph caller node to backward propagation graph
 * @param manager FuncGraph manager
 * @param bprop_fg Backward propagation FuncGraph
 * @param node Node to move
 * @param depend_nodes Dependency nodes
 * @param origin_to_new_nodes [in/out] Mapping from original nodes to new nodes
 * @return New node created in backward graph
 *
 * When K-graph caller node and bprop caller are not in the same graph,
 * need to move K-graph caller node to backward graph for execution, achieving recomputation.
 *
 * Processing flow:
 * 1. If it's recompute K-graph caller, create new node and add dependencies
 * 2. If it's TupleGetItem/MakeTuple/Depend intermediate node, recursively process its inputs
 * 3. Replace original bprop getter with newly created K-graph caller node
 */
CNodePtr Recomputation::MoveKCallerToBprop(const FuncGraphManagerPtr &manager, const FuncGraphPtr &bprop_fg,
                                           const CNodePtr &node, const AnfNodePtr &depend_nodes,
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
    bool recompute_cell_output = false;
    if (IsRecomputeCell(GetValueNode<FuncGraphPtr>(node->input(0)), &recompute_cell_output)) {
      auto new_depend_nodes = depend_nodes;
      if (!recompute_cell_output) {
        // If do not recompute cell output, add the dout input of its bprop function to the dependencies
        // because the recomputed cell will not output to other recomputed nodes.
        auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, node));
        if (bprop_caller != nullptr) {
          std::vector<AnfNodePtr> new_depend_nodes_inputs;
          auto depend_cnode = depend_nodes->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(depend_cnode);
          (void)std::copy(depend_cnode->inputs().begin(), depend_cnode->inputs().end(),
                          std::back_inserter(new_depend_nodes_inputs));
          auto bprop_caller_cnode = bprop_caller->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(bprop_caller_cnode);
          (void)new_depend_nodes_inputs.emplace_back(bprop_caller_cnode->input(1));
          new_depend_nodes = bprop_fg->NewCNode(new_depend_nodes_inputs);
        }
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

/**
 * @brief Add dependency edges for recomputation nodes
 * @param manager FuncGraph manager
 * @param fg Current FuncGraph
 * @param k_fg_caller_cnode K-graph caller node
 *
 * Core function: Ensures recomputation executes at the correct timing
 *
 * Processing flow:
 * 1. Collect dependencies for recomputation node (GetDependencies)
 * 2. Filter out dependency nodes that could create cycles
 * 3. Create MakeTuple node to wrap all dependencies
 * 4. Attach dependencies to recomputation input via Depend node
 *
 * Two scenarios:
 * - Same graph: Insert Depend directly in current graph
 * - Cross graph: Need to move K-graph caller to backward graph (MoveKCallerToBprop)
 */
void Recomputation::AddDependNodes(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg,
                                   const CNodePtr &k_fg_caller_cnode) {
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

/**
 * @brief Replace the final forward result getter
 * @param manager FuncGraph manager
 * @param fg Current FuncGraph
 * @param origin_forward_getter Original forward result getter
 * @param new_forward_getter New forward result getter
 *
 * Recursively replaces all users of original forward result getter to use the new getter.
 * Handles TupleGetItem, Depend and other intermediate node cases.
 */
void Recomputation::ReplaceFinalForwardGetter(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg,
                                              const AnfNodePtr &origin_forward_getter,
                                              const AnfNodePtr &new_forward_getter) {
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
      auto user_inputs = user->cast_ptr<CNode>()->inputs();
      std::vector<AnfNodePtr> new_inputs;
      (void)std::transform(user_inputs.begin(), user_inputs.end(), std::back_inserter(new_inputs),
                           [this, &fg](const AnfNodePtr &input) -> AnfNodePtr {
                             // Make sure there is only one u monad fv.
                             if (input->func_graph() == nullptr || input->func_graph() == fg) {
                               return input;
                             }
                             if (HasAbstractUMonad(input)) {
                               return FindMonadFv(fg, input, &bprop_to_umonad_fv_);
                             } else if (HasAbstractIOMonad(input)) {
                               return FindMonadFv(fg, input, &bprop_to_iomonad_fv_);
                             }
                             return input;
                           });
      auto new_depend = fg->NewCNode(new_inputs);
      new_depend->set_input(IntToSize(node_and_idx.second), new_forward_getter);
      ReplaceFinalForwardGetter(manager, fg, user, new_depend);
      continue;
    }
    MS_LOG(DEBUG) << "Set edge for user: " << user->DebugString();
    manager->SetEdge(user, node_and_idx.second, new_forward_getter);
  }
}

/**
 * @brief Main entry point for recomputation optimization pass
 * @param root Root FuncGraph
 * @param opt Optimizer
 * @return True if graph was modified
 *
 * Overall processing flow:
 * 1. Check if recomputation should execute before inline
 * 2. Traverse all nodes in reverse topological order, find recompute K-graph forward getters
 *    (TupleGetItem index=0)
 * 3. For each recompute K-graph caller node:
 *    a. Get Primal (retrieve original forward computation from K-graph's transforms["primal"])
 *    b. Create new Primal caller node: new_primal = fg->NewCNodeInOrder({primal, inputs...})
 *    c. Replace non-gradient users of forward result with new Primal (AddNewPrimalNode)
 *       - Gradient computation part still uses original K-graph call, ensuring re-execution
 *         during backward
 *    d. Mark K-graph caller node: kAttrReplacedWithPrimal, to avoid duplicate processing
 *    e. If current node has no inputs from other recompute nodes, add dependency edges
 *       (AddDependNodes)
 *       - Dependency edges ensure recomputation executes at correct timing during backward
 * 4. Add CSE attribute to nodes within recomputation scope for subsequent CSE optimization
 *
 * Key aspects of recompute node generation:
 * - K-graph caller node preserved: Original K-graph caller node is not deleted,
 *   it will re-execute during backward propagation
 * - New Primal node: Creates independent forward computation node for non-gradient users,
 *   avoiding saving intermediate results
 * - Dependency edge insertion: Controls recomputation execution timing via Depend node,
 *   depending on dout (backward gradient input)
 *
 * Memory optimization principle:
 * - Original: Forward propagation saves intermediate results → Backward directly uses them
 * - Recomputation: Forward doesn't save → Backward recomputes via K-graph
 */
bool Recomputation::operator()(const FuncGraphPtr &root, const OptimizerPtr &opt) {
  if (!RecomputeBeforeInline()) {
    return false;
  }
  DumpRecomputeIR("before_recompute_root.ir", root);
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
    bool recompute_cell_output = false;
    auto primal = GetPrimal(k_fg, &recompute_cell, &recompute_cell_output);
    if (primal == nullptr) {
      continue;
    }
    // Replace the forward getter with the origin primal.
    MS_LOG(DEBUG) << "Handle recompute k graph forward getter: "
                  << node->DebugString(AnfNode::DebugStringLevel::kLevel2);
    std::vector<AnfNodePtr> inputs{primal};
    (void)inputs.insert(inputs.cend(), k_fg_caller_cnode->inputs().begin() + 1, k_fg_caller_cnode->inputs().end());
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto new_primal = fg->NewCNodeInOrder(inputs);
    if (IsValueNode<Primitive>(primal)) {
      SetPrimalAttrs(new_primal, k_fg);
    }
    std::unordered_map<AnfNodePtr, AnfNodePtr> origin_to_new_primal;
    bool change =
      AddNewPrimalNode(manager, fg, node, new_primal, recompute_cell && !recompute_cell_output, &origin_to_new_primal);
    changed = change || changed;
    if (change && recompute_cell) {
      static int64_t recomput_subgraph_id = 1;
      k_fg->set_attr(kRecomputeSubgraphIdAttr, MakeValue(recomput_subgraph_id++));
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
  DumpRecomputeIR("after_recompute_root.ir", root);
  return changed;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
