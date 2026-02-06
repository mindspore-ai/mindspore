/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/graph_util/parallel_tensordump.h"

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <utility>
#include <string>

#include "base/base.h"
#include "ir/anf.h"
#include "ir/manager.h"
#include "ir/scope.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "primitive/structure_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace parallel {
RedistributionParallelTensorDumpHandler::RedistributionParallelTensorDumpHandler(
  const std::vector<AnfNodePtr> &pre_nodes,
  const std::vector<std::pair<std::pair<AnfNodePtr, int>, std::vector<int>>> &next_nodes,
  const FuncGraphManagerPtr &fg_manager) {
  // vector pre_nodes size is 1
  prenode_redistribution_ = pre_nodes.front();
  nodes_need_redistribution_ = next_nodes;
  for (auto &node_info : nodes_need_redistribution_) {
    auto node = node_info.first.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    size_t pos = IntToSize(node_info.first.second);
    if (pos >= node->inputs().size()) {
      MS_LOG(ERROR) << "Access Index should less than length of node input."
                    << " Length of node input: " << node->inputs().size() << " Access Index: " << pos;
    }
    MS_EXCEPTION_IF_NULL(node->input(pos));
    parent_to_successors_[prenode_redistribution_].push_back(node_info.first);
  }
  fg_manager_ = fg_manager;
}

void RedistributionParallelTensorDumpHandler::HandleDumpAfterRedistributionNode() {
  NodeUsersMap &node_user_map = fg_manager_->node_users();
  for (auto &parent_successors_pair : parent_to_successors_) {
    const AnfNodePtr &parent = parent_successors_pair.first;
    std::vector<std::pair<AnfNodePtr, int>> &successors = parent_successors_pair.second;
    MS_EXCEPTION_IF_NULL(parent);
    const mindspore::CompactSet<std::string> scope_set = GetScopeSetFromNodes(successors);
    std::unordered_set<AnfNodePtr> viewed_dumps;
    std::unordered_set<AnfNodePtr> viewed_bwd_dump_hook;
    for (auto &node_pair : successors) {
      AnfNodePtr node = node_pair.first;
      MS_EXCEPTION_IF_NULL(node);
      FuncGraphPtr func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      FuncGraphManagerPtr manager = func_graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      const AnfNodePtrList path = CollectNodePathBetween(parent, node_pair);
      const AnfNodePtrList collect_dumps = CollectDumpNodesAlongPath(path, manager);
      const AnfNodePtrList collect_bwd_dump_hook = CollectBwdDumpHookAlongPath(path);
      AnfNodePtrList dumps = DoFilterByScopeSet(scope_set, node->scope(), collect_dumps);
      AnfNodePtrList bwd_dump_hooks = DoFilterByScopeSet(scope_set, node->scope(), collect_bwd_dump_hook);
      viewed_dumps.insert(dumps.begin(), dumps.end());
      viewed_bwd_dump_hook.insert(bwd_dump_hooks.begin(), bwd_dump_hooks.end());
      CNodePtr cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      size_t index = IntToSize(node_pair.second);
      if (parent == cnode->input(index)) {
        MS_LOG(INFO) << "No Redistribution Operators Insert.";
      }
      AnfNodePtr last_inserted_redistribution_op = cnode->input(index);
      MS_EXCEPTION_IF_NULL(last_inserted_redistribution_op);
      MS_LOG(INFO) << "Last Insert Redistribution: " << last_inserted_redistribution_op->DebugString();
      (void)MakeOutModeDumpBwdHookAfterRedistribution(bwd_dump_hooks, cnode, index, last_inserted_redistribution_op);
      (void)MakeInModeDumpAfterRedistribution(dumps, cnode, index, last_inserted_redistribution_op, func_graph,
                                              cnode->scope());
    }
    for (const auto &dump_node : viewed_dumps) {
      CNodePtr dump_cnode = dump_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(dump_cnode);
      AnfNodePtr dump_node_parent = dump_cnode->input(kIndex2);
      MS_EXCEPTION_IF_NULL(dump_node_parent);
      auto prim = GetCNodePrimitive(dump_node);
      MS_EXCEPTION_IF_NULL(prim);
      std::string str_input_output = GetDumpInputOutputAttr(dump_node);
      if (str_input_output != IN_MODE) {
        continue;
      }
      (void)fg_manager_->Replace(dump_node, dump_node_parent);
    }
    for (const auto &dump_hook : viewed_bwd_dump_hook) {
      if (!dump_hook->isa<CNode>()) {
        continue;
      }
      CNodePtr hook_cnode = dump_hook->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(hook_cnode);
      AnfNodePtr hook_prenode = hook_cnode->input(kIndex2);
      (void)fg_manager_->Replace(hook_cnode, hook_prenode);
      auto &hook_prenode_users = node_user_map[hook_prenode];
      if (hook_prenode_users.size() == SIZE_ONE) {
        // Eliminate depend which depend(kIndex1) == depend(kIndex2)
        AnfNodePtr depend = hook_prenode_users.front().first;
        if (!depend || !IsSomePrimitive(depend->cast<CNodePtr>(), DEPEND)) {
          continue;
        }
        CNodePtr depend_c = depend->cast<CNodePtr>();
        if (depend_c && depend_c->input(kIndex1) != depend_c->input(kIndex2)) {
          continue;
        }
        (void)fg_manager_->Replace(depend_c, depend_c->input(kIndex1));
      }
    }
  }
}

AnfNodePtrList RedistributionParallelTensorDumpHandler::CollectNodePathBetween(AnfNodePtr start,
                                                                               std::pair<AnfNodePtr, int> end) {
  MS_EXCEPTION_IF_NULL(start);
  AnfNodePtrList path;
  MS_EXCEPTION_IF_NULL(end.first);
  path.push_back(end.first);
  CNodePtr end_cnode = end.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(end_cnode);
  AnfNodePtr cur_node = end_cnode->input(IntToSize(end.second));
  MS_EXCEPTION_IF_NULL(cur_node);
  path.push_back(cur_node);
  if (!cur_node->isa<CNode>()) {
    return path;
  }
  CNodePtr cur_cnode = cur_node->cast<CNodePtr>();
  while (GetCNodePrimitive(cur_cnode) &&
         GetCNodePrimitive(cur_cnode)->instance_name().find(REDISTRIBUTION_OP) != std::string::npos) {
    auto cnode_parent = cur_cnode->input(kIndex1);
    path.push_back(cnode_parent);
    cur_cnode = cnode_parent->cast<CNodePtr>();
  }
  while (IsSomePrimitiveList(cur_cnode, {DEPEND, INSERTGRADIENTOF, CAST, RESHAPE, DUMPGRADIENT})) {
    size_t travel_index = IsSomePrimitive(cur_cnode, DUMPGRADIENT) ? kDumpGradientSkipIndex : kIndex1;
    cur_node = cur_cnode->input(travel_index);
    path.push_back(cur_node);
    cur_cnode = cur_node->cast<CNodePtr>();
  }
  return path;
}

AnfNodePtrList RedistributionParallelTensorDumpHandler::CollectDumpNodesAlongPath(const AnfNodePtrList &path,
                                                                                  const FuncGraphManagerPtr &manager) {
  AnfNodePtrList dumps;
  for (size_t i = 1; i < path.size(); i++) {
    MS_EXCEPTION_IF_NULL(path[i]);
    AnfNodeIndexSet node_users = manager->node_users()[path[i]];
    for (const auto &node_pair : node_users) {
      AnfNodePtr anf_node = node_pair.first;
      CNodePtr cnode = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (IsPrimitiveCNode(cnode, prim::kPrimTensorDump)) {
        dumps.push_back(anf_node);
      }
    }
  }
  return dumps;
}

AnfNodePtrList RedistributionParallelTensorDumpHandler::CollectBwdDumpHookAlongPath(const AnfNodePtrList &path) {
  AnfNodePtrList bwd_dump_hooks;
  std::copy_if(path.begin(), path.end(), std::back_inserter(bwd_dump_hooks), [](const AnfNodePtr &node) -> bool {
    if (!IsSomePrimitive(node->cast<CNodePtr>(), DUMPGRADIENT)) {
      return false;
    }
    if (GetDumpHookInputOutputAttr(node) != OUT_MODE) {
      return false;
    }
    return true;
  });
  return bwd_dump_hooks;
}

mindspore::CompactSet<std::string> RedistributionParallelTensorDumpHandler::GetScopeSetFromNodes(
  const std::vector<std::pair<AnfNodePtr, int>> &nodes) {
  mindspore::CompactSet<std::string> scope_set;
  for (const auto &node_info : nodes) {
    AnfNodePtr node = node_info.first;
    ScopePtr node_scope = node->scope();
    if (node_scope && node_scope->name().size() > 0) {
      scope_set.insert(node_scope->name());
    }
  }
  return scope_set;
}

AnfNodePtrList RedistributionParallelTensorDumpHandler::DoFilterByScopeSet(
  const mindspore::CompactSet<std::string> &scope_set, const ScopePtr &cur_node_scope, const AnfNodePtrList &collects) {
  AnfNodePtrList filtered;
  std::copy_if(collects.begin(), collects.end(), std::back_inserter(filtered),
               [&scope_set, &cur_node_scope](const AnfNodePtr &dump) {
                 const bool is_prefix_in_scope_set =
                   std::any_of(scope_set.begin(), scope_set.end(), [&dump](const std::string &candidate) -> bool {
                     if (dump->scope()) {
                       const std::string dump_scope_name = dump->scope()->name();
                       if (candidate.find(dump_scope_name) != std::string::npos) {
                         return true;
                       }
                     }
                     return false;
                   });
                 if (dump->scope() && cur_node_scope && dump->scope()->name().size() > 0 &&
                     cur_node_scope->name().size() > 0 && is_prefix_in_scope_set &&
                     cur_node_scope->name().find(dump->scope()->name()) == std::string::npos) {
                   return false;
                 }
                 return true;
               });

  return filtered;
}

void RedistributionParallelTensorDumpHandler::InsertNewTensorDump(const CNodePtr &dump_cnode,
                                                                  const AnfNodePtr &last_insert_redistribution_op,
                                                                  const CNodePtr &node, const size_t pos_u,
                                                                  const FuncGraphPtr &func_graph, const ScopePtr &scope,
                                                                  const std::string &dump_mode) {
  // Create New TensorDump Node to display the input of node
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (dump_cnode && dump_cnode->HasPrimalAttr(VISITED_DUMP) &&
      !GetValue<bool>(dump_cnode->GetPrimalAttr(VISITED_DUMP))) {
    return;
  }
  MS_EXCEPTION_IF_NULL(dump_cnode);
  bool is_side_effect_tensordump = dump_cnode->inputs().size() == 4 ? true : false;
  ValuePtr v = GetValueNode(dump_cnode->input(kIndex1));
  const std::string dump_cnode_filepath = GetValue<std::string>(v);
  ValueNodePtr name_value = NewValueNode(dump_cnode_filepath);
  CNodePtr new_dump_node;
  auto dump_cnode_prim = GetCNodePrimitive(dump_cnode);
  MS_EXCEPTION_IF_NULL(dump_cnode_prim);
  auto new_dump_prim = dump_cnode_prim->Clone();
  if (is_side_effect_tensordump) {
    new_dump_node = func_graph->NewCNode(
      {NewValueNode(new_dump_prim), name_value, last_insert_redistribution_op, NewValueNode(kIOMonad)});
  } else {
    new_dump_node = func_graph->NewCNode({NewValueNode(new_dump_prim), name_value, last_insert_redistribution_op});
  }

  // ops of update_state and depend are used to keep sequence
  auto monad_node = NewValueNode(kIOMonad);
  auto new_update_state_node = func_graph->NewCNode({NewValueNode(prim::kPrimUpdateState), monad_node, new_dump_node});
  auto depend_prev = node->input(pos_u);
  auto new_depend_kIndex2_input = is_side_effect_tensordump ? new_update_state_node : new_dump_node;
  auto new_depend_node = func_graph->NewCNode({NewValueNode(prim::kPrimDepend), depend_prev, new_depend_kIndex2_input});
  // config new nodes attributes
  MS_EXCEPTION_IF_NULL(new_dump_prim);
  new_dump_prim->set_instance_name(dump_cnode_prim->instance_name() + "_new_generate");
  new_dump_prim->AddAttr("side_effect_io", MakeValue(is_side_effect_tensordump));
  new_dump_prim->AddAttr("channel_name", MakeValue("ms_tensor_dump"));
  // Avoid CSE optimization
  new_dump_prim->AddAttr("side_effect_hidden", MakeValue(true));
  if (!is_side_effect_tensordump) {
    new_dump_prim->AddAttr("dyn_input_sizes", MakeValue(std::vector<int>{-1, 1}));
  }
  new_dump_node->set_scope(scope);
  new_dump_node->input(0)->set_scope(scope);
  // Add Visited tag for dump cnode.
  new_dump_node->AddPrimalAttr(VISITED_DUMP, MakeValue(true));
  MS_LOG(INFO) << "New Dump Node: " << new_dump_node->DebugString();
  (void)manager->SetEdge(node, pos_u, new_depend_node);
}

void RedistributionParallelTensorDumpHandler::MakeOutModeDumpBwdHookAfterRedistribution(
  const std::vector<AnfNodePtr> &bwd_dump_hooks, const CNodePtr &node, const size_t pos_u,
  const AnfNodePtr &last_insert_op) {
  FuncGraphPtr cur_fg = node->func_graph();
  MS_EXCEPTION_IF_NULL(cur_fg);
  AnfNodePtr pre_node = last_insert_op;
  for (const auto &hook : bwd_dump_hooks) {
    CNodePtr hook_cnode = hook->cast<CNodePtr>();
    if (!IsSomePrimitive(hook_cnode, DUMPGRADIENT)) {
      continue;
    }
    if (hook_cnode->HasPrimalAttr(VISITED_DUMP) && GetValue<bool>(hook_cnode->GetPrimalAttr(VISITED_DUMP))) {
      continue;
    }
    const PrimitivePtr &old_hook_prim = GetCNodePrimitive(hook_cnode);
    MS_EXCEPTION_IF_NULL(old_hook_prim);
    ValuePtr v = GetValueNode(hook_cnode->input(kIndex1));
    const std::string dump_cnode_filepath = GetValue<std::string>(v);
    const std::string bwd_dump_mode = GetValue<std::string>(GetValueNode(hook_cnode->input(kIndex3)));
    ValueNodePtr name_value = NewValueNode(dump_cnode_filepath);
    ValueNodePtr mode_value = NewValueNode(bwd_dump_mode);
    CNodePtr new_hook = cur_fg->NewCNode({NewValueNode(old_hook_prim->Clone()), name_value, pre_node, mode_value});
    new_hook->AddPrimalAttr(VISITED_DUMP, MakeValue(true));
    pre_node = new_hook;
    (void)fg_manager_->SetEdge(node, SizeToInt(pos_u), new_hook);
  }
}

void RedistributionParallelTensorDumpHandler::MakeInModeDumpAfterRedistribution(
  const std::vector<AnfNodePtr> &dumps, const CNodePtr &node, const size_t pos_u, const AnfNodePtr &last_insert_op,
  const FuncGraphPtr &func_graph, const ScopePtr &scope) {
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(scope);
  for (auto &anf_node : dumps) {
    CNodePtr dump_cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(dump_cnode);
    MS_LOG(INFO) << "Current Process Dump Node: " << dump_cnode->DebugString();
    std::string dump_mode = GetDumpInputOutputAttr(dump_cnode);
    if (dump_mode == OUT_MODE) {
      continue;
    } else if (dump_mode == IN_MODE) {
      InsertNewTensorDump(dump_cnode, last_insert_op, node, pos_u, func_graph, scope, "in");
    } else {
      MS_EXCEPTION(ValueError) << "Dump mode of " << dump_mode << "is not supported, only support mode in [out, in]";
    }
  }
}

std::string GetDumpInputOutputAttr(const AnfNodePtr &dump_node) {
  // Get InputOutputAttr from TensorDump cnode->input(kIndex0) (a.k.a PrimitivePtr)
  MS_EXCEPTION_IF_NULL(dump_node);
  auto dump_cnode = dump_node->cast<CNodePtr>();
  if (!IsSomePrimitive(dump_cnode, prim::kPrimTensorDump->name())) {
    MS_EXCEPTION(ValueError) << "When Fetch input_output from primitive attributions, meet not TensorDump.";
  }
  auto prim = GetCNodePrimitive(dump_cnode);
  MS_EXCEPTION_IF_CHECK_FAIL(prim, "Get Primitive from TensorDump failed");
  if (!prim->HasAttr(INPUT_OUTPUT)) {
    MS_EXCEPTION(ValueError) << "TensorDump has no primitive attribute of 'input_output'";
  }
  std::string dump_mode = GetValue<std::string>(prim->GetAttr(INPUT_OUTPUT));
  return dump_mode;
}

std::string GetDumpHookInputOutputAttr(const AnfNodePtr &dump_gradient) {
  // Get InputOutputAttr from PrimFunc_DumpGradient cnode->input(kIndex3) (a.k.a String)
  MS_EXCEPTION_IF_NULL(dump_gradient);
  const CNodePtr dump_gradient_cnode = dump_gradient->cast<CNodePtr>();
  if (!IsSomePrimitive(dump_gradient_cnode, DUMPGRADIENT)) {
    MS_LOG(EXCEPTION) << "Fetch input_output attr failed, not a PrimFunc_DumpGradient Node";
  }
  if (dump_gradient_cnode->size() != SIZE_FOUR) {
    MS_LOG(EXCEPTION) << "For PrimFunc_DumpGradient, input size should be 4, but got: " << dump_gradient_cnode->size();
  }
  const ValuePtr input_output_value = GetValueNode(dump_gradient_cnode->input(kIndex3));
  std::string res = GetValue<std::string>(input_output_value);
  if (res != IN_MODE && res != OUT_MODE) {
    MS_EXCEPTION(ValueError) << "For PrimFunc_DumpGradient 3rd input, only support value in ['in', 'out'], "
                             << "but got: " << res;
  }
  return res;
}

std::string GetInModeSuffixedDumpPath(const std::string &ori_path) {
  std::string modified_path = ori_path;
  const size_t p = modified_path.rfind(".npy");
  if (p != std::string::npos) {
    modified_path.erase(p);
  }
  std::ostringstream oss;
  oss << modified_path << "_in.npy";
  return oss.str();
}

void FwdCommunicationParallelTensorDumpHandler::CollectDumpNodesRecursively(const AnfNodePtr &start,
                                                                            const bool first_recursive,
                                                                            std::set<AnfNodePtr> *collect_visited) {
  MS_EXCEPTION_IF_NULL(start);
  if (first_recursive) {
    collect_visited->clear();
  }
  if (collect_visited->count(start) != 0) return;
  if (!first_recursive &&
      !IsSomePrimitiveList(start->cast<CNodePtr>(), {DEPEND, INSERTGRADIENTOF, DUMPGRADIENT, RESHAPE}))
    return;
  collect_visited->insert(start);
  FuncGraphPtr func_graph = start->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr fg_manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(fg_manager);
  auto &node_users_map = fg_manager->node_users();
  auto users = node_users_map[start];
  for (const auto &user_pair : users) {
    auto user_node = user_pair.first;
    if (IsSomePrimitive(user_node->cast<CNodePtr>(), TENSORDUMP)) {
      dump_nodes_.push_back(user_node);
    } else if (IsSomePrimitive(user_node->cast<CNodePtr>(), DUMPGRADIENT)) {
      bwd_dump_hooks_.push_back(user_node);
    }
    CollectDumpNodesRecursively(user_node, false, collect_visited);
  }
}

void FwdCommunicationParallelTensorDumpHandler::CollectDumpNodes(const AnfNodePtr &anchor, const bool is_multi_output) {
  if (!anchor) {
    return;
  }
  const FuncGraphPtr &func_graph = anchor->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const FuncGraphManagerPtr &fg_manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(fg_manager);
  AnfNodePtr real_search_start;
  if (is_multi_output) {
    if (!IsPrimitiveCNode(anchor, prim::kPrimTupleGetItem)) {
      MS_LOG(EXCEPTION)
        << "When forward communication op has multioutputs, dump nodes searching should start with a TupleGetItem.";
    }
    const CNodePtr &tuple_get_item = anchor->cast<CNodePtr>();
    const NodeUsersMap &user_map = fg_manager->node_users();
    const auto tuple_get_item_users = user_map.at(tuple_get_item);
    if (tuple_get_item_users.size() != SIZE_ONE ||
        !IsPrimitiveCNode(tuple_get_item_users.front().first, prim::kPrimMakeTuple)) {
      MS_LOG(EXCEPTION)
        << "When forward communication op has multioutputs, TupleGetItem user is expected to be a MakeTuple Node.";
    }
    const CNodePtr &make_tuple = tuple_get_item_users.front().first->cast<CNodePtr>();
    const int index_in_tuple = tuple_get_item_users.front().second;
    const auto make_tuple_users = user_map.at(make_tuple);
    const auto target_user_pair =
      std::find_if(make_tuple_users.begin(), make_tuple_users.end(), [index_in_tuple](auto mktuple_user_pair) {
        const AnfNodePtr &user_node = mktuple_user_pair.first;
        if (IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem) &&
            GetTupleGetItemIndex(user_node->cast<CNodePtr>()) + 1 == index_in_tuple) {
          return true;
        }
        return false;
      });
    if (target_user_pair != make_tuple_users.end()) {
      real_search_start = target_user_pair->first;
    }
  } else {
    real_search_start = anchor;
  }
  // Start collect dump nodes recursively
  if (real_search_start) {
    std::set<AnfNodePtr> collect_visited;
    (void)CollectDumpNodesRecursively(real_search_start, true, &collect_visited);
  }
}

void FwdCommunicationParallelTensorDumpHandler::MakeOutModeDumpBeforeFwdComm() {
  AnfNodePtr node_to_insert = prior_;
  for (const auto &dump : dump_nodes_) {
    FuncGraphPtr func_graph = dump->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    FuncGraphManagerPtr fg_manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(fg_manager);
    CNodePtr dump_cnode = dump->cast<CNodePtr>();
    if (!IsSomePrimitive(dump_cnode, TENSORDUMP)) {
      continue;
    }
    auto prim = GetCNodePrimitive(dump_cnode);
    MS_EXCEPTION_IF_NULL(prim);
    const std::string dump_mode = GetValue<std::string>(prim->GetAttr(INPUT_OUTPUT));
    if (dump_mode != OUT_MODE) {
      continue;
    }
    (void)fg_manager->SetEdge(dump, kIndex2, node_to_insert);
  }
}

void FwdCommunicationParallelTensorDumpHandler::MakeInModeBwdHookBeforeFwdComm() {
  AnfNodePtr node_to_insert = prior_;
  for (const auto &bwd_dump_hook : bwd_dump_hooks_) {
    CNodePtr isg_cnode = bwd_dump_hook->cast<CNodePtr>();
    if (!IsSomePrimitive(isg_cnode, DUMPGRADIENT)) {
      continue;
    }
    FuncGraphPtr func_graph = bwd_dump_hook->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    FuncGraphManagerPtr fg_manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(fg_manager);
    const std::string hook_mode = GetDumpHookInputOutputAttr(isg_cnode);
    if (hook_mode != IN_MODE) {
      continue;
    }
    const std::string ori_path = GetValue<std::string>(GetValueNode(isg_cnode->input(kIndex1)));
    AnfNodePtr isg_input = isg_cnode->input(kIndex2);
    (void)fg_manager->Replace(isg_cnode, isg_input);
    (void)fg_manager->Replace(node_to_insert, isg_cnode);
    (void)fg_manager->SetEdge(isg_cnode, kIndex2, node_to_insert);
    node_to_insert = bwd_dump_hook;
  }
}
}  // namespace parallel
}  // namespace mindspore
