/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/auto_monad_eliminate.h"

#include <algorithm>
#include <memory>
#include <string>
#include <optional>
#include <map>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "utils/ordered_map.h"
#include "mindspore/core/ops/core_ops.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace opt {
namespace {
using ParamUserMap = mindspore::HashMap<std::string, std::vector<size_t>>;
using LoadGraphMap = OrderedMap<std::string, std::vector<size_t>>;

std::optional<std::string> GetRefKey(const AnfNodePtr &node) {
  auto abs = node->abstract();
  if (abs == nullptr) {
    // Abstract for some Depends node are not proper set, we follow its input.
    if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
      return GetRefKey(node->cast<CNodePtr>()->input(1));
    }
    // Abstract should be set except UpdateState nodes.
    if (!IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      MS_LOG(WARNING) << "Abstract not set for " << node->DebugString();
    }
    return std::nullopt;
  }
  auto abs_ref = abs->cast<abstract::AbstractRefPtr>();
  if (abs_ref == nullptr) {
    return std::nullopt;
  }
  auto ref_key = abs_ref->ref_key_value()->cast<StringImmPtr>();
  if (ref_key == nullptr) {
    return std::nullopt;
  }
  return ref_key->value();
}

bool HasSideEffect(const CNodePtr &cnode) {
  const auto &inputs = cnode->inputs();
  constexpr size_t kRequiredArgs = 2;
  if (inputs.size() > kRequiredArgs) {
    return HasAbstractMonad(inputs.back());
  }
  return false;
}

bool IsSpecialNode(const CNodePtr &cnode) {
  const auto &first_input = cnode->input(0);
  return IsPrimitiveCNode(first_input, prim::kPrimJ) || IsPrimitiveCNode(first_input, prim::kPrimVmap) ||
         IsPrimitiveCNode(first_input, prim::kPrimTaylor) || IsPrimitiveCNode(first_input, prim::kPrimShard) ||
         IsValueNode<FuncGraph>(first_input) || cnode->IsApply(prim::kPrimCall) || cnode->IsApply(prim::kPrimPartial) ||
         cnode->IsApply(prim::kPrimSwitch) || cnode->IsApply(prim::kPrimSwitchLayer);
}

LoadGraphMap GenerateLoadGroups(const FuncGraphPtr &fg, std::vector<AnfNodePtr> *toposet,
                                std::vector<AnfNodePtr> *need_replace_loads, ParamUserMap *param_users,
                                std::vector<size_t> *special_op_indexes) {
  LoadGraphMap load_groups;
  // Record inputs of load and id of load in toposort.
  // RefKey --> (Monad --> index).
  std::map<std::string, std::map<AnfNodePtr, size_t>> param_monads;
  auto mgr = fg->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  for (size_t i = 0; i < toposet->size(); i++) {
    auto cnode = dyn_cast<CNode>((*toposet)[i]);
    // Exclude free variable node.
    if (cnode == nullptr || cnode->func_graph() != fg) {
      continue;
    }
    // Handle Load node.
    if (cnode->IsApply(prim::kPrimLoad)) {
      auto ref_key = GetRefKey(cnode->input(1));
      if (!ref_key.has_value()) {
        MS_LOG(INFO) << "Load without ref key: " << cnode->DebugString();
        continue;
      }
      // Group load nodes by their input ref key.
      auto &group = load_groups[ref_key.value()];
      constexpr size_t monad_index = 2;
      auto monad = cnode->input(monad_index);
      std::map<AnfNodePtr, size_t> &cur_param_monads = param_monads[ref_key.value()];
      const auto &iter = cur_param_monads.find(monad);
      // Remove duplicate load which has the same inputs, otherwise there may be an error in the load grouping.
      if (iter != cur_param_monads.end()) {
        auto id = iter->second;
        auto &first_load = (*toposet)[id];
        (void)mgr->Replace(cnode, first_load);
        (*toposet)[i] = first_load;
        continue;
      } else {
        cur_param_monads[monad] = i;
        (void)group.emplace_back(i);
      }
      if (group.size() == 1) {
        // The first load user of param in toposort, if it can be replace load(param, ud) with load(param, u),
        // Means there are not nodes which modify param before the load.
        const bool param_not_used = (param_users->find(ref_key.value()) == param_users->end());
        const bool can_replace = (param_not_used && special_op_indexes->empty());
        if (can_replace) {
          (void)need_replace_loads->emplace_back(cnode);
        }
      }
      continue;
    }
    // Record special cnode.
    if (IsSpecialNode(cnode)) {
      (void)special_op_indexes->emplace_back(i);
      continue;
    }
    // Record param user in toposort nodes.
    // We only check side effect cnodes or Depend nodes.
    if (HasSideEffect(cnode) || cnode->IsApply(prim::kPrimDepend)) {
      for (size_t n = 1; n < cnode->size(); ++n) {
        const auto &input = cnode->input(n);
        auto ref_key = GetRefKey(input);
        if (ref_key.has_value()) {
          (void)(*param_users)[ref_key.value()].emplace_back(i);
        }
      }
    }
  }
  return load_groups;
}

bool HasIndexBetween(const std::vector<size_t> &indexes, size_t first, size_t second) {
  return std::any_of(indexes.begin(), indexes.end(),
                     [&first, &second](size_t index) { return index > first && index < second; });
}

std::vector<std::vector<size_t>> SplitGroup(const std::vector<size_t> &group,
                                            const std::vector<size_t> &param_user_indexes,
                                            const std::vector<size_t> &special_op_indexes) {
  if (group.size() <= 1) {
    return {};
  }
  size_t cur_load_index = 1;
  size_t pre_load_index = 0;
  std::vector<size_t> cur_group = {group[pre_load_index]};
  std::vector<std::vector<size_t>> split_groups;
  while (cur_load_index < group.size()) {
    const auto cur_load = group[cur_load_index];
    const auto prev_load = group[pre_load_index];
    // Exist node which is the user of load_param between prev_load and cur_load,
    // Do not divide into the same group.
    if (HasIndexBetween(param_user_indexes, prev_load, cur_load) ||
        HasIndexBetween(special_op_indexes, prev_load, cur_load)) {
      (void)split_groups.emplace_back(std::move(cur_group));
    }
    cur_group.push_back(cur_load);
    pre_load_index++;
    cur_load_index++;
  }
  // push back the last splited group.
  split_groups.push_back(cur_group);
  return split_groups;
}

// Pattern1======================================
// a = Load(para1, u1)
// ...
// b = Load(para1, u2)
// u3 = UpdateState(u2, b)
// ==>
// delete the UpdateState
void DeleteLoadUserUpdateState(const FuncGraphManagerPtr &manager, const AnfNodePtr &load_user) {
  const auto &update_state_cnode = load_user->cast<CNodePtr>();
  constexpr size_t monad_index = 1;
  const auto &monad = update_state_cnode->input(monad_index);
  (void)manager->Replace(load_user, monad);
}

// Pattern2======================================
// a = Load(para1, u1)
// ...
// b = Load(para1, u2)
// t = make_tuple(x, b)
// u3 = UpdateState(u2, t)
// ==>
// a = Load(para1, u1)
// ...
// b = Load(para1, u2)
// u3 = UpdateState(u2, x)
void DeleteLoadUserMakeTuple(const FuncGraphManagerPtr &manager, const CNodePtr &make_tuple, const AnfNodePtr &load) {
  // Initialize the other_input with load in case of all the inputs of the make_tuple is the same load.
  AnfNodePtr other_input = load;
  for (size_t i = 1; i < make_tuple->size(); i++) {
    if (make_tuple->input(i) != load) {
      other_input = make_tuple->input(i);
      break;
    }
  }
  MS_EXCEPTION_IF_NULL(other_input);
  manager->Replace(make_tuple, other_input);
}

// Pattern3======================================
// a = Load(para1, u1)
// ...
// b = Load(para1, u2)
// t = make_tuple(x, y, b, z)
// u3 = UpdateState(u2, t)
// ==>
// a = Load(para1, u1)
// ...
// b = Load(para1, u2)
// t = make_tuple(x, y, z)
// u3 = UpdateState(u2, t)
void ReplaceLoadUserMakeTuple(const FuncGraphManagerPtr &manager, const CNodePtr &make_tuple, const AnfNodePtr &load) {
  auto &make_tuple_inputs = make_tuple->inputs();
  std::vector<AnfNodePtr> new_make_tuple_inputs;
  (void)std::copy_if(make_tuple_inputs.begin(), make_tuple_inputs.end(), std::back_inserter(new_make_tuple_inputs),
                     [load](const AnfNodePtr &input) { return load != input; });
  auto fg = make_tuple->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  const auto &new_make_tuple = fg->NewCNode(new_make_tuple_inputs);
  // Set abstract for the MakeTuple node.
  abstract::AbstractBasePtrList element_abstracts;
  (void)std::transform(new_make_tuple_inputs.begin() + 1, new_make_tuple_inputs.end(),
                       std::back_inserter(element_abstracts),
                       [](const AnfNodePtr &input) { return input->abstract(); });
  new_make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(element_abstracts));
  manager->Replace(make_tuple, new_make_tuple);
}

bool ReplaceLoadUser(const FuncGraphManagerPtr &manager, const AnfNodePtr &load) {
  bool change = false;
  auto load_users = manager->node_users()[load];
  for (const auto &load_user : load_users) {
    // Pattern1
    if (IsPrimitiveCNode(load_user.first, prim::kPrimUpdateState)) {
      DeleteLoadUserUpdateState(manager, load_user.first);
      change = true;
      continue;
    }

    if (IsPrimitiveCNode(load_user.first, prim::kPrimMakeTuple)) {
      const auto &make_tuple = load_user.first->cast<CNodePtr>();
      auto &maketuple_users = manager->node_users()[make_tuple];
      auto maketuple_as_input_of_update =
        maketuple_users.size() == 1 && IsPrimitiveCNode(maketuple_users.back().first, prim::kPrimUpdateState);
      if (!maketuple_as_input_of_update) {
        continue;
      }
      // Pattern2
      if (make_tuple->size() == 3) {
        DeleteLoadUserMakeTuple(manager, make_tuple, load);
        change = true;
        continue;
      }
      // Pattern3
      if (make_tuple->size() > 3) {
        ReplaceLoadUserMakeTuple(manager, make_tuple, load);
        change = true;
      }
    }
  }
  return change;
}

bool ReplaceSameGroupLoad(const FuncGraphManagerPtr &manager, const std::vector<AnfNodePtr> &toposet,
                          const std::vector<size_t> &group) {
  if (group.size() <= 1) {
    return false;
  }
  bool change = false;
  const auto &main = toposet[group[0]];
  for (size_t i = 1; i < group.size(); i++) {
    change = ReplaceLoadUser(manager, toposet[group[i]]);
    manager->Replace(toposet[group[i]], main);
  }
  return change;
}

AnfNodePtr GetFirstMonad(const FuncGraphPtr &fg) {
  auto &params = fg->parameters();
  auto end = (params.size() > 1) ? (params.rbegin() + 2) : params.rend();
  auto iter = std::find_if(params.rbegin(), end, [](const AnfNodePtr &para) { return HasAbstractUMonad(para); });
  if (iter != end) {
    return *iter;
  }
  auto monad = NewValueNode(kUMonad);
  monad->set_abstract(kUMonad->ToAbstract());
  return monad;
}

bool CheckExistSpecialNode(const AnfNodePtr &update_state, const AnfNodePtr &first_monad) {
  if (!update_state->isa<CNode>()) {
    return false;
  }
  auto update_state_cnode = update_state->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(update_state_cnode);
  constexpr size_t monad_input_index = 1;
  constexpr size_t attach_input_index = 2;
  auto monad = update_state_cnode->input(monad_input_index);
  auto attach_node = update_state_cnode->input(attach_input_index);
  MS_EXCEPTION_IF_NULL(attach_node);
  if (attach_node->isa<CNode>() && IsSpecialNode(attach_node->cast<CNodePtr>())) {
    return true;
  }
  if (monad == first_monad) {
    return false;
  }
  return CheckExistSpecialNode(monad, first_monad);
}

// Replace UpdateStates with U for first load.
// Covert:
// u1 = UpdateState(u, c)
// p1 = Load(para1, u1)  // first load for para1, and there are not special node before u1
// To:
// u1 = UpdateState(u, c)
// p1 = Load(para1, u')  // u' is first monad in graph or new monad
bool ReplaceUpdateStateForLoad(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &need_replace_loads) {
  if (need_replace_loads.size() == 0) {
    return false;
  }
  bool change = false;
  constexpr size_t second_input_index = 2;
  auto monad = GetFirstMonad(fg);
  for (const auto &load_node : need_replace_loads) {
    if (!IsPrimitiveCNode(load_node, prim::kPrimLoad)) {
      continue;
    }
    auto update_state = load_node->cast<CNodePtr>()->input(second_input_index);
    auto mgr = fg->manager();
    MS_EXCEPTION_IF_NULL(mgr);
    // If the u1 only used by Load and one other updatestate, no need to replace u1 by u'.
    auto &node_users = mgr->node_users()[update_state];
    constexpr size_t kUserSize = 2;
    if (!IsPrimitiveCNode(update_state, prim::kPrimUpdateState) || node_users.size() == kUserSize) {
      continue;
    }
    // Check whether there is special node before the current load node in the execution sequence.
    // If exist special node(the node may modify the load parameter), should not replace update_state for the load node.
    if (CheckExistSpecialNode(update_state, monad)) {
      continue;
    }
    mgr->SetEdge(load_node, second_input_index, monad);
    change = true;
  }
  return change;
}
}  // namespace

// Node1{primLoad,X,Y1},...,Node{Node's input != X},...,Node2{primLoad,X,Y2},... =>
// Node1{primLoad,X,Y1},...,Node{Nodes' input != X},...,Node1,...
bool AutoMonadEliminator::ReplaceAutoMonadNode(const FuncGraphManagerPtr &manager) const {
  auto changed = false;
  for (const FuncGraphPtr &fg : manager->func_graphs()) {
    std::vector<AnfNodePtr> toposet = TopoSort(fg->get_return());
    // Record the set of the first load of param which no nodes modify param before the load in toposort.
    std::vector<AnfNodePtr> need_replace_loads;
    // Record the param and the toposort id of the unload user of param, they may modify the value of param.
    ParamUserMap param_users;
    // Record the toposort id of special_op(call, partial, switch, switch_layer), they may modify the value of param.
    std::vector<size_t> special_op_indexes;
    auto load_groups = GenerateLoadGroups(fg, &toposet, &need_replace_loads, &param_users, &special_op_indexes);
    // Split group if there is no-load node between two load nodes.
    std::vector<std::vector<size_t>> need_merge_loads;
    for (const auto &load_group : load_groups) {
      auto &ref_key = load_group.first;
      auto &group = load_group.second;
      const auto &param_user_indexes = param_users[ref_key];
      auto groups = SplitGroup(group, param_user_indexes, special_op_indexes);
      (void)need_merge_loads.insert(need_merge_loads.cend(), groups.cbegin(), groups.cend());
    }
    for (auto &group : need_merge_loads) {
      bool replaced = ReplaceSameGroupLoad(manager, toposet, group);
      if (replaced) {
        changed = true;
      }
    }
    bool update_state_replaced = ReplaceUpdateStateForLoad(fg, need_replace_loads);
    if (update_state_replaced) {
      changed = true;
    }
  }
  return changed;
}

// Eliminate auto monad node:
// From:
//    u1 = UpdateState(...);
//    xxx = User(u1); // Other users except below Depend.
//    output = Depend(output, u1);
//    return output;
// To:
//    u1 = UpdateState(...);
//    xxx = User(u1);
//    return output;
bool AutoMonadEliminator::EliminateAutoMonadNode(const FuncGraphManagerPtr &manager) const {
  auto changed = false;
  for (const FuncGraphPtr &fg : manager->func_graphs()) {
    auto output = fg->output();
    if (output == nullptr) {
      continue;
    }
    if (!IsPrimitiveCNode(output, prim::kPrimDepend)) {
      continue;
    }
    constexpr size_t attach_index = 2;
    auto attach = output->cast<CNodePtr>()->input(attach_index);
    if (!IsPrimitiveCNode(attach, prim::kPrimUpdateState)) {
      continue;
    }
    auto &node_users = manager->node_users();
    auto iter = node_users.find(attach);
    if (iter == node_users.end()) {
      MS_LOG(EXCEPTION) << "No user of node: " << attach->DebugString();
    }
    auto &users = iter->second;
    if (users.size() <= 1) {
      continue;
    }
    constexpr size_t input_index = 1;
    auto input = output->cast<CNodePtr>()->input(input_index);
    MS_LOG(DEBUG) << "Change " << output->DebugString() << " -> " << input->DebugString();
    fg->set_output(input);
    changed = true;
  }
  MS_LOG(DEBUG) << "Changed: " << changed;
  return changed;
}
}  // namespace opt
}  // namespace mindspore
