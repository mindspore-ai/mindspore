/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <memory>

#include "base/core_ops.h"

namespace mindspore {
namespace opt {
using MapParamUserIndexs = std::unordered_map<AnfNodePtr, std::vector<size_t>>;
std::vector<std::vector<size_t>> GenerateLoadGroups(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &toposet,
                                                    std::vector<AnfNodePtr> *need_replace_loads,
                                                    MapParamUserIndexs *unload_users_record,
                                                    std::vector<size_t> *special_op_indexs) {
  std::unordered_map<AnfNodePtr, size_t> load_groups_record;
  std::vector<std::vector<size_t>> load_groups;
  for (size_t i = 0; i < toposet.size(); i++) {
    auto &node = toposet[i];
    auto cnode = node->cast<CNodePtr>();
    // Exclude free variable node.
    if (cnode == nullptr || cnode->func_graph() != fg) {
      continue;
    }
    bool is_special_op = IsPrimitiveCNode(cnode, prim::kPrimCall) || IsValueNode<FuncGraph>(cnode->input(0)) ||
                         IsPrimitiveCNode(cnode, prim::kPrimPartial) || IsPrimitiveCNode(cnode, prim::kPrimSwitch) ||
                         IsPrimitiveCNode(cnode, prim::kPrimSwitchLayer);
    if (is_special_op) {
      (void)special_op_indexs->emplace_back(i);
    }

    // Record param user in toposort nodes.
    if (!IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      for (const auto &input : cnode->inputs()) {
        AnfNodePtr cur_param = nullptr;
        if (input->isa<Parameter>()) {
          cur_param = input;
        } else if (IsPrimitiveCNode(input, prim::kPrimDepend) && input->cast<CNodePtr>()->input(1)->isa<Parameter>()) {
          cur_param = input->cast<CNodePtr>()->input(1);
        }
        if (cur_param != nullptr) {
          (void)(*unload_users_record)[cur_param].emplace_back(i);
        }
      }
      continue;
    }

    auto load_param = cnode->input(1);
    // first time get same input1 of load.
    if (load_groups_record.find(load_param) == load_groups_record.end()) {
      load_groups_record[load_param] = load_groups.size();
      load_groups.push_back({i});
      // The first load user of param in toposort, if it can be replace load(param, ud) with load(param, u)
      // Means there are not nodes which modify param before the load
      bool can_replace = (*unload_users_record)[load_param].empty() && special_op_indexs->empty();
      if (can_replace) {
        need_replace_loads->emplace_back(cnode);
      }
    } else {
      // not first time get same input1 of load
      load_groups[load_groups_record[load_param]].push_back(i);
    }
  }
  return load_groups;
}

std::vector<std::vector<size_t>> SplitGroup(const std::vector<size_t> &group,
                                            const std::vector<size_t> &unload_user_indexs,
                                            const std::vector<size_t> &special_op_indexs) {
  if (group.size() <= 1) {
    return {};
  }
  size_t cur_load_index = 1;
  size_t pre_load_index = 0;
  std::vector<size_t> cur_group = {group[pre_load_index]};
  std::vector<std::vector<size_t>> split_groups;
  while (cur_load_index < group.size()) {
    const auto &cur_load = group[cur_load_index];
    const auto &prev_load = group[pre_load_index];
    // Exist node which is the user of load_param between prev_load and cur_load,
    // Do not divide into the same group.
    const auto param_used_by_other =
      std::any_of(unload_user_indexs.begin(), unload_user_indexs.end(),
                  [&cur_load, &prev_load](size_t index) { return index > prev_load && index < cur_load; });
    const auto param_used_by_special_op =
      std::any_of(special_op_indexs.begin(), special_op_indexs.end(),
                  [&cur_load, &prev_load](size_t index) { return index > prev_load && index < cur_load; });
    if (param_used_by_other || param_used_by_special_op) {
      split_groups.push_back(cur_group);
      cur_group.clear();
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
//==>
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
//==>
// a = Load(para1, u1)
// ...
// b = Load(para1, u2)
// t = make_tuple(x, y, z)
// u3 = UpdateState(u2, t)
void ReplaceLoadUserMakeTuple(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const CNodePtr &make_tuple,
                              const AnfNodePtr &load) {
  auto &make_tuple_inputs = make_tuple->inputs();
  std::vector<AnfNodePtr> new_make_tuple_inputs;
  (void)std::copy_if(make_tuple_inputs.begin(), make_tuple_inputs.end(), std::back_inserter(new_make_tuple_inputs),
                     [load](const AnfNodePtr &input) { return load != input; });
  const auto &new_make_tuple = fg->NewCNode(new_make_tuple_inputs);
  // Set abstract for the MakeTuple node.
  abstract::AbstractBasePtrList element_abstracts;
  (void)std::transform(new_make_tuple_inputs.begin() + 1, new_make_tuple_inputs.end(),
                       std::back_inserter(element_abstracts),
                       [](const AnfNodePtr &input) { return input->abstract(); });
  new_make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(element_abstracts));
  manager->Replace(make_tuple, new_make_tuple);
}

bool ReplaceLoadUser(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const AnfNodePtr &load) {
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
        ReplaceLoadUserMakeTuple(manager, fg, make_tuple, load);
        change = true;
      }
    }
  }
  return change;
}

bool ReplaceSameGroupLoad(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg,
                          const std::vector<AnfNodePtr> &toposet, const std::vector<size_t> &group) {
  if (group.size() <= 1) {
    return false;
  }
  bool change = false;
  const auto &main = toposet[group[0]];
  for (size_t i = 1; i < group.size(); i++) {
    change = ReplaceLoadUser(manager, fg, toposet[group[i]]);
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

// Replace UpdateStates with U for first load.
// Covert:
// u1 = UpdateState(u, c)
// p1 = Load(para1, u1)  // first load for para1
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
    mgr->SetEdge(load_node, second_input_index, monad);
    change = true;
  }
  return change;
}

// Node1{primLoad,X,Y1},...,Node{Node's input != X},...,Node2{primLoad,X,Y2},... =>
// Node1{primLoad,X,Y1},...,Node{Nodes' input != X},...,Node1,...
bool AutoMonadEliminator::ReplaceAutoMonadNode(const FuncGraphManagerPtr &manager) const {
  auto changed = false;
  for (const FuncGraphPtr &fg : manager->func_graphs()) {
    std::vector<AnfNodePtr> toposet = TopoSort(fg->get_return());
    // Record the set of the first load of param which no nodes modify param before the load in toposort.
    std::vector<AnfNodePtr> need_replace_loads;
    // Record the param and the toposort id of the unload user of param, they may modify the value of param.
    MapParamUserIndexs unload_users_record;
    // Record the toposort id of special_op(call, partial, switch, switch_layer), they may modify the value of param.
    std::vector<size_t> special_op_indexs;
    std::vector<std::vector<size_t>> load_groups =
      GenerateLoadGroups(fg, toposet, &need_replace_loads, &unload_users_record, &special_op_indexs);
    // split group if there is no-load node between two load nodes.
    std::vector<std::vector<size_t>> need_merge_loads;
    for (auto &group : load_groups) {
      auto load_param = toposet[group.back()]->cast<CNodePtr>()->input(1);
      const auto &unload_user_indexs = unload_users_record[load_param];
      auto groups = SplitGroup(group, unload_user_indexs, special_op_indexs);
      need_merge_loads.insert(need_merge_loads.end(), groups.begin(), groups.end());
    }
    for (auto &group : need_merge_loads) {
      bool replaced = ReplaceSameGroupLoad(manager, fg, toposet, group);
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
