/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/cse.h"

#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include "abstract/abstract_function.h"
#include "utils/flags.h"
#include "utils/utils.h"
#include "base/core_ops.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;

bool WithRecomputedScope(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto full_name_with_scope = node->fullname_with_scope();
  return full_name_with_scope.find(kAttrRecompute) == 0;
}

bool IsSetRecomputed(const CNodePtr &a, const CNodePtr &b) {
  return (WithRecomputedScope(a) && !a->HasAttr(kAttrNeedCseAfterRecompute)) ||
         (WithRecomputedScope(b) && !b->HasAttr(kAttrNeedCseAfterRecompute));
}

BasePtr AbsOf(const AnfNodePtr &node, bool ignore_fg_abs_tracking_id) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_abs = node->abstract();
  // In testcase: TestOptOpt.CSE, node->abstract() is null.
  if (node_abs == nullptr) {
    return kAnyValue;
  }
  if (node_abs->isa<abstract::PrimitiveAbstractClosure>()) {
    // Ignore the tracking_id and prim pointer hash.
    auto prim_abs = node_abs->cast<abstract::PrimitiveAbstractClosurePtr>();
    return prim_abs->prim();
  } else if (ignore_fg_abs_tracking_id && node_abs->isa<abstract::FuncGraphAbstractClosure>()) {
    // Ignore the tracking_id.
    auto new_fg_abs = node_abs->cast<abstract::AbstractFunctionPtr>()->Copy();
    new_fg_abs->set_tracking_id(nullptr);
    return new_fg_abs;
  }
  return node_abs;
}

bool CSE::BuildOrderGroupAndDoReplace(const FuncGraphManagerPtr manager) const {
  bool changed = false;
  for (FuncGraphPtr fg : manager->func_graphs()) {
    MS_EXCEPTION_IF_NULL(fg);
    std::vector<std::size_t> order_group;
    std::unordered_map<std::size_t, std::vector<AnfNodePtr>> groups;
    std::unordered_map<AnfNodePtr, std::size_t> hashes;

    std::vector<AnfNodePtr> toposet = TopoSort(fg->get_return());
    for (auto node : toposet) {
      MS_EXCEPTION_IF_NULL(node);
      if (hashes.find(node) != hashes.end()) {
        continue;
      }

      std::size_t h = 0;
      if (node->isa<ValueNode>()) {
        ValueNodePtr value_node = node->cast<ValueNodePtr>();
        auto value = value_node->value();
        MS_EXCEPTION_IF_NULL(value);
        h = hash_combine(value->hash(), (AbsOf(value_node, true)->hash()));
      } else if (node->isa<CNode>()) {
        auto cnode = node->cast<CNodePtr>();
        auto &inputs = cnode->inputs();
        size_t init = 0;
        h = std::accumulate(inputs.begin(), inputs.end(), init, [&hashes](std::size_t hash, const AnfNodePtr &node_in) {
          return hash_combine(hash, hashes[node_in]);
        });
      } else if (node->isa<Parameter>()) {
        h = node->hash();
      } else {
        MS_LOG(ERROR) << "Unknown node type";
      }

      hashes[node] = h;
      if (groups.find(h) == groups.end()) {
        std::vector<AnfNodePtr> innervec({node});
        groups[h] = innervec;
        order_group.emplace_back(h);
      } else {
        groups[h].push_back(node);
      }
    }

    changed = DoReplace(manager, order_group, &groups) || changed;
  }

  return changed;
}

std::vector<std::vector<size_t>> GenerateLoadGroups(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &toposet,
                                                    std::vector<AnfNodePtr> *need_replace_loads) {
  std::unordered_map<AnfNodePtr, size_t> load_groups_record;
  std::vector<std::vector<size_t>> load_groups;
  std::unordered_set<AnfNodePtr> unload_users_record;
  for (size_t i = 0; i < toposet.size(); i++) {
    auto &node = toposet[i];
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (!IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      for (const auto &input : cnode->inputs()) {
        if (input->isa<Parameter>() ||
            (IsPrimitiveCNode(input, prim::kPrimDepend) && input->cast<CNodePtr>()->input(1)->isa<Parameter>())) {
          unload_users_record.insert(input);
        }
      }
      continue;
    }
    // Exclude free variable node.
    if (cnode->func_graph() != fg) {
      continue;
    }
    auto load_param = cnode->input(1);
    // first time get same input1 of load.
    if (load_groups_record.find(load_param) == load_groups_record.end()) {
      load_groups_record[load_param] = load_groups.size();
      load_groups.push_back({i});
      if (unload_users_record.find(load_param) == unload_users_record.end()) {
        need_replace_loads->emplace_back(cnode);
      }
    } else {
      // not first time get same input1 of load
      load_groups[load_groups_record[load_param]].push_back(i);
    }
  }
  return load_groups;
}

std::vector<std::vector<size_t>> SplitGroup(const std::vector<AnfNodePtr> &toposet, const std::vector<size_t> &group) {
  if (group.size() <= 1) {
    return {};
  }
  auto load_param = toposet[group.back()]->cast<CNodePtr>()->input(1);
  size_t cur_load_index = 1;
  size_t pre_load_index = 0;
  std::vector<size_t> cur_group = {group[pre_load_index]};
  std::vector<std::vector<size_t>> split_groups;
  while (cur_load_index < group.size()) {
    const auto &cur_load = group[cur_load_index];
    const auto &prev_load = group[pre_load_index];
    const auto param_used_by_other =
      std::any_of(toposet.begin() + prev_load, toposet.begin() + cur_load, [&load_param](const AnfNodePtr &node) {
        if (!node->isa<CNode>()) {
          return false;
        }
        if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
          return false;
        }
        auto cnode = node->cast<CNodePtr>();
        auto &inputs = cnode->inputs();
        return std::any_of(inputs.begin(), inputs.end(),
                           [&load_param](const AnfNodePtr &input) { return load_param == input; });
      });
    if (param_used_by_other) {
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
//==>
// delete the UpdateState
void DeleteLoadUserUpdateState(const FuncGraphManagerPtr &manager, const AnfNodePtr &load_user,
                               const AnfNodePtr &load) {
  const auto &load_cnode = load->cast<CNodePtr>();
  const auto &u = load_cnode->input(2);
  manager->Replace(load_user, u);
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
  AnfNodePtr other_input = nullptr;
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
  manager->Replace(make_tuple, new_make_tuple);
}

void ReplaceLoadUser(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const AnfNodePtr &load) {
  auto load_users = manager->node_users()[load];
  for (const auto &load_user : load_users) {
    // Pattern1
    if (IsPrimitiveCNode(load_user.first, prim::kPrimUpdateState)) {
      DeleteLoadUserUpdateState(manager, load_user.first, load);
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
        continue;
      }
      // Pattern3
      if (make_tuple->size() > 3) {
        ReplaceLoadUserMakeTuple(manager, fg, make_tuple, load);
      }
    }
  }
}

bool ReplaceSameGroupLoad(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg,
                          const std::vector<AnfNodePtr> &toposet, const std::vector<size_t> &group) {
  if (group.size() <= 1) {
    return false;
  }
  const auto &main = toposet[group[0]];
  for (size_t i = 1; i < group.size(); i++) {
    ReplaceLoadUser(manager, fg, toposet[group[i]]);
    manager->Replace(toposet[group[i]], main);
  }
  return true;
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
  constexpr size_t second_input_index = 2;
  auto monad = GetFirstMonad(fg);
  for (const auto &load_node : need_replace_loads) {
    if (!IsPrimitiveCNode(load_node, prim::kPrimLoad)) {
      continue;
    }
    auto update_state = load_node->cast<CNodePtr>()->input(second_input_index);
    if (!IsPrimitiveCNode(update_state, prim::kPrimUpdateState)) {
      continue;
    }
    auto mgr = fg->manager();
    mgr->SetEdge(load_node, second_input_index, monad);
  }
  return true;
}

// Node1{primLoad,X,Y1},...,Node{Node's input != X},...,Node2{primLoad,X,Y2},... =>
// Node1{primLoad,X,Y1},...,Node{Nodes' input != X},...,Node1,...
bool CSE::ReplaceAutoMonadNode(const FuncGraphManagerPtr &manager) const {
  auto changed = false;
  for (const FuncGraphPtr &fg : manager->func_graphs()) {
    std::vector<AnfNodePtr> toposet = TopoSort(fg->get_return());
    std::vector<AnfNodePtr> need_replace_loads;
    std::vector<std::vector<size_t>> load_groups = GenerateLoadGroups(fg, toposet, &need_replace_loads);
    const bool update_state_replaced = ReplaceUpdateStateForLoad(fg, need_replace_loads);
    if (update_state_replaced) {
      changed = true;
    }
    // split group if there is no-load node between two load nodes.
    std::vector<std::vector<size_t>> need_merge_loads;
    for (auto &group : load_groups) {
      auto groups = SplitGroup(toposet, group);
      need_merge_loads.insert(need_merge_loads.end(), groups.begin(), groups.end());
    }
    for (auto &group : need_merge_loads) {
      const bool replaced = ReplaceSameGroupLoad(manager, fg, toposet, group);
      if (!changed && replaced) {
        changed = true;
      }
    }
  }
  return changed;
}

// The op like print, summary, or the op do not has true output, and always as a depend node input.
static bool HasSideEffect(const AnfNodePtr &node) {
  auto prim = GetCNodePrimitive(node);
  if (prim == nullptr) {
    return false;
  }
  auto side_effect_v = prim->GetAttr(GRAPH_FLAG_SIDE_EFFECT);
  if (side_effect_v != nullptr && side_effect_v->isa<BoolImm>()) {
    return GetValue<bool>(side_effect_v);
  }
  return false;
}
// If true do not merge the node.
bool CSE::CheckRandomEffect(const AnfNodePtr &main, const AnfNodePtr &node) const {
  bool has_random_effect = false;
  auto prim_main = GetCNodePrimitive(main);
  auto prim_node = GetCNodePrimitive(node);
  // if has random effect, when generate by different op (not same object), do not merge.
  if (prim_main != nullptr) {
    if (prim_main == prim_node) {
      return false;
    }
    auto effect_val = prim_main->GetAttr(GRAPH_FLAG_RANDOM_EFFECT);
    if (effect_val != nullptr && effect_val->isa<BoolImm>()) {
      has_random_effect = GetValue<bool>(effect_val);
    }
  }
  return has_random_effect;
}

bool CSE::CheckReplace(const AnfNodePtr &main, const AnfNodePtr &node, bool check_side_effect) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);

  if (main->isa<ValueNode>() && node->isa<ValueNode>()) {
    auto main_value = GetValueNode(main);
    auto node_value = GetValueNode(node);
    return (AbsOf(main, true) == AbsOf(node, true)) && (*main_value == *node_value);
  } else if (main->isa<CNode>() && node->isa<CNode>()) {
    auto c_main = main->cast<CNodePtr>();
    auto c_node = node->cast<CNodePtr>();
    // Not do cse for the node set recompute before the recompute pass.
    if (IsSetRecomputed(c_main, c_node)) {
      return false;
    }
    // When appsame is true, check if has side effect, do not merge.
    if (check_side_effect && HasSideEffect(main)) {
      return false;
    }
    const auto &inp1 = c_main->inputs();
    const auto &inp2 = c_node->inputs();
    if (inp1.size() != inp2.size()) {
      return false;
    }
    for (size_t j = 0; j < inp1.size(); j++) {
      auto inp1_j = inp1[j];
      auto inp2_j = inp2[j];
      MS_EXCEPTION_IF_NULL(inp1_j);
      MS_EXCEPTION_IF_NULL(inp2_j);
      if (!(*inp1_j == *inp2_j)) {
        // Handle the case of two different Tensor, but with the same value
        if (IsValueNode<tensor::Tensor>(inp1_j) && IsValueNode<tensor::Tensor>(inp2_j)) {
          auto tensor1 = GetValueNode<tensor::TensorPtr>(inp1_j);
          auto tensor2 = GetValueNode<tensor::TensorPtr>(inp2_j);
          if (tensor1->ValueEqual(*tensor2)) {
            continue;
          }
        } else if (HasSideEffect(inp1_j) && HasSideEffect(inp2_j)) {
          // When the same side effect node as another two nodes' inputs, we still merge the node.
          // Because the node only can be the inputs of `depend`, when the `depend` is duplicated merge the depend the
          // node.
          if (CheckReplace(inp1_j, inp2_j, false)) {
            continue;
          }
        }
        return false;
      }
    }
    // When appsame is true, check if has random effect do not merge
    if (CheckRandomEffect(c_main, c_node)) {
      return false;
    }
    return true;
  }
  // a parameter node.
  return false;
}

bool CSE::DoReplace(const FuncGraphManagerPtr manager, const std::vector<std::size_t> &order_group,
                    std::unordered_map<std::size_t, std::vector<AnfNodePtr>> *groups) const {
  bool changes = false;
  std::set<size_t> clear_set;
  for (auto &h : order_group) {
    std::vector<AnfNodePtr> &group = (*groups)[h];
    // If there are more than 2 node in that group, they may be same common expression can be eliminated.
    if (group.size() > 1) {
      for (size_t k = 0; k < group.size() - 1; k++) {
        AnfNodePtr main = group[k];
        MS_EXCEPTION_IF_NULL(main);

        // When all node in group has been replaced
        // or a valuenode node, skip compare in group
        if ((k + 1 + clear_set.size() == group.size()) || (k > 0 && main->isa<ValueNode>())) {
          break;
        }

        // skip node has been replaced
        if (clear_set.find(k) != clear_set.end()) {
          continue;
        }

        // Compare with rest elements in this group.
        for (size_t i = k + 1; i < group.size(); i++) {
          auto node = group[i];
          MS_EXCEPTION_IF_NULL(node);

          if (clear_set.find(i) != clear_set.end()) {
            continue;
          }
          if (main->func_graph() != node->func_graph()) {
            continue;
          }
          if (CheckReplace(node, main)) {
            changes = true;
            (void)manager->Replace(node, main);
            (void)clear_set.insert(i);
          }
        }
      }
      clear_set.clear();
    }
  }

  return changes;
}

bool CSE::Cse(const FuncGraphPtr root, const FuncGraphManagerPtr manager) const {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);
  auto change1 = ReplaceAutoMonadNode(manager);
  auto change2 = BuildOrderGroupAndDoReplace(manager);
  return change1 || change2;
}
}  // namespace opt
}  // namespace mindspore
