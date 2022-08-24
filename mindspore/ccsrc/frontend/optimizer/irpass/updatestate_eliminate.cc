/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/updatestate_eliminate.h"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/pattern_matcher.h"

namespace mindspore::opt::irpass {
namespace {
// data = Load(input, attach)
// data = Depend(input, attach)
// monad = UpdateState(input, attach)
constexpr size_t kFirstInputIndex = 0;
constexpr size_t kInputIndex = 1;
constexpr size_t kAttachIndex = 2;
constexpr size_t kMakeTupleSize = 3;
constexpr size_t kMinDependSize = 3;
constexpr size_t kUpdateStateSize = 3;
constexpr size_t kAssignSize = 4;
constexpr size_t kAssignRefInputIndex = 1;
constexpr size_t kAssignMonadInputIndex = 3;

FuncGraphManagerPtr GetManager(const AnfNodePtr &node) {
  auto fg = node->func_graph();
  if (fg == nullptr) {
    return nullptr;
  }
  return fg->manager();
}

// Return true if the node(be_used_node) is only used by the given node.
bool OnlyUsedByOneNode(const AnfNodePtr &be_used_node, const CNodePtr &given_node) {
  auto mgr = GetManager(given_node);
  if (mgr == nullptr) {
    return false;
  }
  auto &node_users = mgr->node_users();
  auto iter = node_users.find(be_used_node);
  if (iter == node_users.end()) {
    return false;
  }
  auto &partial_users = iter->second;
  return (partial_users.size() == 1) && (partial_users.front().first == given_node);
}

// Return true if the node(be_used_node) is only used by the given two nodes(first_node and second_node).
bool OnlyUsedByTwoNode(const AnfNodePtr &be_used_node, const AnfNodePtr &first_node, const AnfNodePtr &second_node) {
  auto mgr = GetManager(be_used_node);
  if (mgr == nullptr || first_node == second_node) {
    return false;
  }
  auto &node_users = mgr->node_users();
  auto iter = node_users.find(be_used_node);
  if (iter == node_users.end()) {
    return false;
  }
  constexpr size_t partial_users_cnt = 2;
  auto &partial_users = iter->second;
  if (partial_users.size() != partial_users_cnt) {
    return false;
  }
  const auto &first_user = partial_users.front().first;
  const auto &second_user = partial_users.back().first;
  return (first_user == first_node && second_user == second_node) ||
         (first_user == second_node && second_user == first_node);
}

// Determine whether there is a monad in the inputs of the node.
bool CheckHasMonadInput(const CNodePtr &cnode) {
  // If the last input is a monad, means the attach node has side-effect and
  // we should keep UpdateState; otherwise, we will remove the UpdateState.
  if (cnode->size() > 1 && HasAbstractMonad(cnode->inputs().back())) {
    return true;
  }

  // Check the inputs of Call/Switch/SwitchLayer.
  auto first_input_node = cnode->input(kFirstInputIndex);
  if (IsPrimitiveCNode(first_input_node, prim::kPrimSwitch) ||
      IsPrimitiveCNode(first_input_node, prim::kPrimSwitchLayer)) {
    for (auto &input : first_input_node->cast<CNodePtr>()->inputs()) {
      if (HasAbstractMonad(input)) {
        return true;
      }
      auto input_cnode = dyn_cast<CNode>(input);
      if (input_cnode != nullptr && input_cnode->size() > 1 && HasAbstractMonad(input_cnode->inputs().back())) {
        return true;
      }
    }
  }
  return false;
}

AnfNodePtr NewUpdateStateWithAttach(const CNodePtr &update_state, const AnfNodePtr &attach) {
  auto fg = update_state->func_graph();
  if (fg == nullptr) {
    return nullptr;
  }
  auto new_update_state =
    fg->NewCNode({update_state->input(kFirstInputIndex), update_state->input(kInputIndex), attach});
  new_update_state->set_abstract(update_state->abstract());
  new_update_state->set_scope(update_state->scope());
  return new_update_state;
}

AnfNodePtr EliminateUpdateStateWithDepend(const CNodePtr &update_state) {
  auto depend = update_state->input(kAttachIndex)->cast<CNodePtr>();
  constexpr auto recur_2 = 2;
  // If same Depend CNode is used by multiple UpdateState CNode, it may be replaced by previous elimination.
  if (depend == nullptr) {
    MS_LOG(DEBUG) << "UpdateState's input 2 Depend had been replaced: " << update_state->DebugString(recur_2);
    return nullptr;
  }
  auto input_monad = depend->inputs().back();
  if (!HasAbstractMonad(input_monad)) {
    // Skip if Depend attach input is not a monad.
    return nullptr;
  }
  auto update_monad = update_state->input(kInputIndex);
  if (!HasAbstractMonad(update_monad)) {
    // Skip if UpdateState input is not a monad.
    MS_LOG(WARNING) << "Not a monad input: " << update_state->DebugString();
    return nullptr;
  }
  // x1 = Depend(x, u0)        <-not match--   <--match--
  // u2 = UpdateState(u1, x1)                  <--match--
  // u3 = UpdateState(u2, x1)  <-not match--
  // u3 and x1 should not match otherwise u1 will be lost; u2 and x1 can match.
  if (IsPrimitiveCNode(update_monad, prim::kPrimUpdateState) &&
      update_monad->cast<CNodePtr>()->input(kAttachIndex) == depend) {
    MS_LOG(DEBUG) << "UpdateState should not be replaced. node: " << update_state->DebugString(recur_2);
    return nullptr;
  }
  // Check monad inputs.
  const auto &input_monad_abs = *(input_monad->abstract());
  const auto &update_monad_abs = *(update_monad->abstract());
  bool same_monad = (input_monad_abs == update_monad_abs);
  if (!same_monad) {
    // Skip if they are different monad (one is IO, another is U).
    return nullptr;
  }
  // Now we can eliminate the UpdateState and Depend nodes.
  auto mgr = GetManager(update_state);
  if (mgr == nullptr) {
    return nullptr;
  }
  // Replace Depend with its input.
  if (depend->size() == kMinDependSize) {
    auto depend_input = depend->input(kInputIndex);
    mgr->Replace(depend, depend_input);
  } else {
    auto inputs = depend->inputs();
    inputs.pop_back();
    auto fg = depend->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto new_depend = fg->NewCNode(inputs);
    new_depend->set_abstract(depend->abstract());
    mgr->Replace(depend, new_depend);
  }
  // Replace UpdateState node with the input monad of Depend.
  return input_monad;
}

bool ExistEnvironGet(const FuncGraphManagerPtr &manager) {
  const FuncGraphSet &fgs = manager->func_graphs();
  for (auto &fg : fgs) {
    auto &nodes = fg->value_nodes();
    bool exist = std::any_of(nodes.begin(), nodes.end(),
                             [](const auto &node) { return IsPrimitive(node.first, prim::kPrimEnvironGet); });
    if (exist) {
      return true;
    }
  }
  return false;
}

// Convert:
// cnode1 = EnvironSet(EnvCreate(), para1, attach1)
// cnode2 = EnvironSet(cnode1, para2, attach2)
// ...
// cnode_n = EnvironSet(cnode_n-1, para_n-1, attach_n-1)
// maketuple = maketuple(cnode_n, ...)
// updatestate = updatestate(umonad, maketuple)
// To:
// new_maketuple = maketuple(..., attach1, attach2, ..., attach_n-1)
// new_updatestate = updatestate(umonad, new_maketuple)
AnfNodePtr EliminateUpdateStateMakeTupleWithUselessEnv(const CNodePtr &update_state, const CNodePtr &make_tuple) {
  std::vector<AnfNodePtr> env_nodes;
  std::vector<AnfNodePtr> new_maketuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
  size_t input_size = make_tuple->inputs().size();
  bool has_environ_set = false;
  for (size_t i = 1; i < input_size; i++) {
    auto node = make_tuple->input(i);
    if (IsPrimitiveCNode(node, prim::kPrimEnvironSet) && OnlyUsedByOneNode(node, make_tuple)) {
      (void)env_nodes.emplace_back(node);
      has_environ_set = true;
    } else if (node->isa<CNode>() && !IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      (void)new_maketuple_inputs.emplace_back(node);
    }
  }
  if (!has_environ_set) {
    return nullptr;
  }
  // Check EnvironSet in MakeTuple
  auto mgr = GetManager(update_state);
  if (mgr == nullptr) {
    return nullptr;
  }
  // If exist EnvironGet, don't eliminate EnvironSet.
  if (ExistEnvironGet(mgr)) {
    return nullptr;
  }
  const size_t first_index = 1;
  const size_t attach_index = 3;
  const size_t no_env_node_size = new_maketuple_inputs.size();
  while (!env_nodes.empty()) {
    auto env = env_nodes.back();
    env_nodes.pop_back();
    if (!env->isa<CNode>()) {
      continue;
    }
    auto env_cnode = env->cast<CNodePtr>();
    auto env_input = env_cnode->input(first_index);
    auto attach = env_cnode->input(attach_index);
    if (IsPrimitiveCNode(env_input, prim::kPrimEnvironSet) && OnlyUsedByOneNode(env_input, env_cnode)) {
      (void)env_nodes.emplace_back(env_input);
      (void)new_maketuple_inputs.insert(new_maketuple_inputs.cbegin() + SizeToLong(no_env_node_size), attach);
    }
  }
  if (new_maketuple_inputs.size() == 1) {
    return nullptr;
  }
  auto fg = update_state->func_graph();
  if (fg == nullptr) {
    return nullptr;
  }
  abstract::AbstractBasePtrList element_abstracts;
  (void)std::transform(new_maketuple_inputs.begin() + 1, new_maketuple_inputs.end(),
                       std::back_inserter(element_abstracts),
                       [](const AnfNodePtr &input) { return input->abstract(); });
  auto new_make_tuple = fg->NewCNode(new_maketuple_inputs);
  new_make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(element_abstracts));
  auto new_update_state =
    fg->NewCNode({update_state->input(kFirstInputIndex), update_state->input(kInputIndex), new_make_tuple});
  new_update_state->set_abstract(update_state->abstract());
  new_update_state->set_scope(update_state->scope());
  return new_update_state;
}

AnfNodePtr EliminateUpdateStateMakeTupleWithUselessNode(const CNodePtr &update_state, const CNodePtr &make_tuple) {
  if (make_tuple->size() != kMakeTupleSize) {
    return nullptr;
  }
  AnfNodePtr attach_node = nullptr;
  auto &first_input = make_tuple->input(kInputIndex);
  auto &second_input = make_tuple->input(kAttachIndex);

  // Eliminate useless make_tuple with 'Dead Node'.
  // UpdateState(u, MakeTuple(input, "Dead Node")) -> UpdateState(u, input)
  auto abs = second_input->abstract();
  if (abs != nullptr && abs->isa<abstract::AbstractError>()) {
    return NewUpdateStateWithAttach(update_state, first_input);
  }

  // Eliminate useless make_tuple with useless Function.
  // UpdateState(u, MakeTuple(Function, input) -> UpdateState(u, input)
  // UpdateState(u, MakeTuple(input, Function) -> UpdateState(u, input)
  if (IsValueNode<FuncGraph>(first_input) && OnlyUsedByOneNode(first_input, make_tuple)) {
    return NewUpdateStateWithAttach(update_state, second_input);
  }
  if (IsValueNode<FuncGraph>(second_input) && OnlyUsedByOneNode(second_input, make_tuple)) {
    return NewUpdateStateWithAttach(update_state, first_input);
  }
  return nullptr;
}

void GetLoadsFollowLoad(const CNodePtr &update_state, const CNodePtr &load, std::vector<CNodePtr> *update_states,
                        std::vector<CNodePtr> *loads);
void GetLoadsFollowTuple(const CNodePtr &update_state, const CNodePtr &make_tuple, std::vector<CNodePtr> *update_states,
                         std::vector<CNodePtr> *loads);

// Search consecutive load nodes from UpdateState node.
void GetLoadsFromUpdateState(const CNodePtr &update_state, std::vector<CNodePtr> *update_states,
                             std::vector<CNodePtr> *loads) {
  auto &attach = update_state->input(kAttachIndex);
  if (IsPrimitiveCNode(attach, prim::kPrimLoad)) {
    GetLoadsFollowLoad(update_state, attach->cast<CNodePtr>(), update_states, loads);
  } else if (IsPrimitiveCNode(attach, prim::kPrimMakeTuple)) {
    GetLoadsFollowTuple(update_state, attach->cast<CNodePtr>(), update_states, loads);
  }
}

void GetLoadsFollowLoad(const CNodePtr &update_state, const CNodePtr &load, std::vector<CNodePtr> *update_states,
                        std::vector<CNodePtr> *loads) {
  (void)update_states->emplace_back(update_state);
  (void)loads->emplace_back(load);
  auto &load_attach = load->input(kAttachIndex);
  if (IsPrimitiveCNode(load_attach, prim::kPrimUpdateState)) {
    GetLoadsFromUpdateState(load_attach->cast<CNodePtr>(), update_states, loads);
  }
}

void GetLoadsFollowTuple(const CNodePtr &update_state, const CNodePtr &make_tuple, std::vector<CNodePtr> *update_states,
                         std::vector<CNodePtr> *loads) {
  if (!OnlyUsedByOneNode(make_tuple, update_state)) {
    // UpdateState should be the only user of make_tuple.
    return;
  }
  auto &inputs = make_tuple->inputs();
  const auto &monad = update_state->input(kInputIndex);
  bool is_all_load = std::all_of(inputs.begin() + 1, inputs.end(), [&monad](const AnfNodePtr &node) {
    // Tuple element should be Load and use same monad that UpdateState used.
    return (IsPrimitiveCNode(node, prim::kPrimLoad) && node->cast<CNodePtr>()->input(kAttachIndex) == monad);
  });
  if (!is_all_load) {
    // Stop if not all tuple elements are load nodes and use same monad.
    return;
  }
  // Add update_state and load nodes.
  (void)update_states->emplace_back(update_state);
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto &element = inputs.at(i);
    (void)loads->emplace_back(element->cast<CNodePtr>());
  }
  // Follow prev update state if found.
  auto prev_node = update_state->input(kInputIndex);
  if (IsPrimitiveCNode(prev_node, prim::kPrimUpdateState)) {
    GetLoadsFromUpdateState(prev_node->cast<CNodePtr>(), update_states, loads);
  }
}

// Create a MakeTuple node before UpdateState for same nodes, if there are more than 1 node used.
AnfNodePtr MakeTupleForSameNodes(const FuncGraphPtr &fg, const CNodePtr &old_update_state,
                                 const AnfNodePtrList &make_tuple_inputs) {
  constexpr size_t kOneNodeInputSize = 2;
  if (make_tuple_inputs.size() == kOneNodeInputSize) {
    // We don't need make_tuple since there is only one load.
    return make_tuple_inputs.at(1);
  }
  // Create MakeTuple cnode.
  auto make_tuple = fg->NewCNode(make_tuple_inputs);
  // Set abstract for the MakeTuple node.
  abstract::AbstractBasePtrList element_abstracts;
  std::transform(make_tuple_inputs.begin() + 1, make_tuple_inputs.end(), std::back_inserter(element_abstracts),
                 [](const AnfNodePtr &input) { return input->abstract(); });
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(element_abstracts));
  make_tuple->set_scope(old_update_state->scope());
  return make_tuple;
}

// Remove all nodes related to UpdateStates, if they're redundant.
void EliminateUselessNodesForUpdateStates(const std::vector<CNodePtr> &update_states) {
  if (update_states.empty()) {
    return;
  }
  auto mgr = GetManager(update_states[0]);
  if (mgr == nullptr) {
    return;
  }

  // 1. Remove the use of UpdateState nodes, except the last one.
  for (auto i = update_states.size() - 1; i > 0; i--) {
    auto &us = update_states[i];
    mgr->Replace(us, us->input(kInputIndex));
  }

  // 2. Remove the Depend users of last UpdateState node.
  auto &node_users = mgr->node_users();
  auto iter = node_users.find(update_states[0]);
  if (iter == node_users.end()) {
    return;
  }
  auto &us_users = iter->second;
  if (us_users.size() < 2) {
    return;
  }
  std::vector<AnfNodePtr> depend_nodes;
  for (auto &user : us_users) {
    if (IsPrimitiveCNode(user.first, prim::kPrimDepend) && user.second == kAttachIndex) {
      (void)depend_nodes.emplace_back(user.first);
    }
  }
  if (depend_nodes.empty()) {
    return;
  }
  ssize_t end = 0;
  // If all users are Depend CNode.
  if (depend_nodes.size() == us_users.size()) {
    end = 1;
    // Set abstract value for reserved Depend node.
    auto &reserved_depend_node = depend_nodes[0];
    auto &primary_node = reserved_depend_node->cast<CNodePtr>()->input(kInputIndex);
    reserved_depend_node->set_abstract(primary_node->abstract());
  }
  for (ssize_t i = depend_nodes.size() - 1; i >= end; i--) {
    const auto &depend_node = depend_nodes[i];
    const auto &depend_cnode = depend_node->cast<CNodePtr>();
    mgr->Replace(depend_cnode, depend_cnode->input(kInputIndex));
  }
}

// Eliminate UpdateStates for consecutive Loads.
// Convert:
//    x1 = Load(x1, u)
//    u1 = UpdateState(u, x1)
//    x2 = Load(x2, u1)
//    u2 = UpdateState(u1, x2)
//    ...
//    xN = Load(xN, u(N-1))
//    uN = UpdateState(u(N-1), xN)
// To:
//    x1 = Load(x1, u)
//    x2 = Load(x2, u)
//    ...
//    xN = Load(xN, u)
//    t = make_tuple(x1, x2, ... , xN)
//    u1 = UpdateState(u, t)
AnfNodePtr EliminateUpdateStateForLoads(const CNodePtr &old_update_state, const std::vector<CNodePtr> &update_states,
                                        const std::vector<CNodePtr> &loads) {
  auto fg = old_update_state->func_graph();
  if (fg == nullptr) {
    return nullptr;
  }
  auto mgr = fg->manager();
  if (mgr == nullptr) {
    return nullptr;
  }
  // Prepare tuple elements from Load nodes.
  AnfNodePtrList make_tuple_inputs;
  std::set<AnfNodePtr> loaded_para_set;
  make_tuple_inputs.reserve(loads.size() + 1);
  (void)make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  auto input_monad = loads.back()->input(kAttachIndex);
  for (auto iter = loads.rbegin(); iter != loads.rend(); ++iter) {
    auto &load = *iter;
    auto result = loaded_para_set.emplace(load->input(kInputIndex));
    const bool is_new_load = result.second;
    if (is_new_load) {
      // Put Load node as a tuple element, if the parameter is not loaded by other Load.
      (void)make_tuple_inputs.emplace_back(load);
    }
    auto load_attach = load->input(kAttachIndex);
    if (load_attach != input_monad) {
      // Set all load use same input monad.
      mgr->Replace(load_attach, input_monad);
    }
  }

  EliminateUselessNodesForUpdateStates(update_states);

  if (make_tuple_inputs.size() == 1) {
    // This should not happen.
    MS_LOG(WARNING) << "No loads for " << old_update_state->DebugString(2);
    return nullptr;
  }
  // Create the new UpdateState node with a MakeTuple, replace the old UpdateStateNode.
  auto attach = MakeTupleForSameNodes(fg, old_update_state, make_tuple_inputs);
  auto update_state = NewValueNode(prim::kPrimUpdateState);
  auto new_update_state = fg->NewCNode({update_state, input_monad, attach});
  new_update_state->set_abstract(old_update_state->abstract());
  new_update_state->set_scope(old_update_state->scope());
  return new_update_state;
}

// Eliminate UpdateStates between Assign nodes.
// Covert:
// a1 = Assign(para1, value1, u1)
// u2 = UpdateState(u1, a1)
// a2 = Assign(para2, value2, u2)  # para1 != para2, para1 != value2, para2 != value1
// u3 = UpdateState(u2, a2)
// To:
// a1 = Assign(para1, value1, u1)
// a2 = Assign(para2, value2, u1)
// t = MakeTuple(a1, a2)
// u3 = UpdateState(u1, t)
AnfNodePtr EliminateUpdateStateBetweenAssigns(const CNodePtr &update_state, const AnfNodePtr &assign) {
  auto a2_cnode = assign->cast<CNodePtr>();
  auto u2 = a2_cnode->input(kAssignMonadInputIndex);
  auto a1 = u2->cast<CNodePtr>()->input(kAttachIndex);
  if (IsPrimitiveCNode(a1, prim::kPrimAssign)) {
    auto a1_cnode = a1->cast<CNodePtr>();
    if (a1_cnode->size() != kAssignSize) {
      return nullptr;
    }
    auto para1 = a1_cnode->input(kInputIndex);
    auto value1 = a1_cnode->input(kAttachIndex);
    auto para2 = a2_cnode->input(kInputIndex);
    auto value2 = a2_cnode->input(kAttachIndex);
    auto u1 = a1_cnode->input(kAssignMonadInputIndex);
    if (para1 != para2 && para1 != value2 && para2 != value1) {
      auto fg = update_state->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      auto mgr = fg->manager();
      MS_EXCEPTION_IF_NULL(mgr);
      mgr->Replace(u2, u1);

      AnfNodePtrList make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), a1, assign};
      auto make_tuple = MakeTupleForSameNodes(fg, update_state, make_tuple_inputs);
      auto new_update_state = fg->NewCNode({NewValueNode(prim::kPrimUpdateState), u1, make_tuple});
      new_update_state->set_abstract(update_state->abstract());
      new_update_state->set_scope(update_state->scope());
      return new_update_state;
    }
  }
  return nullptr;
}

// Eliminate Load before Assign nodes.
// Covert:
// load = Load(parameter)
// a = Assign(load, value, u)
// To:
// a = Assign(parameter, value, u)
bool EliminateLoadBeforeAssigns(const FuncGraphManagerPtr &manager, const CNodePtr &update_state) {
  auto &attach = update_state->input(kAttachIndex);
  // UpdateState(u, Assign(para, value, u))
  if (IsPrimitiveCNode(attach, prim::kPrimAssign)) {
    auto assign = attach->cast<CNodePtr>();
    if (assign->size() != kAssignSize) {
      return false;
    }
    // If assign's first input is load, eliminate load.
    auto &ref_node = assign->input(kAssignRefInputIndex);
    if (IsPrimitiveCNode(ref_node, prim::kPrimLoad)) {
      auto load = ref_node->cast<CNodePtr>();
      auto &parameter = load->input(kInputIndex);
      // If Load used by other nodes, keep load node.
      auto assign_cnode = assign->cast<CNodePtr>();
      if (OnlyUsedByOneNode(ref_node, assign_cnode)) {
        (void)manager->Replace(ref_node, parameter);
      } else {
        manager->SetEdge(assign, kInputIndex, parameter);
      }
      return true;
    }
  }
  return false;
}

// Eliminate UpdateStates between MakeTuple and Assign.
// Covert:
// a1 = Assign(para1, value1, u1)
// a2 = Assign(para2, value2, u2)  # u2 == u1
// t = MakeTuple(a1, a2)
// u3 = UpdateState(u1, t)
// a3 = Assign(para3, value3, u3)  # para3 != para1, para3 != para2, value3 != para1, value3 != para2
//                                 # value1 != para3, value2 != para3
// u4 = UpdateState(u3, a3)
// To:
// a1 = Assign(para1, value1, u1)
// a2 = Assign(para2, value2, u1)
// a3 = Assign(para3, value3, u1)
// t = MakeTuple(a1, a2, a3)
// u4 = UpdateState(u1, t)
AnfNodePtr EliminateUpdateStateBetweenAssignMakeTuple(const CNodePtr &update_state, const AnfNodePtr &assign) {
  auto a3_cnode = assign->cast<CNodePtr>();
  auto u3 = a3_cnode->input(kAssignMonadInputIndex);
  auto u3_cnode = u3->cast<CNodePtr>();
  auto make_tuple = u3_cnode->input(kAttachIndex);
  if (IsPrimitiveCNode(make_tuple, prim::kPrimMakeTuple) && OnlyUsedByOneNode(make_tuple, u3_cnode)) {
    auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
    if (make_tuple_cnode->size() != kMakeTupleSize) {
      return nullptr;
    }
    auto a1 = make_tuple_cnode->input(kInputIndex);
    auto a2 = make_tuple_cnode->input(kAttachIndex);
    if (IsPrimitiveCNode(a1, prim::kPrimAssign) && IsPrimitiveCNode(a2, prim::kPrimAssign)) {
      auto a1_cnode = a1->cast<CNodePtr>();
      auto a2_cnode = a2->cast<CNodePtr>();
      if (a1_cnode->size() != kAssignSize || a2_cnode->size() != kAssignSize) {
        return nullptr;
      }
      auto para1 = a1_cnode->input(kInputIndex);
      auto value1 = a1_cnode->input(kAttachIndex);
      auto u1 = a1_cnode->input(kAssignMonadInputIndex);
      auto para2 = a2_cnode->input(kInputIndex);
      auto value2 = a2_cnode->input(kAttachIndex);
      auto u2 = a2_cnode->input(kAssignMonadInputIndex);
      auto para3 = a3_cnode->input(kInputIndex);
      auto value3 = a3_cnode->input(kAttachIndex);
      bool replace_judge = (u1 == u2) && (para1 != para3) && (para1 != value3) && (para2 != para3) &&
                           (para2 != value3) && (value1 != para3) && (value2 != para3);
      if (replace_judge) {
        auto fg = update_state->func_graph();
        MS_EXCEPTION_IF_NULL(fg);
        auto mgr = fg->manager();
        MS_EXCEPTION_IF_NULL(mgr);
        mgr->Replace(u3, u1);

        AnfNodePtrList new_make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), make_tuple_cnode->input(kInputIndex),
                                             make_tuple_cnode->input(kAttachIndex), assign};
        auto new_make_tuple = MakeTupleForSameNodes(fg, update_state, new_make_tuple_inputs);
        mgr->Replace(make_tuple, new_make_tuple);
        auto new_update_state = fg->NewCNode({NewValueNode(prim::kPrimUpdateState), u1, new_make_tuple});
        new_update_state->set_abstract(update_state->abstract());
        new_update_state->set_scope(update_state->scope());
        return new_update_state;
      }
    }
  }
  return nullptr;
}

// Eliminate UpdateStates between Assign and MakeTuple.
// Covert:
// a1 = Assign(para1, value1, u1)
// u2 = UpdateState(u1_1, a1)      # u1_1 == u1
// a2 = Assign(para2, value2, u2)
// a3 = Assign(para3, value3, u3)  # u2 == u3
// t = MakeTuple(a2, a3)
// u4 = UpdateState(u3, t)         # para3 != para1, para3 != para2, value3 != para1, value3 != para2
//                                 # value1 != para3, value1 != para3
// To:
// a1 = Assign(para1, value1, u1)
// a2 = Assign(para2, value2, u1)
// a3 = Assign(para3, value3, u1)
// t = MakeTuple(a1, a2, a3)
// u4 = UpdateState(u1, t)
AnfNodePtr EliminateUpdateStateBetweenMakeTupleAssign(const CNodePtr &update_state, const AnfNodePtr &make_tuple) {
  auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
  if (make_tuple_cnode->size() != kMakeTupleSize || !OnlyUsedByOneNode(make_tuple, update_state)) {
    return nullptr;
  }
  auto a2 = make_tuple_cnode->input(kInputIndex);
  auto a3 = make_tuple_cnode->input(kAttachIndex);
  if (IsPrimitiveCNode(a2, prim::kPrimAssign) && IsPrimitiveCNode(a3, prim::kPrimAssign)) {
    auto a2_cnode = a2->cast<CNodePtr>();
    auto a3_cnode = a3->cast<CNodePtr>();
    if (a2_cnode->size() != kAssignSize || a3_cnode->size() != kAssignSize) {
      return nullptr;
    }
    auto para2 = a2_cnode->input(kInputIndex);
    auto value2 = a2_cnode->input(kAttachIndex);
    auto u2 = a2_cnode->input(kAssignMonadInputIndex);
    if (!IsPrimitiveCNode(u2, prim::kPrimUpdateState) || !OnlyUsedByTwoNode(u2, a2, a3)) {
      return nullptr;
    }
    auto para3 = a3_cnode->input(kInputIndex);
    auto value3 = a3_cnode->input(kAttachIndex);
    auto u3 = a3_cnode->input(kAssignMonadInputIndex);
    if (u2 == u3) {
      auto u2_cnode = u2->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(u2_cnode);
      auto u1 = u2_cnode->input(kInputIndex);
      auto a1 = u2_cnode->input(kAttachIndex);
      if (IsPrimitiveCNode(a1, prim::kPrimAssign)) {
        auto a1_cnode = a1->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(a1_cnode);
        if (a1_cnode->size() != kAssignSize) {
          return nullptr;
        }
        auto para1 = a1_cnode->input(kInputIndex);
        auto value1 = a1_cnode->input(kAttachIndex);
        auto u1_1 = a1_cnode->input(kAssignMonadInputIndex);
        bool replace_judge = (u1 == u1_1) && (para1 != para2) && (para1 != para3) && (para1 != value2) &&
                             (para1 != value3) && (para2 != value1) && (para3 != value1);
        if (replace_judge) {
          auto fg = update_state->func_graph();
          MS_EXCEPTION_IF_NULL(fg);
          auto mgr = fg->manager();
          MS_EXCEPTION_IF_NULL(mgr);
          mgr->Replace(u2, u1);
          AnfNodePtrList new_make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), a1,
                                               make_tuple_cnode->input(kInputIndex),
                                               make_tuple_cnode->input(kAttachIndex)};
          auto new_make_tuple = MakeTupleForSameNodes(fg, update_state, new_make_tuple_inputs);
          mgr->Replace(make_tuple, new_make_tuple);
          auto new_update_state = fg->NewCNode({NewValueNode(prim::kPrimUpdateState), u1, new_make_tuple});
          new_update_state->set_abstract(update_state->abstract());
          new_update_state->set_scope(update_state->scope());
          return new_update_state;
        }
      }
    }
  }
  return nullptr;
}

AnfNodePtr EliminateUpdateStateForAssign(const CNodePtr &update_state) {
  // UpdateState(u, MakeTuple)
  auto &attach = update_state->input(kAttachIndex);
  if (IsPrimitiveCNode(attach, prim::kPrimMakeTuple)) {
    return EliminateUpdateStateBetweenMakeTupleAssign(update_state, attach);
  }
  // UpdateState(u, Assign(para, value, u))
  if (IsPrimitiveCNode(attach, prim::kPrimAssign)) {
    auto assign = attach->cast<CNodePtr>();
    if (assign->size() != kAssignSize) {
      return nullptr;
    }
    auto u = assign->input(kAssignMonadInputIndex);
    // u is UpdateState, u only be used by assign and update_state.
    if (IsPrimitiveCNode(u, prim::kPrimUpdateState) && OnlyUsedByTwoNode(u, assign, update_state)) {
      auto u_attach = u->cast<CNodePtr>()->input(kAttachIndex);
      if (IsPrimitiveCNode(u_attach, prim::kPrimAssign)) {
        return EliminateUpdateStateBetweenAssigns(update_state, assign);
      }
      if (IsPrimitiveCNode(u_attach, prim::kPrimMakeTuple)) {
        return EliminateUpdateStateBetweenAssignMakeTuple(update_state, assign);
      }
    }
  }
  return nullptr;
}
}  // namespace

// Eliminate useless node that only used by associated update_state.
AnfNodePtr UpdatestateUselessNodeEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  auto update_state_node = dyn_cast<CNode>(node);
  if (update_state_node == nullptr || update_state_node->size() != kUpdateStateSize) {
    return nullptr;
  }

  // If update_state is the only user of partial/load, replace it with the input monad.
  // UpdateState(u, Partial) -> u
  // UpdateState(u, Load) -> u
  // UpdateState(u, FuncGraph) -> u
  auto &attach = update_state_node->input(kAttachIndex);
  if (IsPrimitiveCNode(attach, prim::kPrimPartial) || IsPrimitiveCNode(attach, prim::kPrimLoad) ||
      IsValueNode<FuncGraph>(attach)) {
    // Replace UpdateState with the input monad.
    if (OnlyUsedByOneNode(attach, update_state_node)) {
      return update_state_node->input(kInputIndex);
    }
    return nullptr;
  }

  // Handling the case where the second input of update_state is make_tuple which contains DeadNode or useless function.
  // UpdateState(u, MakeTuple(input, "Dead Node")) -> UpdateState(u, input)
  // UpdateState(u, MakeTuple(Function, input) -> UpdateState(u, input)
  // UpdateState(u, MakeTuple(input, Function) -> UpdateState(u, input)
  if (IsPrimitiveCNode(attach, prim::kPrimMakeTuple)) {
    auto new_node = EliminateUpdateStateMakeTupleWithUselessNode(update_state_node, attach->cast<CNodePtr>());
    if (new_node != nullptr) {
      return new_node;
    }
    return EliminateUpdateStateMakeTupleWithUselessEnv(update_state_node, attach->cast<CNodePtr>());
  }
  return nullptr;
}

// Eliminate UpdateState that attaches a pure (no-side-effect) node.
// Convert:
//   x = pure_node(args) # no side effect
//   u1 = update_state(u, x)
//   user(u1)
// To:
//   x = pure_node(args)
//   user(u)
AnfNodePtr UpdatestatePureNodeEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  auto update_state_node = dyn_cast<CNode>(node);
  if (update_state_node == nullptr || update_state_node->size() != kUpdateStateSize) {
    return nullptr;
  }

  auto &attach = update_state_node->input(kAttachIndex);
  // update_state(u, param) or update_state(u, value_node) is redundant.
  auto cnode = dyn_cast<CNode>(attach);
  if (cnode == nullptr) {
    return update_state_node->input(kInputIndex);
  }
  const auto &first_input = cnode->input(0);
  bool is_special_ops = cnode->IsApply(prim::kPrimTupleGetItem) || cnode->IsApply(prim::kPrimDepend) ||
                        cnode->IsApply(prim::kPrimPartial) || cnode->IsApply(prim::kPrimMakeTuple) ||
                        cnode->IsApply(prim::kPrimCall) || IsValueNode<FuncGraph>(first_input) ||
                        IsPrimitiveCNode(first_input, prim::kPrimJ) || IsPrimitiveCNode(first_input, prim::kPrimVmap) ||
                        IsPrimitiveCNode(first_input, prim::kPrimTaylor) ||
                        IsPrimitiveCNode(first_input, prim::kPrimShard);
  if (is_special_ops) {
    return nullptr;
  }
  if (CheckHasMonadInput(cnode)) {
    return nullptr;
  }
  return update_state_node->input(kInputIndex);
}

// Eliminate redundant UpdateState/Depend pair nodes caused by inline.
// Convert:
//    x1 = Depend(x, u0)
//    u1 = UpdateState(u', x1)
//    out = x_user(x1)
//    u2 = u_user(u1)
// To:
//    out = x_user(x)
//    u2 = u_user(u0)
bool UpdatestateDependEliminater::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  // Filter nodes that do not match UpdateState(u, Depend).
  auto filter = [](const AnfNodePtr &node) {
    if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      auto update_state = node->cast<CNodePtr>();
      if (update_state->size() != kUpdateStateSize) {
        return true;
      }
      auto &attach = update_state->input(kAttachIndex);
      if (IsPrimitiveCNode(attach, prim::kPrimDepend)) {
        return false;
      }
    }
    return true;
  };

  bool change = false;
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &all_nodes = manager->all_nodes();
  std::vector<AnfNodePtr> todo = DeepScopedGraphSearchWithFilter(func_graph->get_return(), AlwaysInclude, filter);
  for (auto &node : todo) {
    if (node == nullptr || !all_nodes.contains(node)) {
      continue;
    }
    auto new_node = EliminateUpdateStateWithDepend(node->cast<CNodePtr>());
    if (new_node != nullptr) {
      manager->Replace(node, new_node);
      change = true;
    }
  }
  return change;
}

// Eliminate UpdateStates for consecutive Assign.
bool UpdatestateAssignEliminater::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  // Filter nodes that do not match UpdateState(u, Assign) or UpdateState(u, MakeTuple).
  auto filter = [](const AnfNodePtr &node) {
    if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      auto update_state = node->cast<CNodePtr>();
      if (update_state->size() != kUpdateStateSize) {
        return true;
      }
      auto &attach = update_state->input(kAttachIndex);
      if (IsPrimitiveCNode(attach, prim::kPrimAssign) || IsPrimitiveCNode(attach, prim::kPrimMakeTuple)) {
        return false;
      }
    }
    return true;
  };

  bool change = false;
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &all_nodes = manager->all_nodes();
  std::vector<AnfNodePtr> todo = DeepScopedGraphSearchWithFilter(func_graph->get_return(), AlwaysInclude, filter);
  for (auto &node : todo) {
    if (node == nullptr || !all_nodes.contains(node)) {
      continue;
    }
    auto new_node = EliminateUpdateStateForAssign(node->cast<CNodePtr>());
    if (new_node != nullptr) {
      manager->Replace(node, new_node);
      change = true;
    }
    bool load_eliminate = EliminateLoadBeforeAssigns(manager, node->cast<CNodePtr>());
    change = change || load_eliminate;
  }
  return change;
}

// Eliminate UpdateStates for consecutive Loads.
bool UpdatestateLoadsEliminater::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  // Filter nodes that do not match UpdateState(u, Load) or UpdateState(u, MakeTuple).
  auto filter = [](const AnfNodePtr &node) {
    if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      auto update_state = node->cast<CNodePtr>();
      if (update_state->size() != kUpdateStateSize) {
        return true;
      }
      auto &attach = update_state->input(kAttachIndex);
      if (IsPrimitiveCNode(attach, prim::kPrimLoad) || IsPrimitiveCNode(attach, prim::kPrimMakeTuple)) {
        return false;
      }
    }
    return true;
  };

  bool change = false;
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &all_nodes = manager->all_nodes();
  std::vector<AnfNodePtr> todo = DeepScopedGraphSearchWithFilter(func_graph->get_return(), AlwaysInclude, filter);
  for (auto &node : todo) {
    if (node == nullptr || !all_nodes.contains(node)) {
      continue;
    }
    std::vector<CNodePtr> update_states;
    std::vector<CNodePtr> loads;
    auto update_state_node = node->cast<CNodePtr>();
    GetLoadsFromUpdateState(update_state_node, &update_states, &loads);
    if (update_states.size() > 1 && loads.size() > 1) {
      auto new_node = EliminateUpdateStateForLoads(update_state_node, update_states, loads);
      if (new_node != nullptr) {
        manager->Replace(node, new_node);
        change = true;
      }
    }
  }
  return change;
}

// Eliminate Monad parameter for switch call.
// Convert:
//     x = Load(x, u)
//     u = UpdateState(u, x)
//     ...
//     g1 = Partial(...)
//     g2 = Partial(...)
//     s = switch(cond, g1, g2)
//     res = s(u)
// To:
//     x = Load(x, u)
//     u = UpdateState(u, x)
//     ...
//     g1 = Partial(..., u)
//     g2 = Partial(..., u)
//     s = switch(cond, g1, g2)
//     res = s()
AnfNodePtr SwitchCallMonadParameterEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  const CNodePtr &switch_call = dyn_cast<CNode>(node);
  if (switch_call == nullptr) {
    return nullptr;
  }
  auto fg = switch_call->func_graph();
  if (fg == nullptr) {
    return nullptr;
  }
  auto mgr = fg->manager();
  if (mgr == nullptr) {
    return nullptr;
  }
  const size_t switch_call_input_size = 2;
  if (switch_call->inputs().size() < switch_call_input_size) {
    return nullptr;
  }
  constexpr size_t primary_index = 0;
  auto switch_node = switch_call->input(primary_index);
  if (!IsPrimitiveCNode(switch_node, prim::kPrimSwitch)) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "Found switch call with monad parameter, " << switch_call->DebugString();
  auto switch_cnode = dyn_cast<CNode>(switch_node);
  if (switch_cnode == nullptr) {
    MS_LOG(EXCEPTION) << "switch node cast to CNode failed, " << switch_node->DebugString();
  }
  constexpr size_t condition_index = 1;
  constexpr size_t first_fg_index = 2;
  constexpr size_t second_fg_index = 3;
  auto fg1_node = switch_cnode->input(first_fg_index);
  auto fg2_node = switch_cnode->input(second_fg_index);
  auto build_partial = [&fg, &switch_call](const AnfNodePtr &node) {
    CNodePtr new_node;
    if (IsPrimitiveCNode(node, prim::kPrimPartial)) {  // Node is already Partial CNode.
      new_node = fg->NewCNode(node->cast<CNodePtr>()->inputs());
    } else {  // Node is FuncGraph ValueNode.
      new_node = fg->NewCNode({NewValueNode(prim::kPrimPartial), node});
    }
    constexpr size_t args_start_index = 1;
    for (size_t i = args_start_index; i < switch_call->inputs().size(); i++) {
      new_node->add_input(switch_call->input(i));
    }
    // partial's abstract is same with first input.
    new_node->set_abstract(new_node->input(1)->abstract());
    return new_node;
  };
  fg1_node = build_partial(fg1_node);
  fg2_node = build_partial(fg2_node);
  auto cond = switch_cnode->input(condition_index);
  auto new_switch_cnode = fg->NewCNode({NewValueNode(prim::kPrimSwitch), cond, fg1_node, fg2_node});
  auto new_switch_call = fg->NewCNode({new_switch_cnode});
  new_switch_cnode->set_abstract(switch_node->abstract());
  new_switch_call->set_abstract(switch_call->abstract());
  return new_switch_call;
}
}  // namespace mindspore::opt::irpass
