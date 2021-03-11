/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

namespace mindspore::opt::irpass {
namespace {
// data = Load(input, attach)
// data = Depend(input, attach)
// monad = UpdateState(input, attach)
constexpr size_t kInputIndex = 1;
constexpr size_t kAttachIndex = 2;
constexpr size_t kMakeTupleSize = 3;
constexpr size_t kMinDependSize = 3;
constexpr size_t kAssignSize = 4;
constexpr size_t kAssignMonadInputIndex = 3;

FuncGraphManagerPtr GetManager(const AnfNodePtr &node) {
  auto fg = node->func_graph();
  if (fg == nullptr) {
    return nullptr;
  }
  return fg->manager();
}

// Return true if the node is only used by the given update_state node.
bool OnlyUpdateStateUse(const CNodePtr &update_state_node, const AnfNodePtr &node) {
  auto mgr = GetManager(update_state_node);
  if (mgr == nullptr) {
    return false;
  }
  auto &node_users = mgr->node_users();
  auto iter = node_users.find(node);
  if (iter == node_users.end()) {
    return false;
  }
  auto &partial_users = iter->second;
  return (partial_users.size() == 1) && (partial_users.front().first == update_state_node);
}

// Eliminate useless node that only used by associated update_state.
// Convert:
//   x1 = node(x, u)
//   u1 = update_state(u, x1) # update_state is the only user of node
//   user(u1)
// To:
//   user(u)
AnfNodePtr EliminateUpdateStateOnlyUsedNode(const CNodePtr &update_state, const AnfNodePtr &node) {
  if (!OnlyUpdateStateUse(update_state, node)) {
    // Skip if UpdateState is not the only user of cnode.
    return nullptr;
  }
  // Replace UpdateState with the input monad.
  return update_state->input(kInputIndex);
}

// Eliminate UpdateState that attaches a pure (no-side-effect) node.
// Convert:
//   x = pure_node(args) # no side effect
//   u1 = update_state(u, x)
//   user(u1)
// To:
//   x = pure_node(args)
//   user(u)
AnfNodePtr EliminateUpdateStateForPureNode(const CNodePtr &update_state, const AnfNodePtr &attach) {
  if (IsPrimitiveCNode(attach, prim::kPrimTupleGetItem)) {
    // Skip tuple_getitem.
    return nullptr;
  }
  auto cnode = dyn_cast<CNode>(attach);
  if (cnode == nullptr) {
    // Skip value node or parameter.
    return nullptr;
  }
  if (cnode->size() > 1) {
    // If the last input is a monad, means the attach node has side-effect and
    // we should keep UpdateState; otherwise, we will remove the UpdateState.
    if (HasAbstractMonad(cnode->inputs().back())) {
      return nullptr;
    }
  }
  // Remove UpdateState by replace it with its input monad.
  return update_state->input(kInputIndex);
}

// Eliminate redundant UpdateState/Depend pair nodes caused by inline.
// Convert:
//    x1 = Depend(x, u)
//    u1 = UpdateState(u, x1)
//    out = x_user(x1)
//    u2 = u_user(u1)
// To:
//    out = x_user(x)
//    u2 = u_user(u)
AnfNodePtr EliminateUpdateStateWithDepend(const CNodePtr &update_state, const CNodePtr &depend) {
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
    auto new_depend = fg->NewCNode(inputs);
    new_depend->set_abstract(depend->abstract());
    mgr->Replace(depend, new_depend);
  }
  // Replace UpdateState node with the input monad of Depend.
  return input_monad;
}

// Eliminate useless make_tuple with 'Dead Node'.
// Convert:
//   t = make_tuple(input, "Dead Node")
//   u = UpdateState(u, t)
// To:
//   u = UpdateState(u, input)
AnfNodePtr EliminateMakeTupleWithDeadNode(const CNodePtr &update_state, const CNodePtr &make_tuple) {
  if (make_tuple->size() != kMakeTupleSize) {
    return nullptr;
  }
  auto &node = make_tuple->input(kAttachIndex);
  auto node_abs = node->abstract();
  if (node_abs == nullptr || !node_abs->isa<abstract::AbstractError>()) {
    return nullptr;
  }
  auto fg = update_state->func_graph();
  if (fg == nullptr) {
    return nullptr;
  }
  // Create a new UpdateState to replace the old one.
  const auto &attach = make_tuple->input(kInputIndex);
  auto new_update_state = fg->NewCNode({update_state->input(0), update_state->input(1), attach});
  new_update_state->set_abstract(update_state->abstract());
  new_update_state->set_scope(update_state->scope());
  return new_update_state;
}

// Return true if the function is only used by make_tuple.
bool OnlyMakeTupleUseFunc(const CNodePtr &make_tuple, const AnfNodePtr &func_node) {
  auto mgr = GetManager(make_tuple);
  if (mgr == nullptr) {
    return false;
  }
  auto &node_users = mgr->node_users();
  auto iter = node_users.find(func_node);
  if (iter == node_users.end()) {
    return false;
  }
  auto &partial_users = iter->second;
  return (partial_users.size() == 1) && (partial_users.front().first == make_tuple);
}

// Eliminate UpdateState which the second input is MakeTuple, and the second input of MakeTuple is useless Function.
// Convert:
//   t = make_tuple(input, Function) or t = make_tuple(Function, input)
//   u2 = UpdateState(u1, t)
// To:
//   t = make_tuple(input, Function) or t = make_tuple(Function, input)
//   u2 = u1
AnfNodePtr EliminateUpdateStateWithMakeTupleFunc(const CNodePtr &update_state, const CNodePtr &make_tuple) {
  if (make_tuple->size() != kMakeTupleSize) {
    return nullptr;
  }
  auto &first_input = make_tuple->input(kInputIndex);
  if (IsValueNode<FuncGraph>(first_input) && OnlyMakeTupleUseFunc(make_tuple, first_input)) {
    return update_state->input(1);
  }
  auto &second_input = make_tuple->input(kAttachIndex);
  if (IsValueNode<FuncGraph>(second_input) && OnlyMakeTupleUseFunc(make_tuple, second_input)) {
    return update_state->input(1);
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
  update_states->emplace_back(update_state);
  loads->emplace_back(load);
  auto &load_attach = load->input(kAttachIndex);
  if (IsPrimitiveCNode(load_attach, prim::kPrimUpdateState)) {
    GetLoadsFromUpdateState(load_attach->cast<CNodePtr>(), update_states, loads);
  }
}

void GetLoadsFollowTuple(const CNodePtr &update_state, const CNodePtr &make_tuple, std::vector<CNodePtr> *update_states,
                         std::vector<CNodePtr> *loads) {
  if (!OnlyUpdateStateUse(update_state, make_tuple)) {
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
  update_states->emplace_back(update_state);
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto &element = inputs.at(i);
    loads->emplace_back(element->cast<CNodePtr>());
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
      depend_nodes.emplace_back(user.first);
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
  make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  auto input_monad = loads.back()->input(kAttachIndex);
  for (auto iter = loads.rbegin(); iter != loads.rend(); ++iter) {
    auto &load = *iter;
    auto result = loaded_para_set.emplace(load->input(kInputIndex));
    const bool is_new_load = result.second;
    if (is_new_load) {
      // Put Load node as a tuple element, if the parameter is not loaded by other Load.
      make_tuple_inputs.emplace_back(load);
    }
    if (load->input(kAttachIndex) != input_monad) {
      // Set all load use same input monad.
      mgr->SetEdge(load, kAttachIndex, input_monad);
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
  if (a2_cnode->size() != kAssignSize) {
    return nullptr;
  }
  auto para2 = a2_cnode->input(kInputIndex);
  auto value2 = a2_cnode->input(kAttachIndex);
  auto u2 = a2_cnode->input(kAssignMonadInputIndex);
  if (IsPrimitiveCNode(u2, prim::kPrimUpdateState)) {
    auto a1 = u2->cast<CNodePtr>()->input(kAttachIndex);
    if (IsPrimitiveCNode(a1, prim::kPrimAssign)) {
      auto a1_cnode = a1->cast<CNodePtr>();
      if (a1_cnode->size() != kAssignSize) {
        return nullptr;
      }
      auto para1 = a1_cnode->input(kInputIndex);
      auto value1 = a1_cnode->input(kAttachIndex);
      auto u1 = a1_cnode->input(kAssignMonadInputIndex);
      if (para1 != para2 && para1 != value2 && para2 != value1) {
        auto fg = update_state->func_graph();
        MS_EXCEPTION_IF_NULL(fg);
        auto mgr = fg->manager();
        mgr->Replace(u2, u1);
        AnfNodePtrList make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), a1, assign};
        auto make_tuple = MakeTupleForSameNodes(fg, update_state, make_tuple_inputs);
        auto new_update_state = fg->NewCNode({NewValueNode(prim::kPrimUpdateState), u1, make_tuple});
        new_update_state->set_abstract(update_state->abstract());
        new_update_state->set_scope(update_state->scope());
        return new_update_state;
      }
    }
  }
  return nullptr;
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
AnfNodePtr EliminateUpdateStateBetweenMakeTupleAssign(const CNodePtr &update_state, const AnfNodePtr &assign) {
  auto a3_cnode = assign->cast<CNodePtr>();
  if (a3_cnode->size() != kAssignSize) {
    return nullptr;
  }
  auto para3 = a3_cnode->input(kInputIndex);
  auto value3 = a3_cnode->input(kAttachIndex);
  auto u3 = a3_cnode->input(kAssignMonadInputIndex);
  if (IsPrimitiveCNode(u3, prim::kPrimUpdateState)) {
    auto make_tuple = u3->cast<CNodePtr>()->input(kAttachIndex);
    if (IsPrimitiveCNode(make_tuple, prim::kPrimMakeTuple)) {
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
AnfNodePtr EliminateUpdateStateBetweenAssignMakeTuple(const CNodePtr &update_state, const AnfNodePtr &make_tuple) {
  auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
  if (make_tuple_cnode->size() != kMakeTupleSize) {
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
    if (!IsPrimitiveCNode(u2, prim::kPrimUpdateState)) {
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

}  // namespace

AnfNodePtr UpdatestateEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  auto update_state_node = dyn_cast<CNode>(node);
  if (update_state_node == nullptr || update_state_node->inputs().empty()) {
    MS_LOG(WARNING) << "UpdatestateEliminater encounter invalid node: " << node->DebugString();
    return nullptr;
  }
  auto &attach = update_state_node->input(kAttachIndex);

  // Handle UpdateState(u, Depend(...)).
  if (IsPrimitiveCNode(attach, prim::kPrimDepend)) {
    return EliminateUpdateStateWithDepend(update_state_node, attach->cast<CNodePtr>());
  }

  // Handle UpdateState(u, Partial(...)).
  if (IsPrimitiveCNode(attach, prim::kPrimPartial)) {
    return EliminateUpdateStateOnlyUsedNode(update_state_node, attach);
  }

  // Handle UpdateState(u, Assign(...)).
  if (IsPrimitiveCNode(attach, prim::kPrimAssign)) {
    auto new_node = EliminateUpdateStateBetweenAssigns(update_state_node, attach);
    if (new_node != nullptr) {
      return new_node;
    }
    return EliminateUpdateStateBetweenMakeTupleAssign(update_state_node, attach);
  }

  // Handle UpdateState(u, Load(...)).
  const bool attach_is_load = IsPrimitiveCNode(attach, prim::kPrimLoad);
  if (attach_is_load) {
    auto new_node = EliminateUpdateStateOnlyUsedNode(update_state_node, attach);
    if (new_node != nullptr) {
      return new_node;
    }
  }

  // Handle UpdateState(u, MakeTuple(...)).
  const bool attach_is_tuple = IsPrimitiveCNode(attach, prim::kPrimMakeTuple);
  if (attach_is_tuple) {
    auto make_tuple = attach->cast<CNodePtr>();
    auto new_node = EliminateMakeTupleWithDeadNode(update_state_node, make_tuple);
    if (new_node != nullptr) {
      return new_node;
    }
    new_node = EliminateUpdateStateWithMakeTupleFunc(update_state_node, make_tuple);
    if (new_node != nullptr) {
      return new_node;
    }
    new_node = EliminateUpdateStateBetweenAssignMakeTuple(update_state_node, make_tuple);
    if (new_node != nullptr) {
      return new_node;
    }
  }
  // Merge UpdateStates for Loads.
  if (attach_is_load || attach_is_tuple) {
    std::vector<CNodePtr> update_states;
    std::vector<CNodePtr> loads;
    GetLoadsFromUpdateState(update_state_node, &update_states, &loads);
    if (update_states.size() > 1 && loads.size() > 1) {
      return EliminateUpdateStateForLoads(update_state_node, update_states, loads);
    }
    return nullptr;
  }
  // Eliminate UpdateStates that attaches a no-side-effect node.
  return EliminateUpdateStateForPureNode(update_state_node, attach);
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
AnfNodePtr EliminateMonadParameterForSwitchCall(const AnfNodePtr &node) {
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
  if (switch_call->inputs().size() < 2) {
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
    return new_node;
  };
  fg1_node = build_partial(fg1_node);
  fg2_node = build_partial(fg2_node);
  auto cond = switch_cnode->input(condition_index);
  auto new_switch_cnode = fg->NewCNode({NewValueNode(prim::kPrimSwitch), cond, fg1_node, fg2_node});
  auto new_switch_call = fg->NewCNode({new_switch_cnode});
  return new_switch_call;
}

AnfNodePtr SwitchCallMonadParameterEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  return EliminateMonadParameterForSwitchCall(node);
}
}  // namespace mindspore::opt::irpass
