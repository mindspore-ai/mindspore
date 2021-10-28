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

#include "pipeline/jit/static_analysis/order_enforce.h"
#include <algorithm>
#include <map>
#include <queue>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "base/core_ops.h"

namespace mindspore::pipeline {
namespace {
class OrderEnforcer {
 public:
  explicit OrderEnforcer(const FuncGraphPtr &func_graph) : func_graph_(func_graph), manager_(func_graph->manager()) {
    MS_EXCEPTION_IF_NULL(func_graph_);
    MS_EXCEPTION_IF_NULL(manager_);
  }
  ~OrderEnforcer() = default;

  void Run() {
    auto nodes = MakeTopoSortMap();
    for (auto &node : nodes) {
      if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
        HandleUpdateState(node);
      } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
        // op(MakeTuple(Load, ...)) sometimes do not attach update_state,
        // So need special treatment in order to ensure the exec_order of MakeTuple users.
        HandleMakeTupleUsers(node);
      }
    }
  }

 private:
  AnfNodePtrList MakeTopoSortMap() {
    auto nodes = TopoSort(func_graph_->get_return());
    for (size_t i = 0; i < nodes.size(); ++i) {
      (void)topo_sort_map_.emplace(nodes[i], i);
    }
    return nodes;
  }

  void HandleUpdateState(const AnfNodePtr &node) {
    auto update_state = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(update_state);
    const size_t update_state_inputs_size = 3;
    if (update_state->inputs().size() < update_state_inputs_size) {
      MS_LOG(ERROR) << "UpdateState inputs size is less than 3, node is:" << update_state->DebugString();
    }
    if (!HasAbstractUMonad(update_state->input(1))) {
      // Skip UpdateStates for IO.
      return;
    }
    const size_t attach_index = 2;
    auto &attach = update_state->input(attach_index);
    if (IsPrimitiveCNode(attach, prim::kPrimLoad) && IsPrimitiveCNode(attach, prim::kPrimMakeTuple)) {
      // Skip UpdateState for Loads.
      return;
    }
    // Check previous update_state.
    auto &prev_u = update_state->input(1);
    if (!IsPrimitiveCNode(prev_u, prim::kPrimUpdateState)) {
      // Skip if previous is not UpdateState (maybe a U).
      return;
    }
    // Search side effect cnodes that use previous update_state as input.
    auto side_effect_nodes = FindNodeUsers(prev_u, [&update_state](const AnfNodePtr &user_node) {
      return (user_node != update_state) && !IsPrimitiveCNode(user_node, prim::kPrimLoad);
    });
    // For such side effect cnodes, try enfore order for them.
    for (auto &side_effect_node : side_effect_nodes) {
      HandleSideEffectNode(side_effect_node->cast<CNodePtr>(), prev_u->cast<CNodePtr>());
    }
  }

  bool HasLoadInput(const CNodePtr &cnode) {
    auto &inputs = cnode->inputs();
    return std::any_of(inputs.begin() + 1, inputs.end(),
                       [](const AnfNodePtr &input) { return IsPrimitiveCNode(input, prim::kPrimLoad); });
  }

  std::vector<AnfNodePtr> FindUpdateStateUsers(const AnfNodePtr &node) {
    auto &node_users = manager_->node_users();
    auto iter = node_users.find(node);
    if (iter == node_users.end()) {
      return {};
    }
    std::vector<AnfNodePtr> update_states;
    auto &users = iter->second;
    for (auto &user : users) {
      auto &user_node = user.first;
      if (IsPrimitiveCNode(user_node, prim::kPrimUpdateState)) {
        (void)update_states.emplace_back(user_node);
        continue;
      }
      if (IsPrimitiveCNode(user_node, prim::kPrimMakeTuple)) {
        auto make_tuple_users = FindUpdateStateUsers(user_node);
        (void)update_states.insert(update_states.end(), make_tuple_users.begin(), make_tuple_users.end());
      }
    }
    return update_states;
  }

  AnfNodePtr FindLastUpdateState(const CNodePtr &cnode) {
    auto &inputs = cnode->inputs();
    // Find all update_state nodes from the user of input load nodes.
    std::vector<AnfNodePtr> all_update_states;
    for (size_t index = 1; index < inputs.size(); index++) {
      auto &input = inputs[index];
      if (IsPrimitiveCNode(input, prim::kPrimLoad)) {
        std::vector<AnfNodePtr> update_states = FindUpdateStateUsers(input);
        (void)all_update_states.insert(all_update_states.end(), update_states.begin(), update_states.end());
      }
    }
    // Find the last update_state by topo sort order.
    auto last_update_state =
      std::max_element(all_update_states.begin(), all_update_states.end(),
                       [this](const AnfNodePtr &a, const AnfNodePtr &b) { return IsBefore(a, b); });
    if (last_update_state == all_update_states.end()) {
      return nullptr;
    }
    return *last_update_state;
  }

  // Convert:
  // load1 = Load(para1, u1)
  // load2 = Load(para2, u2)
  // maketuple1 = MakeTuple(inputs, load1, load2) # the make_tuple we should handle.
  // addn = AddN(maketupe1) # or other-op, user of the make_tuple
  // maketuple2 = MakeTuple(load1, load2)  # load user
  // u3 = UpdateState(u', maketuple2)  # the last update_state for load users.
  // assign = Assign(para2, inputs, u3)
  // To:
  // load1 = Load(para1, u1)
  // load2 = Load(para2, u2)
  // maketuple1 = MakeTuple(inputs, load1, load2)
  // addn = AddN(maketupe1)
  // maketuple2 = MakeTuple(load1, load2)
  // u3 = UpdateState(u', maketuple2, addn) # need put addn or other-op into u3 inputs
  // assign = Assign(para2, inputs, u3)
  void HandleMakeTupleUsers(const AnfNodePtr &node) {
    auto maketuple = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(maketuple);
    if (!HasLoadInput(maketuple)) {
      // MakeTuple without Load input.
      return;
    }
    // Find the last update_state node from users of input Loads.
    auto update_state = FindLastUpdateState(maketuple);
    if (update_state == nullptr) {
      return;
    }
    // Users of the make_tuple.
    auto maketuple_users = FindNodeUsers(maketuple, [](const AnfNodePtr &user_node) {
      // Push and Pull at the end of the execution order,
      // In order to ensure push and pull operator cut into the same graph,
      // we do not put push operator into updatestate.
      return !IsPrimitiveCNode(user_node, prim::kPrimPush);
    });
    // Attach make_tuple users to the update_state.
    auto update_state_cnode = update_state->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(update_state_cnode);
    AddInputEdges(update_state_cnode, maketuple_users);
  }

  bool IsRef(const AnfNodePtr &node) {
    auto &abs = node->abstract();
    return abs != nullptr && abs->isa<abstract::AbstractRef>();
  }

  bool IsSpecialPrimitive(const AnfNodePtr &node) const {
    return IsPrimitiveCNode(node, prim::kPrimExpandDims) || IsPrimitiveCNode(node, prim::kPrimBatchNormGrad);
  }

  void HandleSideEffectNode(const CNodePtr &cnode, const CNodePtr &update_state) {
    MS_EXCEPTION_IF_NULL(cnode);
    // Find refs from the cnode inputs.
    auto &inputs = cnode->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      auto &input = inputs[i];
      // Skip non-ref input and update_state.
      if (!IsRef(input) || input == update_state) {
        continue;
      }
      // The input is a ref (of parameter), find load nodes for it.
      auto loads = FindLoadNodes(input);
      for (auto &load : loads) {
        // Find user nodes of the Load.
        auto load_users = FindLoadUsers(load);
        std::unordered_set<AnfNodePtr> real_users;
        for (auto &load_user : load_users) {
          // Check the special operator, only one level of user is considered for now.
          if (IsSpecialPrimitive(load_user)) {
            auto special_real_users = FindNodeUsers(load_user);
            real_users.insert(special_real_users.begin(), special_real_users.end());
          } else {
            (void)real_users.insert(load_user);
          }
        }
        AddInputEdges(update_state, real_users);
      }
    }
  }

  bool IsInUpdateState(const AnfNodePtr &load_user, const CNodePtr &update_state) {
    MS_EXCEPTION_IF_NULL(update_state);
    const size_t attach_index = 2;
    const size_t input_size = update_state->inputs().size();
    for (size_t index = attach_index; index < input_size; index++) {
      auto &attach = update_state->input(attach_index);
      if (attach == load_user) {
        return true;
      }
      if (IsPrimitiveCNode(attach, prim::kPrimMakeTuple)) {
        auto attach_cnode = attach->cast<CNodePtr>();
        auto &inputs = attach_cnode->inputs();
        auto iter = std::find(inputs.begin() + 1, inputs.end(), load_user);
        if (iter != inputs.end()) {
          return true;
        }
      }
    }
    return false;
  }

  // Add load users as input edges of the update_state node.
  void AddInputEdges(const CNodePtr &update_state, const std::unordered_set<AnfNodePtr> &load_users) {
    auto sorted_load_users = SortLoadUsers(load_users);
    for (auto &load_user : sorted_load_users) {
      if (IsPrimitiveCNode(load_user, prim::kPrimMakeTuple) || IsPrimitiveCNode(load_user, prim::kPrimUpdateState)) {
        continue;
      }
      if (!IsDependOn(load_user, update_state)) {
        (void)processed_nodes_.insert(load_user);
        if (!IsInUpdateState(load_user, update_state)) {
          manager_->AddEdge(update_state, load_user);
        }
      }
    }
  }

  // Sort load users by their topo sort order.
  std::vector<AnfNodePtr> SortLoadUsers(const std::unordered_set<AnfNodePtr> &load_users) {
    std::vector<AnfNodePtr> vec{load_users.begin(), load_users.end()};
    std::sort(vec.begin(), vec.end(), [this](const AnfNodePtr &a, const AnfNodePtr &b) { return IsBefore(a, b); });
    return vec;
  }

  // Check if the load user node depend on the given UpdateState node.
  bool IsDependOn(const AnfNodePtr &load_user, const AnfNodePtr &update_state) {
    size_t update_state_order = topo_sort_map_[update_state];
    if (topo_sort_map_[load_user] < update_state_order) {
      return false;
    }
    auto user_cnode = dyn_cast<CNode>(load_user);
    if (user_cnode == nullptr) {
      return false;
    }
    size_t seen = NewSeenGeneration();
    std::queue<CNodePtr> q;
    user_cnode->seen_ = seen;
    q.push(user_cnode);
    while (!q.empty()) {
      auto cnode = q.front();
      q.pop();
      for (auto &input : cnode->inputs()) {
        if (input == update_state) {
          // Dependency found.
          return true;
        }
        if (input->seen_ == seen) {
          // Skip visited nodes.
          continue;
        }
        if (topo_sort_map_[input] < update_state_order) {
          // Skip input nodes that before the UpdateState node.
          continue;
        }
        auto input_cnode = dyn_cast<CNode>(input);
        if (input_cnode != nullptr) {
          input_cnode->seen_ = seen;
          q.push(input_cnode);
        }
      }
    }
    return false;
  }

  bool IsBefore(const AnfNodePtr &node1, const AnfNodePtr &node2) {
    return topo_sort_map_[node1] < topo_sort_map_[node2];
  }

  using PredFunc = std::function<bool(const AnfNodePtr &)>;

  // Find user nodes for the given node.
  std::unordered_set<AnfNodePtr> FindNodeUsers(const AnfNodePtr &node, const PredFunc &pred = nullptr) {
    auto &node_users = manager_->node_users();
    auto iter = node_users.find(node);
    if (iter == node_users.end()) {
      return {};
    }
    std::unordered_set<AnfNodePtr> users;
    for (auto &user : iter->second) {
      auto &user_node = user.first;
      if (pred == nullptr || pred(user_node)) {
        (void)users.emplace(user_node);
      }
    }
    return users;
  }

  // Find Load or parameter users as the candidate nodes to enforce order of execution.
  std::unordered_set<AnfNodePtr> FindLoadUsers(const AnfNodePtr &load_or_param) {
    return FindNodeUsers(load_or_param, [this](const AnfNodePtr &user_node) {
      // Skip processed nodes.
      return processed_nodes_.find(user_node) == processed_nodes_.end();
    });
  }

  // Find Load nodes for a parameter.
  std::unordered_set<AnfNodePtr> FindLoadNodes(const AnfNodePtr &param) {
    return FindNodeUsers(param, [this](const AnfNodePtr &user_node) {
      // Search for Load nodes only.
      return IsPrimitiveCNode(user_node, prim::kPrimLoad);
    });
  }

  const FuncGraphPtr &func_graph_;
  FuncGraphManagerPtr manager_;
  std::unordered_map<AnfNodePtr, size_t> topo_sort_map_;
  std::unordered_set<AnfNodePtr> processed_nodes_;
};
}  // namespace

// Enforce order of execution for Load users node.
void OrderEnforce(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  OrderEnforcer enforcer(func_graph);
  enforcer.Run();
  auto fg_used_total = func_graph->func_graphs_used_total();
  for (auto &fg : fg_used_total) {
    OrderEnforcer fg_enforcer(fg);
    fg_enforcer.Run();
  }
}
}  // namespace mindspore::pipeline
