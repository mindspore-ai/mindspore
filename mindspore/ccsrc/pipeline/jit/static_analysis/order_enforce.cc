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
      HandleNode(node);
    }
  }

 private:
  AnfNodePtrList MakeTopoSortMap() {
    auto nodes = TopoSort(func_graph_->get_return());
    for (size_t i = 0; i < nodes.size(); ++i) {
      topo_sort_map_.emplace(nodes[i], i);
    }
    return nodes;
  }

  void HandleNode(const AnfNodePtr &node) {
    if (!IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      // Skip nodes other than UpdateState.
      return;
    }
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
    auto updated_refs = FindUpdatedRefs(update_state);
    if (updated_refs.empty()) {
      // Skip UpdateStates that do not have updated refs.
      return;
    }
    auto &attach = update_state->input(2);
    if (IsPrimitiveCNode(attach, prim::kPrimLoad)) {
      // Handle UpdateState with Load.
      EnforceOrderForLoad(update_state, attach->cast<CNodePtr>(), updated_refs);
    } else if (IsPrimitiveCNode(attach, prim::kPrimMakeTuple)) {
      // Handle UpdateState with MakeTuple.
      EnforceOrderForTuple(update_state, attach->cast<CNodePtr>(), updated_refs);
    }
  }

  std::unordered_set<AnfNodePtr> FindUpdatedRefs(const CNodePtr &update_state) {
    std::unordered_set<AnfNodePtr> updated_refs;
    auto &users = manager_->node_users()[update_state];
    for (auto &user : users) {
      auto cnode = dyn_cast<CNode>(user.first);
      if (cnode == nullptr) {
        continue;
      }
      if (cnode->IsApply(prim::kPrimLoad) || cnode->IsApply(prim::kPrimDepend) ||
          cnode->IsApply(prim::kPrimUpdateState)) {
        continue;
      }
      for (auto &input : cnode->inputs()) {
        if (IsRef(input)) {
          updated_refs.insert(input);
        }
      }
    }
    return updated_refs;
  }

  bool IsRef(const AnfNodePtr &node) {
    auto &abs = node->abstract();
    return abs != nullptr && abs->isa<abstract::AbstractRef>();
  }

  void EnforceOrderForLoad(const CNodePtr &update_state, const CNodePtr &load,
                           const std::unordered_set<AnfNodePtr> &refs) {
    auto parameter = load->input(1);
    if (refs.find(parameter) == refs.end()) {
      // Skip if loaded parameter is not updated.
      return;
    }
    // Find load users, ignore processed nodes.
    auto load_users = FindUsers(load, update_state);
    auto parameter_users = FindUsers(parameter, update_state);
    load_users.insert(parameter_users.begin(), parameter_users.end());
    // Find load users that not depend on the UpdateState,
    // and than let UpdateState depend on them.
    AddInputEdges(update_state, load_users);
  }

  void EnforceOrderForTuple(const CNodePtr &update_state, const CNodePtr &make_tuple,
                            const std::unordered_set<AnfNodePtr> &refs) {
    // The UpdateState should be the only one user of the make_tuple.
    // for performance, we only check the number of output edges.
    if (manager_->node_users()[make_tuple].size() != 1) {
      return;
    }
    // Find load users from the tuple of Load nodes.
    std::unordered_set<AnfNodePtr> all_load_users;
    auto &inputs = make_tuple->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      auto &input = inputs.at(i);
      if (!IsPrimitiveCNode(input, prim::kPrimLoad)) {
        // Skip non-Load nodes.
        continue;
      }
      auto load = input->cast<CNodePtr>();
      auto parameter = load->input(1);
      if (refs.find(parameter) == refs.end()) {
        // Skip if loaded parameter is not updated.
        continue;
      }
      auto load_users = FindUsers(load, make_tuple);
      auto parameter_users = FindUsers(parameter, make_tuple);
      all_load_users.insert(parameter_users.begin(), parameter_users.end());
      all_load_users.insert(load_users.begin(), load_users.end());
    }
    // Find load users that not depend on the UpdateState,
    // and than let UpdateState depend on them.
    AddInputEdges(update_state, all_load_users);
  }

  bool IsInUpdateState(const AnfNodePtr &load_user, const CNodePtr &update_state) {
    const size_t attach_index = 2;
    const size_t input_size = update_state->inputs().size();
    for (size_t index = attach_index; index < input_size; index++) {
      auto attach = update_state->input(attach_index);
      if (IsPrimitiveCNode(attach, prim::kPrimMakeTuple)) {
        auto attach_cnode = attach->cast<CNodePtr>();
        auto inputs = attach_cnode->inputs();
        bool has_load_user =
          std::any_of(inputs.begin() + 1, inputs.end(), [load_user](const auto &input) { return input == load_user; });
        if (has_load_user) {
          return true;
        }
      } else if (attach == load_user) {
        return true;
      }
    }
    return false;
  }

  // Add load users as input edges of the update_state node.
  void AddInputEdges(const CNodePtr &update_state, const std::unordered_set<AnfNodePtr> &load_users) {
    auto sorted_load_users = SortLoadUsers(load_users);
    for (auto &load_user : sorted_load_users) {
      if (!IsDependOn(load_user, update_state)) {
        processed_nodes_.insert(load_user);
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

  // Find Load or parameter users as the candidate nodes to enforce order of execution.
  std::unordered_set<AnfNodePtr> FindUsers(const AnfNodePtr &load_or_param, const AnfNodePtr &exclude) {
    auto &node_users = manager_->node_users();
    auto iter = node_users.find(load_or_param);
    if (iter == node_users.end()) {
      return {};
    }
    std::unordered_set<AnfNodePtr> load_param_users;
    auto &users = iter->second;
    for (auto &user : users) {
      auto &user_node = user.first;
      if (user_node == exclude) {
        // Skip excluded node.
        continue;
      }
      if (processed_nodes_.find(user_node) != processed_nodes_.end()) {
        // Skip processed nodes.
        continue;
      }
      auto cnode = dyn_cast<CNode>(user_node);
      MS_EXCEPTION_IF_NULL(cnode);
      load_param_users.insert(cnode);
    }
    return load_param_users;
  }

 private:
  const FuncGraphPtr &func_graph_;
  FuncGraphManagerPtr manager_;
  std::unordered_map<AnfNodePtr, size_t> topo_sort_map_;
  std::unordered_set<AnfNodePtr> processed_nodes_;
};

}  // namespace

//
// Enforce order of execution for Load users node.
//
void OrderEnforce(const FuncGraphPtr &func_graph) {
  OrderEnforcer enforcer(func_graph);
  enforcer.Run();
  auto fg_used_total = func_graph->func_graphs_used_total();
  for (auto &fg : fg_used_total) {
    OrderEnforcer fg_enforcer(fg);
    fg_enforcer.Run();
  }
}
}  // namespace mindspore::pipeline
