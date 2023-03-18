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

#include "pipeline/jit/static_analysis/order_enforce.h"
#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/compact_set.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"

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
    // After ensuring the correct control edge relationship, then insert the TensorMove operator.
    // In order to store current value of parameter, insert TensorMove for Load:
    // whose refkey appears more than once,
    // or the load is input of call or partial,
    // or the first input of load is call or partial.
    std::vector<CNodePtr> need_insert_loads = GetNeedInsertLoads();
    for (auto &node : need_insert_loads) {
      InsertTensorMoveForLoad(node->cast<CNodePtr>());
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

  bool HasLoadInput(const CNodePtr &cnode) const {
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

  bool IsRef(const AnfNodePtr &node) const {
    auto &abs = node->abstract();
    return abs != nullptr && abs->isa<abstract::AbstractRefTensor>();
  }

  bool IsSpecialPrimitive(const AnfNodePtr &node) const {
    return IsPrimitiveCNode(node, prim::kPrimExpandDims) || IsPrimitiveCNode(node, prim::kPrimBatchNormGrad);
  }

  bool IsSpecialParallelPrimitive(const AnfNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitiveWithoutDoSignature(cnode);
    if (prim == nullptr) {
      return false;
    }
    if (prim->HasAttr(GRAPH_FLAG_ORDER_ENFORCE_SKIP)) {
      return true;
    }
    return false;
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
        mindspore::CompactSet<AnfNodePtr> real_users;
        for (auto &load_user : load_users) {
          // Check the special operator, only one level of user is considered for now.
          if (IsSpecialPrimitive(load_user)) {
            auto special_real_users = FindNodeUsers(load_user);
            real_users.insert(special_real_users.begin(), special_real_users.end());
          } else if (IsSpecialParallelPrimitive(load_user)) {
            auto parallel__users = FindParallelNodeUsers(load_user);
            real_users.insert(parallel__users.begin(), parallel__users.end());
          } else {
            real_users.insert(load_user);
          }
        }
        AddInputEdges(update_state, real_users);
      }
    }
  }

  bool IsInUpdateState(const AnfNodePtr &load_user, const CNodePtr &update_state) const {
    MS_EXCEPTION_IF_NULL(update_state);
    const size_t attach_index = 2;
    const size_t input_size = update_state->inputs().size();
    for (size_t index = attach_index; index < input_size; index++) {
      auto &attach = update_state->input(index);
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
  void AddInputEdges(const CNodePtr &update_state, const mindspore::CompactSet<AnfNodePtr> &load_users) {
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
  std::vector<AnfNodePtr> SortLoadUsers(const mindspore::CompactSet<AnfNodePtr> &load_users) {
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
    auto seen = NewSeenGeneration();
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
  mindspore::CompactSet<AnfNodePtr> FindNodeUsers(const AnfNodePtr &node, const PredFunc &pred = nullptr) {
    auto &node_users = manager_->node_users();
    auto iter = node_users.find(node);
    if (iter == node_users.end()) {
      return {};
    }
    mindspore::CompactSet<AnfNodePtr> users;
    for (auto &user : iter->second) {
      auto &user_node = user.first;
      if (pred == nullptr || pred(user_node)) {
        users.insert(user_node);
      }
    }
    return users;
  }

  // Find real user nodes for the given parallel nodes.
  mindspore::CompactSet<AnfNodePtr> FindParallelNodeUsers(const AnfNodePtr &node) {
    auto &node_users = manager_->node_users();
    auto iter = node_users.find(node);
    if (iter == node_users.end()) {
      return {};
    }
    mindspore::CompactSet<AnfNodePtr> users;
    for (auto &user : iter->second) {
      auto &user_node = user.first;
      if (!IsSpecialParallelPrimitive(user_node)) {
        users.insert(user_node);
      } else {
        mindspore::CompactSet<AnfNodePtr> real_users;
        real_users = FindParallelNodeUsers(user_node);
        users.insert(real_users.begin(), real_users.end());
      }
    }
    return users;
  }

  // Find Load or parameter users as the candidate nodes to enforce order of execution.
  mindspore::CompactSet<AnfNodePtr> FindLoadUsers(const AnfNodePtr &load_or_param) {
    return FindNodeUsers(load_or_param, [this](const AnfNodePtr &user_node) {
      // Skip processed nodes.
      return processed_nodes_.find(user_node) == processed_nodes_.end();
    });
  }

  // Find Load nodes for a parameter.
  mindspore::CompactSet<AnfNodePtr> FindLoadNodes(const AnfNodePtr &param) {
    return FindNodeUsers(param, [this](const AnfNodePtr &user_node) {
      // Search for Load nodes only.
      return IsPrimitiveCNode(user_node, prim::kPrimLoad);
    });
  }

  std::string GetRefKey(const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    auto abs = node->abstract();
    if (abs == nullptr) {
      if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
        return GetRefKey(node->cast<CNodePtr>()->input(1));
      }
      return "";
    }
    auto abs_ref = abs->cast<abstract::AbstractRefPtr>();
    if (abs_ref == nullptr) {
      return "";
    }
    auto ref_key = abs_ref->ref_key_value()->cast<StringImmPtr>();
    if (ref_key == nullptr) {
      return "";
    }
    return ref_key->value();
  }

  std::vector<CNodePtr> GetAllLoads(const AnfNodePtrList &check_nodes) const {
    std::vector<CNodePtr> need_insert_loads;
    for (auto &node : check_nodes) {
      if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
        auto load = node->cast<CNodePtr>();
        (void)need_insert_loads.emplace_back(load);
      }
    }
    return need_insert_loads;
  }

  using RefLoads = std::map<std::string, std::vector<CNodePtr>>;

  void AppendLoads(const RefLoads &loads_map, std::vector<CNodePtr> *need_insert_loads) const {
    for (auto &refkey_load_special : loads_map) {
      auto &loads = refkey_load_special.second;
      // If loads size > 1, mean has exist in refkey_loads.
      if (loads.size() == 1) {
        (void)need_insert_loads->emplace_back(loads[0]);
      }
    }
  }

  std::vector<CNodePtr> GetSpecialLoads(const RefLoads &loads_map1, const RefLoads &loads_map2,
                                        const RefLoads &loads_map3, const RefLoads &loads_map4,
                                        const std::set<CNodePtr> &call_nodes) const {
    std::vector<CNodePtr> need_insert_loads;
    for (auto &refkey_load : loads_map1) {
      auto &loads = refkey_load.second;
      if (loads.size() > 1) {
        (void)std::transform(loads.begin(), loads.end(), std::back_inserter(need_insert_loads),
                             [](const CNodePtr &load) { return load; });
      }
    }
    AppendLoads(loads_map2, &need_insert_loads);
    AppendLoads(loads_map3, &need_insert_loads);
    AppendLoads(loads_map4, &need_insert_loads);
    // Add call node will output is a AbstractRefTensor and ref_key is kValueAny.
    for (const auto &call_node : call_nodes) {
      if (std::find(need_insert_loads.begin(), need_insert_loads.end(), call_node) == need_insert_loads.end()) {
        need_insert_loads.push_back(call_node);
      }
    }
    return need_insert_loads;
  }

  bool CheckLoadInput(const AnfNodePtr &input) const {
    return IsPrimitiveCNode(input, prim::kPrimCall) || IsPrimitiveCNode(input, prim::kPrimPartial) ||
           (input->isa<CNode>() && (IsValueNode<FuncGraph>(input->cast<CNodePtr>()->input(0)) ||
                                    IsPrimitiveCNode(input->cast<CNodePtr>()->input(0), prim::kPrimSwitch) ||
                                    IsPrimitiveCNode(input->cast<CNodePtr>()->input(0), prim::kPrimSwitchLayer)));
  }

  void ProcessReturnLoad(const AnfNodePtr &node, const RefLoads &refkey_loads, RefLoads *refkey_loads_return_is_load) {
    auto return_input = node->cast<CNodePtr>()->input(1);
    while (IsPrimitiveCNode(return_input, prim::kPrimDepend)) {
      return_input = return_input->cast<CNodePtr>()->input(1);
    }
    auto check_load = [this, &refkey_loads, &refkey_loads_return_is_load](const AnfNodePtr &inp_node) {
      auto load = inp_node->cast<CNodePtr>();
      auto refkey = GetRefKey(load->input(1));
      if (refkey == "") {
        MS_LOG(INFO) << "Load without ref key:" << load->DebugString();
        return;
      }
      auto iter = refkey_loads.find(refkey);
      if (iter != refkey_loads.end()) {
        size_t load_size = iter->second.size();
        if (load_size > 1) {
          return;
        }
      }
      (void)(*refkey_loads_return_is_load)[refkey].emplace_back(load);
    };
    if (IsPrimitiveCNode(return_input, prim::kPrimMakeTuple)) {
      const auto &make_tuple = return_input->cast<CNodePtr>();
      const auto &make_tuple_inputs = make_tuple->inputs();
      if (make_tuple_inputs.size() <= 1) {
        return;
      }
      for (size_t i = 1; i < make_tuple_inputs.size(); ++i) {
        if (IsPrimitiveCNode(make_tuple_inputs[i], prim::kPrimLoad)) {
          check_load(make_tuple_inputs[i]);
        }
      }
    } else if (IsPrimitiveCNode(return_input, prim::kPrimLoad)) {
      check_load(return_input);
    }
  }

  std::vector<CNodePtr> GetNeedInsertLoads() {
    auto check_nodes = TopoSort(func_graph_->get_return());
    static const bool enable_all_load = common::GetEnv("MS_DEV_ENABLE_LOAD_INSERT_TENSORMOVE") == "1";
    // Insert TensorMove for all Load nodes
    if (enable_all_load) {
      return GetAllLoads(check_nodes);
    }
    RefLoads refkey_loads;
    RefLoads refkey_loads_in_call_or_partial;
    RefLoads refkey_loads_input_is_call_or_partial;
    RefLoads refkey_loads_return_is_load;
    std::set<CNodePtr> ref_call_nodes;
    for (auto &node : check_nodes) {
      // Record load refkey
      if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
        auto load = node->cast<CNodePtr>();
        auto input = load->input(1);
        if (CheckLoadInput(input)) {
          (void)ref_call_nodes.insert(load);
        }
        auto refkey = GetRefKey(input);
        if (refkey == "") {
          MS_LOG(INFO) << "Load without ref key:" << load->DebugString();
          continue;
        }
        (void)refkey_loads[refkey].emplace_back(load);
        while (IsPrimitiveCNode(input, prim::kPrimDepend)) {
          input = input->cast<CNodePtr>()->input(1);
        }
        // If Load(call/partial, monad), we should insert TensorMove for the load node.
        if (CheckLoadInput(input)) {
          (void)refkey_loads_input_is_call_or_partial[refkey].emplace_back(load);
        }
      }

      // Check if the return node is a load.
      if (IsPrimitiveCNode(node, prim::kPrimReturn)) {
        ProcessReturnLoad(node, refkey_loads, &refkey_loads_return_is_load);
      }

      // Find special load which is in call or partial
      if (!IsPrimitiveCNode(node, prim::kPrimCall) && !IsPrimitiveCNode(node, prim::kPrimPartial) &&
          !(node->isa<CNode>() && IsValueNode<FuncGraph>(node->cast<CNodePtr>()->input(0)))) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      for (size_t index = 1; index < cnode->inputs().size(); ++index) {
        auto input = cnode->input(index);
        if (IsPrimitiveCNode(input, prim::kPrimLoad)) {
          auto load = input->cast<CNodePtr>();
          auto refkey = GetRefKey(load->input(1));
          if (refkey == "") {
            MS_LOG(INFO) << "Load without ref key:" << load->DebugString();
            continue;
          }
          if (refkey_loads[refkey].size() > 1) {
            continue;
          }
          (void)refkey_loads_in_call_or_partial[refkey].emplace_back(load);
        }
      }
    }
    return GetSpecialLoads(refkey_loads, refkey_loads_in_call_or_partial, refkey_loads_input_is_call_or_partial,
                           refkey_loads_return_is_load, ref_call_nodes);
  }

  void InsertTensorMoveForLoad(const CNodePtr &node) {
    if (!IsPrimitiveCNode(node, prim::kPrimLoad)) {
      return;
    }
    auto prim = std::make_shared<Primitive>(kTensorMoveOpName);
    std::vector<AnfNodePtr> new_inputs{NewValueNode(prim)};
    (void)new_inputs.emplace_back(node);
    auto real_load = func_graph_->NewCNode(new_inputs);
    auto load_abs = node->abstract();
    auto abs_ref = dyn_cast_ptr<abstract::AbstractRefTensor>(load_abs);
    if (abs_ref != nullptr) {
      real_load->set_abstract(abs_ref->CloneAsTensor());
    } else {
      real_load->set_abstract(load_abs);
    }
    MS_LOG(DEBUG) << "Insert TensorMove " << real_load->DebugString() << " for load " << node->DebugString();
    (void)manager_->Replace(node, real_load);
  }

  const FuncGraphPtr &func_graph_;
  FuncGraphManagerPtr manager_;
  mindspore::HashMap<AnfNodePtr, size_t> topo_sort_map_;
  // As of now it's no requirement for insertion order, so use the unordered set.
  mindspore::HashSet<AnfNodePtr> processed_nodes_;
};
}  // namespace

// Enforce order of execution for Load users node.
void OrderEnforce(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  OrderEnforcer enforcer(func_graph);
  enforcer.Run();
  auto fg_used_total = func_graph->func_graphs_used_total();
  for (const auto &fg : fg_used_total) {
    OrderEnforcer fg_enforcer(fg);
    fg_enforcer.Run();
  }
}
}  // namespace mindspore::pipeline
