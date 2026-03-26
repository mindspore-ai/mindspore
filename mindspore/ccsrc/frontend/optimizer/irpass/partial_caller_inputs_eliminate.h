/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARTIAL_CALLER_INPUTS_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARTIAL_CALLER_INPUTS_ELIMINATE_H_

#include <vector>
#include <unordered_set>

#include "frontend/optimizer/irpass.h"
#include "ir/manager.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
namespace irpass {
class PartialCallerInputsEliminate {
 public:
  PartialCallerInputsEliminate() = default;
  ~PartialCallerInputsEliminate() = default;

  bool operator()(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    bool changed = false;
    const auto &all_nodes = manager->all_nodes();
    for (const auto &node : all_nodes) {
      if (node == nullptr || !node->isa<CNode>()) {
        continue;
      }
      if (TryEliminateOnCNode(node->cast<CNodePtr>())) {
        changed = true;
      }
    }
    return changed;
  }

 private:
  static constexpr size_t kCallArgStartPos = 1;
  static constexpr size_t kEliminateMinimumSize = 2;  // Need retain one input for scheduler
  size_t partial_capture_arg_size_ = 0;

  bool TryEliminateOnCNode(const CNodePtr &partial) {
    MS_EXCEPTION_IF_NULL(partial);
    if (partial->abstract() == nullptr || !IsPrimitiveCNode(partial, prim::kPrimTupleGetItem) ||
        !partial->abstract()->isa<abstract::PartialAbstractClosure>()) {
      return false;
    }
    auto target_fg = GetPartialFuncGraph(partial);
    if (target_fg == nullptr) {
      return false;
    }
    auto manager = partial->func_graph()->manager();
    MS_EXCEPTION_IF_NULL(manager);
    bool changed = target_fg->has_user_data("unused_parameter_indexes");
    auto unused_parameter_indexes = GetUnusedParameterIndexes(target_fg);
    if (unused_parameter_indexes.empty()) {
      return false;
    }

    const auto users = manager->node_users()[partial];
    bool user_replaced = false;
    for (const auto &[user_node, user_index] : users) {
      auto user_cnode = user_node->cast<CNodePtr>();
      if (user_cnode == nullptr || user_index != 0) {
        continue;
      }
      if (user_cnode->size() < unused_parameter_indexes.size() + kEliminateMinimumSize) {
        return false;
      }
      auto new_user = BuildNewUser(user_cnode, partial, unused_parameter_indexes);
      (void)manager->Replace(user_cnode, new_user);
      user_replaced = true;
    }
    if (!changed && user_replaced) {
      EraseUnusedParameters(target_fg, unused_parameter_indexes);
    }
    return user_replaced;
  }

  FuncGraphPtr GetPartialFuncGraph(const CNodePtr &partial) {
    MS_EXCEPTION_IF_NULL(partial);
    if (partial->size() <= 1) {
      return nullptr;
    }
    auto abs = partial->abstract()->cast<abstract::PartialAbstractClosurePtr>();
    if (abs != nullptr) {
      partial_capture_arg_size_ = abs->args().size();
      auto abstract_fn = abs->fn();
      if (abstract_fn != nullptr && abstract_fn->isa<abstract::FuncGraphAbstractClosure>()) {
        return abstract_fn->cast<abstract::FuncGraphAbstractClosurePtr>()->func_graph();
      }
    }
    return nullptr;
  }

  std::unordered_set<size_t> GetUnusedParameterIndexes(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    if (func_graph->has_user_data("unused_parameter_indexes")) {
      return *(func_graph->user_data<std::unordered_set<size_t>>("unused_parameter_indexes"));
    }
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    const auto &node_users = manager->node_users();
    const auto &parameters = func_graph->parameters();
    std::unordered_set<size_t> unused_parameter_indexes;
    for (size_t i = partial_capture_arg_size_; i < parameters.size(); ++i) {
      const auto &param = parameters[i];
      auto iter = node_users.find(param);
      if (iter == node_users.end() || iter->second.empty()) {
        (void)unused_parameter_indexes.insert(i);
      }
    }
    func_graph->set_user_data<std::unordered_set<size_t>>(
      "unused_parameter_indexes", std::make_shared<std::unordered_set<size_t>>(unused_parameter_indexes));
    return unused_parameter_indexes;
  }

  static void EraseUnusedParameters(const FuncGraphPtr &func_graph,
                                    const std::unordered_set<size_t> &unused_parameter_indexes) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    const auto &parameters = func_graph->parameters();
    std::vector<AnfNodePtr> new_parameters;
    new_parameters.reserve(parameters.size() - unused_parameter_indexes.size());
    for (size_t i = 0; i < parameters.size(); ++i) {
      if (unused_parameter_indexes.count(i) == 0) {
        (void)new_parameters.emplace_back(parameters[i]);
      }
    }
    manager->SetParameters(func_graph, new_parameters);
  }

  CNodePtr BuildNewUser(const CNodePtr &old_user, const CNodePtr &old_partial,
                        const std::unordered_set<size_t> &unused_parameter_indexes) {
    MS_EXCEPTION_IF_NULL(old_user);
    MS_EXCEPTION_IF_NULL(old_partial);
    std::vector<AnfNodePtr> new_inputs{old_partial};
    for (size_t i = kCallArgStartPos; i < old_user->size(); ++i) {
      const auto parameter_index = partial_capture_arg_size_ + (i - kCallArgStartPos);
      if (unused_parameter_indexes.count(parameter_index) == 0) {
        (void)new_inputs.emplace_back(old_user->input(i));
      }
    }
    auto new_user = old_user->func_graph()->NewCNode(new_inputs);
    new_user->set_abstract(old_user->abstract());
    new_user->set_scope(old_user->scope());
    new_user->set_attrs(old_user->attrs());
    new_user->set_primal_attrs(old_user->primal_attrs());
    return new_user;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARTIAL_CALLER_INPUTS_ELIMINATE_H_
