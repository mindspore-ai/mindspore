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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAMETER_ELIMINATE_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAMETER_ELIMINATE_H
#include <vector>
#include <utility>
#include <unordered_set>
#include <memory>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/manager.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {

class ParameterEliminator {
 public:
  ParameterEliminator() = default;
  virtual ~ParameterEliminator() = default;
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
    const auto &manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    bool changes = false;
    while (true) {
      const auto &[fg, callers] = SearchFuncGraphCallers(func_graph);
      if (fg == nullptr) {
        break;
      }
      auto manager = fg->manager();
      MS_EXCEPTION_IF_NULL(manager);
      const auto &erase_indexes = EraseUnusedParameters(fg, manager);
      for (auto caller : callers) {
        // Erase the corresponding args.
        EraseArgs(caller, erase_indexes, manager);
      }
      changes = true;
    }
    return changes;
  }

 private:
  static std::vector<CNodePtr> GetCallers(const FuncGraphPtr &fg) {
    const auto &fg_caller_and_indexes = fg->func_graph_cnodes_index();
    std::vector<CNodePtr> caller_cnodes = {};
    // Find all caller of fg.
    for (const auto &it : fg_caller_and_indexes) {
      const auto &fg_caller_and_index = it.first;
      auto caller_cnode = fg_caller_and_index->first;
      auto index = fg_caller_and_index->second;
      // If index != 0, the caller is a indirect caller, can't erase the parameter of graph.Because
      // in this situation ValueNode<FuncGraph> is a input of Return or of MakeTuple.
      if (index != 0) {
        return {};
      }
      caller_cnodes.push_back(caller_cnode->cast<CNodePtr>());
    }
    return caller_cnodes;
  }

  static std::pair<FuncGraphPtr, std::vector<CNodePtr>> SearchFuncGraphCallers(const FuncGraphPtr &func_graph) {
    for (const auto &fg : func_graph->func_graphs_used_total()) {
      if (fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE)) {
        continue;
      }
      const auto &parameters = fg->parameters();
      MS_EXCEPTION_IF_NULL(fg->manager());
      const auto &manager_node_users = fg->manager()->node_users();
      bool exist_param_unused =
        std::any_of(parameters.begin(), parameters.end(), [&manager_node_users](const AnfNodePtr &parameter) {
          const auto &node_users_it = manager_node_users.find(parameter);
          return node_users_it == manager_node_users.end() || node_users_it->second.empty();
        });
      if (exist_param_unused) {
        const auto &callers = GetCallers(fg);
        if (!callers.empty()) {
          return {fg, callers};
        }
      }
    }
    return {nullptr, {}};
  }

  static std::unordered_set<size_t> EraseUnusedParameters(const FuncGraphPtr &fg, const FuncGraphManagerPtr &manager) {
    MS_EXCEPTION_IF_NULL(fg->manager());
    const auto &manager_node_users = fg->manager()->node_users();
    const auto &parameters = fg->parameters();
    std::unordered_set<size_t> unused_parameter_indexes;
    // Traverse to find all unused parameters.
    size_t index = 0;
    for (const auto &parameter : parameters) {
      const auto &node_users_it = manager_node_users.find(parameter);
      if (node_users_it == manager_node_users.end() || node_users_it->second.empty()) {
        unused_parameter_indexes.insert(index);
      }
      index++;
    }
    // Erase unused parameters.
    std::vector<AnfNodePtr> new_parameters;
    for (size_t i = 0; i < parameters.size(); i++) {
      if (unused_parameter_indexes.find(i) == unused_parameter_indexes.end()) {
        new_parameters.push_back(parameters[i]);
      } else {
        MS_LOG(DEBUG) << "Erase parameter:" << parameters[i]->DebugString() << ",index:" << i;
      }
    }
    manager->SetParameters(fg, new_parameters);
    return unused_parameter_indexes;
  }

  static void EraseArgs(const CNodePtr &caller, const std::unordered_set<size_t> &unused_parameter_indexes,
                        const FuncGraphManagerPtr &manager) {
    std::vector<AnfNodePtr> new_args = {caller->inputs()[0]};
    for (size_t i = 0; i < caller->inputs().size() - 1; i++) {
      if (unused_parameter_indexes.find(i) == unused_parameter_indexes.end()) {
        new_args.push_back(caller->inputs()[i + 1]);
      } else {
        MS_LOG(DEBUG) << "Erase arg:" << caller->inputs()[i + 1]->DebugString() << ",index:" << i;
      }
    }
    TraceGuard trace_guard(std::make_shared<TraceCopy>(caller->debug_info()));
    auto new_caller = caller->func_graph()->NewCNode(new_args);
    new_caller->set_abstract(caller->abstract());
    manager->Replace(caller, new_caller);
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAMETER_ELIMINATE_H
