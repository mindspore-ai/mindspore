/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include <memory>

#include "abstract/abstract_value.h"
#include "utils/trace_info.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "frontend/optimizer/irpass.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/manager.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"
#include "ir/func_graph_flag.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace opt {
namespace irpass {
static inline void CheckSwitchCallValid(const CNodePtr &switch_call) {
  MS_EXCEPTION_IF_NULL(switch_call);
  if (switch_call->size() > 1) {
    // Means call switch(arg1, ...) has args.
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, switch_call)
      << "After switch_call_monad_eliminater pass, the call switch node should not has args."
      << " The call_switch_cnode is: " << switch_call->DebugString(AnfNode::DebugStringLevel::kLevel2);
  }
}

static inline std::vector<CNodePtr> GetCallers(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  const auto &fg_caller_and_indexes = fg->func_graph_cnodes_index();
  std::vector<CNodePtr> caller_cnodes = {};
  // Find all caller of fg.
  auto manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  for (const auto &it : fg_caller_and_indexes) {
    const auto &fg_caller_and_index = it.first;
    auto caller_cnode = fg_caller_and_index->first;
    auto index = fg_caller_and_index->second;
    // If index != 0, the caller is a indirect caller, can't erase the parameter of graph.
    // Because in this situation ValueNode<FuncGraph> is a input of Return or of MakeTuple.
    MS_LOG(DEBUG) << "index: " << index;
    // Process has partial func_graph with Primitive
    // %1 = Partial(func_graph, arg1, arg2, ...)
    if (index == 1 && IsPrimitiveCNode(caller_cnode, prim::kPrimPartial)) {
      auto iter = node_users.find(caller_cnode);
      for (auto &user : iter->second) {
        auto &user_node = user.first;
        auto user_cnode = user_node->cast<CNodePtr>();
        // Check user of partial (switch), the numbers of args should be 0.
        if (IsPrimitiveCNode(user_cnode, prim::kPrimSwitch)) {
          // Call switch()
          auto call_switchs = node_users[user_cnode];
          for (auto call_switch_iter : call_switchs) {
            CheckSwitchCallValid(call_switch_iter.first->cast<CNodePtr>());
          }
          if (std::find(caller_cnodes.begin(), caller_cnodes.end(), caller_cnode) == caller_cnodes.end()) {
            (void)caller_cnodes.emplace_back(caller_cnode->cast<CNodePtr>());
          }
        }
      }
    } else if (index != 0) {
      return {};
    } else {
      // Process call func_graph: %1 = func_graph(arg1, arg2, ...)
      (void)caller_cnodes.emplace_back(caller_cnode->cast<CNodePtr>());
    }
  }
  return caller_cnodes;
}

static inline std::pair<FuncGraphPtr, std::vector<CNodePtr>> SearchFuncGraphCallers(
  const FuncGraphPtr &func_graph, bool eliminate_only_returned_parameter) {
  for (const auto &fg : func_graph->func_graphs_used_total()) {
    if (fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) || fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH) ||
        fg->has_flag(FUNC_GRAPH_FLAG_ROLLED_HEADER)) {
      continue;
    }
    const auto &parameters = fg->parameters();
    MS_EXCEPTION_IF_NULL(fg->manager());
    const auto &manager_node_users = fg->manager()->node_users();
    // Check if no user parameter or only one user in output tuple.
    bool exist_param_unused =
      std::any_of(parameters.begin(), parameters.end(),
                  [&manager_node_users, &fg, eliminate_only_returned_parameter](const AnfNodePtr &parameter) {
                    const auto &node_users_it = manager_node_users.find(parameter);
                    // No user parameter.
                    if (node_users_it == manager_node_users.end() || node_users_it->second.empty()) {
                      return true;
                    }
                    // We will check the tuple output, if only one user.
                    if (eliminate_only_returned_parameter && fg->has_flag(FUNC_GRAPH_FLAG_NO_INLINE) &&
                        node_users_it->second.size() == 1) {
                      auto user = node_users_it->second.begin()->first;
                      // The parameter only used as returned MakeTuple's element.
                      if (IsPrimitiveCNode(user, prim::kPrimMakeTuple) && fg->output() == user) {
                        return true;
                      }
                    }
                    return false;
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

static inline void RemoveUnusedParametersFromGraph(const FuncGraphPtr &fg,
                                                   mindspore::HashSet<size_t> &unused_parameter_indexes) {
  MS_EXCEPTION_IF_NULL(fg);
  const FuncGraphManagerPtr &manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &parameters = fg->parameters();
  std::vector<AnfNodePtr> new_parameters;
  const auto &var_arg_node = fg->GetVariableArgParameter();
  const auto &kw_arg_node = fg->GetVariableKwargParameter();
  const auto &kw_only_args = fg->GetKwOnlyArgsParameters();
  const size_t fv_position = parameters.size() - fg->fv_param_count();

  auto is_kw_only_arg = [&kw_only_args](const AnfNodePtr &param) {
    return std::any_of(kw_only_args.cbegin(), kw_only_args.cend(),
                       [&param](const auto &kw_only_arg) { return kw_only_arg == param; });
  };

  for (size_t i = 0; i < parameters.size(); i++) {
    const auto &param_i = parameters[i];
    if (unused_parameter_indexes.find(i) == unused_parameter_indexes.end()) {
      (void)new_parameters.emplace_back(param_i);
      continue;
    }
    // VarArgs, KwArgs, KwOnlyArgs may not following the index as the Positional Arguments.
    if (param_i == var_arg_node) {
      fg->set_has_vararg(false);
      (void)unused_parameter_indexes.erase(i);
    } else if (param_i == kw_arg_node) {
      fg->set_has_kwarg(false);
      (void)unused_parameter_indexes.erase(i);
    } else if (is_kw_only_arg(param_i)) {
      if (fg->kwonlyargs_count() <= 0) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, fg->return_node())
          << "The kw_only_args_count is 0 when a kw_only_arg should be removed";
      }
      fg->set_kwonlyargs_count(fg->kwonlyargs_count() - 1);
      (void)unused_parameter_indexes.erase(i);
    }
    if (i >= fv_position) {
      fg->set_fv_param_count(fg->fv_param_count() - 1);
    }
    MS_LOG(DEBUG) << "Erase parameter: " << param_i->DebugString() << ", index: " << i;
  }
  manager->SetParameters(fg, new_parameters);
}

static inline std::pair<mindspore::HashSet<size_t>, mindspore::HashMap<size_t, size_t>> EraseUnusedParameters(
  const FuncGraphPtr &fg, bool eliminate_only_returned_parameter) {
  MS_EXCEPTION_IF_NULL(fg);
  const FuncGraphManagerPtr &manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &manager_node_users = manager->node_users();
  const auto &parameters = fg->parameters();
  mindspore::HashSet<size_t> unused_parameter_indexes;
  mindspore::HashMap<size_t, size_t> only_return_parameter_indexes;
  // Traverse to find all unused parameters.
  size_t index = 0;
  for (const auto &parameter : parameters) {
    const auto &node_users_it = manager_node_users.find(parameter);
    if (node_users_it == manager_node_users.end() || node_users_it->second.empty()) {
      (void)unused_parameter_indexes.emplace(index);
    } else if (eliminate_only_returned_parameter && fg->has_flag(FUNC_GRAPH_FLAG_NO_INLINE) &&
               node_users_it->second.size() == 1) {
      auto user = node_users_it->second.begin()->first;
      auto pos = node_users_it->second.begin()->second;
      // The parameter only used as returned MakeTuple's element.
      if (IsPrimitiveCNode(user, prim::kPrimMakeTuple) && fg->output() == user) {
        MS_LOG(DEBUG) << "Found only returned parameter[" << index << "] at output index[" << pos << "] of "
                      << user->DebugString();
        (void)only_return_parameter_indexes.emplace(pos, index);
        (void)unused_parameter_indexes.emplace(index);
        // Erase the unused element in returned MakeTuple CNode.
        auto user_cnode = dyn_cast<CNode>(user);
        MS_EXCEPTION_IF_NULL(user_cnode);
        auto zero_value = NewValueNode(MakeValue<int64_t>(0));
        zero_value->set_abstract(std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(0)));
        user_cnode->set_input(IntToSize(pos), zero_value);
      }
    }
    index++;
  }
  // Erase unused parameters.
  if (!unused_parameter_indexes.empty()) {
    RemoveUnusedParametersFromGraph(fg, unused_parameter_indexes);
  }

  return {unused_parameter_indexes, only_return_parameter_indexes};
}

static inline void UpdateAbstractFunctions(const CNodePtr &caller, const CNodePtr &new_caller) {
  auto origin_abs = caller->abstract();
  MS_EXCEPTION_IF_NULL(origin_abs);
  if (IsPrimitiveCNode(caller, prim::kPrimPartial) && origin_abs->isa<abstract::PartialAbstractClosure>()) {
    auto original_partial_func = origin_abs->cast<abstract::PartialAbstractClosurePtr>();
    original_partial_func->set_node(new_caller);
  }
  const FuncGraphManagerPtr &manager = caller->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_users = manager->node_users();
  const auto &iter = node_users.find(caller);
  if (iter == node_users.end() || iter->second.empty()) {
    return;
  }
  auto &all_users = iter->second;
  constexpr int switch_branch_pos = 2;
  for (auto &user : all_users) {
    auto node = user.first;
    auto index = user.second;
    MS_EXCEPTION_IF_NULL(node);
    auto union_abs = node->abstract();
    if (IsPrimitiveCNode(node, prim::kPrimSwitch) && union_abs->isa<abstract::AbstractFuncUnion>()) {
      auto func_union_abstract = dyn_cast<abstract::AbstractFuncUnion>(union_abs);
      const auto &func_list = func_union_abstract->func_list();
      if (SizeToInt(func_list.size()) <= index - switch_branch_pos) {
        MS_LOG(EXCEPTION) << "Func list size: " << func_list.size()
                          << " is not compatible with function position: " << index - switch_branch_pos;
      }
      auto branch_abs = func_list[index - switch_branch_pos];
      MS_EXCEPTION_IF_NULL(branch_abs);
      if (branch_abs->isa<abstract::PartialAbstractClosure>()) {
        auto branch_partial_func = branch_abs->cast<abstract::PartialAbstractClosurePtr>();
        branch_partial_func->set_node(new_caller);
      }
    }
  }
}

// Adjust the call arguments of func graph whose parameter's eliminated.
static inline void AdjustCallerArgs(const FuncGraphPtr &called, const CNodePtr &caller,
                                    const mindspore::HashSet<size_t> &unused_parameter_indexes) {
  size_t arg_start_index = 1;
  MS_EXCEPTION_IF_NULL(caller->func_graph());
  const FuncGraphManagerPtr &manager = caller->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> new_args = {caller->input(0)};
  if (IsPrimitiveCNode(caller, prim::kPrimPartial)) {
    (void)new_args.emplace_back(caller->input(1));
    arg_start_index = arg_start_index + 1;
  }
  for (size_t i = 0; i < caller->size() - arg_start_index; i++) {
    if (unused_parameter_indexes.find(i) == unused_parameter_indexes.end()) {
      (void)new_args.emplace_back(caller->input(i + arg_start_index));
    } else {
      MS_LOG(DEBUG) << "Erase arg: " << caller->input(i + arg_start_index)->DebugString();
    }
  }
  // Remove any Args which may be packed into VarArgs if VarArgs is not used in called FuncGraph;
  // Note: 1. If there is any *args or key=value argument in call site, it will be converted to unpack_call
  // CNode. So in this direct call case, all arguments should be plain arguments.
  //       2. The arguments in caller may be less than the formal parameters in called as some parameters can have
  //       default value.
  if (!called->has_vararg() &&
      caller->size() > (1 + IntToSize(called->GetPositionalArgsCount()) + called->fv_param_count())) {
    size_t start_offset = IntToSize(called->GetPositionalArgsCount()) + arg_start_index;
    size_t end_offset = called->fv_param_count();
    if (start_offset > new_args.size()) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, caller)
        << "The start_offset is " << start_offset << ", which exceeds the number of new args " << new_args.size()
        << ".";
    }
    (void)new_args.erase(new_args.cbegin() + SizeToLong(start_offset), new_args.cend() - SizeToLong(end_offset));
  }

  TraceGuard trace_guard(MakeTraceInfo<TraceCopy>(caller->debug_info()));
  auto new_caller = caller->func_graph()->NewCNode(new_args);
  new_caller->set_primal_attrs(caller->primal_attrs());
  new_caller->set_attrs(caller->attrs());
  new_caller->set_scope(caller->scope());
  UpdateAbstractFunctions(caller, new_caller);
  new_caller->set_abstract(caller->abstract());
  // Should be done before manager. Replace as caller CNode will be dropped after Replace, the ReplaceInOrder will be
  // no effect.
  caller->func_graph()->ReplaceInOrder(caller, new_caller);
  (void)manager->Replace(caller, new_caller);
}

// Adjust the caller(returned tuple)'s caller(getitem call)'s caller of func graph.
// Since the elements in returned tuple maybe eliminated,
// we should convert getitem(returned_tuple, x) into the eliminating argument itself.
static inline void AdjustGetItemCall(const CNodePtr &caller,
                                     const mindspore::HashMap<size_t, size_t> &only_return_parameter_indexes) {
  MS_EXCEPTION_IF_NULL(caller->func_graph());
  const FuncGraphManagerPtr &manager = caller->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (only_return_parameter_indexes.empty()) {
    return;
  }
  const auto &node_users = manager->node_users();
  const auto &iter = node_users.find(caller);
  if (iter == node_users.end() || iter->second.empty()) {
    return;
  }
  std::vector<std::pair<AnfNodePtr, AnfNodePtr>> replacing_nodes;
  auto &all_users = iter->second;
  for (auto &user : all_users) {
    auto node = user.first;
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      MS_LOG(ERROR) << "We expect a GetItem from the return tuple, but got " << node->DebugString();
      continue;
    }
    auto getitem_cnode = dyn_cast<CNode>(node);
    MS_EXCEPTION_IF_NULL(getitem_cnode);
    // Check if it's the eliminated element of returned tuple.
    constexpr size_t getitem_index_pos = 2;
    auto &index_node = getitem_cnode->input(getitem_index_pos);
    auto index_value = GetValueNode<Int64ImmPtr>(index_node);
    if (index_value == nullptr || index_value->value() < 0) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, index_node) << "The index_value is incorrect, " << index_node->DebugString();
    }
    size_t index_value_imm = LongToSize(index_value->value());
    const auto &index_pos = only_return_parameter_indexes.find(index_value_imm + 1);
    if (index_pos == only_return_parameter_indexes.end()) {
      continue;
    }

    // Found the tuple element, to replace it.
    auto eliminating_argument_pos = index_pos->second;
    MS_LOG(DEBUG) << "Found unused getitem CNode: " << getitem_cnode->DebugString() << ", index: " << index_value_imm
                  << ", eliminating_argument_pos: " << eliminating_argument_pos;
    // Replace the getitem CNode with the eliminated argument.
    auto &arg = caller->input(eliminating_argument_pos + 1);
    (void)replacing_nodes.emplace_back(std::pair(getitem_cnode, arg));
  }
  for (auto &nodes : replacing_nodes) {
    MS_LOG(DEBUG) << "Replace: " << nodes.first->DebugString() << ", with: " << nodes.second->DebugString();
    (void)manager->Replace(nodes.first, nodes.second);
  }
}

class ParameterEliminator {
 public:
  ParameterEliminator() = default;
  virtual ~ParameterEliminator() = default;
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &opt) {
    bool changes = false;
    while (true) {
      const auto &[fg, callers] = SearchFuncGraphCallers(func_graph, eliminate_only_returned_parameter_);
      if (fg == nullptr) {
        break;
      }
      const auto &[unused_parameter_indexes, only_return_parameter_indexes] =
        EraseUnusedParameters(fg, eliminate_only_returned_parameter_);
      for (auto caller : callers) {
        MS_LOG(DEBUG) << "caller: " << caller->DebugString();
        // Replace the getitem CNodes with the arguments.
        if (eliminate_only_returned_parameter_) {
          AdjustGetItemCall(caller, only_return_parameter_indexes);
        }
        // Erase the arguments for eliminated parameters.
        AdjustCallerArgs(fg, caller, unused_parameter_indexes);
      }
      changes = true;
    }
    return changes;
  }

  void set_eliminate_only_returned_parameter(bool eliminate_only_returned_parameter) {
    eliminate_only_returned_parameter_ = eliminate_only_returned_parameter;
  }

 private:
  bool eliminate_only_returned_parameter_{false};
};

class NoneParameterEliminator {
 public:
  NoneParameterEliminator() = default;
  ~NoneParameterEliminator() = default;
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &) {
    bool is_changed = false;
    for (const auto &fg : func_graph->func_graphs_used_total()) {
      if (fg == nullptr || fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) || fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH) ||
          fg->has_flag(FUNC_GRAPH_FLAG_ROLLED_HEADER)) {
        continue;
      }

      if (!IsNoneParameterExists(fg)) {
        MS_LOG(DEBUG) << "No None parameter in fg: " << fg->ToString();
        continue;
      }

      const auto &callers = GetCallers(fg);
      if (callers.empty()) {
        MS_LOG(DEBUG) << "Caller of fg is empty fg: " << fg->ToString();
        continue;
      }

      const auto &unused_parameter_indexes = EraseUnusedNoneParameters(fg);
      if (!unused_parameter_indexes.empty()) {
        is_changed = true;
        for (auto caller : callers) {
          AdjustCallerArgs(fg, caller, unused_parameter_indexes);
        }
      }
    }
    return is_changed;
  }

 private:
  static bool IsNoneParameterExists(const FuncGraphPtr &fg) {
    MS_EXCEPTION_IF_NULL(fg);
    const auto &parameters = fg->parameters();
    return std::any_of(parameters.begin(), parameters.end(), [](const AnfNodePtr &param) {
      return param != nullptr && param->abstract() != nullptr && param->abstract()->isa<abstract::AbstractNone>();
    });
  }

  static std::pair<FuncGraphPtr, int> GetCalledGraphAndParamIndex(const AnfNodePtr &node_user, int user_index) {
    if (node_user == nullptr || !node_user->isa<CNode>()) {
      return {nullptr, 0};
    }
    auto cnode = node_user->cast<CNodePtr>();
    if (cnode->inputs().empty()) {
      return {nullptr, 0};
    }

    constexpr auto kPartialFirstArgIndex = 2;
    constexpr auto kCallFirstArgIndex = 1;
    if (IsPrimitiveCNode(node_user, prim::kPrimPartial)) {
      if (cnode->inputs().size() > 1) {
        return {GetValueNode<FuncGraphPtr>(cnode->input(1)), user_index - kPartialFirstArgIndex};
      }
      return {nullptr, 0};
    }
    auto fg_input = cnode->input(0);
    return {GetValueNode<FuncGraphPtr>(fg_input), user_index - kCallFirstArgIndex};
  }

  static bool IsParameterOnlyUsedInRecursiveCall(const FuncGraphPtr &source_graph, const FuncGraphPtr &current_graph,
                                                 int parameter_index,
                                                 mindspore::HashSet<FuncGraphPtr> &visited_graphs) {
    if (source_graph == nullptr || current_graph == nullptr) {
      return false;
    }

    if (current_graph == source_graph) {
      return true;
    }

    // Prevent local recursive calls: fg0->fg1->fg2->fg1
    // fg0->fg1->fg2->fg1 will eliminate None parameter through the following steps: first fg1->fg2->fg1, then fg0.
    if (visited_graphs.find(current_graph) != visited_graphs.end()) {
      MS_LOG(DEBUG) << "Exist local recursive call current_graph: " << current_graph->ToString()
                    << ", source_graph: " << source_graph->ToString();
      return false;
    }
    visited_graphs.insert(current_graph);

    const FuncGraphManagerPtr &manager = current_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    const auto &node_users_map = manager->node_users();
    const auto &current_parameters = current_graph->parameters();

    if (parameter_index < 0 || IntToSize(parameter_index) >= current_parameters.size()) {
      return false;
    }

    const auto &none_parameter = current_parameters[parameter_index];
    MS_LOG(DEBUG) << "Checking none parameter: " << none_parameter->DebugString(AnfNode::DebugStringLevel::kLevel2);
    const auto &users_iterator = node_users_map.find(none_parameter);
    if (users_iterator == node_users_map.end() || users_iterator->second.empty()) {
      return true;
    }

    const auto &parameter_users = users_iterator->second;
    for (const auto &user_entry : parameter_users) {
      auto user_node = user_entry.first;
      MS_EXCEPTION_IF_NULL(user_node);
      MS_LOG(DEBUG) << "Parameter user node: " << user_node->DebugString(AnfNode::DebugStringLevel::kLevel2);
      auto user_index = user_entry.second;
      auto [called_graph, param_index] = GetCalledGraphAndParamIndex(user_node, user_index);
      if (called_graph == nullptr) {
        return false;
      }
      if (!IsParameterOnlyUsedInRecursiveCall(source_graph, called_graph, param_index, visited_graphs)) {
        return false;
      }
    }

    return true;
  }

  static mindspore::HashSet<size_t> EraseUnusedNoneParameters(const FuncGraphPtr &fg) {
    MS_EXCEPTION_IF_NULL(fg);
    const FuncGraphManagerPtr &manager = fg->manager();
    MS_EXCEPTION_IF_NULL(manager);
    const auto &manager_node_users = manager->node_users();
    const auto &parameters = fg->parameters();
    mindspore::HashSet<size_t> unused_parameter_indexes;
    // Pattern: fg0->fg1->fg2->...->fg0
    // In fg0: none_param only used in Partial(fg1, param_0, none_param)
    // In fg1: none_param only used in call(fg2, param_1, param_2, none_param)
    // In fg2: none_param only used in call(fg0, param_3, none_param)
    for (size_t index = 0; index < parameters.size(); ++index) {
      const auto &parameter = parameters[index];
      if (parameter == nullptr || parameter->abstract() == nullptr ||
          !parameter->abstract()->isa<abstract::AbstractNone>()) {
        continue;
      }

      const auto &node_users_it = manager_node_users.find(parameter);
      if (node_users_it == manager_node_users.end() || node_users_it->second.empty()) {
        MS_LOG(DEBUG) << "None parameter has no user: " << parameter->DebugString(AnfNode::DebugStringLevel::kLevel2);
        unused_parameter_indexes.emplace(index);
        continue;
      }

      // Check if all uses of the parameter form a recursive call chain
      const auto &node_users = node_users_it->second;
      bool used_by_real_op = false;
      bool used_by_recursive_call = false;
      for (const auto &node_user_it : node_users) {
        auto node_user = node_user_it.first;
        MS_EXCEPTION_IF_NULL(node_user);
        auto user_index = node_user_it.second;
        auto [called_graph, param_index] = GetCalledGraphAndParamIndex(node_user, user_index);
        if (called_graph == nullptr) {
          used_by_real_op = true;
          break;
        }

        mindspore::HashSet<FuncGraphPtr> visited_graphs;
        if (IsParameterOnlyUsedInRecursiveCall(fg, called_graph, param_index, visited_graphs)) {
          MS_LOG(DEBUG) << "None parameter is only used in recursive call node_user: "
                        << node_user->DebugString(AnfNode::DebugStringLevel::kLevel2);
          used_by_recursive_call = true;
        } else {
          used_by_recursive_call = false;
          break;
        }
      }

      if (!used_by_real_op && used_by_recursive_call) {
        MS_LOG(DEBUG) << "None parameter unused: " << parameter->DebugString(AnfNode::DebugStringLevel::kLevel2);
        unused_parameter_indexes.emplace(index);
      }
    }

    // Erase unused parameters.
    if (!unused_parameter_indexes.empty()) {
      RemoveUnusedParametersFromGraph(fg, unused_parameter_indexes);
    }

    return unused_parameter_indexes;
  }
};

class PartialUnusedArgsEliminate {
 public:
  PartialUnusedArgsEliminate() = default;
  virtual ~PartialUnusedArgsEliminate() = default;
  bool operator()(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    bool changed = false;
    auto fgs = func_graph->func_graphs_used_total();
    for (const auto &fg : fgs) {
      MS_EXCEPTION_IF_NULL(fg);
      std::vector<CNodePtr> partial_nodes;
      if (!GetUserPartialNodes(fg, &partial_nodes)) {
        continue;
      }
      std::vector<size_t> unused_parameter_idx;
      std::vector<AnfNodePtr> new_parameters;
      const auto &node_users = manager->node_users();
      const auto &origin_parameters = fg->parameters();
      bool added_forward_u = fg->has_flag(kFuncGraphFlagAddedForwardU);
      AnfNodePtr unused_arg_u = nullptr;
      for (size_t i = 0; i < origin_parameters.size(); ++i) {
        auto origin_para = origin_parameters[i];
        auto iter = node_users.find(origin_para);
        // Currently, we don't eliminate the function parameter node because it will produce DeadNode after renormalize.
        if (!HasAbstractFunction(origin_para) && (iter == node_users.end() || iter->second.empty())) {
          (void)unused_parameter_idx.emplace_back(i);
        } else if (added_forward_u && HasAbstractUMonad(origin_para) && i < origin_parameters.size() - 1) {
          // The fv u monad from fprop should be replaced with the forward u added by pass 'add_forward_monad_depend.h'.
          (void)unused_parameter_idx.emplace_back(i);
          unused_arg_u = origin_para;
        } else {
          (void)new_parameters.emplace_back(origin_para);
        }
      }
      if (unused_parameter_idx.empty()) {
        continue;
      }
      mindspore::HashMap<AnfNodePtr, AnfNodePtr> repl;
      if (!GetPartialRepl(partial_nodes, unused_parameter_idx, &repl)) {
        continue;
      }
      if (unused_arg_u != nullptr) {
        (void)manager->Replace(unused_arg_u, origin_parameters[origin_parameters.size() - 1]);
      }
      fg->set_parameters(new_parameters);
      auto tr = manager->Transact();
      for (auto &item : repl) {
        (void)tr.Replace(item.first, item.second);
      }
      tr.Commit();
      changed = true;
    }
    return changed;
  }

 private:
  static bool HasAbstractFunction(const AnfNodePtr &node) {
    return node->abstract() != nullptr && node->abstract()->isa<abstract::AbstractFunction>();
  }

  static bool GetUserPartialNodes(const FuncGraphPtr &fg, std::vector<CNodePtr> *partial_nodes) {
    for (const auto &node_and_idx : fg->func_graph_cnodes_index()) {
      auto user_node = node_and_idx.first->first;
      if (!IsPrimitiveCNode(user_node, prim::kPrimPartial)) {
        return false;
      }
      (void)partial_nodes->emplace_back(user_node->cast<CNodePtr>());
    }
    return true;
  }

  static bool GetPartialRepl(const std::vector<CNodePtr> &partial_nodes,
                             const std::vector<size_t> &unused_parameter_idx,
                             mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl) {
    constexpr auto kPartialFirstArgIndex = 2;
    for (const auto &partial : partial_nodes) {
      const auto &origin_partial_inputs = partial->inputs();
      std::vector<AnfNodePtr> new_partial_inputs;
      size_t j = 0;
      for (size_t i = 0; i < origin_partial_inputs.size(); ++i) {
        if (j < unused_parameter_idx.size() && i >= kPartialFirstArgIndex &&
            i - kPartialFirstArgIndex == unused_parameter_idx[j]) {
          ++j;
          continue;
        } else {
          (void)new_partial_inputs.emplace_back(origin_partial_inputs[i]);
        }
      }
      // The unused parameter should be one of the partial inputs.
      if (j < unused_parameter_idx.size()) {
        return false;
      }
      auto partial_fg = partial->func_graph();
      MS_EXCEPTION_IF_NULL(partial_fg);
      auto new_partial = partial_fg->NewCNode(new_partial_inputs);
      (void)repl->emplace(partial, new_partial);
    }
    return true;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAMETER_ELIMINATE_H
