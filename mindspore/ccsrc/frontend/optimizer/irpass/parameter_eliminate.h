/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "utils/hash_set.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/manager.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
static inline void CheckSwitchCallValid(const CNodePtr &switch_call) {
  if (switch_call->size() > 1) {
    // Means call switch(arg1, ...) has args.
    constexpr auto recursive_count = 2;
    MS_LOG(INTERNAL_EXCEPTION) << "After switch_call_monad_eliminater pass, the call switch node should not has args."
                               << " The call_switch_cnode is: " << switch_call->DebugString(recursive_count);
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
    if (fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) || fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
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
  std::vector<AnfNodePtr> new_parameters;
  const auto &var_arg_node = fg->GetVariableArgParameter();
  const auto &kw_arg_node = fg->GetVariableKwargParameter();
  const auto &kw_only_args = fg->GetKwOnlyArgsParameters();
  const size_t fv_position = parameters.size() - fg->fv_param_count();
  for (size_t i = 0; i < parameters.size(); i++) {
    const auto &param_i = parameters[i];
    if (unused_parameter_indexes.find(i) == unused_parameter_indexes.end()) {
      (void)new_parameters.emplace_back(param_i);
    } else {
      // VarArgs, KwArgs, KwOnlyArgs may not following the index as the Positional Arguments.
      if (param_i == var_arg_node) {
        fg->set_has_vararg(false);
        (void)unused_parameter_indexes.erase(i);
      } else if (param_i == kw_arg_node) {
        fg->set_has_kwarg(false);
        (void)unused_parameter_indexes.erase(i);
      } else {
        bool is_kw_only_arg = std::any_of(kw_only_args.cbegin(), kw_only_args.cend(),
                                          [param_i](const auto &kw_only_arg) { return kw_only_arg == param_i; });
        if (is_kw_only_arg) {
          if (fg->kwonlyargs_count() <= 0) {
            MS_LOG(INTERNAL_EXCEPTION) << "The kw_only_args_count is 0 when a kw_only_arg should be removed";
          }
          fg->set_kwonlyargs_count(fg->kwonlyargs_count() - 1);
          (void)unused_parameter_indexes.erase(i);
        }
      }
      if (i >= fv_position) {
        fg->set_fv_param_count(fg->fv_param_count() - 1);
      }
      MS_LOG(DEBUG) << "Erase parameter: " << param_i->DebugString() << ", index: " << i;
    }
  }
  manager->SetParameters(fg, new_parameters);
  return {unused_parameter_indexes, only_return_parameter_indexes};
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
      MS_LOG(INTERNAL_EXCEPTION) << "The start_offset is " << start_offset << ", which exceeds the number of new args "
                                 << new_args.size() << ".";
    }
    (void)new_args.erase(new_args.cbegin() + SizeToLong(start_offset), new_args.cend() - SizeToLong(end_offset));
  }

  TraceGuard trace_guard(std::make_shared<TraceCopy>(caller->debug_info()));
  auto new_caller = caller->func_graph()->NewCNode(new_args);
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
      MS_LOG(INTERNAL_EXCEPTION) << "The index_value is incorrect, " << index_node->DebugString();
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
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &) {
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
