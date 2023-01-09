/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "ir/func_graph_cloner.h"

#include <algorithm>

#include "ir/manager.h"
#include "ir/param_info.h"
#include "abstract/abstract_function.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/profile.h"
#include "utils/ms_context.h"
#include "ir/graph_utils.h"
#include "utils/parallel_node_check.h"
#include "mindspore/core/ops/core_ops.h"

// namespace to support intermediate representation definition
namespace mindspore {
namespace {
NodeDebugInfoPtr CloneNodeDebugInfo(const NodeDebugInfoPtr &debug_info, const TraceInfoPtr &relation) {
  auto trace_info = relation->clone();
  trace_info->set_debug_info(debug_info);
  return std::make_shared<NodeDebugInfo>(std::move(trace_info));
}

NodeDebugInfoPtr CloneNodeDebugInfo(const NodeDebugInfoPtr &debug_info) {
  auto trace_info = std::make_shared<TraceCopy>(debug_info);
  return std::make_shared<NodeDebugInfo>(std::move(trace_info));
}

GraphDebugInfoPtr CloneGraphDebugInfo(const GraphDebugInfoPtr &debug_info, const TraceInfoPtr &relation) {
  auto trace_info = relation->clone();
  trace_info->set_debug_info(debug_info);
  return std::make_shared<GraphDebugInfo>(std::move(trace_info));
}
}  // namespace

Cloner::Cloner(const FuncGraphVector &func_graphs, bool clone_all_valuenodes, bool clone_all_child_graphs,
               bool clone_all_used_graphs, const TraceInfoPtr &relation, const TraceInfoPtr &target_relation)
    : clone_all_valuenodes_(clone_all_valuenodes),
      clone_all_child_graphs_(clone_all_child_graphs),
      clone_all_used_graphs_(clone_all_used_graphs),
      relation_(relation),
      target_relation_(target_relation == nullptr ? relation : target_relation),
      scope_(kDefaultScope),
      type_(kBasic) {
  for (auto &func_graph : func_graphs) {
    AddClone(func_graph);
  }
}

void Cloner::AddClone(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph,
                      const AnfNodePtrList &params, CloneType type) {
  if (func_graph != nullptr) {
    (void)todo_.emplace_back(CloneInfo{func_graph, target_func_graph, params});
    type_ = type;
  }
}

void Cloner::CloneNode(const AnfNodePtr &node, const FuncGraphPtr &target) {
  MS_EXCEPTION_IF_NULL(node);
  if (repl_node_.find(node) != repl_node_.end()) {
    return;
  }
  if (node->isa<CNode>()) {
    CloneCNode(node, target);
  } else if (node->isa<Parameter>()) {
    CloneParameter(node, target, false);
  }
}

void Cloner::CloneParameter(const AnfNodePtr &node, const FuncGraphPtr &target, bool is_add) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(target);
  auto old_param = node->cast_ptr<Parameter>();
  MS_EXCEPTION_IF_NULL(old_param);
  auto debug_info = CloneNodeDebugInfo(node->debug_info(), relation_);
  auto new_param = (is_add ? target->add_parameter(std::move(debug_info))
                           : std::make_shared<Parameter>(target, std::move(debug_info)));
  new_param->set_abstract(old_param->abstract());
  new_param->set_name(old_param->name());
  if (old_param->has_default()) {
    // Default parameter can be shared since it is readonly.
    new_param->set_default_param(old_param->default_param());
  }
  new_param->set_is_top_graph_param(old_param->is_top_graph_param());
  ScopePtr scope = ((node->scope() == kDefaultScope) && (this->scope() != nullptr)) ? this->scope() : node->scope();
  new_param->set_scope(scope);
  repl_node_[node] = std::move(new_param);
}

void Cloner::CloneCNode(const AnfNodePtr &node, const FuncGraphPtr &target) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(target);
  auto old_node = node->cast<CNodePtr>();
  AnfNodePtrList inputs;
  inputs.reserve(old_node->size());
  NodeDebugInfoPtr debug_info;
  if (this->update_info() != nullptr && this->update_info()->debug_info_ != nullptr) {
    debug_info = this->update_info()->debug_info_;
  } else {
    debug_info = node->debug_info();
  }
  auto cloned_debug_info = CloneNodeDebugInfo(debug_info, relation_);
  CNodePtr new_node = std::make_shared<CNode>(std::move(inputs), target, std::move(cloned_debug_info));
  new_node->CloneCNodeInfo(old_node);
  ScopePtr scope;
  if (this->update_info() != nullptr && this->update_info()->scope_ != nullptr) {
    scope = this->update_info()->scope_;
  } else {
    scope = ((node->scope() == kDefaultScope) && (this->scope() != nullptr)) ? this->scope() : node->scope();
  }
  new_node->set_scope(scope);
  repl_node_[node] = std::move(new_node);
}

void Cloner::CloneValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast_ptr<ValueNode>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto debug_info = CloneNodeDebugInfo(node->debug_info(), relation_);
  ValueNodePtr new_const = NewValueNode(GetValueNode(node), std::move(debug_info));
  ScopePtr scope = ((node->scope() == kDefaultScope) && (this->scope() != nullptr)) ? this->scope() : node->scope();
  new_const->set_scope(scope);
  new_const->set_abstract(node->abstract());
  new_const->set_has_new_value(value_node->has_new_value());
  repl_node_[node] = std::move(new_const);
}

void Cloner::CloneFuncGraphValueNode(const AnfNodePtr &node, const FuncGraphPtr &target) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(target);
  auto value_node = node->cast_ptr<ValueNode>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto debug_info = CloneNodeDebugInfo(node->debug_info(), relation_);
  ValueNodePtr new_const = NewValueNode(target, std::move(debug_info));
  ScopePtr scope = ((node->scope() == kDefaultScope) && (this->scope() != nullptr)) ? this->scope() : node->scope();
  new_const->set_scope(scope);
  new_const->set_abstract(node->abstract());
  new_const->set_has_new_value(value_node->has_new_value());
  repl_node_[node] = std::move(new_const);
}

void Cloner::CloneValueNodes(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!clone_all_valuenodes_) {
    return;
  }
  auto &value_nodes = func_graph->value_nodes();
  for (auto &value_node : value_nodes) {
    auto &old_node = value_node.first;
    if (repl_node_.find(old_node) == repl_node_.end()) {
      CloneValueNode(old_node);
    }
  }
}

void Cloner::AddChildGraphs(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(manager_);
  if (!clone_all_child_graphs_) {
    return;
  }
  auto &scopes = manager_->scopes(func_graph);
  std::set<const FuncGraph *> memo;
  for (auto &graph : scopes) {
    // Avoid to insert duplicate function.
    if (graph != func_graph && memo.emplace(graph.get()).second) {
      (void)todo_.emplace_back(CloneInfo{graph, nullptr, {}});
    }
  }
}

void Cloner::AddTotalGraphs(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!clone_all_used_graphs_) {
    return;
  }
  std::set<const FuncGraph *> memo;
  auto &used = func_graph->func_graphs_used();
  for (auto &fg : used) {
    // Avoid to insert duplicate function.
    if (memo.emplace(fg.first.get()).second) {
      (void)todo_.emplace_back(CloneInfo{fg.first, nullptr, {}});
    }
  }
}

void Cloner::CloneFuncGraphDefaultValues(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  for (auto &item : func_graph->parameter_default_value()) {
    auto nodes = DeepLinkedGraphSearch(item.second);
    for (auto &node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      if (node->isa<CNode>()) {
        CloneNode(node, target_func_graph);
      } else if (node->isa<ValueNode>()) {
        CloneValueNode(node);
      }
    }
  }
}

void Cloner::CloneFuncGraphValueNodes(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);

  target_func_graph->set_stage(func_graph->stage());
  auto &old_return = func_graph->return_node();
  if (old_return != nullptr) {
    auto iter = repl_node_.find(old_return);
    if (iter == repl_node_.end()) {
      MS_LOG(EXCEPTION) << "Can't find replicate node for return.";
    }
    MS_EXCEPTION_IF_NULL(iter->second);
    auto return_node = iter->second->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(return_node);
    target_func_graph->set_return(return_node);
  }

  auto &cnodes = func_graph->func_graph_cnodes_index();
  for (auto &cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode.first);
    MS_EXCEPTION_IF_NULL(cnode.first->first);
    auto parent = cnode.first->first->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(parent);
    const auto &valuenode = parent->input(IntToSize(cnode.first->second));
    CloneFuncGraphValueNode(valuenode, target_func_graph);
  }
}

void Cloner::InlineCloneParameters(const FuncGraphPtr &func_graph, const AnfNodePtrList &params) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto &old_params = func_graph->parameters();
  if (old_params.size() != params.size()) {
    MS_EXCEPTION(TypeError) << "Origin params size[" << old_params.size() << "], inline params size[" << params.size()
                            << "]";
  }
  for (size_t i = 0; i < old_params.size(); ++i) {
    repl_node_[old_params[i]] = params[i];
  }
}

void Cloner::SetFuncGraphInfo(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  target_func_graph->set_attrs(func_graph->attrs());
  target_func_graph->set_transforms(func_graph->transforms());
  target_func_graph->set_has_vararg(func_graph->has_vararg());
  target_func_graph->set_has_kwarg(func_graph->has_kwarg());
  target_func_graph->set_kwonlyargs_count(func_graph->kwonlyargs_count());
  target_func_graph->set_fv_param_count(func_graph->fv_param_count());
  target_func_graph->set_is_generate(func_graph->is_generated());
  target_func_graph->set_stub(func_graph->stub());
  target_func_graph->set_switch_input(func_graph->switch_input());
  target_func_graph->set_switch_layer_input(func_graph->switch_layer_input());
  target_func_graph->set_python_obj(func_graph->python_obj());
}

void Cloner::CloneParameters(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  auto &params = func_graph->parameters();
  for (auto &param : params) {
    CloneParameter(param, target_func_graph, true);
  }
  repl_func_graph_[func_graph] = target_func_graph;
}

void Cloner::GenParameters(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto &free_vars = manager_->free_variables_total();
  auto iter = free_vars.find(func_graph);
  if (iter == free_vars.end()) {
    return;
  }

  CloneInfo item = todo_.back();
  auto lift_top_func_graph = item.origin;
  for (auto &fv_map : iter->second) {
    auto &free_var = fv_map.first;
    if (!utils::isa<AnfNodePtr>(free_var)) {
      continue;
    }
    auto free_var_node = utils::cast<AnfNodePtr>(free_var);
    // Don't lift weight parameter to top func_graph.
    if (IsLiftTopFuncGraph(func_graph) && free_var_node->isa<Parameter>()) {
      auto free_var_param = free_var_node->cast_ptr<Parameter>();
      if (free_var_param->has_default()) {
        MS_LOG(DEBUG) << "Bypass weight param: " << free_var_param->DebugString()
                      << " for top_func_graph: " << lift_top_func_graph->ToString();
        continue;
      }
    }
    auto &repl_node = repl_map_node_[func_graph];
    if (repl_node.find(free_var_node) != repl_node.end()) {
      MS_LOG(DEBUG) << "Param exists: " << free_var_node->DebugString()
                    << " for func_graph: " << func_graph->ToString();
      continue;
    }

    MS_LOG(DEBUG) << "Gen param: " << free_var_node->ToString() << " for func_graph: " << func_graph->ToString();
    auto fv_parameter = AddParameter(func_graph, free_var_node);
    fv_parameter->set_user_data<bool>("lifted_from_fv", std::make_shared<bool>(true));
    auto &fg_params = repl_func_graph_params_[func_graph];
    (void)fg_params.emplace_back(fv_parameter);
  }
}

void Cloner::CloneParameter(const ParameterPtr &param, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(node);
  param->set_abstract(node->abstract());
  if (node->isa<Parameter>()) {
    auto old_param = node->cast_ptr<Parameter>();
    if (old_param->has_default()) {
      // Default parameter can be shared since it is readonly.
      param->set_default_param(old_param->default_param());
    }
    param->set_name(old_param->name());
    constexpr char lifted_user_data_key[] = "lifted_from_fv";
    auto lifted = param->user_data<bool>(lifted_user_data_key);
    if (lifted != nullptr && *lifted) {
      param->set_user_data<bool>(lifted_user_data_key, std::make_shared<bool>(true));
    }
  }
}

ParameterPtr Cloner::AddParameter(const FuncGraphPtr &func_graph, const AnfNodePtr &node, bool is_add) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto debug_info = CloneNodeDebugInfo(node->debug_info());
  ParameterPtr param = std::make_shared<Parameter>(func_graph, std::move(debug_info));
  CloneParameter(param, node);
  if (is_add) {
    func_graph->add_parameter(param);
  }
  repl_node_[param] = node;
  repl_map_node_[func_graph][node] = param;
  return param;
}

void Cloner::AddParameters(const FuncGraphPtr &func_graph, const AnfNodePtrList &params,
                           AnfNodePtrList *const lift_params, AnfNodePtrList *const input_params) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(lift_params);
  MS_EXCEPTION_IF_NULL(input_params);
  AnfNodePtrList parameters;
  mindspore::HashSet<AnfNodePtr> old_params;
  for (auto &param : func_graph->parameters()) {
    auto iter = repl_node_.find(param);
    if (iter != repl_node_.end()) {
      (void)old_params.insert(iter->second);
      parameters.push_back(param);
    } else {
      parameters.push_back(AddParameter(func_graph, param, false));
      (void)old_params.insert(param);
    }
  }
  AnfNodePtr new_param = nullptr;
  for (auto &param : params) {
    auto old_param = repl_node_[param];
    if (old_param->isa<CNode>() && old_param->func_graph() == func_graph) {
      repl_node_[old_param] = old_param;
      repl_map_node_[func_graph][old_param] = old_param;
      input_params->push_back(old_param);
      continue;
    }
    if (old_params.find(old_param) != old_params.end()) {
      new_param = repl_map_node_[func_graph][old_param];
      if (new_param == nullptr) {
        MS_LOG(EXCEPTION) << "map_node, func_graph: " << func_graph->ToString()
                          << ", old_param: " << old_param->DebugString() << " cannot found";
      }
      input_params->push_back(new_param);
      continue;
    }
    if (IsLiftTopFuncGraph(func_graph)) {
      // Don't lift parameter from used_graphs to my parameter if I am the top;
      repl_node_[old_param] = old_param;
      repl_map_node_[func_graph][old_param] = old_param;
      MS_EXCEPTION_IF_NULL(old_param->func_graph());
      repl_map_node_[old_param->func_graph()][old_param] = old_param;
      input_params->push_back(old_param);
      MS_LOG(DEBUG) << "Bypass param: " << old_param->DebugString()
                    << " for top_func_graph: " << func_graph->ToString();
      continue;
    }
    new_param = AddParameter(func_graph, old_param, false);
    parameters.push_back(new_param);
    lift_params->push_back(new_param);
    input_params->push_back(new_param);
  }
  func_graph->set_parameters(std::move(parameters));
}

namespace {
void FilterMonadInput(const AnfNodePtrList &old_inputs, AnfNodePtrList *new_inputs, AnfNodePtr *possible_u_monad,
                      AnfNodePtr *possible_io_monad) {
  AnfNodePtr local_u_monad = nullptr, local_io_monad = nullptr;
  (void)std::copy_if(old_inputs.cbegin(), old_inputs.cend(), std::back_inserter(*new_inputs),
                     [&local_u_monad, &local_io_monad](const auto &input) -> bool {
                       if (HasAbstractUMonad(input)) {
                         if (local_u_monad != nullptr) {
                           MS_LOG(EXCEPTION)
                             << "Cannot have multiple U Monad in one call, first: " << local_u_monad->ToString()
                             << ", second: " << input->ToString();
                         }
                         local_u_monad = input;
                         return false;
                       }
                       if (HasAbstractIOMonad(input)) {
                         if (local_io_monad != nullptr) {
                           MS_LOG(EXCEPTION)
                             << "Cannot have multiple IO Monad in one call, first: " << local_io_monad->ToString()
                             << ", second: " << input->ToString();
                         }
                         local_io_monad = input;
                         return false;
                       }
                       return true;
                     });
  *possible_u_monad = local_u_monad;
  *possible_io_monad = local_io_monad;
}

// After lift, func_graph will not refer any free variable, so DummyContext is proper.
AnfNodePtr BuildFuncGraphValueNode(const FuncGraphPtr &func_graph) {
  auto new_node = NewValueNode(func_graph);
  auto abstract = std::make_shared<abstract::FuncGraphAbstractClosure>(
    func_graph, abstract::AnalysisContext::DummyContext(), new_node, true);
  new_node->set_abstract(abstract);
  return new_node;
}

AnfNodePtr BuildPrimitiveValueNode(const PrimitivePtr &primitive) {
  auto new_node = NewValueNode(primitive);
  auto abstract = std::make_shared<abstract::PrimitiveAbstractClosure>(primitive, new_node);
  new_node->set_abstract(abstract);
  return new_node;
}
}  // namespace

bool Cloner::IsLiftTopFuncGraph(const FuncGraphPtr &func_graph) {
  const auto &iter = std::find_if(todo_.begin(), todo_.end(),
                                  [func_graph](const CloneInfo &item) -> bool { return item.origin == func_graph; });
  if (iter == todo_.end()) {
    return false;
  }
  return true;
}

void Cloner::AddInputs(const FuncGraphPtr &func_graph_user, const FuncGraphPtr &func_graph,
                       const AnfNodePtrList &params) {
  auto &repl_func_graph = repl_map_func_graph_[func_graph_user];
  auto [iter, inserted] = repl_func_graph.emplace(func_graph, nullptr);
  if (inserted) {
    auto value_node = BuildPrimitiveValueNode(prim::kPrimPartial);
    AnfNodePtrList cnode_inputs{value_node, BuildFuncGraphValueNode(func_graph)};
    auto partial_node = func_graph_user->NewCNode(std::move(cnode_inputs));
    iter->second = partial_node;
  }
  auto cnode = dyn_cast<CNode>(iter->second);
  if (cnode == nullptr) {
    return;
  }
  AnfNodePtr input_u_monad;
  AnfNodePtr input_io_monad;
  AnfNodePtr param_u_monad;
  AnfNodePtr param_io_monad;
  AnfNodePtrList inputs;
  std::vector<AnfNodePtr> add_params;
  FilterMonadInput(cnode->inputs(), &inputs, &input_u_monad, &input_io_monad);
  FilterMonadInput(params, &add_params, &param_u_monad, &param_io_monad);

  constexpr auto caller_first_arg_index = 2;
  for (size_t i = caller_first_arg_index; i < inputs.size(); i++) {
    auto pos = std::find(add_params.begin(), add_params.end(), inputs[i]);
    if (pos != add_params.end()) {
      (void)add_params.erase(pos);
    }
  }
  if (input_u_monad != nullptr && param_u_monad != nullptr && input_u_monad != param_u_monad) {
    MS_LOG(EXCEPTION) << "Cannot have multiple U Monad in one call, first: " << input_u_monad->ToString()
                      << ", second: " << param_u_monad->ToString();
  }
  if (input_io_monad != nullptr && param_io_monad != nullptr && input_io_monad != param_io_monad) {
    MS_LOG(EXCEPTION) << "Cannot have multiple IO Monad in one call, first: " << input_io_monad->ToString()
                      << ", second: " << param_io_monad->ToString();
  }
  (void)inputs.insert(inputs.end(), add_params.begin(), add_params.end());
  auto &u_monad = (input_u_monad != nullptr ? input_u_monad : param_u_monad);
  auto &io_monad = (input_io_monad != nullptr ? input_io_monad : param_io_monad);
  if (u_monad != nullptr) {
    inputs.push_back(u_monad);
  }
  if (io_monad != nullptr) {
    inputs.push_back(io_monad);
  }
  cnode->set_inputs(inputs);
  OrderParameters(func_graph, inputs, caller_first_arg_index);

  AbstractBasePtrList args_spec_list;
  (void)std::for_each(inputs.begin() + caller_first_arg_index, inputs.end(),
                      [&args_spec_list](const AnfNodePtr &node) { args_spec_list.push_back(node->abstract()); });
  MS_EXCEPTION_IF_NULL(func_graph->ToAbstract());
  auto abs = std::make_shared<abstract::PartialAbstractClosure>(
    func_graph->ToAbstract()->cast<abstract::AbstractFuncAtomPtr>(), args_spec_list, cnode);
  cnode->set_abstract(abs);
}

void Cloner::OrderParameters(const FuncGraphPtr &func_graph, const AnfNodePtrList &inputs, size_t arg_start_index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  mindspore::HashSet<AnfNodePtr> old_params;
  for (auto &param : func_graph->parameters()) {
    (void)old_params.insert(repl_node_[param]);
  }
  mindspore::HashSet<AnfNodePtr> new_params;
  AnfNodePtrList parameters;
  // Ignore the 1st and 2nd param of inputs(such as. partial graph)
  for (size_t i = arg_start_index; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    const auto &param = repl_node_[input];
    if (old_params.find(param) != old_params.end()) {
      auto &new_param = repl_map_node_[func_graph][param];
      parameters.push_back(new_param);
      (void)new_params.insert(new_param);
    }
  }
  for (auto &param : func_graph->parameters()) {
    if (new_params.find(param) == new_params.end()) {
      parameters.push_back(param);
    }
  }
  func_graph->set_parameters(std::move(parameters));
}

void Cloner::SetEdges(const FuncGraphPtr &func_graph, FuncGraphTransaction *tx) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(tx);
  for (auto &node : func_graph->nodes()) {
    auto cnode = dyn_cast<CNode>(node);
    // Only cnode needed to be handled
    if (cnode == nullptr) {
      continue;
    }
    auto &inputs = cnode->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
      auto &input = inputs[i];
      if (IsValueNode<FuncGraph>(input)) {
        auto graph = GetValueNode<FuncGraphPtr>(input);
        auto &repl_func_graph = repl_map_func_graph_[func_graph];
        if (repl_func_graph.find(graph) != repl_func_graph.end()) {
          tx->SetEdge(cnode, static_cast<int>(i), repl_func_graph[graph]);
        }
      } else {
        auto &repl_node = repl_map_node_[func_graph];
        if (repl_node.find(input) != repl_node.end()) {
          tx->SetEdge(cnode, static_cast<int>(i), repl_node[input]);
        }
      }
    }
  }
}

void Cloner::LiftParameters(const FuncGraphPtr &func_graph_user, const FuncGraphPtr &func_graph,
                            const AnfNodePtrList &params) {
  MS_EXCEPTION_IF_NULL(func_graph_user);
  AnfNodePtrList lift_params;
  AnfNodePtrList input_params;
  AddParameters(func_graph_user, params, &lift_params, &input_params);
  AddInputs(func_graph_user, func_graph, input_params);
  if (lift_params.empty()) {
    return;
  }
  for (auto &cnode : func_graph_user->func_graph_cnodes_index()) {
    MS_EXCEPTION_IF_NULL(cnode.first);
    MS_EXCEPTION_IF_NULL(cnode.first->first);
    LiftParameters(cnode.first->first->func_graph(), func_graph_user, lift_params);
  }
}

void Cloner::Lift(const std::vector<FuncGraphPtr> &sorted) {
  // lift inner graph first
  for (auto r_iter = sorted.rbegin(); r_iter != sorted.rend(); ++r_iter) {
    auto func_graph = *r_iter;
    auto iter = repl_func_graph_params_.find(func_graph);
    if (iter != repl_func_graph_params_.end()) {
      auto &params = iter->second;
      for (auto &cnode : func_graph->func_graph_cnodes_index()) {
        MS_EXCEPTION_IF_NULL(cnode.first);
        MS_EXCEPTION_IF_NULL(cnode.first->first);
        LiftParameters(cnode.first->first->func_graph(), func_graph, params);
      }
    }
  }
}

void Cloner::SetEdgesBfs(const FuncGraphPtr &root_fg, FuncGraphTransaction *tx) {
  MS_EXCEPTION_IF_NULL(root_fg);
  const auto &func_graphs = BroadFirstSearchGraphUsed(root_fg);
  for (auto &func_graph : func_graphs) {
    SetEdges(func_graph, tx);
  }
}

void Cloner::LiftParameters(const FuncGraphVector &todo_func_graphs) {
  MS_EXCEPTION_IF_NULL(manager_);
  auto tx = manager_->Transact();
  for (const auto &todo_func_graph : todo_func_graphs) {
    const auto &func_graphs = BroadFirstSearchGraphUsed(todo_func_graph);
    for (auto &func_graph : func_graphs) {
      GenParameters(func_graph);
    }
    Lift(func_graphs);
  }
  const auto &roots = manager_->roots();
  // Roots in manager is noot set in Pynative mode.
  if (roots.empty()) {
    for (const auto &todo_func_graph : todo_func_graphs) {
      SetEdgesBfs(todo_func_graph, &tx);
    }
  } else {
    for (const auto &root_func_graph : roots) {
      SetEdgesBfs(root_func_graph, &tx);
    }
  }
  tx.Commit();
}

bool Cloner::CheckStatus(const FuncGraphPtr &func_graph, bool is_inline) {
  MS_EXCEPTION_IF_NULL(func_graph);
  // Make sure only inline once
  auto iter = status_.find(func_graph);
  if (iter != status_.end()) {
    if (is_inline == iter->second) {
      return false;
    }
    if (clone_all_used_graphs_) {
      MS_LOG(ERROR) << "Try setting the `clone_all_used_graphs` option to False.";
      return false;
    }
  }
  return true;
}

void Cloner::CloneAllNodes(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  const AnfNodeSet &nodes = func_graph->nodes();
  repl_node_.reserve(repl_node_.size() + nodes.size());
  for (auto &node : nodes) {
    CloneNode(node, target_func_graph);
  }
  // Only func_graph is inlined, it cannot be found in repl;
  if (repl_func_graph_.find(func_graph) != repl_func_graph_.end()) {
    CloneOrderList(func_graph, target_func_graph);
  }
}

void Cloner::CloneOrderList(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) {
  for (auto &cnode : func_graph->order_list()) {
    auto it = repl_node_.find(cnode);
    if (it == repl_node_.end()) {
      // For cnode which generated in Analyze phase, it cannot got from nodes API of func_graph,
      // so it cannot be cloned in normal Clone API.
      // If we ignore it, the order will be lost.
      // Therefore we put this old node as placeholder to the order list of target func_graph to
      // keep the order.
      // It may be replaced in ProgramSpecialize.
      // If this disconnected node is not used in target func_graph, it will be cleared after
      // ProgramSpecialize;
      target_func_graph->AppendOrderList(cnode);
      continue;
    }
    auto repl_cnode = dyn_cast<CNode>(it->second);
    if (repl_cnode != nullptr) {
      target_func_graph->AppendOrderList(repl_cnode);
    }
  }
}

void Cloner::Run() {
  if (todo_.empty()) {
    return;
  }

  FuncGraphVector func_graphs;
  (void)std::transform(todo_.begin(), todo_.end(), std::back_inserter(func_graphs),
                       [](const CloneInfo &item) -> FuncGraphPtr { return item.origin; });
  if (type_ < kLifting) {
    // Basic and Inline Clone
    manager_ = Manage(func_graphs, false);
    CloneNodes();
    LinkEdges();
    SetDefaults();
  } else {
    // Lifting Clone
    manager_ = Manage(func_graphs);
    LiftParameters(func_graphs);
  }
}

void Cloner::CloneNodes() {
  while (!todo_.empty()) {
    CloneInfo item = std::move(todo_.back());
    todo_.pop_back();

    const bool is_inline = (item.target != nullptr);
    FuncGraphPtr &func_graph = item.origin;
    (void)graph_set_.insert(func_graph);

    if (!CheckStatus(func_graph, is_inline)) {
      continue;
    }

    if (is_inline) {
      InlineCloneParameters(func_graph, item.params);
      CloneAllNodes(func_graph, item.target);
    } else {
      auto debug_info = CloneGraphDebugInfo(func_graph->debug_info(), target_relation_);
      auto target_func_graph = std::make_shared<FuncGraph>(std::move(debug_info));
      SetFuncGraphInfo(func_graph, target_func_graph);
      CloneParameters(func_graph, target_func_graph);
      CloneAllNodes(func_graph, target_func_graph);
      CloneFuncGraphValueNodes(func_graph, target_func_graph);
      CloneFuncGraphDefaultValues(func_graph, target_func_graph);
    }

    CloneValueNodes(func_graph);
    AddChildGraphs(func_graph);
    AddTotalGraphs(func_graph);
    status_[func_graph] = is_inline;
  }
}

void Cloner::LinkEdges() {
  for (auto &repl : repl_node_) {
    auto old_node = dyn_cast_ptr<CNode>(repl.first);
    if (old_node == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(repl.second);
    auto new_node = repl.second->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(new_node);
    for (auto &input : old_node->inputs()) {
      auto iter = repl_node_.find(input);
      auto &new_input = (iter == repl_node_.end() ? input : iter->second);
      new_node->add_input(new_input);
    }
  }
}

// For the graphs cloned, update its default value map to the cloned nodes.
void Cloner::SetDefaults() {
  for (auto &old_fg : graph_set_) {
    MS_EXCEPTION_IF_NULL(old_fg);
    auto iter = repl_func_graph_.find(old_fg);
    if (iter == repl_func_graph_.end()) {
      continue;
    }
    auto &new_fg = iter->second;
    MS_EXCEPTION_IF_NULL(new_fg);
    for (auto &param_def : old_fg->parameter_default_value()) {
      auto repl_iter = repl_node_.find(param_def.second);
      auto &value_node = (repl_iter == repl_node_.end() ? param_def.second : repl_iter->second);
      new_fg->set_param_default_value(param_def.first, value_node);
    }
  }
}

AnfNodePtr Cloner::CloneDisconnected(const AnfNodePtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto fg_iter = repl_func_graph_.find(root->func_graph());
  if (fg_iter == repl_func_graph_.end()) {
    MS_EXCEPTION_IF_NULL(root->func_graph());
    MS_LOG(EXCEPTION) << "Cannot find func graph " << root->func_graph()->ToString() << " in cloner.";
  }
  CloneNode(root, fg_iter->second);
  auto iter = repl_node_.find(root);
  if (iter == repl_node_.end()) {
    MS_LOG(EXCEPTION) << "Failed in clone for node " << root->DebugString() << ".";
  }
  return iter->second;
}

AnfNodePtr Cloner::operator[](const AnfNodePtr &node) {
#ifdef ENABLE_PROFILE
  double time = GetTime();
#endif
  Run();
#ifdef ENABLE_PROFILE
  MsProfile::StatTime("func_graph_cloner_run.FuncGraphClonerNode", GetTime() - time);
#endif
  auto iter = repl_node_.find(node);
  return ((iter == repl_node_.end()) ? node : iter->second);
}

FuncGraphPtr Cloner::operator[](const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
#ifdef ENABLE_PROFILE
  double time = GetTime();
#endif
  Run();
#ifdef ENABLE_PROFILE
  MsProfile::StatTime("func_graph_cloner_run.FuncGraphClonerGraph", GetTime() - time);
#endif
  auto iter = repl_func_graph_.find(func_graph);
  auto ret = ((iter == repl_func_graph_.end()) ? func_graph : iter->second);
  ret->set_python_obj(func_graph->python_obj());
  return ret;
}

FuncGraphPtr BasicClone(const FuncGraphPtr &func_graph, bool clone_value_nodes, const UpdateInfoPtr update_info) {
  MS_EXCEPTION_IF_NULL(func_graph);
  Cloner cloner({func_graph}, clone_value_nodes, true, true);
  if (update_info != nullptr) {
    cloner.set_update_info(update_info);
  }
  return cloner[func_graph];
}

AnfNodePtr InlineClone(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph,
                       const AnfNodePtrList &func_graph_args, const ScopePtr &scope) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  Cloner cloner({}, false);
  if (scope != nullptr) {
    cloner.set_scope(scope);
  }
  cloner.AddClone(func_graph, target_func_graph, func_graph_args, kInline);
  return cloner[func_graph->output()];
}

FuncGraphPtr LiftingClone(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  Cloner cloner({}, false);
  cloner.AddClone(func_graph, nullptr, {}, kLifting);
  return cloner[func_graph];
}

FuncGraphVector LiftingCloneMulti(const FuncGraphVector &func_graphs) {
  Cloner cloner({}, false);
  for (const auto &func_graph : func_graphs) {
    cloner.AddClone(func_graph, nullptr, {}, kLifting);
  }
  cloner.Run();

  FuncGraphVector lifted_func_graphs;
  const auto &repl_func_graphs = cloner.cloned_func_graphs();
  for (const auto &func_graph : func_graphs) {
    auto iter = repl_func_graphs.find(func_graph);
    auto ret = ((iter == repl_func_graphs.end()) ? func_graph : iter->second);
    MS_EXCEPTION_IF_NULL(ret);
    ret->set_python_obj(func_graph->python_obj());
    lifted_func_graphs.push_back(ret);
  }

  return lifted_func_graphs;
}

ClonerPtr SpecializerClone(const FuncGraphPtr &func_graph, const TraceInfoPtr &relation) {
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphVector func_graphs = {func_graph};
  ClonerPtr cloner =
    std::make_shared<Cloner>(func_graphs, false, false, false, std::make_shared<TraceCopy>(), relation);
#ifdef ENABLE_PROFILE
  double time = GetTime();
#endif
  cloner->Run();
#ifdef ENABLE_PROFILE
  MsProfile::StatTime("func_graph_cloner_run.FuncGraphSpecializer", GetTime() - time);
#endif
  return cloner;
}

FuncGraphPtr TransformableClone(const FuncGraphPtr &func_graph, const TraceInfoPtr &relation) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto debug_info = CloneGraphDebugInfo(func_graph->debug_info(), relation);
  auto new_func_graph = std::make_shared<FuncGraph>(std::move(debug_info));
  for (auto &param : func_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_debug_info = CloneNodeDebugInfo(param->debug_info());
    new_func_graph->add_parameter(std::move(param_debug_info))->set_abstract(param->abstract());
  }

  Cloner cloner{};
  cloner.AddClone(func_graph, new_func_graph, new_func_graph->parameters());
  AnfNodePtr output = cloner[func_graph->output()];
  new_func_graph->set_output(output);
  new_func_graph->set_has_vararg(func_graph->has_vararg());
  new_func_graph->set_has_kwarg(func_graph->has_kwarg());
  new_func_graph->set_kwonlyargs_count(func_graph->kwonlyargs_count());
  new_func_graph->set_fv_param_count(func_graph->fv_param_count());
  new_func_graph->set_is_generate(func_graph->is_generated());
  new_func_graph->set_stub(func_graph->stub());
  new_func_graph->set_switch_input(func_graph->switch_input());
  new_func_graph->set_switch_layer_input(func_graph->switch_layer_input());
  for (auto &item : func_graph->parameter_default_value()) {
    new_func_graph->set_param_default_value(item.first, cloner[item.second]);
  }
  if (func_graph->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
    new_func_graph->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
  }
  if (func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
    new_func_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  }
  new_func_graph->set_stage(func_graph->stage());
  return new_func_graph;
}
}  // namespace mindspore
