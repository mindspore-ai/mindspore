/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include <set>

#include "abstract/abstract_function.h"
#include "ir/graph_utils.h"
#include "ir/manager.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/parallel_node_check.h"
#include "utils/profile.h"
#include "utils/trace_base.h"

// namespace to support intermediate representation definition
namespace mindspore {
namespace {
NodeDebugInfoPtr CloneNodeDebugInfo(const DebugInfoPtr &debug_info, const TraceInfoPtr &relation) {
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
  if (replicated_node_.find(node) != replicated_node_.end()) {
    return;
  }
  if (node->isa<CNode>()) {
    CloneCNodeWithoutInputs(node, target);
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
  if (preset_abstract()) {
    new_param->set_abstract(old_param->abstract());
  }
  new_param->set_name(old_param->name());
  if (old_param->has_default()) {
    // Default parameter can be shared since it is readonly.
    new_param->set_default_param(old_param->default_param());
  }
  new_param->set_is_top_graph_param(old_param->is_top_graph_param());
  ScopePtr scope = ((node->scope() == kDefaultScope) && (this->scope() != nullptr)) ? this->scope() : node->scope();
  new_param->set_scope(scope);
  replicated_node_[node] = std::move(new_param);
}

// Create a new empty CNode for old one, and bind them.
// Also see LinkCNodeEdges().
void Cloner::CloneCNodeWithoutInputs(const AnfNodePtr &node, const FuncGraphPtr &target) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(target);
  auto old_node = node->cast<CNodePtr>();
  AnfNodeWeakPtrList inputs;
  inputs.reserve(old_node->size());
  DebugInfoPtr debug_info;
  if (this->update_info() != nullptr && this->update_info()->debug_info_ != nullptr) {
    debug_info = this->update_info()->debug_info_;
  } else {
    debug_info = node->debug_info();
  }

  auto cloned_debug_info = CloneNodeDebugInfo(debug_info, relation_);
  CNodePtr new_node = std::make_shared<CNode>(std::move(inputs), target, std::move(cloned_debug_info));
  if (inline_call_node_ != nullptr) {
    MS_LOG(DEBUG) << "inline_call_node_: " << inline_call_node_ << "/" << inline_call_node_->DebugString()
                  << ", new_node: " << new_node << "/" << new_node->DebugString();
    UpdateInlineCNodeDebugInfo(inline_call_node_, new_node);
  } else {
    // Synchronize callers' shadow debug infos.
    auto &new_shadow_debug_infos = new_node->debug_info()->shadow_debug_infos_map();
    const auto &old_shadow_debug_infos = debug_info->shadow_debug_infos_map();
    new_shadow_debug_infos.insert(old_shadow_debug_infos.cbegin(), old_shadow_debug_infos.cend());
  }
  new_node->CloneCNodeInfo(old_node);
  // Copy to target graph
  if (new_node->forward().first != nullptr) {
    target->set_used_forward_nodes({new_node});
  }
  ScopePtr scope;
  if (this->update_info() != nullptr && this->update_info()->scope_ != nullptr) {
    scope = this->update_info()->scope_;
  } else {
    scope = ((node->scope() == kDefaultScope) && (this->scope() != nullptr)) ? this->scope() : node->scope();
  }
  new_node->set_scope(scope);
  auto new_cnode = new_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_has_side_effect_node(old_node->has_side_effect_node());
  if (old_node->has_side_effect_node()) {
    target->set_has_side_effect_node(true);
    MS_LOG(DEBUG) << "Set isolated side-effect node flag for " << target->ToString();
  }
  replicated_node_[node] = std::move(new_node);
}

void Cloner::CloneValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast_ptr<ValueNode>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto debug_info = CloneNodeDebugInfo(node->debug_info(), relation_);
  ValueNodePtr new_const = NewValueNode(GetValueNode(node), std::move(debug_info));
  ScopePtr scope = ((node->scope() == kDefaultScope) && (this->scope() != nullptr)) ? this->scope() : node->scope();
  new_const->set_scope(scope);
  if (preset_abstract()) {
    new_const->set_abstract(node->abstract());
  }
  new_const->set_has_new_value(value_node->has_new_value());
  replicated_node_[node] = std::move(new_const);
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
  if (preset_abstract()) {
    new_const->set_abstract(node->abstract());
  }
  new_const->set_has_new_value(value_node->has_new_value());
  replicated_node_[node] = std::move(new_const);
}

void Cloner::CloneValueNodes(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!clone_all_valuenodes_) {
    return;
  }
  auto &value_nodes = func_graph->value_nodes();
  for (auto &value_node : value_nodes) {
    auto &old_node = value_node.first;
    if (replicated_node_.find(old_node) == replicated_node_.end()) {
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
  // The graph marked 'no_child_graph' has no child graph.
  if (func_graph->has_flag(FUNC_GRAPH_FLAG_NO_CHILD_GRAPH)) {
    return;
  }
  auto &scopes = manager_->scopes(func_graph);
  std::set<const FuncGraph *> memo;
  for (auto &graph : scopes) {
    // Avoid to insert duplicate function.
    if (graph == func_graph || !memo.emplace(graph.get()).second) {
      continue;
    }
    (void)todo_.emplace_back(CloneInfo{graph, nullptr, {}});
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
    if (!memo.emplace(fg.first.get()).second) {
      continue;
    }
    (void)todo_.emplace_back(CloneInfo{fg.first, nullptr, {}});
  }
}

void Cloner::CloneFuncGraphDefaultValues(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  for (auto &item : func_graph->parameter_default_value()) {
    auto nodes = TopoSort(item.second, SuccDeeperSimple);
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
  target_func_graph->set_segment(func_graph->segment());
  auto &old_return = func_graph->return_node();
  if (old_return != nullptr) {
    auto iter = replicated_node_.find(old_return);
    if (iter == replicated_node_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Can't find replicate node for return.";
    }
    MS_EXCEPTION_IF_NULL(iter->second);
    auto return_node = iter->second->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(return_node);
    target_func_graph->set_return(return_node);
  } else {
    MS_LOG(ERROR) << "Has no return node, func_graph: " << func_graph << "/" << func_graph->ToString();
  }

  auto &cnodes = func_graph->func_graph_cnodes_index();
  for (auto &cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode.first);
    MS_EXCEPTION_IF_NULL(cnode.first->first);
    auto user_cnode = cnode.first->first->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    const auto &valuenode = user_cnode->input(IntToSize(cnode.first->second));
    if (valuenode == nullptr) {
      continue;
    }
    CloneFuncGraphValueNode(valuenode, target_func_graph);
  }
}

void Cloner::InlineCloneParameters(const FuncGraphPtr &func_graph, const AnfNodePtrList &params) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto &old_params = func_graph->parameters();
  if (old_params.size() != params.size()) {
    MS_INTERNAL_EXCEPTION(TypeError) << "Origin params size[" << old_params.size() << "], inline params size["
                                     << params.size() << "]";
  }
  for (size_t i = 0; i < old_params.size(); ++i) {
    replicated_node_[old_params[i]] = params[i];
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
  target_func_graph->set_indirect(func_graph->indirect());
  target_func_graph->set_python_obj(func_graph->python_obj());
  target_func_graph->set_has_side_effect_node(func_graph->has_side_effect_node());
}

void Cloner::CloneParameters(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  auto &params = func_graph->parameters();
  for (auto &param : params) {
    CloneParameter(param, target_func_graph, true);
  }
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
    auto &replicated_node = replicated_map_node_[func_graph];
    if (replicated_node.find(free_var_node) != replicated_node.end()) {
      MS_LOG(DEBUG) << "Param exists: " << free_var_node->DebugString()
                    << " for func_graph: " << func_graph->ToString();
      continue;
    }

    MS_LOG(DEBUG) << "Gen param: " << free_var_node->ToString() << " for func_graph: " << func_graph->ToString();
    auto fv_parameter = AddParameter(func_graph, free_var_node);
    fv_parameter->set_user_data<bool>("lifted_from_fv", std::make_shared<bool>(true));
    auto &fg_params = replicated_func_graph_params_[func_graph];
    (void)fg_params.emplace_back(fv_parameter);
  }
}

void Cloner::CloneParameter(const ParameterPtr &param, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(node);
  if (preset_abstract()) {
    param->set_abstract(node->abstract());
  }
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
  replicated_node_[param] = node;
  replicated_map_node_[func_graph][node] = param;
  return param;
}

namespace {
bool FilterMonadInput(const AnfNodeWeakPtrList &old_inputs, AnfNodeWeakPtrList *new_inputs,
                      AnfNodePtr *possible_u_monad, AnfNodePtr *possible_io_monad) {
  AnfNodePtr local_u_monad = nullptr;
  AnfNodePtr local_io_monad = nullptr;
  for (const auto &weak_input : old_inputs) {
    auto input = weak_input.lock();
    MS_EXCEPTION_IF_NULL(input);
    // Should be only one U Monad input.
    if (HasAbstractUMonad(input)) {
      if (local_u_monad != nullptr) {
        MS_LOG(ERROR) << "Cannot have multiple U Monad in one call, first: " << local_u_monad->ToString()
                      << ", second: " << input->ToString();
        return false;
      }
      local_u_monad = input;
      continue;
    }
    // Should be only one IO Monad input.
    if (HasAbstractIOMonad(input)) {
      if (local_io_monad != nullptr) {
        MS_LOG(ERROR) << "Cannot have multiple IO Monad in one call, first: " << local_io_monad->ToString()
                      << ", second: " << input->ToString();
        return false;
      }
      local_io_monad = input;
      continue;
    }
    // Collect all non-monad inputs.
    (void)new_inputs->emplace_back(weak_input);
  }
  *possible_u_monad = local_u_monad;
  *possible_io_monad = local_io_monad;
  return true;
}

// After lift, func_graph will not refer any free variable, so DummyContext is proper.
AnfNodePtr BuildFuncGraphValueNode(const FuncGraphPtr &func_graph, bool preset_abstract) {
  auto new_node = NewValueNode(func_graph);
  auto abstract = std::make_shared<abstract::FuncGraphAbstractClosure>(
    func_graph, abstract::AnalysisContext::DummyContext(), new_node, preset_abstract);
  new_node->set_abstract(abstract);
  return new_node;
}

AnfNodePtr BuildPrimitiveValueNode(const PrimitivePtr &primitive) {
  auto new_node = NewValueNode(primitive);
  auto abstract = std::make_shared<abstract::PrimitiveAbstractClosure>(primitive, new_node);
  new_node->set_abstract(abstract);
  return new_node;
}

void PresetPartialAbstractClosure(const CNodePtr &cnode, const FuncGraphPtr &func_graph,
                                  const AnfNodeWeakPtrList &weak_inputs, bool preset_abstract) {
  if (!preset_abstract) {
    return;
  }
  constexpr auto ignore_partial_fg_count = 2;
  AbstractBasePtrList args_abs_list;
  (void)std::for_each(weak_inputs.cbegin() + ignore_partial_fg_count, weak_inputs.cend(),
                      [&args_abs_list](const AnfNodeWeakPtr &weak_node) {
                        auto node = weak_node.lock();
                        MS_EXCEPTION_IF_NULL(node);
                        (void)args_abs_list.emplace_back(node->abstract());
                      });
  MS_EXCEPTION_IF_NULL(func_graph->ToAbstract());
  auto abs = std::make_shared<abstract::PartialAbstractClosure>(
    func_graph->ToAbstract()->cast<abstract::AbstractFuncAtomPtr>(), args_abs_list, cnode);
  cnode->set_abstract(abs);
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

void Cloner::OrderParameters(const FuncGraphPtr &func_graph, const AnfNodeWeakPtrList &inputs, size_t arg_start_index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  mindspore::HashSet<AnfNodePtr> old_params;
  for (auto &param : func_graph->parameters()) {
    (void)old_params.insert(replicated_node_[param]);
  }
  mindspore::HashSet<AnfNodePtr> new_params;
  AnfNodePtrList parameters;
  // Ignore the 1st and 2nd param of inputs(such as. partial graph)
  for (size_t i = arg_start_index; i < inputs.size(); ++i) {
    const auto &input = inputs[i].lock();
    MS_EXCEPTION_IF_NULL(input);
    const auto &param = replicated_node_[input];
    if (old_params.find(param) != old_params.end()) {
      auto &new_param = replicated_map_node_[func_graph][param];
      (void)parameters.emplace_back(new_param);
      (void)new_params.insert(new_param);
    }
  }
  for (auto &param : func_graph->parameters()) {
    if (new_params.find(param) == new_params.end()) {
      (void)parameters.emplace_back(param);
    }
  }
  func_graph->set_parameters(std::move(parameters));
}

// Avoid to create nested partial CNode.
CNodePtr Cloner::SetPartialEdges(const FuncGraphPtr &func_graph, const CNodePtr &cnode, FuncGraphTransaction *tx) {
  if (!IsPrimitiveCNode(cnode, prim::kPrimPartial) || !IsValueNode<FuncGraph>(cnode->input(1))) {
    return nullptr;
  }
  auto graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
  MS_EXCEPTION_IF_NULL(graph);
  auto &replicated_func_graph = replicated_map_func_graph_[func_graph];
  if (replicated_func_graph.find(graph) == replicated_func_graph.end()) {
    return nullptr;
  }

  auto partial_node = replicated_func_graph[graph];
  if (!IsPrimitiveCNode(partial_node, prim::kPrimPartial)) {
    return nullptr;
  }
  auto partial_cnode = dyn_cast<CNode>(partial_node);
  MS_EXCEPTION_IF_NULL(partial_cnode);
  auto value_node = BuildPrimitiveValueNode(prim::kPrimPartial);
  MS_EXCEPTION_IF_NULL(value_node);
  auto func_graph_node = BuildFuncGraphValueNode(graph, preset_abstract());
  MS_EXCEPTION_IF_NULL(func_graph_node);
  AnfNodeWeakPtrList new_inputs = {value_node, func_graph_node};
  constexpr auto ignore_partial_fg_count = 2;
  (void)std::copy(partial_cnode->weak_inputs().cbegin() + ignore_partial_fg_count, partial_cnode->weak_inputs().cend(),
                  std::back_inserter(new_inputs));
  (void)std::copy(cnode->weak_inputs().cbegin() + ignore_partial_fg_count, cnode->weak_inputs().cend(),
                  std::back_inserter(new_inputs));
  auto new_cnode = func_graph->NewCNodeWeak(std::move(new_inputs));
  MS_EXCEPTION_IF_NULL(new_cnode);
  PresetPartialAbstractClosure(new_cnode, graph, new_cnode->weak_inputs(), preset_abstract());

  MS_LOG(DEBUG) << "Rebuild partial CNode, old_node: " << cnode->DebugString()
                << ", partial_cnode: " << partial_cnode->DebugString() << ", new_node: " << new_cnode->DebugString()
                << ", new_node abs: " << (new_cnode->abstract() != nullptr ? new_cnode->abstract()->ToString() : "null")
                << ", partial " << graph->ToString() << " in " << func_graph->ToString();
  (void)tx->Replace(cnode, new_cnode);
  return new_cnode;
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

    // Avoid to create nested partial CNode.
    auto old_cnode = cnode;
    auto new_cnode = SetPartialEdges(func_graph, cnode, tx);
    if (new_cnode != nullptr) {
      cnode = new_cnode;
    }

    const auto &inputs = cnode->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto &input = inputs[i];
      if (IsValueNode<FuncGraph>(input)) {
        if (i == 1 && new_cnode != nullptr) {
          continue;
        }
        auto graph = GetValueNode<FuncGraphPtr>(input);
        auto &replicated_func_graph = replicated_map_func_graph_[func_graph];
        if (replicated_func_graph.find(graph) != replicated_func_graph.end()) {
          auto partial_node = replicated_func_graph[graph];
          tx->SetEdge(cnode, static_cast<int>(i), partial_node);
        }
      } else {
        auto &replicated_node = replicated_map_node_[func_graph];
        if (replicated_node.find(input) != replicated_node.end()) {
          tx->SetEdge(cnode, static_cast<int>(i), replicated_node[input]);
        }
      }
    }
  }
}

void Cloner::AddParameters(const FuncGraphPtr &func_graph, const AnfNodeWeakPtrList &params,
                           AnfNodeWeakPtrList *const lift_params, AnfNodeWeakPtrList *const input_params) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(lift_params);
  MS_EXCEPTION_IF_NULL(input_params);
  AnfNodePtrList parameters;
  mindspore::HashSet<AnfNodePtr> old_params;
  for (auto &param : func_graph->parameters()) {
    auto iter = replicated_node_.find(param);
    if (iter != replicated_node_.end()) {
      (void)old_params.insert(iter->second);
      (void)parameters.emplace_back(param);
    } else {
      (void)parameters.emplace_back(AddParameter(func_graph, param, false));
      (void)old_params.insert(param);
    }
  }
  AnfNodePtr new_param = nullptr;
  for (auto &weak_param : params) {
    const auto &param = weak_param.lock();
    auto old_param = replicated_node_[param];
    MS_EXCEPTION_IF_NULL(old_param);
    if (old_param->isa<CNode>() && old_param->func_graph() == func_graph) {
      replicated_node_[old_param] = old_param;
      replicated_map_node_[func_graph][old_param] = old_param;
      (void)input_params->emplace_back(old_param);
      continue;
    }
    if (old_params.find(old_param) != old_params.end()) {
      new_param = replicated_map_node_[func_graph][old_param];
      if (new_param == nullptr) {
        MS_LOG(INTERNAL_EXCEPTION) << "map_node, func_graph: " << func_graph->ToString()
                                   << ", old_param: " << old_param->DebugString() << " cannot found";
      }
      (void)input_params->emplace_back(new_param);
      continue;
    }
    if (IsLiftTopFuncGraph(func_graph)) {
      // Don't lift parameter from used_graphs to my parameter if I am the top;
      replicated_node_[old_param] = old_param;
      replicated_map_node_[func_graph][old_param] = old_param;
      MS_EXCEPTION_IF_NULL(old_param->func_graph());
      replicated_map_node_[old_param->func_graph()][old_param] = old_param;
      (void)input_params->emplace_back(old_param);
      MS_LOG(DEBUG) << "Bypass " << old_param->DebugString() << " for top func_graph: " << func_graph->ToString();
      continue;
    }
    new_param = AddParameter(func_graph, old_param, false);
    (void)parameters.emplace_back(new_param);
    (void)lift_params->emplace_back(new_param);
    (void)input_params->emplace_back(new_param);
  }
  func_graph->set_parameters(std::move(parameters));
}

void Cloner::AddInputs(const FuncGraphPtr &func_graph_user, const FuncGraphPtr &func_graph,
                       const AnfNodeWeakPtrList &params) {
  auto &replicated_func_graph = replicated_map_func_graph_[func_graph_user];
  auto [iter, inserted] = replicated_func_graph.emplace(func_graph, nullptr);
  if (inserted) {
    const auto value_node = BuildPrimitiveValueNode(prim::kPrimPartial);
    const auto fg_value = BuildFuncGraphValueNode(func_graph, preset_abstract());
    AnfNodeWeakPtrList cnode_inputs{value_node, fg_value};
    auto partial_node = func_graph_user->NewCNodeWeak(std::move(cnode_inputs));
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
  AnfNodeWeakPtrList inputs;
  AnfNodeWeakPtrList add_params;
  if (!FilterMonadInput(cnode->weak_inputs(), &inputs, &input_u_monad, &input_io_monad)) {
    constexpr auto recursive_level = 2;
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot have multiple U Monad or multiple IO Monad in one CNode, cnode: "
                               << cnode->DebugString(recursive_level);
  }
  if (!FilterMonadInput(params, &add_params, &param_u_monad, &param_io_monad)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot have multiple U Monad or multiple IO Monad in Parameters list, func_graph: "
                               << func_graph->ToString();
  }

  // Append new inputs from free variable.
  constexpr auto caller_first_arg_index = 2;
  for (size_t i = caller_first_arg_index; i < inputs.size(); i++) {
    auto input = inputs[i].lock();
    auto pos = std::find_if(add_params.cbegin(), add_params.cend(), [&input](const auto &weak_param) {
      if (weak_param.lock() != nullptr && weak_param.lock() == input) {
        return true;
      }
      return false;
    });
    if (pos != add_params.end()) {
      (void)add_params.erase(pos);
    }
  }
  (void)inputs.insert(inputs.end(), add_params.cbegin(), add_params.cend());

  // Append monad inputs.
  if (input_u_monad != nullptr && param_u_monad != nullptr && input_u_monad != param_u_monad) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot have multiple U Monad in one call, first: " << input_u_monad->ToString()
                               << ", second: " << param_u_monad->ToString();
  }
  if (input_io_monad != nullptr && param_io_monad != nullptr && input_io_monad != param_io_monad) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot have multiple IO Monad in one call, first: " << input_io_monad->ToString()
                               << ", second: " << param_io_monad->ToString();
  }
  auto &u_monad = (input_u_monad != nullptr ? input_u_monad : param_u_monad);
  auto &io_monad = (input_io_monad != nullptr ? input_io_monad : param_io_monad);
  if (u_monad != nullptr) {
    (void)inputs.emplace_back(u_monad);
  }
  if (io_monad != nullptr) {
    (void)inputs.emplace_back(io_monad);
  }

  cnode->set_weak_inputs(inputs);
  OrderParameters(func_graph, inputs, caller_first_arg_index);
  PresetPartialAbstractClosure(cnode, func_graph, inputs, preset_abstract());
  MS_LOG(DEBUG) << "Create new partial CNode: " << cnode->DebugString();
}

void Cloner::LiftParameters(const FuncGraphPtr &func_graph_user, const FuncGraphPtr &func_graph,
                            const AnfNodeWeakPtrList &params) {
  MS_EXCEPTION_IF_NULL(func_graph_user);
  AnfNodeWeakPtrList lift_params;
  AnfNodeWeakPtrList input_params;
  AddParameters(func_graph_user, params, &lift_params, &input_params);
  AddInputs(func_graph_user, func_graph, input_params);
  if (lift_params.empty()) {
    return;
  }
  for (auto &cnode_index : func_graph_user->func_graph_cnodes_index()) {
    MS_EXCEPTION_IF_NULL(cnode_index.first);
    const auto &user_node = cnode_index.first->first;
    MS_EXCEPTION_IF_NULL(user_node);
    LiftParameters(user_node->func_graph(), func_graph_user, lift_params);
  }
}

void Cloner::Lift(const std::vector<FuncGraphPtr> &sorted) {
  // lift inner graph first
  for (auto r_iter = sorted.rbegin(); r_iter != sorted.rend(); ++r_iter) {
    auto func_graph = *r_iter;
    auto iter = replicated_func_graph_params_.find(func_graph);
    if (iter != replicated_func_graph_params_.end()) {
      auto &params = iter->second;
      for (auto &cnode_index : func_graph->func_graph_cnodes_index()) {
        MS_EXCEPTION_IF_NULL(cnode_index.first);
        const auto &user_node = cnode_index.first->first;
        MS_EXCEPTION_IF_NULL(user_node);
        LiftParameters(user_node->func_graph(), func_graph, params);
      }
    }
  }
}

void Cloner::SetEdgesBfs(const FuncGraphPtr &root_fg, FuncGraphTransaction *tx) {
  MS_EXCEPTION_IF_NULL(root_fg);
  const auto &func_graphs = BroadFirstSearchGraphUsed(root_fg, lifting_func_graph_filter());
  for (auto &func_graph : func_graphs) {
    SetEdges(func_graph, tx);
  }
}

void Cloner::LiftParameters(const FuncGraphVector &todo_func_graphs) {
  MS_EXCEPTION_IF_NULL(manager_);
  auto tx = manager_->Transact();
  for (const auto &todo_func_graph : todo_func_graphs) {
    const auto &func_graphs = BroadFirstSearchGraphUsed(todo_func_graph, lifting_func_graph_filter());
    for (auto &func_graph : func_graphs) {
      GenParameters(func_graph);
    }
    Lift(func_graphs);
  }
  const auto &roots = manager_->roots();
  // Roots in manager is not set in Pynative mode.
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
  replicated_node_.reserve(replicated_node_.size() + nodes.size());
  for (auto &node : nodes) {
    CloneNode(node, target_func_graph);
  }
  // Only func_graph is inlined, it cannot be found in repl;
  if (replicated_func_graph_.find(func_graph) != replicated_func_graph_.end()) {
    CloneOrderList(func_graph, target_func_graph);
  }
}

void Cloner::CloneOrderList(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) {
  for (auto &weak_cnode : func_graph->order_list()) {
    const auto &cnode = weak_cnode.lock();
    if (cnode == nullptr) {
      continue;
    }
    auto it = replicated_node_.find(cnode);
    if (it == replicated_node_.end()) {
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
    auto replicated_cnode = dyn_cast<CNode>(it->second);
    if (replicated_cnode != nullptr) {
      target_func_graph->AppendOrderList(replicated_cnode);
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
    LinkCNodeEdges();
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
      replicated_func_graph_[func_graph] = target_func_graph;
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

// Link the CNode with its inputs.
// Also see CloneCNodeWithoutInputs()
void Cloner::LinkCNodeEdges() {
  for (auto &repl : replicated_node_) {
    auto old_node = dyn_cast_ptr<CNode>(repl.first);
    if (old_node == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(repl.second);
    auto new_node = repl.second->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(new_node);
    for (auto &weak_input : old_node->weak_inputs()) {
      auto input = weak_input.lock();
      MS_EXCEPTION_IF_NULL(input);
      auto iter = replicated_node_.find(input);
      auto &new_input = (iter == replicated_node_.end() ? input : iter->second);
      new_node->add_input(new_input);
    }
  }
}

// For the graphs cloned, update its default value map to the cloned nodes.
void Cloner::SetDefaults() {
  for (auto &old_fg : graph_set_) {
    MS_EXCEPTION_IF_NULL(old_fg);
    auto iter = replicated_func_graph_.find(old_fg);
    if (iter == replicated_func_graph_.end()) {
      continue;
    }
    auto &new_fg = iter->second;
    MS_EXCEPTION_IF_NULL(new_fg);
    for (auto &param_def : old_fg->parameter_default_value()) {
      auto replicated_iter = replicated_node_.find(param_def.second);
      auto &value_node = (replicated_iter == replicated_node_.end() ? param_def.second : replicated_iter->second);
      new_fg->set_param_default_value(param_def.first, value_node);
    }
  }
}

AnfNodePtr Cloner::CloneDisconnected(const AnfNodePtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto fg_iter = replicated_func_graph_.find(root->func_graph());
  if (fg_iter == replicated_func_graph_.end()) {
    MS_EXCEPTION_IF_NULL(root->func_graph());
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot find func graph " << root->func_graph()->ToString() << " in cloner.";
  }
  CloneNode(root, fg_iter->second);
  auto iter = replicated_node_.find(root);
  if (iter == replicated_node_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Failed in clone for node " << root->DebugString() << ".";
  }
  return iter->second;
}

AnfNodePtr Cloner::operator[](const AnfNodePtr &node) {
  {
    MsProfileStatGuard stat_guard("func_graph_cloner_run.FuncGraphClonerNode");
    Run();
  }

  auto iter = replicated_node_.find(node);
  return ((iter == replicated_node_.end()) ? node : iter->second);
}

FuncGraphPtr Cloner::operator[](const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  {
    MsProfileStatGuard stat_guard("func_graph_cloner_run.FuncGraphClonerGraph");
    Run();
  }

  auto iter = replicated_func_graph_.find(func_graph);
  auto ret = ((iter == replicated_func_graph_.end()) ? func_graph : iter->second);
  ret->set_python_obj(func_graph->python_obj());
  return ret;
}

FuncGraphPtr BasicClone(const FuncGraphPtr &func_graph, bool clone_value_nodes, const UpdateInfoPtr update_info) {
  MS_EXCEPTION_IF_NULL(func_graph);
  Cloner cloner({func_graph}, clone_value_nodes, true, true);
  if (update_info != nullptr) {
    cloner.set_update_info(update_info);
  }
  auto target_func_graph = cloner[func_graph];
  if (func_graph->has_flag(GRAPH_FLAG_IS_WHILE_HEADER)) {
    MS_EXCEPTION_IF_NULL(target_func_graph);
    target_func_graph->set_flag(GRAPH_FLAG_IS_WHILE_HEADER, true);
  }
  return target_func_graph;
}

AnfNodePtr InlineClone(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph,
                       const AnfNodePtrList &func_graph_args, const AnfNodePtr &call_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  Cloner cloner({}, false);
  if (call_node != nullptr) {
    auto call_cnode = dyn_cast<CNode>(call_node);
    MS_EXCEPTION_IF_NULL(call_cnode);
    if (call_cnode->input(0)->scope() != nullptr) {
      cloner.set_scope(call_cnode->input(0)->scope());
    }
  }
  cloner.set_inline_call_node(call_node);
  cloner.AddClone(func_graph, target_func_graph, func_graph_args, kInline);
  if (func_graph->has_flag(GRAPH_FLAG_IS_WHILE_HEADER)) {
    target_func_graph->set_flag(GRAPH_FLAG_IS_WHILE_HEADER, true);
  }
  if (func_graph->has_flag(kTraining)) {
    target_func_graph->set_flag(kTraining, true);
  }
  return cloner[func_graph->output()];
}

FuncGraphPtr LiftingClone(const FuncGraphPtr &func_graph, bool preset_abstract,
                          const GraphFilterFunc &lifting_func_graph_filter) {
  MS_EXCEPTION_IF_NULL(func_graph);
  Cloner cloner({}, false);
  cloner.set_preset_abstract(preset_abstract);
  cloner.set_lifting_func_graph_filter(lifting_func_graph_filter);
  cloner.AddClone(func_graph, nullptr, {}, kLifting);
  auto target_func_graph = cloner[func_graph];
  if (func_graph->has_flag(GRAPH_FLAG_IS_WHILE_HEADER)) {
    target_func_graph->set_flag(GRAPH_FLAG_IS_WHILE_HEADER, true);
  }
  return target_func_graph;
}

FuncGraphVector LiftingCloneMulti(const FuncGraphVector &func_graphs) {
  Cloner cloner({}, false);
  for (const auto &func_graph : func_graphs) {
    cloner.AddClone(func_graph, nullptr, {}, kLifting);
  }
  cloner.Run();

  FuncGraphVector lifted_func_graphs;
  const auto &replicated_func_graphs = cloner.cloned_func_graphs();
  for (const auto &func_graph : func_graphs) {
    auto iter = replicated_func_graphs.find(func_graph);
    auto ret = ((iter == replicated_func_graphs.end()) ? func_graph : iter->second);
    MS_EXCEPTION_IF_NULL(ret);
    ret->set_python_obj(func_graph->python_obj());
    (void)lifted_func_graphs.emplace_back(ret);
  }

  return lifted_func_graphs;
}

ClonerPtr SpecializerClone(const FuncGraphPtr &func_graph, const TraceInfoPtr &relation) {
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphVector func_graphs = {func_graph};
  ClonerPtr cloner =
    std::make_shared<Cloner>(func_graphs, false, false, false, std::make_shared<TraceCopy>(), relation);
  {
    MsProfileStatGuard stat_guard("func_graph_cloner_run.FuncGraphSpecializer");
    cloner->Run();
  }
  return cloner;
}

FuncGraphPtr TransformableClone(const FuncGraphPtr &func_graph, const TraceInfoPtr &relation) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto debug_info = CloneGraphDebugInfo(func_graph->debug_info(), relation);
  auto new_func_graph = std::make_shared<FuncGraph>(std::move(debug_info));
  for (auto &param : func_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_debug_info = CloneNodeDebugInfo(param->debug_info());
    auto new_param = new_func_graph->add_parameter(std::move(param_debug_info));
    new_param->set_abstract(param->abstract());
  }

  Cloner cloner({}, true);
  cloner.AddClone(func_graph, new_func_graph, new_func_graph->parameters());
  AnfNodePtr output = cloner[func_graph->output()];
  new_func_graph->set_output(output);
  new_func_graph->set_has_vararg(func_graph->has_vararg());
  new_func_graph->set_has_kwarg(func_graph->has_kwarg());
  new_func_graph->set_kwonlyargs_count(func_graph->kwonlyargs_count());
  new_func_graph->set_fv_param_count(func_graph->fv_param_count());
  new_func_graph->set_is_generate(func_graph->is_generated());
  new_func_graph->set_indirect(func_graph->indirect());
  new_func_graph->set_stub(func_graph->stub());
  for (auto &item : func_graph->parameter_default_value()) {
    new_func_graph->set_param_default_value(item.first, cloner[item.second]);
  }
  if (func_graph->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
    new_func_graph->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
  }
  if (func_graph->has_flag(GRAPH_FLAG_IS_WHILE_HEADER)) {
    new_func_graph->set_flag(GRAPH_FLAG_IS_WHILE_HEADER, true);
  }
  if (func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
    new_func_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  }
  new_func_graph->set_stage(func_graph->stage());
  new_func_graph->set_segment(func_graph->segment());
  return new_func_graph;
}
}  // namespace mindspore
