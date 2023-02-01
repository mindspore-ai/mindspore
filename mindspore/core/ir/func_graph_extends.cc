/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ir/func_graph.h"

#include "ir/manager.h"
#include "utils/ordered_set.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"
#include "mindspore/core/ops/core_ops.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;
using mindspore::abstract::AnalysisContextPtr;
using mindspore::abstract::PrimitiveAbstractClosure;
using mindspore::abstract::VirtualAbstractClosure;

AbstractFunctionPtr FuncGraph::abstract() {
  AbstractBasePtrList args_spec_list;

  for (auto &para : parameters_) {
    MS_EXCEPTION_IF_NULL(para);
    if (para->abstract() == nullptr) {
      MS_LOG(ERROR) << "Error!!";
      return nullptr;
    }
    args_spec_list.push_back(para->abstract());
  }

  if (output() == nullptr) {
    MS_LOG(ERROR) << "Error func graph no output";
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(output());
  return std::make_shared<VirtualAbstractClosure>(args_spec_list, output()->abstract());
}

void FuncGraph::set_output(const AnfNodePtr &value, bool force_new_ret) {
  MS_EXCEPTION_IF_NULL(value);
  if (force_new_ret || return_ == nullptr) {
    std::vector<AnfNodePtr> params({NewValueNode(prim::kPrimReturn), value});
    FuncGraphPtr this_graph = shared_from_base<FuncGraph>();
    return_ = this_graph->NewCNodeInOrder(std::move(params));
  } else {
    if (manager_.lock()) {
      manager_.lock()->SetEdge(return_, 1, value);
    } else {
      constexpr auto first_data_index = 1;
      return_->set_input(first_data_index, value);
    }
  }

  return_->set_abstract(value->abstract());
  AnfNodePtr input0 = return_->input(0);
  auto f = std::make_shared<PrimitiveAbstractClosure>(prim::kPrimReturn, input0);
  input0->set_abstract(f);
}

void FuncGraph::GenerateVarParams(const FuncGraphPtr &specialized_graph, int variable_args_count,
                                  int pos_args_input_count, std::vector<AnfNodePtr> *specialized_parameter_list,
                                  mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const {
  MS_EXCEPTION_IF_NULL(specialized_graph);
  if (!specialized_graph->has_vararg()) {
    if (variable_args_count > 0) {
      MS_LOG(EXCEPTION) << "Function:" << this->ToString() << " takes " << GetPositionalArgsCount()
                        << " positional arguments, but " << pos_args_input_count << " were given.";
    }
    // Only copy parameters other than default arguments.
    for (size_t i = 0; i < IntToSize(pos_args_input_count); ++i) {
      specialized_parameter_list->push_back(specialized_graph->parameters()[i]);
    }
    return;
  }

  // If there is variable argument.
  if (variable_args_count < 0) {
    MS_LOG(EXCEPTION) << "For function:" << this->ToString() << ", its argument size: " << pos_args_input_count
                      << " is less or equal to parameter size: " << GetPositionalArgsCount();
  }
  // Copy other parameters than vararg's firstly.
  for (size_t i = 0; i < IntToSize(GetPositionalArgsCount()); ++i) {
    specialized_parameter_list->push_back(specialized_graph->parameters()[i]);
  }
  TraceGuard trace_guard(
    std::make_shared<TraceGenerateVarArg>(specialized_graph->GetVariableArgParameter()->debug_info()));
  std::vector<AnfNodePtr> var_param_tuple_nodes;
  var_param_tuple_nodes.push_back(NewValueNode(prim::kPrimMakeTuple));

  auto varg_name = specialized_graph->GetVariableArgName();
  // For python variable argument input, there is no upper limit.
  for (int i = 0; i < variable_args_count; ++i) {
    ParameterPtr para = std::make_shared<Parameter>(specialized_graph);
    std::string param_name = varg_name + std::to_string(i);
    para->set_name(param_name);
    MS_EXCEPTION_IF_NULL(para->debug_info());
    para->debug_info()->set_name(param_name);
    var_param_tuple_nodes.push_back(para);
    MS_EXCEPTION_IF_NULL(specialized_parameter_list);
    specialized_parameter_list->push_back(para);
  }
  auto var_tuple_param = specialized_graph->NewCNode(std::move(var_param_tuple_nodes));
  MS_EXCEPTION_IF_NULL(repl_nodes);
  (void)repl_nodes->emplace(specialized_graph->GetVariableArgParameter(), var_tuple_param);
}

void FuncGraph::GenerateKwParams(const FuncGraphPtr &specialized_graph,
                                 const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list,
                                 int pos_args_input_count, std::vector<AnfNodePtr> *specialized_parameter_list,
                                 mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const {
  MS_EXCEPTION_IF_NULL(specialized_parameter_list);
  MS_EXCEPTION_IF_NULL(repl_nodes);
  MS_EXCEPTION_IF_NULL(specialized_graph);
  std::vector<AnfNodePtr> kwarg_keys_tuple_nodes = {NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AnfNodePtr> kwarg_values_tuple_nodes = {NewValueNode(prim::kPrimMakeTuple)};

  std::set<AnfNodePtr> kwarg_nodes;
  for (size_t i = 0; i < kwarg_list.size(); ++i) {
    auto kwarg = kwarg_list[i];
    MS_EXCEPTION_IF_NULL(kwarg);
    std::string kw_param_name = kwarg->get_key();
    AnfNodePtr param_node = specialized_graph->GetParameterByName(kw_param_name);
    // If not find corresponding parameter node.
    if (param_node == nullptr) {
      if (!has_kwarg()) {
        if (pos_args_input_count + i > specialized_graph->parameters().size() - 1) {
          MS_LOG(EXCEPTION) << "Got unexpected keyword argument: " << kw_param_name;
        }
        specialized_parameter_list->push_back(specialized_graph->parameters()[pos_args_input_count + i]);
      } else {
        ParameterPtr para = std::make_shared<Parameter>(specialized_graph);
        std::string param_name = specialized_graph->GetVariableKwargName() + "[" + kw_param_name + "]";
        auto find_kw_arg_in_list = std::any_of(specialized_parameter_list->begin(), specialized_parameter_list->end(),
                                               [param_name](const AnfNodePtr &node) {
                                                 MS_EXCEPTION_IF_NULL(node);
                                                 auto param = node->cast_ptr<Parameter>();
                                                 return param != nullptr && param->name() == param_name;
                                               });
        if (find_kw_arg_in_list) {
          MS_EXCEPTION(TypeError) << "Multiply values for keyword argument: " << kw_param_name;
        }
        para->set_name(param_name);
        MS_EXCEPTION_IF_NULL(para->debug_info());
        para->debug_info()->set_name(param_name);
        kwarg_keys_tuple_nodes.push_back(NewValueNode(kw_param_name));
        auto extract_node =
          specialized_graph->NewCNode({NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(kw_param_name), para});
        kwarg_values_tuple_nodes.push_back(extract_node);
        specialized_parameter_list->push_back(para);
      }
    } else {
      auto node_itr = std::find(specialized_parameter_list->begin(), specialized_parameter_list->end(), param_node);
      // Multiply values found given for parameter.
      if (node_itr != specialized_parameter_list->end() && kwarg_nodes.find(param_node) == kwarg_nodes.end()) {
        MS_EXCEPTION(TypeError) << "Multiply values for specific argument: " << kw_param_name;
      } else {
        specialized_parameter_list->push_back(param_node);
        auto extract_node = specialized_graph->NewCNode(
          {NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(kw_param_name), param_node});
        kwarg_nodes.insert(param_node);
        (void)repl_nodes->emplace(param_node, extract_node);
      }
    }
  }

  GenerateKwargReplNode(specialized_graph, kwarg_keys_tuple_nodes, kwarg_values_tuple_nodes, repl_nodes);
}

void FuncGraph::GenerateKwargReplNode(const FuncGraphPtr &specialized_graph,
                                      const std::vector<AnfNodePtr> &kwarg_keys_tuple_nodes,
                                      const std::vector<AnfNodePtr> &kwarg_values_tuple_nodes,
                                      mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const {
  if (has_kwarg() && !kwarg_keys_tuple_nodes.empty()) {
    MS_EXCEPTION_IF_NULL(specialized_graph);
    TraceGuard guard(
      std::make_shared<TraceGenerateKwArg>(specialized_graph->GetVariableKwargParameter()->debug_info()));
    auto make_tuple_keys = specialized_graph->NewCNode(kwarg_keys_tuple_nodes);
    auto make_tuple_values = specialized_graph->NewCNode(kwarg_values_tuple_nodes);
    auto make_dict_node =
      specialized_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), make_tuple_keys, make_tuple_values});
    MS_EXCEPTION_IF_NULL(repl_nodes);
    (void)repl_nodes->emplace(specialized_graph->GetVariableKwargParameter(), make_dict_node);
  }
}

bool FuncGraph::NeedGenerate(const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list) {
  // If the function does not have any vararg/kwarg/kwonly/default value/kw args input
  // return the original graph
  if (!has_vararg() && kwonlyargs_count() == 0 && !has_kwarg() && GetDefaultValueCount() == 0 && kwarg_list.empty()) {
    return false;
  }

  // If the graph is generated for specific input, do not need to generate again
  return !is_generated();
}

void FuncGraph::GenerateDefaultValue(const FuncGraphPtr &specialized_graph,
                                     const std::vector<AnfNodePtr> &specialized_parameter_list,
                                     mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const {
  MS_EXCEPTION_IF_NULL(specialized_graph);
  for (size_t i = 0; i < specialized_graph->parameters().size() - fv_param_count(); ++i) {
    MS_EXCEPTION_IF_NULL(specialized_graph->parameters()[i]);
    auto param_node = specialized_graph->parameters()[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);
    auto param_name = param_node->name();
    auto node_itr = std::find(specialized_parameter_list.begin(), specialized_parameter_list.end(), param_node);
    if (node_itr != specialized_parameter_list.end()) {
      continue;
    }
    if (param_name == specialized_graph->GetVariableArgName() ||
        param_name == specialized_graph->GetVariableKwargName()) {
      continue;
    }
    auto default_value = specialized_graph->GetDefaultValueByName(param_name);
    if (default_value == nullptr) {
      MS_LOG(EXCEPTION) << "Miss argument input for parameter:" << param_name;
    }
    MS_EXCEPTION_IF_NULL(repl_nodes);
    (void)repl_nodes->emplace(param_node, default_value);
  }
}

FuncGraphPtr FuncGraph::GenerateGraph(const AbstractBasePtrList &args_spec_list) {
  std::vector<abstract::AbstractKeywordArgPtr> kwarg_list;
  std::vector<size_t> pos_arg_indexes;
  size_t arguments_count = args_spec_list.size();
  if (fv_param_count_ > arguments_count) {
    MS_LOG(EXCEPTION) << "The number of parameters in funcgraph cannot exceed the number of arguments.";
  }
  for (size_t i = 0; i < arguments_count - fv_param_count_; i++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[i]);
    if (args_spec_list[i]->isa<abstract::AbstractKeywordArg>()) {
      kwarg_list.push_back(args_spec_list[i]->cast<abstract::AbstractKeywordArgPtr>());
    } else {
      pos_arg_indexes.push_back(i);
    }
  }

  if (!NeedGenerate(kwarg_list)) {
    return shared_from_base<FuncGraph>();
  }
  auto iter = func_graph_cache_.find(args_spec_list);
  if (iter != func_graph_cache_.end()) {
    return iter->second;
  }
  FuncGraphPtr specialized_graph = BasicClone(shared_from_base<FuncGraph>());
  size_t kwarg_count = kwarg_list.size();
  // Get the variable args count from caller.
  int pos_args_input_count = SizeToInt((arguments_count - kwarg_count) - fv_param_count_);
  int variable_args_count = pos_args_input_count - GetPositionalArgsCount();
  std::vector<AnfNodePtr> specialized_parameter_list;
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> repl_nodes;
  GenerateVarParams(specialized_graph, variable_args_count, pos_args_input_count, &specialized_parameter_list,
                    &repl_nodes);
  GenerateKwParams(specialized_graph, kwarg_list, pos_args_input_count, &specialized_parameter_list, &repl_nodes);

  GenerateDefaultValue(specialized_graph, specialized_parameter_list, &repl_nodes);

  // Append hyper parameter to specialized_parameter_list
  MS_EXCEPTION_IF_NULL(specialized_graph);
  auto params = specialized_graph->parameters();
  (void)specialized_parameter_list.insert(specialized_parameter_list.end(), params.end() - SizeToInt(fv_param_count_),
                                          params.end());
  std::vector<AnfNodePtr> specialized_parameter_list_update(specialized_parameter_list.begin() + pos_arg_indexes.size(),
                                                            specialized_parameter_list.end());
  for (size_t i = 0; i < pos_arg_indexes.size(); i++) {
    (void)specialized_parameter_list_update.insert(specialized_parameter_list_update.begin() + pos_arg_indexes[i],
                                                   specialized_parameter_list[i]);
  }

  std::shared_ptr<mindspore::FuncGraphManager> manager = mindspore::Manage(specialized_graph, false);
  auto tr = manager->Transact();
  for (auto &node_pair : repl_nodes) {
    MS_EXCEPTION_IF_NULL(node_pair.first);
    MS_EXCEPTION_IF_NULL(node_pair.second);
    MS_LOG(DEBUG) << "GenerateGraph replace:" << node_pair.first->DebugString() << "-"
                  << node_pair.second->DebugString();
    (void)tr.Replace(node_pair.first, node_pair.second);
  }
  tr.SetParameters(specialized_graph, specialized_parameter_list_update);
  tr.Commit();
  specialized_graph->set_has_kwarg(false);
  specialized_graph->set_has_vararg(false);
  specialized_graph->set_kwonlyargs_count(0);
  specialized_graph->ClearDefaultValues();
  specialized_graph->set_is_generate(true);
  func_graph_cache_[args_spec_list] = specialized_graph;
  return specialized_graph;
}

std::shared_ptr<OrderedSet<CNodePtr>> FindRoots(const std::vector<CNodePtr> &segment) {
  std::shared_ptr<OrderedSet<CNodePtr>> roots = std::make_shared<OrderedSet<CNodePtr>>(segment);
  for (const auto &node : segment) {
    if (roots->size() == 1) {
      return roots;
    }
    auto input_size = node->size();
    for (size_t i = 0; i < input_size; i++) {
      auto in_node = node->input(i);
      auto in_cnode = in_node->cast<CNodePtr>();
      if (in_cnode != nullptr) {
        (void)roots->erase(in_cnode);
      }
    }
  }
  return roots;
}

std::shared_ptr<OrderedSet<CNodePtr>> FindLeaves(const std::vector<CNodePtr> &segment) {
  std::shared_ptr<OrderedSet<CNodePtr>> nodes = std::make_shared<OrderedSet<CNodePtr>>(segment);
  for (const auto &node : segment) {
    if (nodes->size() == 1) {
      return nodes;
    }
    if (IsPrimitiveCNode(node, prim::kPrimSwitch)) {
      (void)nodes->erase(node);
      continue;
    }
    auto input_size = node->size();
    for (size_t i = 0; i < input_size; i++) {
      auto in_node = node->input(i);
      if (!in_node->isa<CNode>()) {
        continue;
      }
      auto in_cnode = in_node->cast<CNodePtr>();
      if (in_cnode != nullptr) {
        if (std::find(segment.begin(), segment.end(), in_cnode) != segment.end()) {
          (void)nodes->erase(node);
          break;
        }
      }
    }
  }
  return nodes;
}
}  // namespace mindspore
