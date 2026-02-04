/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "ir/core_ops_primitive.h"
#include "ir/manager.h"
#include "utils/ordered_set.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"
#include "ir/func_graph_cloner.h"
#include "utils/trace_info.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;
using mindspore::abstract::AnalysisContextPtr;
using mindspore::abstract::PrimitiveAbstractClosure;
using mindspore::abstract::VirtualAbstractClosure;

AbstractFunctionPtr FuncGraph::abstract() {
  AbstractBasePtrList args_abs_list;

  for (auto &para : parameters_) {
    MS_EXCEPTION_IF_NULL(para);
    if (para->abstract() == nullptr) {
      MS_LOG(ERROR) << "Error!!";
      return nullptr;
    }
    args_abs_list.push_back(para->abstract());
  }

  if (output() == nullptr) {
    MS_LOG(ERROR) << "Error func graph no output";
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(output());
  return std::make_shared<VirtualAbstractClosure>(args_abs_list, output()->abstract());
}

void FuncGraph::set_output(const AnfNodePtr &value, bool force_new_ret) {
  MS_EXCEPTION_IF_NULL(value);
  if (force_new_ret || return_node() == nullptr) {
    AnfNodePtrList params({NewValueNode(prim::kPrimReturn), value});
    FuncGraphPtr this_graph = shared_from_base<FuncGraph>();
    set_return(this_graph->NewCNodeInOrder(std::move(params)));
  } else {
    if (manager_.lock()) {
      manager_.lock()->SetEdge(return_node(), 1, value);
    } else {
      constexpr auto first_data_index = 1;
      return_node()->set_input(first_data_index, value);
    }
  }

  return_node()->set_abstract(value->abstract());
  AnfNodePtr input0 = return_node()->input(0);
  auto f = std::make_shared<PrimitiveAbstractClosure>(prim::kPrimReturn, input0);
  input0->set_abstract(f);
}

void FuncGraph::GenerateVarParams(const FuncGraphPtr &specialized_graph, int pos_params_count, int pos_args_input_count,
                                  AnfNodePtrList *specialized_parameter_list,
                                  mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const {
  MS_EXCEPTION_IF_NULL(specialized_graph);
  MS_EXCEPTION_IF_NULL(specialized_parameter_list);
  if (!specialized_graph->has_vararg()) {
    // No *args in parameter, varaible_args_count need to be 0
    if (pos_args_input_count > pos_params_count) {
      MS_LOG(EXCEPTION) << "Function:" << this->ToString() << " takes " << pos_params_count
                        << " positional arguments, but " << pos_args_input_count << " were given.";
    }
    // Only copy parameters other than default arguments.
    size_t pos_size = IntToSize(pos_args_input_count);
    for (size_t i = 0; i < pos_size; ++i) {
      specialized_parameter_list->push_back(specialized_graph->parameters()[i]);
    }
    return;
  }

  // If there is variable argument.
  int max_pos_args_count = pos_args_input_count + SizeToInt(GetDefaultValueCount());
  if (max_pos_args_count < pos_params_count) {
    MS_LOG(EXCEPTION) << "For function:" << this->ToString() << ", its max argument size: " << max_pos_args_count
                      << " is less than parameter size: " << pos_params_count
                      << " , actual argument size from caller: " << pos_args_input_count;
  }
  // Copy other parameters than vararg's firstly.
  int actual_args_count_num = pos_args_input_count < pos_params_count ? pos_args_input_count : pos_params_count;
  for (size_t i = 0; i < IntToSize(actual_args_count_num); ++i) {
    specialized_parameter_list->push_back(specialized_graph->parameters()[i]);
  }
  MS_EXCEPTION_IF_NULL(specialized_graph->GetVariableArgParameter());
  TraceGuard trace_guard(
    MakeTraceInfo<TraceGenerateVarArg>(specialized_graph->GetVariableArgParameter()->debug_info()));
  AnfNodePtrList var_param_tuple_nodes;
  var_param_tuple_nodes.push_back(NewValueNode(prim::kPrimMakeTuple));

  auto varg_name = specialized_graph->GetVariableArgName();
  // For python variable argument input, there is no upper limit.
  int variable_args_count = pos_args_input_count < pos_params_count ? 0 : pos_args_input_count - pos_params_count;
  for (int i = 0; i < variable_args_count; ++i) {
    ParameterPtr para = std::make_shared<Parameter>(specialized_graph);
    std::string param_name = varg_name + std::to_string(i);
    para->set_name(param_name);
    if (para->debug_info() != nullptr) {
      para->debug_info()->set_name(param_name);
    }
    var_param_tuple_nodes.push_back(para);
    specialized_parameter_list->push_back(para);
  }
  auto var_tuple_param = specialized_graph->NewCNode(std::move(var_param_tuple_nodes));
  MS_EXCEPTION_IF_NULL(repl_nodes);
  (void)repl_nodes->emplace(specialized_graph->GetVariableArgParameter(), var_tuple_param);
}

void FuncGraph::GenerateKwParams(const FuncGraphPtr &specialized_graph,
                                 const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list,
                                 int pos_args_input_count, AnfNodePtrList *specialized_parameter_list,
                                 mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const {
  MS_EXCEPTION_IF_NULL(specialized_parameter_list);
  MS_EXCEPTION_IF_NULL(repl_nodes);
  MS_EXCEPTION_IF_NULL(specialized_graph);
  AnfNodePtrList kwarg_keys_tuple_nodes = {NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList kwarg_values_tuple_nodes = {NewValueNode(prim::kPrimMakeTuple)};

  std::set<AnfNodePtr> kwarg_nodes;
  for (size_t i = 0; i < kwarg_list.size(); ++i) {
    auto kwarg = kwarg_list[i];
    MS_EXCEPTION_IF_NULL(kwarg);
    std::string kw_param_name = kwarg->get_key();
    AnfNodePtr param_node = specialized_graph->GetParameterByName(kw_param_name);
    // If not find corresponding parameter node.
    if (param_node == nullptr) {
      if (!has_kwarg()) {
        MS_LOG(DEBUG) << "Not found parameter by name '" << kw_param_name << "'";
        if (IntToSize(pos_args_input_count) + i < specialized_graph->parameters().size()) {
          auto kw_param = dyn_cast<Parameter>(specialized_graph->parameters()[IntToSize(pos_args_input_count) + i]);
          if (kw_param != nullptr && (specialized_graph->has_flag(FUNC_GRAPH_FLAG_ARGS_NO_EXPAND) ||
                                      kw_param->name() == "kwargs[" + kw_param_name + "]")) {
            specialized_parameter_list->push_back(kw_param);
            continue;
          }
        }
        MS_LOG(EXCEPTION) << "Got an unexpected keyword argument '" << kw_param_name << "'";
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
        if (para->debug_info() != nullptr) {
          para->debug_info()->set_name(param_name);
        }
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
                                      const AnfNodePtrList &kwarg_keys_tuple_nodes,
                                      const AnfNodePtrList &kwarg_values_tuple_nodes,
                                      mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const {
  if (has_kwarg() && !kwarg_keys_tuple_nodes.empty()) {
    MS_EXCEPTION_IF_NULL(specialized_graph);
    MS_EXCEPTION_IF_NULL(specialized_graph->GetVariableKwargParameter());
    TraceGuard guard(MakeTraceInfo<TraceGenerateKwArg>(specialized_graph->GetVariableKwargParameter()->debug_info()));
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
                                     const AnfNodePtrList &specialized_parameter_list,
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
      MS_LOG(INTERNAL_EXCEPTION) << "Miss argument input for parameter:" << param_name;
    }
    MS_EXCEPTION_IF_NULL(repl_nodes);
    (void)repl_nodes->emplace(param_node, default_value);
  }
}

FuncGraphPtr FuncGraph::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  if (has_attr(FUNC_GRAPH_FLAG_PROXY_GRAPH)) {
    MS_LOG(DEBUG) << "proxy function graph: " << ToString();
    auto original_params_size = parameters().size();
    auto args_size = args_abs_list.size();
    MS_LOG(DEBUG) << "original_params_size:" << original_params_size << ", args_size:" << args_size;
    if (args_size != original_params_size) {
      std::vector<AnfNodePtr> new_params;
      set_parameters(new_params);
      MS_EXCEPTION_IF_NULL(output());
      auto call_node = output()->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(call_node);
      std::vector<AnfNodePtr> new_inputs{call_node->input(0)};
      for (size_t index = 0; index < args_size; ++index) {
        auto new_param = add_parameter();
        (void)new_params.emplace_back(new_param);
        (void)new_inputs.emplace_back(new_param);
      }
      set_parameters(new_params);
      auto new_out = NewCNodeInOrder(new_inputs);
      set_output(new_out);
      MS_LOG(DEBUG) << "The proxy truncates the parameters to match the size. fg: " << ToString()
                    << ", original args: " << original_params_size << ", call args: " << args_size
                    << ", new args: " << parameters().size() << ", call inputs: " << new_out->inputs().size();
    }
    return shared_from_base<FuncGraph>();
  }

  std::vector<abstract::AbstractKeywordArgPtr> kwarg_list;
  std::vector<size_t> pos_arg_indexes;
  size_t arguments_count = args_abs_list.size();
  if (fv_param_count_ > arguments_count) {
    MS_LOG(INTERNAL_EXCEPTION) << "The number of parameters in funcgraph cannot exceed the number of arguments.";
  }
  for (size_t i = 0; i < arguments_count - fv_param_count_; i++) {
    MS_EXCEPTION_IF_NULL(args_abs_list[i]);
    if (args_abs_list[i]->isa<abstract::AbstractKeywordArg>()) {
      kwarg_list.push_back(args_abs_list[i]->cast<abstract::AbstractKeywordArgPtr>());
    } else {
      pos_arg_indexes.push_back(i);
    }
  }

  if (!NeedGenerate(kwarg_list)) {
    MS_LOG(DEBUG) << "No need generate, " << ToString();
    return shared_from_base<FuncGraph>();
  }
  auto iter = func_graph_cache_.find(args_abs_list);
  if (iter != func_graph_cache_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    MS_LOG(DEBUG) << "Found in cache, " << iter->second->ToString() << ", for " << ToString();
    return iter->second;
  }
  FuncGraphPtr specialized_graph = BasicClone(shared_from_base<FuncGraph>());
  MS_EXCEPTION_IF_NULL(specialized_graph);
  size_t kwarg_count = kwarg_list.size();
  // Get the variable args count from caller.
  // Relationship between pos_args_input_count and pos_params_count
  // Callee pos_params_count:  param number before *args, eg: [a, b, c=1, d=2, *args] = 4
  // Caller pos_args_input_count :
  //    Range from [pos_params_count - default_param_count, pos_params_count + varaible_args_count]
  // Actual varaible_args_count: max(0, pos_args_input_count - pos_params_count)
  int pos_args_input_count = SizeToInt((arguments_count - kwarg_count) - fv_param_count_);
  int pos_params_count = GetPositionalArgsCount();
  int variable_args_count = pos_args_input_count < pos_params_count ? 0 : pos_args_input_count - pos_params_count;
  AnfNodePtrList specialized_parameter_list;
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> repl_nodes;
  MS_LOG(DEBUG) << "specialized_graph: " << specialized_graph->ToString()
                << ", variable_args_count: " << variable_args_count << ", pos_params_count: " << pos_params_count
                << ", pos_args_input_count: " << pos_args_input_count << ", arguments_count: " << arguments_count
                << ", kwarg_count: " << kwarg_count << ", fv_param_count_: " << fv_param_count_;
  GenerateVarParams(specialized_graph, pos_params_count, pos_args_input_count, &specialized_parameter_list,
                    &repl_nodes);
  GenerateKwParams(specialized_graph, kwarg_list, pos_args_input_count, &specialized_parameter_list, &repl_nodes);

  GenerateDefaultValue(specialized_graph, specialized_parameter_list, &repl_nodes);

  // Append hyper parameter to specialized_parameter_list
  auto params = specialized_graph->parameters();
  (void)specialized_parameter_list.insert(specialized_parameter_list.end(), params.end() - SizeToInt(fv_param_count_),
                                          params.end());
  AnfNodePtrList specialized_parameter_list_update(
    specialized_parameter_list.begin() + SizeToLong(pos_arg_indexes.size()), specialized_parameter_list.end());
  for (size_t i = 0; i < pos_arg_indexes.size(); i++) {
    (void)specialized_parameter_list_update.insert(specialized_parameter_list_update.begin() + pos_arg_indexes[i],
                                                   specialized_parameter_list[i]);
  }

  ud_chain::Preprocess(specialized_graph);
  for (auto &node_pair : repl_nodes) {
    MS_EXCEPTION_IF_NULL(node_pair.first);
    MS_EXCEPTION_IF_NULL(node_pair.second);
    MS_LOG(DEBUG) << "GenerateFuncGraph replace: " << node_pair.first->DebugString() << ", "
                  << node_pair.second->DebugString();
    ud_chain::Replace(node_pair.first, node_pair.second);
  }
  specialized_graph->set_parameters(specialized_parameter_list_update);
  specialized_graph->set_has_kwarg(false);
  specialized_graph->set_has_vararg(false);
  specialized_graph->set_kwonlyargs_count(0);
  specialized_graph->ClearDefaultValues();
  specialized_graph->set_is_generate(true);
  func_graph_cache_[args_abs_list] = specialized_graph;
  MS_LOG(DEBUG) << "Generated, " << specialized_graph->ToString() << ", for " << ToString();
  return specialized_graph;
}
}  // namespace mindspore
