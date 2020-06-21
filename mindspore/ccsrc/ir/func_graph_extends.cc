/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <sstream>
#include <utility>

#include "ir/manager.h"
#include "ir/func_graph_cloner.h"
#include "operator/ops.h"
#include "utils/ordered_set.h"
#include "pipeline/static_analysis/abstract_value.h"
#include "pipeline/static_analysis/static_analysis.h"
#include "pipeline/static_analysis/abstract_function.h"

#include "debug/anf_ir_dump.h"
#include "debug/trace.h"
#include "debug/draw.h"
#include "debug/label.h"

namespace mindspore {
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;
using mindspore::abstract::AnalysisContextPtr;
using mindspore::abstract::PrimitiveAbstractClosure;
using mindspore::abstract::VirtualAbstractClosure;

AbstractFunctionPtr FuncGraph::abstract() {
  AbstractBasePtrList args_spec_list;

  for (auto &p : parameters_) {
    MS_EXCEPTION_IF_NULL(p);
    if (p->abstract() == nullptr) {
      MS_LOG(ERROR) << "Error!!";
      return nullptr;
    }
    args_spec_list.push_back(p->abstract());
  }

  if (nullptr == output()) {
    MS_LOG(ERROR) << "Error func graph no output";
    return nullptr;
  }

  return std::make_shared<VirtualAbstractClosure>(args_spec_list, output()->abstract());
}

abstract::AbstractBasePtr FuncGraph::MakeAbstractClosure(const abstract::AnalysisContextPtr &context) {
  AnalysisContextPtr temp_context = context;
  if (temp_context == nullptr) {
    temp_context = abstract::AnalysisContext::DummyContext();
  }
  return std::make_shared<abstract::FuncGraphAbstractClosure>(shared_from_base<FuncGraph>(), temp_context);
}

void FuncGraph::set_output(const AnfNodePtr &value, bool force_new_ret) {
  if (force_new_ret || return_ == nullptr) {
    std::vector<AnfNodePtr> params({NewValueNode(prim::kPrimReturn), value});
    FuncGraphPtr this_graph = shared_from_base<FuncGraph>();
    return_ = this_graph->NewCNode(params);
  } else {
    if (manager_.lock()) {
      manager_.lock()->SetEdge(return_, 1, value);
    } else {
      return_->set_input(1, value);
    }
  }

  return_->set_abstract(value->abstract());

  AnfNodePtr input0 = return_->input(0);

  PrimitivePtr return_prim = prim::kPrimReturn;
  auto f = std::make_shared<PrimitiveAbstractClosure>(return_prim, input0);
  input0->set_abstract(f);
}

void FuncGraph::DumpFuncGraph(const std::string &path) { draw::Draw(path + ".dot", shared_from_base<FuncGraph>()); }

void FuncGraph::GenerateVarParams(const FuncGraphPtr &specialized_graph,
                                  std::vector<AnfNodePtr> *specialized_parameter_list,
                                  std::unordered_map<AnfNodePtr, AnfNodePtr> *repl_nodes, int variable_args_count,
                                  int pos_args_input_count) {
  // if there is variable argument, pass the input arguments that does not match positional args to it as a tuple
  if (specialized_graph->has_vararg()) {
    TraceManager::DebugTrace(
      std::make_shared<TraceGenerateVarArg>(specialized_graph->GetVariableArgParameter()->debug_info()));
    std::vector<AnfNodePtr> var_param_tuple_nodes;
    var_param_tuple_nodes.push_back(NewValueNode(prim::kPrimMakeTuple));

    if (variable_args_count < 0) {
      MS_LOG(EXCEPTION) << "Function:" << this->ToString() << ", variable_args_count " << variable_args_count
                        << " were given.";
    }
    // for python variable argument input , there is no upper limit
    for (int i = 0; i < variable_args_count; ++i) {
      ParameterPtr p = std::make_shared<Parameter>(specialized_graph);
      std::string param_name = specialized_graph->GetVariableArgName() + std::to_string(i);
      p->set_name(param_name);
      MS_EXCEPTION_IF_NULL(p->debug_info());
      p->debug_info()->set_name(param_name);
      var_param_tuple_nodes.push_back(p);
      MS_EXCEPTION_IF_NULL(specialized_parameter_list);
      specialized_parameter_list->push_back(p);
    }
    auto var_tuple_param = specialized_graph->NewCNode(var_param_tuple_nodes);
    (void)repl_nodes->emplace(specialized_graph->GetVariableArgParameter(), var_tuple_param);
    TraceManager::EndTrace();
  } else if (variable_args_count > 0) {
    MS_LOG(EXCEPTION) << "Function:" << this->ToString() << " takes " << this->GetPositionalArgsCount()
                      << " positional arguments, but " << pos_args_input_count << " were given.";
  }
}

void FuncGraph::GenerateKwParams(const FuncGraphPtr &specialized_graph,
                                 std::vector<AnfNodePtr> *specialized_parameter_list,
                                 const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list,
                                 std::unordered_map<AnfNodePtr, AnfNodePtr> *repl_nodes) {
  std::vector<AnfNodePtr> kwarg_keys_tuple_nodes = {NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AnfNodePtr> kwarg_values_tuple_nodes = {NewValueNode(prim::kPrimMakeTuple)};

  for (const auto &kwarg : kwarg_list) {
    MS_EXCEPTION_IF_NULL(kwarg);
    std::string kw_param_name = kwarg->get_key();
    MS_EXCEPTION_IF_NULL(specialized_graph);
    AnfNodePtr param_node = specialized_graph->GetParameterByName(kw_param_name);
    // if not find correspoding parameter node
    if (param_node == nullptr) {
      if (!has_kwarg()) {
        MS_LOG(EXCEPTION) << "Got unexpected keyword argument: " << kw_param_name;
      } else {
        ParameterPtr p = std::make_shared<Parameter>(specialized_graph);
        std::string param_name = specialized_graph->GetVariableKwargName() + "[" + kw_param_name + "]";
        MS_EXCEPTION_IF_NULL(specialized_parameter_list);
        auto find_kw_arg_in_list = std::any_of(specialized_parameter_list->begin(), specialized_parameter_list->end(),
                                               [param_name](const AnfNodePtr &node) {
                                                 MS_EXCEPTION_IF_NULL(node);
                                                 auto param = node->cast<ParameterPtr>();
                                                 return param != nullptr && param->name() == param_name;
                                               });
        if (find_kw_arg_in_list) {
          MS_LOG(EXCEPTION) << "Multiply values for keyword argument:" << kw_param_name;
        }
        p->set_name(param_name);
        p->debug_info()->set_name(param_name);
        kwarg_keys_tuple_nodes.push_back(NewValueNode(kw_param_name));
        auto extract_node =
          specialized_graph->NewCNode({NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(kw_param_name), p});
        kwarg_values_tuple_nodes.push_back(extract_node);
        specialized_parameter_list->push_back(p);
      }
    } else {
      auto node_itr = std::find(specialized_parameter_list->begin(), specialized_parameter_list->end(), param_node);
      // multiply values found given for parameter
      if (node_itr != specialized_parameter_list->end()) {
        MS_LOG(EXCEPTION) << "Multiply values for specific argument:" << kw_param_name;
      } else {
        specialized_parameter_list->push_back(param_node);
        auto extract_node = specialized_graph->NewCNode(
          {NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(kw_param_name), param_node});
        (void)repl_nodes->emplace(param_node, extract_node);
      }
    }
  }

  GenerateKwargReplNode(specialized_graph, repl_nodes, kwarg_keys_tuple_nodes, kwarg_values_tuple_nodes);
}

void FuncGraph::GenerateKwargReplNode(const FuncGraphPtr &specialized_graph,
                                      std::unordered_map<AnfNodePtr, AnfNodePtr> *repl_nodes,
                                      const std::vector<AnfNodePtr> &kwarg_keys_tuple_nodes,
                                      const std::vector<AnfNodePtr> &kwarg_values_tuple_nodes) {
  if (has_kwarg()) {
    MS_EXCEPTION_IF_NULL(specialized_graph);
    TraceManager::DebugTrace(
      std::make_shared<TraceGenerateKwArg>(specialized_graph->GetVariableKwargParameter()->debug_info()));
    auto make_tuple_keys = specialized_graph->NewCNode(kwarg_keys_tuple_nodes);
    auto make_tuple_values = specialized_graph->NewCNode(kwarg_values_tuple_nodes);
    auto make_dict_node =
      specialized_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), make_tuple_keys, make_tuple_values});
    MS_EXCEPTION_IF_NULL(repl_nodes);
    (void)repl_nodes->emplace(specialized_graph->GetVariableKwargParameter(), make_dict_node);
    TraceManager::EndTrace();
  }
}

bool FuncGraph::NeedGenerate(const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list) {
  // if the function does not have any vararg/kwarg/kwonly/default value/kw args input
  // return the original graph
  if (!has_vararg() && kwonlyargs_count() == 0 && !has_kwarg() && GetDefaultValueCount() == 0 && kwarg_list.empty()) {
    return false;
  }

  // if the graph is generated for specific input, do not need to generate again
  if (is_generated()) {
    return false;
  }
  return true;
}

void FuncGraph::GenerateDefaultValue(const FuncGraphPtr &specialized_graph,
                                     const std::vector<AnfNodePtr> &specialized_parameter_list,
                                     std::unordered_map<AnfNodePtr, AnfNodePtr> *repl_nodes) {
  MS_EXCEPTION_IF_NULL(specialized_graph);
  for (size_t i = 0; i < specialized_graph->parameters().size() - hyper_param_count(); ++i) {
    auto param_node = specialized_graph->parameters()[i];
    MS_EXCEPTION_IF_NULL(param_node);
    auto param_name = param_node->cast<ParameterPtr>()->name();
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
  size_t arguments_count = args_spec_list.size();
  for (const auto &arg : args_spec_list) {
    // if it is a keyword argument
    MS_EXCEPTION_IF_NULL(arg);
    if (arg->isa<abstract::AbstractKeywordArg>()) {
      kwarg_list.push_back(dyn_cast<abstract::AbstractKeywordArg>(arg));
    }
  }
  if (!NeedGenerate(kwarg_list)) {
    return shared_from_base<FuncGraph>();
  }
  FuncGraphPtr specialized_graph = BasicClone(shared_from_base<FuncGraph>());
  size_t kwarg_count = kwarg_list.size();
  int pos_args_input_count = SizeToInt(arguments_count - kwarg_count - hyper_param_count());
  int pos_args_count = std::min(pos_args_input_count, this->GetPositionalArgsCount());
  int variable_args_count = pos_args_input_count - pos_args_count;
  std::vector<AnfNodePtr> specialized_parameter_list;
  std::unordered_map<AnfNodePtr, AnfNodePtr> repl_nodes;
  // the parameters that has arg input, copy from original parameters
  for (size_t i = 0; i < IntToSize(pos_args_count); ++i) {
    specialized_parameter_list.push_back(specialized_graph->parameters()[i]);
  }

  GenerateVarParams(specialized_graph, &specialized_parameter_list, &repl_nodes, variable_args_count,
                    pos_args_input_count);

  GenerateKwParams(specialized_graph, &specialized_parameter_list, kwarg_list, &repl_nodes);

  GenerateDefaultValue(specialized_graph, specialized_parameter_list, &repl_nodes);

  // append hyper parameter to specialized_parameter_list
  MS_EXCEPTION_IF_NULL(specialized_graph);
  auto params = specialized_graph->parameters();
  (void)std::transform(params.end() - SizeToInt(hyper_param_count()), params.end(),
                       std::back_inserter(specialized_parameter_list), [](const AnfNodePtr &node) { return node; });

  std::shared_ptr<mindspore::FuncGraphManager> manager = mindspore::Manage(specialized_graph, false);
  auto tr = manager->Transact();
  for (auto &node_pair : repl_nodes) {
    MS_LOG(DEBUG) << "GenerateGraph replace:" << node_pair.first->DebugString() << "-"
                  << node_pair.second->DebugString();
    (void)tr.Replace(node_pair.first, node_pair.second);
  }
  tr.SetParameters(specialized_graph, specialized_parameter_list);
  tr.Commit();
  specialized_graph->set_has_kwarg(false);
  specialized_graph->set_has_vararg(false);
  specialized_graph->set_kwonlyargs_count(0);
  specialized_graph->ClearDefaultValues();
  specialized_graph->set_is_generate(true);
  return specialized_graph;
}

const char kPrimHasEffect[] = "_side_effect_flag";

bool FuncGraph::HasEffect(const CNodePtr &cnode) {
  auto prim = GetCNodePrimitive(cnode);
  if (prim != nullptr && prim->isa<prim::DoSignaturePrimitive>()) {
    auto do_sig = prim->cast<prim::DoSignaturePrimitivePtr>();
    auto prim_val = do_sig->function();
    if (prim_val != nullptr && prim_val->isa<Primitive>()) {
      prim = prim_val->cast<PrimitivePtr>();
    } else {
      prim = nullptr;
    }
  }
  if (prim != nullptr) {
    auto effect_val = prim->GetAttr(kPrimHasEffect);
    if (effect_val && effect_val->isa<BoolImm>()) {
      auto effect_bool = GetValue<bool>(effect_val);
      return effect_bool;
    }
  }
  return false;
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

void FuncGraph::ReleaseFullOrderToEffectOrder() {
  MS_LOG(DEBUG) << "Flag has_effect " << has_flag(GRAPH_FLAG_HAS_EFFECT) << ".";
  if (has_flag(GRAPH_FLAG_HAS_EFFECT)) {
    std::list<AnfNodePtr> depends_order;
    std::vector<CNodePtr> segment;
    for (const auto &cnode : order_) {
      if (IsPrimitiveCNode(cnode, prim::kPrimReturn)) {
        continue;
      }
      if (HasEffect(cnode)) {
        MS_LOG(DEBUG) << "Meet a effect node " << cnode->DebugString() << ".";
        if (segment.size() > 0) {
          auto roots = FindRoots(segment);
          for (auto iter = roots->begin(); iter != roots->end(); (void)iter++) {
            depends_order.push_back(*iter);
          }
        }
        segment.clear();
        depends_order.push_back(cnode);
      } else {
        MS_LOG(DEBUG) << "Meet a general node " << cnode->DebugString() << ".";
        segment.push_back(cnode);
      }
    }
    if (segment.size() > 1) {
      auto roots = FindRoots(segment);
      for (auto iter = roots->begin(); iter != roots->end(); (void)iter++) {
        depends_order.push_back(*iter);
      }
    }
    std::vector<AnfNodePtr> depend_inputs;
    auto old_ret = output();
    for (auto iter = depends_order.rbegin(); iter != depends_order.rend(); (void)iter++) {
      if (*iter != old_ret) {
        depend_inputs.push_back(*iter);
      }
    }
    set_flag(GRAPH_FLAG_HAS_EFFECT, false);
    set_flag(GRAPH_FLAG_EFFECT_PATIAL_ORDER, true);
    if (!depend_inputs.empty()) {
      SetEffectDepends(depend_inputs);
    }
  }
}

void FuncGraph::SetEffectDepends(const std::vector<AnfNodePtr> &depend_inputs) {
  auto old_ret = output();
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimDepend), old_ret};
  (void)inputs.insert(inputs.end(), depend_inputs.begin(), depend_inputs.end());
  auto new_ret = NewCNode(inputs);
  auto mng = manager();
  if (mng) {
    (void)mng->Replace(old_ret, new_ret);
  } else {
    return_->set_input(1, new_ret);
  }
}
}  // namespace mindspore
