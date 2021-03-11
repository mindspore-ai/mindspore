/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/parse/resolve.h"

#include <string>
#include <memory>
#include <vector>

#include "ir/param_info.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "utils/any.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace parse {
abstract::AbstractBasePtr ClassObject::ToAbstract() {
  ClassPtr cls_ptr = ParseDataClass(obj());
  auto abs_scalar = std::make_shared<abstract::AbstractScalar>();
  abs_scalar->set_type(std::make_shared<TypeType>());
  abs_scalar->set_value(cls_ptr);

  AbstractBasePtrList args_spec_list = {abs_scalar};
  auto func_ptr = std::make_shared<abstract::PrimitiveAbstractClosure>(prim::kPrimMakeRecord);
  return std::make_shared<abstract::PartialAbstractClosure>(func_ptr, args_spec_list);
}

abstract::AbstractBasePtr ClassType::ToAbstract() {
  auto abs_scalar =
    std::make_shared<abstract::AbstractScalar>(shared_from_base<ClassType>(), std::make_shared<TypeType>());
  AbstractBasePtrList args_spec_list = {abs_scalar};

  auto func_ptr = std::make_shared<abstract::PrimitiveAbstractClosure>(prim::kPrimCreateInstance);
  auto ret_val = std::make_shared<abstract::PartialAbstractClosure>(func_ptr, args_spec_list);
  ret_val->set_value_desc(ToString());
  return ret_val;
}

// call python PYTHON_MOD_RESOLVE_FUNCTION interface to resolve the symbol in corresponding namespace
bool SymbolResolver::Resolve() {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);

  py::object obj = namespace_->obj();
  std::string symbol = symbol_->symbol();
  if (py::isinstance<py::none>(obj)) {
    MS_EXCEPTION(NameError) << "The name \'" << symbol << "\' is not defined.";
  }
  result_ = python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_FUNCTION, obj, common::SafeCStr(symbol));
  return true;
}

namespace {
// if any mixed precision flag add a cast node after the parameter node.
// argument obj should be python Parameter object
// it will be converted to Parameter node here
AnfNodePtr ResolveParameterObj(const FuncGraphPtr &func_graph, const py::object &obj) {
  MS_EXCEPTION_IF_NULL(func_graph);

  // parameter object should not be none
  if (py::isinstance<py::none>(obj)) {
    MS_LOG(EXCEPTION) << "Resolve class Parameter error because obj is null.";
  }

  if (!py::hasattr(obj, "name")) {
    MS_LOG(EXCEPTION) << "Resolve class Parameter error: cannot find name attr for obj";
  }

  // get the parameter name from parameter object
  auto name_attr = python_adapter::GetPyObjAttr(obj, "name");
  if (py::isinstance<py::none>(name_attr)) {
    MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
  }

  auto param_name = py::cast<std::string>(name_attr);
  auto top_graph = Parser::GetTopFuncGraph();
  // if the parameter node has been created , return it
  AnfNodePtr para_node = nullptr;
  for (auto const &param : top_graph->parameters()) {
    auto param_node = dyn_cast<Parameter>(param);
    if (param_node != nullptr && param_node->name() == param_name) {
      para_node = param;
      break;
    }
  }
  if (para_node == nullptr) {
    auto node = top_graph->AddWeightParameter(param_name);
    auto value = py::cast<tensor::MetaTensorPtr>(obj);
    node->set_default_param(value);
    // set_abstract for parameter
    auto abs = value->ToAbstract();
    node->set_abstract(abs);
    para_node = node;
  }
  func_graph->add_used_global_parameters(para_node);
  return para_node;
}

void BroadenCNodeAbstract(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  for (const AnfNodePtr &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto abstract = node->abstract();
    if (abstract != nullptr) {
      node->set_abstract(abstract->Broaden());
    }
  }
}

void ConvertLoadedGraph(const FuncGraphPtr &func_graph, const ValuePtr &value) {
  if (!value->isa<FuncGraph>()) {
    return;
  }
  auto resolved_graph = value->cast<FuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(resolved_graph);
  if (!resolved_graph->has_attr("is_load")) {
    return;
  }
  auto top_graph = Parser::GetTopFuncGraph();
  std::vector<AnfNodePtr> input_params;
  for (auto const &param : resolved_graph->parameters()) {
    auto param_ptr = dyn_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->has_default()) {
      param_ptr->set_func_graph(top_graph);
      func_graph->add_used_global_parameters(param_ptr);

      // update top_graph
      top_graph->add_parameter(param_ptr);
      size_t hyper_param_count = top_graph->hyper_param_count();
      top_graph->set_hyper_param_count(hyper_param_count + 1);
    } else {
      input_params.push_back(param_ptr);
    }
  }
  resolved_graph->set_parameters(input_params);
  BroadenCNodeAbstract(resolved_graph);
}

bool ResolveObjectToNode(const FuncGraphPtr &func_graph, const py::object &obj, AnfNodePtr *const node) {
  AnfNodePtr output = nullptr;
  if (py::hasattr(obj, "__parameter__") && py::isinstance<tensor::MetaTensor>(obj)) {
    auto param = ResolveParameterObj(func_graph, obj);
    if (param == nullptr) {
      MS_LOG(ERROR) << "Resolve parameter object failed, got nullptr";
      return false;
    }
    MS_LOG(DEBUG) << "Add param graph:" << func_graph->ToString() << ", " << param->DebugString();

    output = param;
  } else if (py::hasattr(obj, "__parameter_tuple__")) {
    auto tuple = obj.cast<py::tuple>();
    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t it = 0; it < tuple.size(); ++it) {
      AnfNodePtr out = nullptr;
      bool success = ResolveObjectToNode(func_graph, tuple[it], &out);
      if (!success) {
        MS_LOG(ERROR) << "Resolve object to node failed";
        return false;
      }
      args.push_back(out);
    }
    output = NewCNode(args, func_graph);
  } else {
    ValuePtr convert_result = nullptr;
    bool converted = ConvertData(obj, &convert_result, parse::python_adapter::UseSignatureInResolve());
    if (!converted) {
      MS_LOG(ERROR) << "Convert data failed";
      return false;
    }
    MS_EXCEPTION_IF_NULL(convert_result);
    ConvertLoadedGraph(func_graph, convert_result);
    output = NewValueNode(convert_result);
    if (convert_result->isa<tensor::Tensor>()) {
      output = GetMixedPrecisionCastHelp(func_graph, output);
    }
  }
  *node = output;
  return true;
}

bool IsAllFuncInValueSequence(const std::vector<ValuePtr> &value_vec) {
  if (value_vec.empty()) {
    return false;
  }
  for (auto &elem : value_vec) {
    if (elem->isa<ValueTuple>() || elem->isa<ValueList>()) {
      const auto &vec = GetValue<ValuePtrList>(elem);
      auto is_graph = IsAllFuncInValueSequence(vec);
      if (!is_graph) {
        return false;
      }
    } else if (!elem->isa<FuncGraph>() && !elem->isa<Primitive>()) {
      return false;
    }
  }
  return true;
}

AnfNodePtr TransformToMakeTupleNodes(const FuncGraphManagerPtr &manager, const FuncGraphPtr &func_graph,
                                     const std::vector<ValuePtr> &value_vec) {
  std::vector<AnfNodePtr> nodes;
  nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  for (auto &elem : value_vec) {
    AnfNodePtr node = nullptr;
    if (elem->isa<ValueTuple>() || elem->isa<ValueList>()) {
      const auto &vec = GetValue<std::vector<ValuePtr>>(elem);
      node = TransformToMakeTupleNodes(manager, func_graph, vec);
    } else if (elem->isa<FuncGraph>()) {
      FuncGraphPtr new_fg = elem->cast<FuncGraphPtr>();
      manager->AddFuncGraph(new_fg);
      node = NewValueNode(new_fg);
    } else if (elem->isa<Primitive>()) {
      node = NewValueNode(elem);
    } else {
      MS_LOG(EXCEPTION) << "TransformToMakeTupleNodes error, expect funcgraph, got " << elem->ToString();
    }
    nodes.emplace_back(node);
  }
  auto cnode = func_graph->NewCNode(nodes);
  return cnode;
}

// Transform the ValueTuple or ValueList of graph/primitive node to make tuple of const graph/primitive node
bool TransformVectorFuncValueNode(const FuncGraphManagerPtr &manager, const FuncGraphPtr &func_graph,
                                  const ValueNodePtr &value_node, AnfNodePtr *const transformed) {
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &value_vec = GetValue<ValuePtrList>(value_node->value());
  if (!IsAllFuncInValueSequence(value_vec)) {
    return false;
  }

  // (1) The celllist or ordered_cell will be parsed as valuetuple of const graph in it,
  // So if has graph in list, try to replace the node with make tuple of graph value node.
  // We do this because the graph manager won't investigate the graph inside valuetuple,
  // change the vector of graph to be make_tuple of graph value node.
  // (2) the primitive valuetuple or valuelist may encounter to abstract error, make it all
  // independent nodes.
  auto node_tuple_graphs = TransformToMakeTupleNodes(manager, func_graph, value_vec);
  // Replace the ret ptr to be make tuple of graph value node
  *transformed = node_tuple_graphs;

  return true;
}

// Resolve the python obj, and if the resovled node is valuenode with graphs, add the graphs to manager.
AnfNodePtr ResolveObjectAndAddToManager(const FuncGraphManagerPtr &manager, const py::object &obj,
                                        const AnfNodePtr &node) {
  ScopeGuard scope_guard(node->scope());
  AnfNodePtr resolved_node = nullptr;
  bool success = ResolveObjectToNode(node->func_graph(), obj, &resolved_node);
  if (!success) {
    MS_LOG(EXCEPTION) << "Parse Resolve covert failed NodeInfo.";
  }
  if (IsValueNode<FuncGraph>(resolved_node)) {
    auto new_fg = GetValueNode<FuncGraphPtr>(resolved_node);
    manager->AddFuncGraph(new_fg);
  }

  // If the constant node is constant of vector of graph, add graph to manager.
  if (IsValueNode<ValueTuple>(resolved_node) || IsValueNode<ValueList>(resolved_node)) {
    (void)TransformVectorFuncValueNode(manager, node->func_graph(), resolved_node->cast<ValueNodePtr>(),
                                       &resolved_node);
  }
  return resolved_node;
}
}  // namespace

AnfNodePtr ResolveSymbol(const FuncGraphManagerPtr &manager, const NameSpacePtr &name_space, const SymbolPtr &symbol,
                         const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  TraceGuard trace_guard(std::make_shared<TraceResolve>(node->debug_info()));
  if (node->func_graph() == nullptr || manager == nullptr) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " graph or manager is nullptr";
  }
  SymbolResolver symbol_resolver(name_space, symbol, node);
  symbol_resolver.Resolve();
  py::object obj = symbol_resolver.result();
  AnfNodePtr resolved_node = ResolveObjectAndAddToManager(manager, obj, node);
  TraceManager::ClearParseOrResolveDebugInfo();
  return resolved_node;
}

AnfNodePtr ResolveCellwithAttr(const FuncGraphManagerPtr &manager, const NameSpacePtr &name_space,
                               const SymbolPtr &symbol, const AnfNodePtr &node, const std::string &attr) {
  MS_EXCEPTION_IF_NULL(node);
  TraceGuard trace_guard(std::make_shared<TraceResolve>(node->debug_info()));
  if (node->func_graph() == nullptr || manager == nullptr) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " graph or manager is nullptr";
  }
  SymbolResolver symbol_resolver(name_space, symbol, node);
  if (!symbol_resolver.Resolve()) {
    MS_LOG(EXCEPTION) << "Parse Resolve node failed NodeInfo.";
  }

  py::object obj = symbol_resolver.result();
  if (!data_converter::IsCellInstance(obj)) {
    return nullptr;
  }

  const std::string fn = PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL;
  const std::string module = "mindspore._extends.parse.parser";
  py::object namespace_obj = parse::python_adapter::GetPyFn(module, fn)(obj);
  auto new_namespace = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_obj);
  auto new_symbol = std::make_shared<Symbol>(attr);

  AnfNodePtrList inputs = {NewValueNode(prim::kPrimResolve), NewValueNode(new_namespace), NewValueNode(new_symbol)};
  AnfNodePtr resolved_node = node->func_graph()->NewCNode(inputs);
  TraceManager::ClearParseOrResolveDebugInfo();
  return resolved_node;
}

namespace {
opt::OptPassGroupMap GetOptResolvePasses(const opt::irpass::ResolveIRPassLib &irpass) {
  opt::OptPassGroupMap map({
    {"resolve",
     {
       // for resolve and getattr primitive;
       irpass.resolver_resolve_and_getattr_,
     }},
  });
  return map;
}
}  // namespace

bool ResolveFuncGraph(const FuncGraphPtr &func_graph, const pipeline::ResourceBasePtr &res, bool use_profile) {
  if (func_graph == nullptr || res == nullptr) {
    MS_LOG(ERROR) << "func_graph or resource is null";
    return false;
  }
  opt::irpass::ResolveIRPassLib irpass;
  opt::OptimizerPtr opt_resolve =
    opt::Optimizer::MakeOptimizer("opt_resolve", res, GetOptResolvePasses(irpass), false, false, false);

  (void)parse::python_adapter::set_python_scoped();

  MS_EXCEPTION_IF_NULL(opt_resolve);
  (void)opt_resolve->step(func_graph, use_profile);
  return true;
}

bool ResolveAll(const FuncGraphManagerPtr &manager) {
  if (manager == nullptr) {
    MS_LOG(ERROR) << "func graph manager is null";
    return false;
  }

  if (manager->roots().size() > 1) {
    MS_LOG(WARNING)
      << "After call ResolveAll, only one graph will be kept in GraphManager. ResolveAll can resolve graphs"
         "called from root graph, so it's not necessary to pass all graphs as roots. "
         "Please ensure your usage.";
  }
  // should not use pipeline::Resource as Resource::Clean will clean some
  // global variable such as ScopeManager, it will cause JExpandedGraphs::GetBprop
  // fail as valid scope has been cleaned.
  auto res = std::make_shared<pipeline::ResourceBase>();
  res->set_manager(manager);

  auto roots = manager->roots();
  for (auto &fg : roots) {
    bool ret = ResolveFuncGraph(fg, res, false);
    if (!ret) {
      MS_EXCEPTION_IF_NULL(fg);
      MS_LOG(ERROR) << "Resolve fg " << fg->ToString() << " failed";
    }
  }
  return true;
}
}  // namespace parse
}  // namespace mindspore
