/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/parse/resolve.h"

#include <string>
#include <memory>
#include <vector>
#include <algorithm>

#include "pipeline/parse/data_converter.h"
#include "pipeline/parse/parse.h"
#include "pipeline/parse/python_adapter.h"
#include "utils/any.h"
#include "operator/ops.h"
#include "optimizer/opt.h"
#include "optimizer/irpass.h"
#include "./common.h"

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
    MS_LOG(ERROR) << "Unresolved symbol: " << symbol;
    return false;
  }
  result_ = python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_FUNCTION, obj, common::SafeCStr(symbol));
  return true;
}

namespace {
// argument obj should be python Parameter object
// it will be converted to Parameter node here
AnfNodePtr ResolveParameterObj(const FuncGraphPtr& func_graph, const py::object& obj) {
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

  std::string param_name = py::cast<std::string>(name_attr);
  auto top_graph = Parser::GetTopFuncGraph();
  // if the parameter node has been created , return it
  AnfNodePtr para_node = nullptr;
  for (auto param : top_graph->parameters()) {
    auto param_node = dyn_cast<Parameter>(param);
    if (param_node != nullptr && param_node->name() == param_name) {
      para_node = param;
      break;
    }
  }
  if (para_node == nullptr) {
    ParameterPtr node = top_graph->AddWeightParameter(param_name);
    node->set_default_param(obj);
    para_node = node;
  }
  auto iter = func_graph->make_ref_params().find(para_node);
  if (iter == func_graph->make_ref_params().end()) {
    AnfNodePtr value = GetMixedPrecisionCastHelp(func_graph, para_node);

    AnfNodePtr make_ref = NewValueNode(prim::kPrimMakeRef);
    AnfNodePtr ref_key = NewValueNode(std::make_shared<RefKey>(param_name));
    AnfNodePtr ref_node = func_graph->NewCNode({make_ref, ref_key, value, para_node});
    func_graph->make_ref_params()[para_node] = ref_node;
    func_graph->add_parameter_obj_node(ref_node);
    return ref_node;
  } else {
    return iter->second;
  }
}

bool ResolveObjectToNode(const FuncGraphPtr& func_graph, const py::object& obj, AnfNodePtr* const node) {
  AnfNodePtr output = nullptr;
  if (py::hasattr(obj, "__parameter__")) {
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
    output = NewValueNode(convert_result);
    if (convert_result->isa<tensor::Tensor>()) {
      output = GetMixedPrecisionCastHelp(func_graph, output);
    }
  }
  *node = output;
  return true;
}

// transform the ValueTuple or ValueList of graph node to make tuple of const graph node
bool TransformVectorGraphValueNode(const FuncGraphManagerPtr& manager, const AnfNodePtr& node,
                                   const ValueNodePtr& value_node, AnfNodePtr* const transformed) {
  MS_EXCEPTION_IF_NULL(value_node);
  const auto& value_vec = GetValue<std::vector<ValuePtr>>(value_node->value());
  bool has_graph_in_list = false;
  for (auto& elemv : value_vec) {
    MS_EXCEPTION_IF_NULL(elemv);
    if (elemv->isa<FuncGraph>()) {
      FuncGraphPtr new_fg = elemv->cast<FuncGraphPtr>();
      manager->AddFuncGraph(new_fg);
      has_graph_in_list = true;
      continue;
    }
    if (has_graph_in_list) {
      MS_LOG(EXCEPTION) << "List has graph in it, but not all is graph";
    }
  }
  // The celllist or ordered_cell will be parsed as valuetuple of const graph in it,
  // So if has graph in list, try to replace the node with make tuple of graph value node.
  if (has_graph_in_list) {
    // change the vector of graph to be make_list of graph value node
    std::vector<AnfNodePtr> list_vec;
    auto make_list_op = NewValueNode(prim::kPrimMakeTuple);
    list_vec.emplace_back(make_list_op);
    (void)std::transform(std::begin(value_vec), std::end(value_vec), std::back_inserter(list_vec),
                         [](const ValuePtr& value) { return NewValueNode(value); });
    FuncGraphPtr cnode_graph = nullptr;
    auto users = manager->node_users()[node];
    for (auto& use : users) {
      auto use_node = use.first;
      MS_EXCEPTION_IF_NULL(use_node);
      if (use_node->isa<CNode>()) {
        cnode_graph = use_node->func_graph();
      }
    }

    if (cnode_graph) {
      CNodePtr list_app = cnode_graph->NewCNode(list_vec);
      // replace the ret ptr to be make_list of graph value node
      *transformed = list_app;
    } else {
      MS_LOG(EXCEPTION) << "Can not find apply for node use when replacing node of vector of graph";
    }
  }

  return true;
}
}  // namespace

AnfNodePtr ResolveSymbol(const FuncGraphManagerPtr& manager, const NameSpacePtr& name_space, const SymbolPtr& symbol,
                         const AnfNodePtr& node) {
  if (node->func_graph() == nullptr || manager == nullptr) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " graph or manager is nullptr";
  }
  SymbolResolver symbol_resolver(name_space, symbol, node);
  if (!symbol_resolver.Resolve()) {
    MS_LOG(EXCEPTION) << "Parse Resolve node failed NodeInfo: " << trace::GetDebugInfo(node->debug_info());
  }

  py::object obj = symbol_resolver.result();
  ScopeGuard scope_guard(node->scope());
  AnfNodePtr resolved_node = nullptr;
  TraceManager::DebugTrace(std::make_shared<TraceResolve>(node->debug_info()));
  bool success = ResolveObjectToNode(node->func_graph(), obj, &resolved_node);
  if (!success) {
    MS_LOG(EXCEPTION) << "Parse Resolve covert failed NodeInfo: " << trace::GetDebugInfo(node->debug_info());
  }
  if (IsValueNode<FuncGraph>(resolved_node)) {
    auto new_fg = GetValueNode<FuncGraphPtr>(resolved_node);
    manager->AddFuncGraph(new_fg);
  }

  // if the constant node is constant of vector of graph ,add graph to manager
  if (IsValueNode<ValueTuple>(resolved_node) || IsValueNode<ValueList>(resolved_node)) {
    (void)TransformVectorGraphValueNode(manager, node, resolved_node->cast<ValueNodePtr>(), &resolved_node);
  }

  TraceManager::EndTrace();
  return resolved_node;
}

namespace {
opt::OptPassGroupMap GetOptResolvePasses(const opt::irpass::ResolveIRPassLib& irpass) {
  opt::OptPassGroupMap map({
    {"resolve",
     {
       // for resolve and getattr primitive;
       irpass.resolver_resolve_,
       irpass.resolver_getattr_,
     }},
  });
  return map;
}
}  // namespace

bool ResolveFuncGraph(const FuncGraphPtr& func_graph, const pipeline::ResourceBasePtr& res, bool use_profile) {
  if (func_graph == nullptr || res == nullptr) {
    MS_LOG(ERROR) << "func_graph or resource is null";
    return false;
  }
  opt::irpass::ResolveIRPassLib irpass;
  opt::OptimizerPtr opt_resolve = opt::Optimizer::MakeOptimizer("opt_resolve", res, GetOptResolvePasses(irpass));

  (void)parse::python_adapter::set_python_scoped();

  abstract::AbstractBasePtrList args_spec;
  MS_EXCEPTION_IF_NULL(opt_resolve);
  (void)opt_resolve->step(func_graph, args_spec, use_profile);
  return true;
}

bool ResolveAll(const FuncGraphManagerPtr& manager) {
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
  for (auto& fg : roots) {
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
