/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include <utility>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "ir/param_info.h"
#include "ir/value.h"
#include "ir/map_tensor.h"
#include "pipeline/jit/fallback.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/parse/parse.h"
#include "include/common/utils/python_adapter.h"
#include "utils/any.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/symbol_resolver.h"
#include "include/common/debug/anf_dump_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parse {
namespace {
std::string ReplaceSpecialChar(const std::string &str) {
  std::ostringstream oss;
  for (size_t i = 0; i < str.size(); i++) {
    if (str[i] == '<') {
      // ⎡: \u23A1
      oss << "\u23A1";
    } else if (str[i] == '>') {
      // ⎦: \u23A6
      oss << "\u23A6";
    } else {
      oss << str[i];
    }
  }
  return oss.str();
}

struct AnfDumpHandlerRegister {
  AnfDumpHandlerRegister() {
    AnfDumpHandler::SetValueNodeStrHandler([](const std::shared_ptr<ValueNode> &node) -> std::string {
      if (node == nullptr) {
        return "";
      }
      if (IsValueNode<MetaFuncGraph>(node)) {
        return node->value()->cast<MetaFuncGraphPtr>()->name();
      } else if (IsValueNode<parse::NameSpace>(node)) {
        return node->value()->cast<parse::NameSpacePtr>()->name();
      } else if (IsValueNode<parse::Symbol>(node)) {
        return ReplaceSpecialChar(node->value()->cast<parse::SymbolPtr>()->name());
      }
      return "";
    });
  }
} callback_register;
}  // namespace

abstract::AbstractBasePtr MsClassObject::ToAbstract() {
  py::gil_scoped_acquire acquire;
  auto abs_scalar =
    std::make_shared<abstract::AbstractScalar>(shared_from_base<MsClassObject>(), std::make_shared<MsClassType>());
  AbstractBasePtrList args_spec_list = {abs_scalar};
  abstract::PrimitiveAbstractClosurePtr func_ptr = nullptr;
  bool is_class_type = parse::data_converter::IsClassType(obj());
  if (is_class_type) {
    // Class type as func, such as Net(x, y)
    func_ptr = std::make_shared<abstract::PrimitiveAbstractClosure>(prim::kPrimCreateInstance);
  } else {
    // Class instance as func, such as net(x, y)
    func_ptr = std::make_shared<abstract::PrimitiveAbstractClosure>(prim::kPrimCallInstance);
  }
  auto ret_val = std::make_shared<abstract::PartialAbstractClosure>(func_ptr, args_spec_list);
  ret_val->set_value_desc(ToString());
  return ret_val;
}

static inline bool IsSupportedCreateInstanceType(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  auto res = python_adapter::CallPyModFn(mod, PYTHON_MOD_IS_SUPPORTED_CREATE_INSTANCE_TYPE, obj);
  if (!py::isinstance<py::bool_>(res)) {
    MS_LOG(ERROR) << "Expect a bool type, but got " << py::str(res);
    return false;
  }
  return res.cast<bool>();
}

abstract::AbstractBasePtr ClassType::ToAbstract() {
  py::gil_scoped_acquire acquire;
  auto abs_scalar =
    std::make_shared<abstract::AbstractScalar>(shared_from_base<ClassType>(), std::make_shared<TypeType>());

  // The fallback feature is enabled in default.
  // Not support change the flag during the process is alive.
  static const auto use_fallback = (common::GetEnv("MS_DEV_ENABLE_FALLBACK") != "0");
  if (use_fallback && !IsSupportedCreateInstanceType(obj())) {
    return abs_scalar;
  }
  AbstractBasePtrList args_spec_list = {abs_scalar};

  auto func_ptr = std::make_shared<abstract::PrimitiveAbstractClosure>(prim::kPrimCreateInstance);
  auto ret_val = std::make_shared<abstract::PartialAbstractClosure>(func_ptr, args_spec_list);
  ret_val->set_value_desc(ToString());
  return ret_val;
}

namespace {
std::string GetPyObjId(const py::object &obj) {
  py::object out = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  if (py::isinstance<py::none>(out)) {
    MS_LOG(EXCEPTION) << "Get pyobj failed";
  }
  return out.cast<std::string>();
}
using tensor::MapTensorPtr;
// Get parameter value from a python parameter object.
// If it is a map parameter, return the map tensor value in it,
// otherwise, return parameter itself as a meta tensor value.
ValuePtr GetParameterValue(const py::object &param_obj) {
  constexpr char attr_map_tensor[] = "_map_tensor";
  constexpr char attr_param_info[] = "param_info";
  if (py::hasattr(param_obj, attr_map_tensor)) {
    auto map_tensor = py::cast<MapTensorPtr>(python_adapter::GetPyObjAttr(param_obj, attr_map_tensor));
    MS_EXCEPTION_IF_NULL(map_tensor);
    auto param_info = py::cast<ParamInfoPtr>(python_adapter::GetPyObjAttr(param_obj, attr_param_info));
    MS_EXCEPTION_IF_NULL(param_info);
    map_tensor->set_param_info(param_info);
    return map_tensor;
  }
  return py::cast<tensor::MetaTensorPtr>(param_obj);
}

// If any mixed precision flag add a cast node after the parameter node.
// argument obj should be python Parameter object
// it will be converted to Parameter node here
AnfNodePtr ResolveParameterObj(const FuncGraphPtr &func_graph, const py::object &obj) {
  MS_EXCEPTION_IF_NULL(func_graph);

  // Parameter object should not be none
  if (py::isinstance<py::none>(obj)) {
    MS_LOG(EXCEPTION) << "Resolve class Parameter error because obj is null.";
  }

  if (!py::hasattr(obj, "name")) {
    MS_LOG(EXCEPTION) << "Resolve class Parameter error: cannot find name attr for obj";
  }

  // Get the parameter name from parameter object
  auto name_attr = python_adapter::GetPyObjAttr(obj, "name");
  if (py::isinstance<py::none>(name_attr)) {
    MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
  }
  auto obj_id = GetPyObjId(obj);
  static std::vector<std::string> param_obj_ids;
  auto param_name = py::cast<std::string>(name_attr);
  auto top_func_graph = Parser::GetTopFuncGraph();
  // If the parameter node has been created , return it.
  AnfNodePtr para_node = nullptr;
  for (auto const &param : top_func_graph->parameters()) {
    auto param_node = dyn_cast<Parameter>(param);
    if (param_node != nullptr && param_node->name() == param_name) {
      if (param_node->is_top_graph_param()) {
        // If the name of the input of construct is same as the parameters,
        // add suffix to the name of the input of construct.
        string suffix_name = param_node->name() + "_$";
        param_node->set_name(suffix_name);
        param_node->debug_info()->set_name(suffix_name);
        MS_LOG(DEBUG) << "Add suffix to the name of the input of construct " << func_graph->ToString()
                      << ", input: " << param_node->DebugString();
      } else {
        // Exist two parameter object which name is the same.
        if (std::find(param_obj_ids.begin(), param_obj_ids.end(), obj_id) == param_obj_ids.end()) {
          MS_LOG(EXCEPTION) << "The parameter " << param_node->DebugString() << " , its name '" << param_name
                            << "' already exists. Please set a unique name for the parameter.";
        }
        para_node = param;
        MS_LOG(DEBUG) << "Found existing parameter for " << func_graph->ToString()
                      << ", param: " << para_node->DebugString() << ", top_func_graph: " << top_func_graph->ToString();
        break;
      }
    }
  }
  if (para_node == nullptr) {
    auto value = GetParameterValue(obj);
    para_node = top_func_graph->AddFvParameter(param_name, value);
    (void)param_obj_ids.emplace_back(obj_id);
    MS_LOG(DEBUG) << "Created a new weight parameter for " << func_graph->ToString()
                  << ", param: " << para_node->DebugString() << ", top_func_graph: " << top_func_graph->ToString();
  }
  func_graph->add_parameter_obj_node(para_node);
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
  auto resolved_graph_count = resolved_graph->fv_param_count();
  std::vector<ParameterPtr> drop_node_list;
  for (auto const &param : resolved_graph->parameters()) {
    auto param_ptr = dyn_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->has_default()) {
      param_ptr->set_func_graph(top_graph);
      func_graph->add_parameter_obj_node(param_ptr);
      // Update top_graph
      top_graph->add_parameter(param_ptr);
      size_t fv_param_count = top_graph->fv_param_count();
      top_graph->set_fv_param_count(++fv_param_count);
      (void)drop_node_list.emplace_back(param_ptr);
      resolved_graph->set_fv_param_count(--resolved_graph_count);
    } else {
      input_params.push_back(param_ptr);
    }
  }
  for (const auto &param_ptr : drop_node_list) {
    resolved_graph->DropNode(param_ptr);
  }
  resolved_graph->set_parameters(input_params);
  BroadenCNodeAbstract(resolved_graph);
}

bool HasConstArgAttr(const py::object &obj) {
  constexpr char const_arg_attr[] = "const_arg";
  return py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr));
}

bool HasMutableAttr(const py::object &obj) {
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

bool HasVariableLenAttr(const py::object &obj) {
  constexpr char variable_len_attr[] = "__ms_dynamic_len__";
  return py::hasattr(obj, variable_len_attr) && py::cast<bool>(py::getattr(obj, variable_len_attr));
}

AnfNodePtr ConvertInterpretedObjForResolve(const AnfNodePtr &origin_node, const ValuePtr &convert_result,
                                           const FuncGraphPtr &func_graph) {
  if (convert_result->isa<InterpretedObject>() && !origin_node->has_user_data("__py_interpret_local_value_flag__")) {
    constexpr auto recursive_level = 2;
    MS_LOG(DEBUG) << "Convert InterpretedObj for resolve, node: " << origin_node->DebugString(recursive_level);
    return ConvertInterpretedObjectToPyExecute(func_graph, convert_result, origin_node);
  }
  return nullptr;
}

AnfNodePtr ConvertObjectToNode(const AnfNodePtr &origin_node, const py::object &obj, const FuncGraphPtr &func_graph) {
  // When the cell is set recomputed, it should not use old scope from cache.
  MS_EXCEPTION_IF_NULL(origin_node);
  auto scope = origin_node->scope();
  bool has_recompute_scope = (scope == nullptr) ? false : scope->name().find(kAttrRecompute) == 0;
  ValuePtr convert_result = nullptr;
  bool converted =
    ConvertData(obj, &convert_result, python_adapter::UseSignatureInResolve(), nullptr, has_recompute_scope);
  if (!converted) {
    MS_LOG(ERROR) << "Convert data failed";
    return nullptr;
  }
  // For the bprop which is loaded from mindir file, the sub function should be resolved after loading.
  if (common::GetEnv("MS_DEV_EXPORT_BPROP_MINDIR") == "1" && IsPrimitiveCNode(origin_node, prim::kPrimResolve)) {
    auto name_space = GetValueNode<NameSpacePtr>(origin_node->cast<CNodePtr>()->input(1));
    MS_EXCEPTION_IF_NULL(name_space);
    auto obj_type = data_converter::GetObjType(obj);
    if (obj_type == RESOLVE_TYPE_FUNCTION && convert_result->isa<FuncGraph>() &&
        name_space->module() != RESOLVE_NAMESPACE_NAME_COMMON_OPS) {
      return origin_node;
    }
  }

  bool interpret_without_internal =
    (IsPrimitiveCNode(origin_node, prim::kPrimPyInterpret) && !origin_node->interpret_internal_type()) ||
    origin_node->interpret();
  static const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
  MS_EXCEPTION_IF_NULL(convert_result);
  if (support_fallback_runtime) {
    AnfNodePtr interpreted_output = ConvertInterpretedObjForResolve(origin_node, convert_result, func_graph);
    if (interpreted_output != nullptr) {
      return interpreted_output;
    }
  } else if (!interpret_without_internal && convert_result->isa<InterpretedObject>()) {
    auto type_str = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_PARSE_GET_TYPE, obj);
    MS_EXCEPTION(TypeError) << "Do not support to convert " << py::str(type_str) << " object into graph node."
                            << ".\nFor more details, please refer to "
                            << "https://mindspore.cn/docs/zh-CN/master/search.html?q=Do+not+support+to+convert+object"
                            << "+into+graph+node&check_keywords=yes&area=default\n";
  }

  if (convert_result->isa<FuncGraph>() && has_recompute_scope) {
    UpdateDebugInfo(convert_result->cast<FuncGraphPtr>(), origin_node->scope(), origin_node->debug_info());
  }
  ConvertLoadedGraph(func_graph, convert_result);
  AnfNodePtr output = NewValueNode(convert_result);
  if (convert_result->isa<tensor::Tensor>()) {
    output = GetMixedPrecisionCastHelp(func_graph, output);
    if (HasConstArgAttr(obj)) {
      MS_LOG(WARNING) << "The tensor " << convert_result->ToString()
                      << " which is not used for network input argument should not be set const.";
    }
  }
  if (HasMutableAttr(obj)) {
    auto dynamic_len = HasVariableLenAttr(obj);
    output = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMutable), output, NewValueNode(dynamic_len)});
  }
  return output;
}

bool ResolveObjectToNode(const AnfNodePtr &origin_node, const py::object &obj, AnfNodePtr *const node) {
  MS_EXCEPTION_IF_NULL(origin_node);
  auto func_graph = origin_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr output = nullptr;
  if (py::hasattr(obj, "__parameter__") && py::isinstance<tensor::MetaTensor>(obj)) {
    auto param = ResolveParameterObj(func_graph, obj);
    if (param == nullptr) {
      MS_LOG(ERROR) << "Resolve parameter object failed, got nullptr";
      return false;
    }
    MS_LOG(DEBUG) << "Add param graph:" << func_graph->ToString() << ", " << param->DebugString();
    output = param;
    *node = output;
    return true;
  } else if (py::hasattr(obj, "__parameter_tuple__")) {
    auto tuple = obj.cast<py::tuple>();
    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < tuple.size(); ++i) {
      AnfNodePtr out = nullptr;
      bool success = ResolveObjectToNode(origin_node, tuple[i], &out);
      if (!success) {
        MS_LOG(ERROR) << "Resolve object to node failed";
        return false;
      }
      args.push_back(out);
    }
    // The ParameterTuple will not be added in order list,
    // since we don't want to deal with its RefTensor elements during auto_monad procedure.
    output = NewCNode(std::move(args), func_graph);
    *node = output;
    return true;
  } else if ((py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) && py::len(obj) != 0) {
    auto tuple = obj.cast<py::tuple>();
    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));
    bool all_parameter_sequence = true;
    for (size_t i = 0; i < tuple.size(); ++i) {
      if (!py::hasattr(tuple[i], "__parameter__") || !py::isinstance<tensor::MetaTensor>(tuple[i])) {
        all_parameter_sequence = false;
        break;
      }
      AnfNodePtr out = nullptr;
      bool success = ResolveObjectToNode(origin_node, tuple[i], &out);
      if (!success) {
        MS_LOG(ERROR) << "Resolve object to node failed";
        return false;
      }
      args.push_back(out);
    }
    if (all_parameter_sequence) {
      // The Parameter tuple/list will not be added in order list,
      // since we don't want to deal with its RefTensor elements during auto_monad procedure.
      output = NewCNode(std::move(args), func_graph);
      *node = output;
      return true;
    }
  }
  output = ConvertObjectToNode(origin_node, obj, func_graph);
  if (output == nullptr) {
    return false;
  }
  *node = output;
  return true;
}

bool TransformVectorFuncValueNode(const FuncGraphManagerPtr &manager, const FuncGraphPtr &func_graph,
                                  const ValuePtr &value, AnfNodePtr *const transformed) {
  MS_EXCEPTION_IF_NULL(value);
  const auto &value_vec = GetValue<ValuePtrList>(value);
  if (value_vec.empty()) {
    return false;
  }
  std::vector<AnfNodePtr> nodes;
  (void)nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  bool is_all_func = true;
  for (auto &elem : value_vec) {
    MS_EXCEPTION_IF_NULL(elem);
    AnfNodePtr node = nullptr;
    if (elem->isa<ValueTuple>() || elem->isa<ValueList>()) {
      is_all_func = is_all_func && TransformVectorFuncValueNode(manager, func_graph, elem, &node);
    } else if (elem->isa<FuncGraph>()) {
      FuncGraphPtr new_fg = elem->cast<FuncGraphPtr>();
      manager->AddFuncGraph(new_fg);
      node = NewValueNode(new_fg);
    } else if (elem->isa<Primitive>()) {
      node = NewValueNode(elem);
    } else {
      is_all_func = false;
    }
    (void)nodes.emplace_back(node);
  }
  if (is_all_func) {
    // (1) The celllist or ordered_cell will be parsed as valuetuple of const graph in it,
    // So if has graph in list, try to replace the node with make tuple of graph value node.
    // We do this because the graph manager won't investigate the graph inside valuetuple,
    // change the vector of graph to be make_tuple of graph value node.
    // (2) the primitive valuetuple or valuelist may encounter to abstract error, make it all
    // independent nodes.
    *transformed = func_graph->NewCNode(std::move(nodes));
  }
  return is_all_func;
}

// Resolve the python obj, and if the resovled node is valuenode with graphs, add the graphs to manager.
AnfNodePtr ResolveObjectAndAddToManager(const FuncGraphManagerPtr &manager, const py::object &obj,
                                        const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ScopeGuard scope_guard(node->scope());
  AnfNodePtr resolved_node = nullptr;
  bool success = ResolveObjectToNode(node, obj, &resolved_node);
  if (!success) {
    MS_LOG(EXCEPTION) << "Parse Resolve covert failed.";
  }
  if (IsValueNode<FuncGraph>(resolved_node)) {
    auto new_fg = GetValueNode<FuncGraphPtr>(resolved_node);
    manager->AddFuncGraph(new_fg);
  }

  // If the constant node is constant of vector of graph, add graph to manager.
  if (IsValueNode<ValueTuple>(resolved_node) || IsValueNode<ValueList>(resolved_node)) {
    auto value = resolved_node->cast<ValueNodePtr>()->value();
    (void)TransformVectorFuncValueNode(manager, node->func_graph(), value, &resolved_node);
  }
  return resolved_node;
}
}  // namespace

std::pair<NameSpacePtr, SymbolPtr> GetNamespaceAndSymbol(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsPrimitiveCNode(node, prim::kPrimResolve)) {
    auto resolve_cnode = node->cast<CNodePtr>();
    constexpr size_t namespace_index = 1;
    auto namespace_node = resolve_cnode->input(namespace_index);
    constexpr size_t symbol_index = 2;
    auto symbol_node = resolve_cnode->input(symbol_index);
    if (!IsValueNode<NameSpace>(namespace_node) || !IsValueNode<Symbol>(symbol_node)) {
      MS_LOG(EXCEPTION) << "Unexpected type, namespace: " << namespace_node->ToString()
                        << ", symbol: " << symbol_node->ToString();
    }
    // Deal with the case of GetAttr from a class member,
    // and avoid the case of GetAttr from self (the result of ParseSuper).
    auto name_space = GetValueNode<NameSpacePtr>(namespace_node);
    auto symbol = GetValueNode<SymbolPtr>(symbol_node);
    return {name_space, symbol};
  }
  constexpr auto recursive_level = 2;
  MS_LOG(EXCEPTION) << "It's not prim::Resolve CNode, node: " << node->DebugString(recursive_level);
}

py::object GetSymbolObject(const NameSpacePtr &name_space, const SymbolPtr &symbol, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " graph is nullptr.";
  }
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  auto &obj = name_space->namespace_obj();
  if (py::isinstance<py::none>(obj)) {
    MS_EXCEPTION(NameError) << "The name \'" << symbol << "\' is not defined.";
  }
  return python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_FUNCTION, obj, common::SafeCStr(symbol->symbol()));
}

AnfNodePtr ResolveSymbol(const FuncGraphManagerPtr &manager, const NameSpacePtr &name_space, const SymbolPtr &symbol,
                         const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Manager is nullptr.";
  }
  TraceGuard trace_guard(std::make_shared<TraceResolve>(node->debug_info()));
  auto obj = GetSymbolObject(name_space, symbol, node);
  AnfNodePtr resolved_node = ResolveObjectAndAddToManager(manager, obj, node);
  return resolved_node;
}

// Resolve Cell GetAttr operation.
AnfNodePtr ResolveCellWithAttr(const FuncGraphManagerPtr &manager, const py::object &obj,
                               const AnfNodePtr &resolve_node, const AnfNodePtr &attr,
                               const AnfNodePtr &get_attr_node) {
  MS_EXCEPTION_IF_NULL(resolve_node);
  MS_EXCEPTION_IF_NULL(attr);
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Manager is nullptr.";
  }
  MS_LOG(DEBUG) << "obj: " << py::str(obj) << ", attr: " << attr->ToString();
  TraceGuard trace_guard(std::make_shared<TraceResolve>(get_attr_node->debug_info()));
  if (!data_converter::IsCellInstance(obj)) {
    AnfNodePtr resolved_node = ResolveObjectAndAddToManager(manager, obj, resolve_node);
    AnfNodePtrList inputs = {NewValueNode(prim::kPrimGetAttr), resolved_node, attr};
    MS_EXCEPTION_IF_NULL(get_attr_node->func_graph());
    AnfNodePtr res_node = get_attr_node->func_graph()->NewCNodeInOrder(std::move(inputs));
    return res_node;
  }

  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object namespace_obj = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, obj);
  auto new_namespace = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_obj);
  auto attr_string = GetValuePtr<StringImm>(attr);
  MS_EXCEPTION_IF_NULL(attr_string);
  const std::string &attr_as_string = attr_string->value();
  auto new_symbol = std::make_shared<Symbol>(attr_as_string);
  MS_LOG(DEBUG) << "name_space: " << new_namespace->ToString() << ", symbol: " << new_symbol->ToString();

  AnfNodePtrList inputs = {NewValueNode(prim::kPrimResolve), NewValueNode(new_namespace), NewValueNode(new_symbol)};
  MS_EXCEPTION_IF_NULL(get_attr_node->func_graph());
  AnfNodePtr resolved_node = get_attr_node->func_graph()->NewCNodeInOrder(std::move(inputs));
  return resolved_node;
}

AnfNodePtr ResolveSequenceWithAttr(const FuncGraphManagerPtr &manager, const py::object &obj,
                                   const AnfNodePtr &resolve_node, const AnfNodePtr &attr,
                                   const CNodePtr &get_attr_node) {
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  auto sequence = obj.cast<py::sequence>();
  // Incorporate if all elements of the sequence are Cell instances or MsClass instances.
  size_t count_cell = 0;
  size_t count_msclass = 0;
  size_t sequence_size = sequence.size();
  for (size_t i = 0; i < sequence_size; ++i) {
    if (data_converter::IsCellInstance(sequence[i])) {
      ++count_cell;
    } else if (data_converter::IsMsClassInstance(sequence[i])) {
      ++count_msclass;
    }
  }
  if (count_cell == sequence_size) {
    // Resolve Cell instances.
    for (size_t i = 0; i < sequence_size; ++i) {
      auto res = ResolveCellWithAttr(manager, sequence[i], resolve_node, attr, get_attr_node);
      (void)inputs.emplace_back(res);
    }
  } else if (count_msclass == sequence_size) {
    // Resolve MsClass instances.
    for (size_t i = 0; i < sequence_size; ++i) {
      auto attr_str_ptr = GetValuePtr<StringImm>(attr);
      MS_EXCEPTION_IF_NULL(attr_str_ptr);
      const auto &attr_str = attr_str_ptr->value();
      auto res = ResolveMsClassWithAttr(manager, sequence[i], attr_str, get_attr_node);
      if (res == nullptr || IsValueNode<None>(res)) {
        MS_EXCEPTION(AttributeError) << py::str(sequence[i]) << " object has no attribute: " << attr_str << ".";
      }
      (void)inputs.emplace_back(res);
    }
  } else {
    return nullptr;
  }

  constexpr auto prim_index = 0;
  constexpr auto index_index = 2;
  MS_EXCEPTION_IF_NULL(get_attr_node);
  auto fg = get_attr_node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto make_tuple_node = fg->NewCNodeInOrder(inputs);
  return fg->NewCNodeInOrder({get_attr_node->input(prim_index), make_tuple_node, get_attr_node->input(index_index)});
}

AnfNodePtr ResolveSymbolWithAttr(const FuncGraphManagerPtr &manager, const AnfNodePtr &object_node,
                                 const AnfNodePtr &attr_node, const AnfNodePtr &get_attr_node) {
  // {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
  auto [name_space, symbol] = GetNamespaceAndSymbol(object_node);
  MS_EXCEPTION_IF_NULL(name_space);
  MS_EXCEPTION_IF_NULL(symbol);
  auto module_name = name_space->module();
  constexpr std::string_view parse_super_name = "namespace";
  if (module_name.find(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER) != std::string::npos &&
      symbol->symbol() != parse_super_name) {
    auto symbol_obj = GetSymbolObject(name_space, symbol, get_attr_node);
    return ResolveCellWithAttr(manager, symbol_obj, object_node, attr_node, get_attr_node);
  }
  return nullptr;
}

// Get python object with index from a list or the whole list if the index is not fixed.
py::object GetObjectFromSequence(const NameSpacePtr &name_space, const SymbolPtr &symbol, const AnfNodePtr &node,
                                 const AnfNodePtr &index_node) {
  MS_EXCEPTION_IF_NULL(node);
  TraceGuard trace_guard(std::make_shared<TraceResolve>(node->debug_info()));
  py::object obj = GetSymbolObject(name_space, symbol, node);
  // If obj is nn.CellList, convert it to sequence.
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  bool is_celllist = py::cast<bool>(python_adapter::CallPyModFn(mod, PYTHON_MOD_IS_CELL_LIST, obj));
  if (is_celllist) {
    obj = python_adapter::CallPyModFn(mod, PYTHON_MOD_CONVERT_CELL_LIST_TO_SEQUENCE, obj);
  }
  if (!py::isinstance<py::list>(obj) && !py::isinstance<py::tuple>(obj)) {
    return py::none();
  }

  MS_LOG(DEBUG) << "obj: " << py::str(obj) << ", index_node: " << index_node->ToString();
  auto imm_value = GetValueNode<Int64ImmPtr>(index_node);
  if (imm_value == nullptr) {
    MS_LOG(DEBUG) << "The index is not a value node, so we return the whole list, node: " << node->DebugString()
                  << ", index_node: " << index_node->DebugString();
    // Index is not fixed, return the whole list.
    return obj;
  }
  // It index is a value node, get the item of index directly.
  py::object item_obj =
    python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_ITEM_FROM_SEQUENCE, obj, py::int_(imm_value->value()));
  return item_obj;
}

bool IsResolveNodeWithGetItem(const AnfNodePtr &node) {
  // Check if the node matches: {prim::kPrim::Resolve, ..., 'getitem'}.
  if (IsPrimitiveCNode(node, prim::kPrimResolve)) {
    constexpr size_t symbol_index = 2;
    constexpr auto getitem_symbol = "getitem";
    auto cnode = node->cast<CNodePtr>();
    auto symbol = GetValueNode<SymbolPtr>(cnode->input(symbol_index));
    return symbol->symbol() == getitem_symbol;
  }
  return false;
}

bool IsGetItemCNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  constexpr size_t getitem_inputs_size = 3;
  if (cnode->size() != getitem_inputs_size) {
    return false;
  }
  constexpr auto prim_index = 0;
  return IsResolveNodeWithGetItem(cnode->input(prim_index));
}

AnfNodePtr ResolveGetItemWithAttr(const FuncGraphManagerPtr &manager, const AnfNodePtr &getitem_node,
                                  const AnfNodePtr &attr_node, const AnfNodePtr &node) {
  // {prim::kPrimGetAttr, {getitem, {prim::kPrimResolve, namespace, symbol}, index}, attr}
  // {prim::kPrimGetAttr, {getitem, {prim::kPrimGetAttr, ResolveNode, member}, index}, attr}
  constexpr auto data_index = 1;
  constexpr auto index_index = 2;
  auto getitem_cnode = getitem_node->cast<CNodePtr>();
  auto data_node = getitem_cnode->input(data_index);
  auto index_node = getitem_cnode->input(index_index);
  if (IsPrimitiveCNode(data_node, prim::kPrimResolve)) {
    auto [name_space, symbol] = GetNamespaceAndSymbol(data_node);
    auto obj = GetObjectFromSequence(name_space, symbol, data_node, index_node);
    if (py::isinstance<py::none>(obj)) {
      return nullptr;
    }
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
      return ResolveSequenceWithAttr(manager, obj, data_node, attr_node, getitem_cnode);
    }
    return ResolveCellWithAttr(manager, obj, data_node, attr_node, node);
  }
  if (IsPrimitiveCNode(data_node, prim::kPrimGetAttr)) {
    auto getattr_cnode = data_node->cast<CNodePtr>();
    auto resolve_node = getattr_cnode->input(data_index);
    auto member_node = getattr_cnode->input(index_index);
    if (IsPrimitiveCNode(resolve_node, prim::kPrimResolve)) {
      // Check if the result is a new resolve node.
      auto item_node = ResolveSymbolWithAttr(manager, resolve_node, member_node, node);
      if (IsPrimitiveCNode(item_node, prim::kPrimResolve)) {
        auto [name_space, symbol] = GetNamespaceAndSymbol(item_node);
        auto obj = GetObjectFromSequence(name_space, symbol, item_node, index_node);
        if (py::isinstance<py::none>(obj)) {
          return nullptr;
        }
        if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
          return ResolveSequenceWithAttr(manager, obj, item_node, attr_node, getitem_cnode);
        }
        return ResolveCellWithAttr(manager, obj, item_node, attr_node, node);
      }
    }
  }
  return nullptr;
}

AnfNodePtr ResolveMsClassWithAttr(const FuncGraphManagerPtr &manager, const py::object &cls_obj,
                                  const std::string &attr, const AnfNodePtr &get_attr_node) {
  // Get attribute or method from ms_class obj.
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(get_attr_node);
  MS_LOG(DEBUG) << "Resolve ms_class obj (" << py::str(cls_obj) << ") with attr " << attr << ".";
  TraceGuard trace_guard(std::make_shared<TraceResolve>(get_attr_node->debug_info()));

  constexpr size_t prefix_index = 0;
  std::vector<std::string> support_attr{"__enter__", "__exit__"};
  auto iter = find(support_attr.begin(), support_attr.end(), attr);
  if (!attr.empty() && attr[prefix_index] == '_' && iter == support_attr.end()) {
    MS_LOG(EXCEPTION) << attr << " is a private variable or magic method, which is not supported.";
  }
  if (!py::hasattr(cls_obj, common::SafeCStr(attr))) {
    return nullptr;
  }
  py::object attr_obj = py::getattr(cls_obj, common::SafeCStr(attr));
  AnfNodePtr res_node = ResolveObjectAndAddToManager(manager, attr_obj, get_attr_node);
  return res_node;
}

namespace {
opt::OptPassGroupMap GetOptResolvePasses(const opt::irpass::ResolveIRPassLib &irpass) {
  // For resolve and getattr primitive.
  opt::OptPassGroupMap map({
    {"resolve",
     {
       irpass.resolver_,
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

  (void)python_adapter::set_python_scoped();

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
  // Should not use pipeline::Resource as Resource::Clean will clean some
  // global variable such as ScopeManager, it will cause JExpandedGraphs::GetBprop
  // fail as valid scope has been cleaned.
  auto res = std::make_shared<pipeline::ResourceBase>();
  res->set_manager(manager);

  auto roots = manager->roots();
  for (const auto &fg : roots) {
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
