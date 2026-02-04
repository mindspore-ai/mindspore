/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "frontend/jit/ps/parse/resolve.h"

#include <utility>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "primitive/structure_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "ir/param_info.h"
#include "ir/value.h"
#include "ir/map_tensor.h"
#include "ir/func_graph_cloner.h"
#include "frontend/jit/ps/action.h"
#include "frontend/jit/ps/fallback.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "frontend/jit/ps/parse/parse.h"
#include "include/utils/python_adapter.h"
#include "include/utils/parallel_context.h"
#include "utils/any.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/symbol_resolver.h"
#include "include/utils/pipeline/fallback.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_dump_utils.h"
#include "utils/log_adapter.h"
#include "utils/trace_info.h"
#include "include/utils/tensor_py.h"
#include "include/utils/tensor_py_wrapper.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "include/utils/pynative/variable.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace parse {
static std::unordered_map<std::string, std::string> param_obj_ids;  // param_name : obj_id
void CleanParameterNameCache() {
  MS_LOG(DEBUG) << "Clean parameter name cache.";
  param_obj_ids.clear();
}
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

InterpretedObject::InterpretedObject(const py::object &obj) : PyObjectWrapper(obj) {
  std::stringstream buf;
  auto type_str = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_PARSE_GET_TYPE, obj);
  buf << "PythonObject(type: " << ToPyStr(type_str, "<unknown>") << ", value: " << ToPyStr(obj, "<unknown>") << ")";
  this->set_name(buf.str());
}

abstract::AbstractBasePtr MsClassObject::ToAbstract() {
  py::gil_scoped_acquire acquire;
  bool is_class_type = parse::data_converter::IsClassType(obj());
  if (is_class_type) {
    // Class type as func, such as Net(x, y)
    auto abs_class = std::make_shared<abstract::AbstractClass>(shared_from_base<MsClassObject>());
    AbstractBasePtrList args_abs_list = {abs_class};
    auto func = std::make_shared<abstract::PrimitiveAbstractClosure>(prim::kPrimCreateInstance);
    auto res_val = std::make_shared<abstract::PartialAbstractClosure>(func, args_abs_list);
    res_val->set_value_desc(ToString());
    return res_val;
  } else {
    // Class instance as func, such as net(x, y)
    return std::make_shared<abstract::AbstractClass>(shared_from_base<MsClassObject>());
  }
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

  if (!IsSupportedCreateInstanceType(obj())) {
    return abs_scalar;
  }
  AbstractBasePtrList args_abs_list = {abs_scalar};

  auto func = std::make_shared<abstract::PrimitiveAbstractClosure>(prim::kPrimCreateInstance);
  auto res_val = std::make_shared<abstract::PartialAbstractClosure>(func, args_abs_list);
  res_val->set_value_desc(ToString());
  return res_val;
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

  return tensor::ConvertToTensorPyWrapper(param_obj);
}

void ClearCNodeAbstract(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
  for (const auto &node : nodes) {
    if (node == nullptr || node->isa<Parameter>()) {
      continue;
    }
    auto primitive = GetCNodePrimitive(node);
    if (primitive != nullptr) {
      auto is_load = primitive->GetAttr("is_load");
      if (abstract::GetPrimEvaluator(primitive, nullptr) == nullptr && is_load != nullptr && GetValue<bool>(is_load)) {
        MS_LOG(INFO) << "The primitive is not defined in front end. Primitive: " << primitive->ToString();
        if (node->abstract() != nullptr) {
          node->set_abstract(node->abstract()->Broaden());
        }
        continue;
      }
    }
    auto prev_inferred = node->abstract();
    // Keep previous inferred value for ValueNode if the inferred value is not AbstractFunction.
    if (!node->isa<ValueNode>() || (prev_inferred != nullptr && prev_inferred->isa<abstract::AbstractFunction>())) {
      // Reset tuple/list abstract use flags.
      if (enable_eliminate_unused_element && prev_inferred != nullptr &&
          prev_inferred->isa<abstract::AbstractSequence>()) {
        SetSequenceNodeElementsUseFlags(node, nullptr);
      }
      node->set_abstract(nullptr);
      MS_LOG(DEBUG) << "Abstract of node " << node->DebugString() << " is set to nullptr";
    }
  }
}

namespace {
std::string GetPyObjId(const py::object &obj) {
  py::object out = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  if (py::isinstance<py::none>(out)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Get pyobj failed";
  }
  return out.cast<std::string>();
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
    MS_LOG(DEBUG) << "Convert InterpretedObj for resolve, node: "
                  << origin_node->DebugString(AnfNode::DebugStringLevel::kLevel2);
    auto interpreted_value = dyn_cast<InterpretedObject>(convert_result);
    const auto &key = interpreted_value->name();
    if (interpreted_value->has_converted()) {
      return fallback::ConvertPyObjectToPyExecute(func_graph, key, interpreted_value->obj(), origin_node, true);
    }
    return fallback::ConvertPyObjectToPyInterpret(func_graph, key, interpreted_value->obj(), origin_node, true);
  }
  return nullptr;
}

AnfNodePtr TransformFuncValueNode(const FuncGraphManagerPtr &manager, const FuncGraphPtr &func_graph,
                                  const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<FuncGraph>()) {
    auto fg = value->cast<FuncGraphPtr>();
    manager->AddFuncGraph(fg);
    return NewValueNode(fg);
  }
  if (value->isa<Primitive>()) {
    return NewValueNode(value);
  }
  // (1) The CellList or CellDict will be parsed as value_sequence or value_dict of const graph in it,
  // So if there is graph in list, try to replace the node with make_tuple or make_dict of graph value node.
  // We do this because the graph manager won't investigate the graph inside value_sequence or value_dict,
  // change the vector of graph to be make_tuple or make_dict of graph value node.
  // (2) the primitive value_tuple or value_sequence or value_dict may encounter to abstract error, make it all
  // independent nodes.
  if (value->isa<ValueSequence>()) {
    std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
    bool is_all_func = true;
    auto value_sequence = value->cast<ValueSequencePtr>();
    if (value_sequence->size() == 0) {
      return nullptr;
    }
    for (auto &elem : value_sequence->value()) {
      auto node = TransformFuncValueNode(manager, func_graph, elem);
      if (node == nullptr) {
        is_all_func = false;
      }
      (void)inputs.emplace_back(node);
    }
    if (is_all_func) {
      return func_graph->NewCNode(std::move(inputs));
    }
    return nullptr;
  }
  if (value->isa<ValueDictionary>()) {
    std::vector<AnfNodePtr> keys{NewValueNode(prim::kPrimMakeTuple)};
    std::vector<AnfNodePtr> values{NewValueNode(prim::kPrimMakeTuple)};
    bool is_all_func = true;
    for (auto &elem : value->cast<ValueDictionaryPtr>()->value()) {
      (void)keys.emplace_back(NewValueNode(elem.first));
      auto node = TransformFuncValueNode(manager, func_graph, elem.second);
      if (node == nullptr) {
        is_all_func = false;
      }
      (void)values.emplace_back(node);
    }
    if (is_all_func) {
      return func_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), func_graph->NewCNode(std::move(keys)),
                                   func_graph->NewCNode(std::move(values))});
    }
    return nullptr;
  }

  return nullptr;
}

AnfNodePtr HandleConstValueNodeInner(const AnfNodePtr &node, const FuncGraphPtr &func_graph,
                                     const FuncGraphManagerPtr &manager) {
  if (IsValueNode<ValueSequence>(node) || IsValueNode<ValueDictionary>(node)) {
    auto value = node->cast<ValueNodePtr>()->value();
    auto new_node = TransformFuncValueNode(manager, func_graph, value);
    return new_node;
  }
  return nullptr;
}

AnfNodePtr HandleConstValueNode(const AnfNodePtr &resolved_node, const FuncGraphPtr &func_graph,
                                const FuncGraphManagerPtr &manager) {
  // Handle const value node CellList/CellDict
  auto convert_node = HandleConstValueNodeInner(resolved_node, func_graph, manager);
  if (convert_node != nullptr) {
    return convert_node;
  }
  // Handle CNode such as {prim::kPrimMakeList/prim::kPrimMakeTuple, CellList/CellDict, elem2, ...}
  if (IsPrimitiveCNode(resolved_node, prim::kPrimMakeTuple) || IsPrimitiveCNode(resolved_node, prim::kPrimMakeList)) {
    auto cnode = resolved_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    for (size_t index = kIndex1; index < inputs.size(); ++index) {
      auto new_node = HandleConstValueNodeInner(inputs[index], func_graph, manager);
      if (new_node != nullptr) {
        cnode->set_input(index, new_node);
      }
    }
    // Handle CNode such as {prim::kPrimMakeDict, (key1, key2, ...), (value1, CellList/CellDict, ...)}
  } else if (IsPrimitiveCNode(resolved_node, prim::kPrimMakeDict)) {
    auto cnode = resolved_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    auto new_node = HandleConstValueNode(inputs[kIndex2], func_graph, manager);
    if (new_node != nullptr) {
      cnode->set_input(kIndex2, new_node);
    }
  }
  return nullptr;
}
}  // namespace

void Resolver::ConvertLoadedGraph(const FuncGraphPtr &func_graph, const ValuePtr &value) {
  if (!value->isa<FuncGraph>()) {
    return;
  }
  auto resolved_graph = value->cast<FuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(resolved_graph);
  if (!resolved_graph->has_attr("is_load")) {
    return;
  }
  std::vector<AnfNodePtr> input_params;
  auto resolved_graph_count = resolved_graph->fv_param_count();
  std::vector<ParameterPtr> drop_node_list;
  for (auto const &param : resolved_graph->parameters()) {
    auto param_ptr = dyn_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->has_default()) {
      param_ptr->set_func_graph(top_func_graph_);
      func_graph->add_parameter_obj_node(param_ptr);
      // Update top_graph
      top_func_graph_->add_parameter(param_ptr);
      size_t fv_param_count = top_func_graph_->fv_param_count();
      top_func_graph_->set_fv_param_count(++fv_param_count);
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
  ClearCNodeAbstract(resolved_graph);
}

AnfNodePtr Resolver::ConvertObjectToNode(const AnfNodePtr &origin_node, const py::object &obj,
                                         const FuncGraphPtr &func_graph, bool is_element_obj) {
  // When the cell is set recomputed, it should not use old scope from cache.
  MS_EXCEPTION_IF_NULL(origin_node);
  auto origin_cnode = dyn_cast<CNode>(origin_node);
  MS_EXCEPTION_IF_NULL(origin_cnode);
  bool is_resolve = IsPrimitiveCNode(origin_node, prim::kPrimResolve);
  auto scope = origin_node->scope();
  bool has_recompute_scope =
    (scope != nullptr && scope->name().compare(0, strlen(kAttrRecompute), kAttrRecompute) == 0);
  ValuePtr convert_result = nullptr;
  constexpr auto resolve_with_args_inputs_size = 4;
  MS_LOG(DEBUG) << "origin_cnode: " << origin_cnode->DebugString();
  if (is_resolve && origin_cnode->size() == resolve_with_args_inputs_size) {  // (resolve, namespace, symbol, arguments)
    constexpr auto args_input_pos = 3;
    auto args_node = origin_cnode->input(args_input_pos);
    auto args_value = GetValueNode<ValueTuplePtr>(args_node);
    MS_EXCEPTION_IF_NULL(args_value);
    parse::DataConverter data_converter(args_value->value(), python_adapter::UseSignatureInResolve());
    convert_result = data_converter.ConvertData(obj);
    if (convert_result == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert error with Python object: " << std::string(py::str(obj));
    }
  } else {  // (resolve/getattr, namespace, symbol, optional[getattr])
    bool converted =
      ConvertData(obj, &convert_result, python_adapter::UseSignatureInResolve(), nullptr, has_recompute_scope);
    if (!converted) {
      MS_LOG(ERROR) << "Convert data failed";
      return nullptr;
    }
  }

  // If obj is an element, do not convert InterpretedObj.
  if (!is_element_obj) {
    AnfNodePtr interpreted_output = ConvertInterpretedObjForResolve(origin_node, convert_result, func_graph);
    if (interpreted_output != nullptr) {
      return interpreted_output;
    }
  }

  if (convert_result->isa<FuncGraph>() && has_recompute_scope) {
    UpdateRecomputeScope(convert_result->cast<FuncGraphPtr>());
  }
  ConvertLoadedGraph(func_graph, convert_result);
  AnfNodePtr output = NewValueNode(convert_result);
  if (IsPrimitive(output, prim::kPrimTraceGraph)) {
    auto func_obj = std::make_shared<InterpretedObject>(obj);
    origin_cnode->add_input(NewValueNode(func_obj));
  }
  if (convert_result->isa<ValueDictionary>()) {
    output->set_user_data<py::object>("origin_object", std::make_shared<py::object>(obj));
  }
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

// Resolve the python obj, and if the resolved node is valuenode with graphs, add the graphs to manager.
AnfNodePtr Resolver::ResolveObjectAndAddToManager(const FuncGraphManagerPtr &manager, const py::object &obj,
                                                  const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ScopeGuard scope_guard(node->scope());
  AnfNodePtr resolved_node = nullptr;
  bool success = ResolveObjectToNode(node, obj, &resolved_node);
  if (!success) {
    MS_LOG(INTERNAL_EXCEPTION) << "Parse Resolve convert failed.";
  }
  if (IsValueNode<FuncGraph>(resolved_node)) {
    auto new_fg = GetValueNode<FuncGraphPtr>(resolved_node);
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    // If func_graph has amp strategy (not to_float) but its python obj does not,
    // it may be from cache and need to be cloned.
    if (new_fg->amp_strategy() != nullptr && new_fg->amp_strategy()->IsEnable() &&
        !py::hasattr(obj, FUNC_GRAPH_FLAG_AMP_STRATEGY) && new_fg->amp_strategy() != fg->amp_strategy()) {
      new_fg = BasicClone(new_fg);
      new_fg->set_amp_strategy(fg->amp_strategy());
      resolved_node = NewValueNode(new_fg);
    }
    // If it's the sub func graph resolved in a reserved func graph.
    if (fg->reserved()) {
      new_fg->set_reserved(true);
    }
    manager->AddFuncGraph(new_fg);
  }

  // If the constant node is constant of vector of graph, add graph to manager.
  auto new_node = HandleConstValueNode(resolved_node, node->func_graph(), manager);
  if (new_node != nullptr) {
    resolved_node = new_node;
  }
  fallback::SetPyObjectToNode(resolved_node, obj);
  return resolved_node;
}

bool IsParameterObject(const py::object &obj) {
  return obj.ptr() != nullptr && py::hasattr(obj, "__parameter__") && tensor::IsTensorPy(obj);
}

bool ContainsParameter(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    return false;
  }
  if (IsParameterObject(obj) || py::hasattr(obj, "__parameter_tuple__")) {
    return true;
  }
  if ((py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) && py::len(obj) != 0) {
    // NamedTuple
    if (py::hasattr(obj, "_fields")) {
      return false;
    }
    auto tuple = obj.cast<py::tuple>();
    for (size_t i = 0; i < tuple.size(); ++i) {
      if (ContainsParameter(tuple[i])) {
        return true;
      }
    }
  } else if (py::isinstance<py::dict>(obj)) {
    auto dict = obj.cast<py::dict>();
    for (auto item : dict) {
      auto item_value = py::cast<py::object>(item.second);
      if (ContainsParameter(item_value)) {
        return true;
      }
    }
  }
  return false;
}

bool Resolver::ResolveObjectToNode(const AnfNodePtr &origin_node, const py::object &obj, AnfNodePtr *const node,
                                   bool is_element_obj) {
  MS_EXCEPTION_IF_NULL(origin_node);
  auto func_graph = origin_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!ContainsParameter(obj)) {
    auto output = ConvertObjectToNode(origin_node, obj, func_graph, is_element_obj);
    if (output == nullptr) {
      return false;
    }
    *node = output;
    return true;
  }
  if (IsParameterObject(obj)) {
    auto param = ResolveParameterObj(func_graph, obj);
    if (param == nullptr) {
      MS_LOG(ERROR) << "Resolve parameter object failed, got nullptr";
      return false;
    }
    MS_LOG(DEBUG) << "Add param graph:" << func_graph->ToString() << ", " << param->DebugString();
    *node = param;
    return true;
  }
  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj) || py::hasattr(obj, "__parameter_tuple__")) {
    bool all_parameter_sequence = true;
    std::vector<AnfNodePtr> args;
    auto tuple = obj.cast<py::tuple>();
    for (size_t i = 0; i < tuple.size(); ++i) {
      if (!IsParameterObject(tuple[i])) {
        all_parameter_sequence = false;
      }
      AnfNodePtr out = nullptr;
      bool success = ResolveObjectToNode(origin_node, tuple[i], &out, true);
      if (!success) {
        MS_LOG(ERROR) << "Resolve object to node failed";
        return false;
      }
      args.push_back(out);
    }
    // Convert [param1, param2, ..., paramN] to tuple.
    bool need_convert_to_tuple = !is_element_obj && all_parameter_sequence && py::isinstance<py::list>(obj);
    if (py::isinstance<py::tuple>(obj) || py::hasattr(obj, "__parameter_tuple__") || need_convert_to_tuple) {
      (void)args.insert(args.begin(), NewValueNode(prim::kPrimMakeTuple));
    } else {
      (void)args.insert(args.begin(), NewValueNode(prim::kPrimMakeList));
    }
    // The ParameterTuple/tuple/list will not be added in order list,
    // since we don't want to deal with its RefTensor elements during auto_monad procedure.
    *node = NewCNode(std::move(args), func_graph);
    return true;
  }
  if (py::isinstance<py::dict>(obj)) {
    auto dict = obj.cast<py::dict>();
    std::vector<AnfNodePtr> keys_tuple{NewValueNode(prim::kPrimMakeTuple)};
    std::vector<AnfNodePtr> values_tuple{NewValueNode(prim::kPrimMakeTuple)};
    for (auto item : dict) {
      AnfNodePtr key = nullptr;
      AnfNodePtr value = nullptr;
      bool success = ResolveObjectToNode(origin_node, py::cast<py::object>(item.first), &key, true) &&
                     ResolveObjectToNode(origin_node, py::cast<py::object>(item.second), &value, true);
      if (!success) {
        MS_LOG(ERROR) << "Resolve object to node failed";
        return false;
      }
      (void)keys_tuple.emplace_back(key);
      (void)values_tuple.emplace_back(value);
    }
    *node = func_graph->NewCNode(
      {NewValueNode(prim::kPrimMakeDict), func_graph->NewCNode(keys_tuple), func_graph->NewCNode(values_tuple)});
    return true;
  }
  MS_EXCEPTION(TypeError) << "The Parameter in obj '" << py::str(obj) << "' with nested structure is not supported."
                          << "\nCurrently only single Parameter, ParameterTuple or Parameters in tuple/list/dict "
                             "are supported. Or do you want to use Tensor instead?";
}

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
  MS_LOG(INTERNAL_EXCEPTION) << "It's not prim::Resolve CNode, node: "
                             << node->DebugString(AnfNode::DebugStringLevel::kLevel2);
}

py::object GetSymbolObject(const NameSpacePtr &name_space, const SymbolPtr &symbol, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->func_graph() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Node " << node->DebugString() << " graph is nullptr.";
  }
  if (name_space->module() == RESOLVE_NAMESPACE_NAME_ENTRY) {
    return name_space->module_obj();
  } else if (name_space->module() == RESOLVE_NAMESPACE_NAME_CLASS_OBJECT) {
    MS_LOG(DEBUG) << "namespace: " << py::str(name_space->namespace_obj()) << ", symbol: " << symbol;
    return name_space->namespace_obj();
  }
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  auto &obj = name_space->namespace_obj();
  if (py::isinstance<py::none>(obj)) {
    MS_EXCEPTION(NameError) << "The name \'" << symbol << "\' is not defined.";
  }
  const auto &res =
    python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_FUNCTION, obj, common::SafeCStr(symbol->symbol()));
  MS_LOG(DEBUG) << "namespace: " << py::str(obj) << ", symbol: " << symbol << ", result: " << py::str(res);
  return res;
}

bool IsCellHookExist(const py::object &obj, const std::string &hook_dict_name) {
  const auto &hook_obj = py::getattr(obj, hook_dict_name.c_str(), py::none());
  if (!py::isinstance<py::dict>(hook_obj)) {
    return false;
  }

  const auto &dict = py::cast<py::dict>(hook_obj);
  return (dict.size() != 0);
}

bool IsCellHookExistAny(const py::object &obj, const std::vector<std::string> &hook_dict_names) {
  return std::any_of(hook_dict_names.begin(), hook_dict_names.end(),
                     [&obj](const auto &name) { return IsCellHookExist(obj, name); });
}

FuncGraphPtr ResolveCellHook(const py::object &obj, const std::string &hook_dict_name,
                             const std::string &hook_func_name) {
  if (!IsCellHookExist(obj, hook_dict_name)) {
    return nullptr;
  }

  const auto &value = py::getattr(obj, hook_func_name.c_str());
  auto hook_func = parse::ParsePythonCode(value);
  MS_EXCEPTION_IF_NULL(hook_func);
  return hook_func;
}

FuncGraphPtr ResolveSaveTensorHook(const py::object &obj, const std::string &hook_name) {
  auto hook_obj = py::getattr(obj, hook_name.c_str(), py::none());
  if (py::isinstance<py::none>(hook_obj)) {
    return nullptr;
  }

  if (!py::isinstance<py::function>(hook_obj)) {
    MS_EXCEPTION(TypeError) << "The attribute '" << hook_name << "' from '" << py::str(obj)
                            << "' should be a function.";
  }
  auto hook_func = parse::ParsePythonCode(hook_obj);
  MS_EXCEPTION_IF_NULL(hook_func);
  return hook_func;
}

namespace {
FuncGraphPtr MakeDummyForwardPreHook() {
  auto forward_pre_hook = std::make_shared<FuncGraph>();
  std::ostringstream ss;
  ss << "dummy_jit_forward_pre_hook_";
  if (forward_pre_hook->debug_info() != nullptr) {
    forward_pre_hook->debug_info()->set_name(ss.str());
  }
  auto input = forward_pre_hook->add_parameter();
  forward_pre_hook->set_output(input);
  return forward_pre_hook;
}

AnfNodePtr ApplyForwardPreHook(const FuncGraphPtr &func_graph, const py::object &obj,
                               const FuncGraphManagerPtr &manager, const AnfNodePtr &tupled_input) {
  MS_EXCEPTION_IF_NULL(func_graph);

  FuncGraphPtr hook_func = ResolveCellHook(obj, CELL_FORWARD_PRE_HOOK, CELL_JIT_FORWARD_PRE_HOOK);
  if (hook_func == nullptr) {
    if (!IsCellHookExist(obj, CELL_BACKWARD_HOOK)) {
      return tupled_input;
    } else {
      hook_func = MakeDummyForwardPreHook();
    }
  }

  hook_func->set_manager(manager);
  hook_func->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  hook_func->set_flag(FUNC_GRAPH_FLAG_FORWARD_PRE_HOOK, true);

  auto pre_hooked_input = func_graph->NewCNodeInOrder({NewValueNode(hook_func), tupled_input});
  func_graph->AddNode(pre_hooked_input);

  for (size_t i = 0; i < func_graph->parameters().size(); ++i) {
    auto users_sub = manager->node_users()[func_graph->parameters()[i]];
    auto getitem_cnode = func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), pre_hooked_input, NewValueNode(SizeToLong(i))});
    func_graph->AddNode(getitem_cnode);
    for (const auto &user_sub : users_sub) {
      auto user_subc = user_sub.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(user_subc);
      user_subc->set_input(user_sub.second, getitem_cnode);
    }
  }

  return pre_hooked_input;
}

void ApplyForwardHook(const FuncGraphPtr &func_graph, const py::object &obj, const FuncGraphManagerPtr &manager,
                      const AnfNodePtr &hook_input_node) {
  if (func_graph == nullptr) {
    return;
  }

  auto hook_func = ResolveCellHook(obj, CELL_FORWARD_HOOK, CELL_JIT_FORWARD_HOOK);
  if (hook_func == nullptr) {
    return;
  }

  hook_func->set_manager(manager);

  const auto &output = func_graph->output();
  auto hook_cnode = func_graph->NewCNodeInOrder({NewValueNode(hook_func), hook_input_node, output});
  func_graph->set_output(hook_cnode);
}

void MarkDeferInlineForHookedCell(const FuncGraphPtr &func_graph, const py::object &obj) {
  auto mark_when_hook_is = [&func_graph, &obj](const char *hook_type) -> void {
    MS_EXCEPTION_IF_NULL(hook_type);

    if (!py::hasattr(obj, hook_type)) {
      return;
    }

    const auto &dict = py::cast<py::dict>(py::getattr(obj, hook_type));
    if (dict.size() == 0) {
      return;
    }

    func_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  };

  mark_when_hook_is(CELL_BACKWARD_PRE_HOOK);
  mark_when_hook_is(CELL_BACKWARD_HOOK);
}

void ApplyCellHooks(const AnfNodePtr &resolved_node, const py::object &obj, const FuncGraphManagerPtr &manager) {
  if (!IsCellHookExistAny(obj,
                          {CELL_FORWARD_PRE_HOOK, CELL_FORWARD_HOOK, CELL_BACKWARD_PRE_HOOK, CELL_BACKWARD_HOOK})) {
    return;
  }
  auto resolved_graph = GetValueNode<FuncGraphPtr>(resolved_node);
  if (resolved_graph == nullptr) {
    return;
  }

  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < resolved_graph->parameters().size(); ++i) {
    (void)inputs.emplace_back(resolved_graph->parameters()[i]);
  }
  auto tupled_input = resolved_graph->NewCNodeInOrder(inputs);
  resolved_graph->AddNode(tupled_input);

  auto pre_hooked_input = ApplyForwardPreHook(resolved_graph, obj, manager, tupled_input);
  ApplyForwardHook(resolved_graph, obj, manager, pre_hooked_input);

  MarkDeferInlineForHookedCell(resolved_graph, obj);
}
}  // namespace

void UnsetTraceTag(const py::object &obj) {
  const std::string trace_tag = "__trace_func__";
  const std::string construct_func = "construct";
  if (py::hasattr(obj, construct_func.c_str())) {
    const auto &construct_obj = py::getattr(obj, construct_func.c_str());
    if (py::hasattr(construct_obj, trace_tag.c_str())) {
      py::delattr(construct_obj, trace_tag.c_str());
      MS_LOG(WARNING)
        << "Trace mode in ast mode does not support getting kwargs. Pure ast mode will be executed instead";
    }
  } else if (py::hasattr(obj, trace_tag.c_str())) {
    py::delattr(obj, trace_tag.c_str());
    MS_LOG(WARNING) << "Trace mode in ast mode does not support getting kwargs. Pure ast mode will be executed instead";
  }
}

void MarkDeferInlineForSavedTensorsHooks(const AnfNodePtr &resolved_node, const py::object &obj) {
  MS_EXCEPTION_IF_NULL(resolved_node);
  auto func_graph = GetValueNode<FuncGraphPtr>(resolved_node);
  if (func_graph == nullptr) {
    return;
  }

  auto mark_when_has_hook = [&func_graph, &obj](const char *py_attr_name) {
    auto attr = py::getattr(obj, py_attr_name, py::none());
    if (!attr.is_none()) {
      func_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
      return;
    }

    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    auto construct_obj = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_ORIGINAL_CELL_CONSTRUCT, obj);
    if (construct_obj.is_none()) {
      return;
    }
    attr = py::getattr(construct_obj, py_attr_name, py::none());
    if (attr.is_none()) {
      return;
    }
    func_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  };

  mark_when_has_hook(FUNC_GRAPH_FLAG_PACK_HOOK);
  mark_when_has_hook(FUNC_GRAPH_FLAG_UNPACK_HOOK);
}

AnfNodePtr Resolver::ResolveSymbol(const FuncGraphManagerPtr &manager, const NameSpacePtr &name_space,
                                   const SymbolPtr &symbol, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (manager == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Manager is nullptr.";
  }
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString()
                << ", loc: " << trace::GetDebugInfoStr(node->debug_info());
  TraceGuard trace_guard(MakeTraceInfo<TraceResolve>(trace::GetSourceCodeDebugInfo(node->debug_info())));
  auto obj = GetSymbolObject(name_space, symbol, node);
  if (node->has_user_data(CONTAIN_KWARGS_INPUT)) {
    UnsetTraceTag(obj);
  }
  AnfNodePtr resolved_node = ResolveObjectAndAddToManager(manager, obj, node);
  if (IsPrimitive(resolved_node, prim::kPrimTraceGraph)) {
    auto cell_obj = name_space->module_obj();
    if (py::isinstance<Cell>(cell_obj)) {
      auto cell_interpteted = std::make_shared<InterpretedObject>(cell_obj);
      auto cnode = dyn_cast<CNode>(node);
      MS_EXCEPTION_IF_NULL(cnode);
      cnode->add_input(NewValueNode(cell_interpteted));
    }
  }
  if (IsValueNode<NameSpace>(resolved_node) && !py::isinstance<py::none>(name_space->module_obj())) {
    auto name_value = GetValueNode(resolved_node);
    auto nameptr = name_value->cast<NameSpacePtr>();
    nameptr->set_module_obj(name_space->module_obj());
  }
  fallback::SetPyObjectToNode(resolved_node, obj);
  // Update top graph debug info with user top graph's
  if (name_space->module() == RESOLVE_NAMESPACE_NAME_ENTRY && IsValueNode<FuncGraph>(resolved_node)) {
    auto user_top_fg = GetValueNode<FuncGraphPtr>(resolved_node);
    MS_EXCEPTION_IF_NULL(user_top_fg);
    auto top_fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(top_fg);
    top_fg->set_debug_info(user_top_fg->debug_info());
    top_fg->return_node()->set_debug_info(user_top_fg->return_node()->debug_info());
    MS_LOG(DEBUG) << "Update top graph's and node's debug infos with user top graph's. top_fg: " << top_fg->ToString()
                  << ", user_top_fg: " << user_top_fg->ToString();
    top_fg->set_attrs(user_top_fg->attrs());
    // Update top graph parameters' name
    auto top_params = top_fg->parameters();
    auto resolve_params = user_top_fg->parameters();
    auto top_arg_size = top_fg->GetPositionalArgsCount();
    auto user_arg_size = user_top_fg->GetPositionalArgsCount();
    if (top_arg_size > user_arg_size) {
      MS_LOG(INFO) << "Top graph's parameter size: " << top_arg_size
                   << " should not be greater than resolved func_graph's parameter size: " << user_arg_size;
    } else {
      for (int i = 0; i < top_arg_size; ++i) {
        auto param_ptr = top_params[i]->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(param_ptr);
        auto user_param_ptr = resolve_params[i]->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(user_param_ptr);
        param_ptr->set_debug_info(user_param_ptr->debug_info());
        param_ptr->set_name(user_param_ptr->name());
      }
      MS_LOG(DEBUG) << "Update top graph's parameters debug info with user top graph's parameters";
    }
  }

  const auto compile_cell_hook = "__compile_cell_hook__";
  if (name_space->module() != RESOLVE_NAMESPACE_NAME_ENTRY || py::hasattr(obj, compile_cell_hook)) {
    ApplyCellHooks(resolved_node, obj, manager);
  }

  MarkDeferInlineForSavedTensorsHooks(resolved_node, obj);

  return resolved_node;
}

AnfNodePtr CreateResolveNode(const py::object &obj, const AnfNodePtr &attr, const AnfNodePtr &get_attr_node) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object namespace_obj = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, obj);
  auto new_namespace = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_obj, obj);
  auto attr_string = GetValuePtr<StringImm>(attr);
  MS_EXCEPTION_IF_NULL(attr_string);
  const std::string &attr_as_string = attr_string->value();
  auto new_symbol = std::make_shared<Symbol>(attr_as_string);
  MS_LOG(DEBUG) << "name_space: " << new_namespace->ToString() << ", symbol: " << new_symbol->ToString();

  auto fg = get_attr_node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  AnfNodePtr resolved_node =
    fg->NewCNode({NewValueNode(prim::kPrimResolve), NewValueNode(new_namespace), NewValueNode(new_symbol)});
  resolved_node->set_debug_info(get_attr_node->debug_info());
  fg->ReplaceInOrder(get_attr_node, resolved_node);
  return resolved_node;
}

// Resolve Cell GetAttr operation.
AnfNodePtr Resolver::ResolveCellWithAttr(const FuncGraphManagerPtr &manager, const py::object &obj,
                                         const AnfNodePtr &resolve_node, const AnfNodePtr &attr,
                                         const AnfNodePtr &get_attr_node) {
  MS_EXCEPTION_IF_NULL(resolve_node);
  MS_EXCEPTION_IF_NULL(attr);
  MS_EXCEPTION_IF_NULL(manager);
  MS_LOG(DEBUG) << "obj: " << py::str(obj) << ", attr: " << attr->ToString();
  if (IsValueNode<StringImm>(attr)) {
    const auto &attr_name = GetValue<std::string>(GetValueNode(attr));
    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    bool is_property =
      (python_adapter::CallPyModFn(mod, parse::PYTHON_PARSE_CHECK_ATTR_IS_PROPERTY, obj, attr_name)).cast<bool>();
    if (is_property) {
      auto get_attr_cnode = get_attr_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(get_attr_cnode);
      constexpr size_t kInputIdx1 = 1;
      AnfNodePtr node = get_attr_cnode->input(kInputIdx1);
      auto cur_func = get_attr_node->func_graph();
      auto call_func_node = parse::TransPropertyToFunc(cur_func, node, obj, attr_name);
      MS_LOG(DEBUG) << "call_func_node:" << call_func_node->DebugString();
      return call_func_node;
    }
  }
  TraceGuard trace_guard(MakeTraceInfo<TraceResolve>(get_attr_node->debug_info()));
  if (!data_converter::IsCellInstance(obj)) {
    AnfNodePtr resolved_node = ResolveObjectAndAddToManager(manager, obj, resolve_node);
    AnfNodePtrList inputs = {NewValueNode(prim::kPrimGetAttr), resolved_node, attr};
    auto cur_func = get_attr_node->func_graph();
    MS_EXCEPTION_IF_NULL(cur_func);
    AnfNodePtr res_node = cur_func->NewCNode(std::move(inputs));
    res_node->set_debug_info(get_attr_node->debug_info());
    cur_func->ReplaceInOrder(get_attr_node, res_node);
    return res_node;
  }

  constexpr auto tensors_queue_attr = "__is_tensors_queue__";
  if (py::hasattr(obj, tensors_queue_attr) && IsValueNode<StringImm>(attr)) {
    const auto &attr_name = GetValue<std::string>(GetValueNode(attr));
    constexpr auto pop_attr = "pop";
    if (attr_name == pop_attr) {
      constexpr auto graph_pop_attr = "__graph_pop__";
      MS_LOG(DEBUG) << "Replace " << pop_attr << " to " << graph_pop_attr << " for " << py::str(obj);
      return CreateResolveNode(obj, NewValueNode(graph_pop_attr), get_attr_node);
    }
  }
  return CreateResolveNode(obj, attr, get_attr_node);
}

// Get attribute or method from ms_class obj or cell obj.
AnfNodePtr ResolveClassObjectWithAttr(const py::object &cls_obj, const AnfNodePtr &attr,
                                      const AnfNodePtr &get_attr_node) {
  MS_EXCEPTION_IF_NULL(get_attr_node);
  MS_LOG(DEBUG) << "Resolve ms_class obj (" << py::str(cls_obj) << ") with attr " << attr->ToString() << ".";
  TraceGuard trace_guard(MakeTraceInfo<TraceResolve>(get_attr_node->debug_info()));
  return CreateResolveNode(cls_obj, attr, get_attr_node);
}

AnfNodePtr Resolver::ResolveSequenceWithAttr(const FuncGraphManagerPtr &manager, const py::object &obj,
                                             const AnfNodePtr &resolve_node, const AnfNodePtr &attr,
                                             const CNodePtr &get_attr_node) {
  MS_EXCEPTION_IF_NULL(get_attr_node);
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
      auto res = ResolveClassObjectWithAttr(sequence[i], attr, get_attr_node);
      (void)inputs.emplace_back(res);
    }
  } else {
    return nullptr;
  }

  constexpr auto prim_index = 0;
  constexpr auto index_index = 2;
  auto fg = get_attr_node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto make_tuple_node = fg->NewCNodeInOrder(inputs);
  return fg->NewCNodeInOrder({get_attr_node->input(prim_index), make_tuple_node, get_attr_node->input(index_index)});
}

AnfNodePtr Resolver::ResolveSymbolWithAttr(const FuncGraphManagerPtr &manager, const AnfNodePtr &object_node,
                                           const AnfNodePtr &attr_node, const AnfNodePtr &get_attr_node) {
  // {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
  auto [name_space, symbol] = GetNamespaceAndSymbol(object_node);
  MS_EXCEPTION_IF_NULL(name_space);
  MS_EXCEPTION_IF_NULL(symbol);
  constexpr std::string_view parse_super_name = "namespace";
  if (symbol->symbol() == parse_super_name) {
    return nullptr;
  }
  const auto &module_name = name_space->module();
  auto symbol_obj = GetSymbolObject(name_space, symbol, get_attr_node);
  if (module_name == RESOLVE_NAMESPACE_NAME_CLASS_MEMBER || data_converter::IsCellInstance(symbol_obj)) {
    auto res = ResolveCellWithAttr(manager, symbol_obj, object_node, attr_node, get_attr_node);
    res->set_user_data<py::object>("__getattr__", std::make_shared<py::object>(symbol_obj));
    return res;
  }
  return nullptr;
}

// Get python object with index from a list or the whole list if the index is not fixed.
py::object GetObjectFromSequence(const NameSpacePtr &name_space, const SymbolPtr &symbol, const AnfNodePtr &node,
                                 const AnfNodePtr &index_node) {
  MS_EXCEPTION_IF_NULL(node);
  TraceGuard trace_guard(MakeTraceInfo<TraceResolve>(node->debug_info()));
  py::object obj = GetSymbolObject(name_space, symbol, node);
  // If obj is nn.CellList, convert it to sequence.
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  bool is_cell_list = py::hasattr(obj, PYTHON_CELL_AS_LIST);
  if (is_cell_list) {
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

AnfNodePtr Resolver::ResolveGetItemWithAttr(const FuncGraphManagerPtr &manager, const AnfNodePtr &getitem_node,
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

AnfNodePtr ResolveInterpretedObjectOfSetAttr(const AnfNodePtr &target_node, const AnfNodePtr &attr_node,
                                             const AnfNodePtr &value_node) {
  auto [name_space, symbol] = GetNamespaceAndSymbol(target_node);
  MS_EXCEPTION_IF_NULL(name_space);
  MS_EXCEPTION_IF_NULL(symbol);
  auto symbol_obj = GetSymbolObject(name_space, symbol, target_node);
  auto interpreted_obj = std::make_shared<InterpretedObject>(symbol_obj);
  MS_EXCEPTION_IF_NULL(interpreted_obj);
  MS_LOG(DEBUG) << "Created a interpreted object: " << interpreted_obj->ToString();
  const auto &resolve_node = ConvertInterpretedObjForResolve(target_node, interpreted_obj, target_node->func_graph());

  AnfNodePtrList inputs = {NewValueNode(prim::kPrimSetAttr), resolve_node, attr_node, value_node};
  return target_node->func_graph()->NewCNodeInOrder(std::move(inputs));
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
  func_graph->PresetCertainSideEffect();
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
  auto res = std::make_shared<pipeline::Resource>();
  res->set_manager(manager);

  auto roots = manager->roots();
  for (const auto &fg : roots) {
    res->set_func_graph(fg);
    bool ret = ResolveFuncGraph(fg, res, false);
    if (!ret) {
      MS_EXCEPTION_IF_NULL(fg);
      MS_LOG(ERROR) << "Resolve fg " << fg->ToString() << " failed";
    }
  }
  return true;
}

TensorHookMapPtr ResolveTensorHooks(const pipeline::ResourceBasePtr &resource, const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &grad_meta_data = tensor->auto_grad_meta_data();
  if (grad_meta_data == nullptr) {
    MS_LOG(DEBUG) << "The grad_meta_data of " << tensor->ToString() << " is nullptr.";
    return nullptr;
  }

  const auto &grad_node = grad_meta_data->UnsafeGetGradNodeImpl();
  if (grad_node == nullptr) {
    return nullptr;
  }

  TensorHookMapPtr hooks = std::make_shared<TensorHookMap>();
  if (const auto &backward_hooks = grad_node->py_tensor_pre_hooks()) {
    for (const auto &[id, hook] : *backward_hooks) {
      py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
      py::object wrappered_hook = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_HOOK_WRAPPER, hook->hook_fn_);
      auto func = parse::ParsePythonCode(wrappered_hook);
      MS_EXCEPTION_IF_NULL(func);

      // Hook may have side effects.
      func->set_flag(mindspore::kFuncGraphFlagBackPropEntry, true);
      func->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);

      pipeline::ResourceBasePtr res = (resource != nullptr ? resource : std::make_shared<pipeline::Resource>());
      (void)parse::ResolveFuncGraph(func, res, false);

      func->set_reserved(true);
      for (auto &fg : func->func_graphs_used_total()) {
        MS_EXCEPTION_IF_NULL(fg);
        fg->set_reserved(true);
      }

      hooks->emplace(id, func);
    }
  }

  return hooks;
}

// If any mixed precision flag add a cast node after the parameter node.
// argument obj should be python Parameter object
// it will be converted to Parameter node here
AnfNodePtr Resolver::ResolveParameterObj(const FuncGraphPtr &func_graph, const py::object &obj) {
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
  auto param_name = py::cast<std::string>(name_attr);
  // If the parameter node has been created , return it.
  ParameterPtr para_node = nullptr;
  MS_EXCEPTION_IF_NULL(top_func_graph_);
  for (auto const &param : top_func_graph_->parameters()) {
    auto param_node = dyn_cast<Parameter>(param);
    if (param_node != nullptr && param_node->name() == param_name) {
      if (param_node->is_top_graph_param()) {
        // If the name of the input of construct is same as the parameters,
        // add suffix to the name of the input of construct.
        string suffix_name = param_node->name() + "_$";
        param_node->set_name(suffix_name);
        if (param_node->debug_info() != nullptr) {
          param_node->debug_info()->set_name(suffix_name);
        }
        MS_LOG(DEBUG) << "Add suffix to the name of the input of construct " << func_graph->ToString()
                      << ", input: " << param_node->DebugString();
      } else {
        // Exist two parameter object which name is the same.
        auto iter = param_obj_ids.find(param_name);
        if (iter != param_obj_ids.end() && iter->second != obj_id) {
          MS_LOG(EXCEPTION)
            << "The parameter " << param_node->DebugString() << " , its name '" << param_name
            << "' already exists. Please set a unique name for the parameter."
            << "\nFor more details with the name of parameter, please refer to "
            << "https://mindspore.cn/search?inputValue=Please%20set%20a%20unique%20name%20for%20the%20parameter";
        }
        para_node = param_node;
        MS_LOG(DEBUG) << "Found existing parameter for " << func_graph->ToString()
                      << ", param: " << para_node->DebugString() << ", top_func_graph: " << top_func_graph_->ToString();
        break;
      }
    }
  }
  if (para_node == nullptr) {
    auto value = GetParameterValue(obj);
    para_node = top_func_graph_->AddFvParameter(param_name, value);
    param_obj_ids[param_name] = obj_id;
    MS_LOG(DEBUG) << "Created a new weight parameter for " << func_graph->ToString()
                  << ", param: " << para_node->DebugString() << ", top_func_graph: " << top_func_graph_->ToString();
    auto context = parallel::ParallelContext::GetInstance();
    if (context != nullptr && para_node->has_default()) {
      auto param_abs = pipeline::GetDefaultValueAbstract(para_node);
      context->ParallelParameterContextRestoreShape(top_func_graph_, para_node, param_abs);
      para_node->set_abstract(param_abs);
    }
  }
  func_graph->add_parameter_obj_node(para_node);
  return para_node;
}

std::optional<std::string> ToPyStr(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    MS_LOG(DEBUG) << "py::object is null";
    return std::nullopt;
  }
  try {
    return py::isinstance<py::str>(obj) ? obj.cast<std::string>() : py::str(obj).cast<std::string>();
  } catch (py::error_already_set &e) {
    MS_LOG(INFO) << "py::str() throws exception: " << e.what();
    return std::nullopt;
  }
}

std::string ToPyStr(const py::object &obj, const std::string &default_value) {
  return ToPyStr(obj).value_or(default_value);
}

}  // namespace parse
}  // namespace mindspore
