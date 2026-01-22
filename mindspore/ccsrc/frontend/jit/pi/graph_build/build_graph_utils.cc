/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_build/build_graph_utils.h"

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <utility>
#include <algorithm>
#include "utils/flags.h"
#include "utils/ms_context.h"
#include "ir/cell.h"
#include "ir/meta_func_graph.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/utils/tensor_py.h"
#include "include/utils/pynative/common_utils.h"
#include "frontend/jit/pi/external.h"
#include "frontend/jit/ps/action.h"
#include "frontend/jit/ps/parse/resolve.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "include/frontend/operator/primitive_py.h"
#include "frontend/jit/pi/utils/utils.h"
#include "include/utils/pynative/variable.h"
#include "frontend/operator/composite/auto_generate/functional_map.h"
#include "primitive/framework_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/structure_ops.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace pijit {
namespace {
bool ShouldFallBackInRuntime(const PrimitivePtr &prim) {
  static HashSet<std::string> prims_should_fallback_in_runtime = {kListInplaceExtendOpName,
                                                                  kListInplaceInsertOpName,
                                                                  kListInplacePopOpName,
                                                                  kListInplaceReverseOpName,
                                                                  kListInplaceClearOpName,
                                                                  kDictInplaceSetItemOpName,
                                                                  kRaiseOpName,
                                                                  kJoinedStrOpName,
                                                                  kFormatOpName};
  return prims_should_fallback_in_runtime.find(prim->name()) != prims_should_fallback_in_runtime.end();
}

bool IsPrimitiveObject(const py::object &obj) {
  return py::hasattr(obj, PYTHON_PRIMITIVE_FLAG) &&
         parse::data_converter::GetObjType(obj) != parse::RESOLVE_TYPE_CLASS_TYPE;
}

bool IsPrimitiveFunctionalObject(const py::object &obj) {
  return py::isinstance<PrimitiveFunctionAdapter>(obj) || py::isinstance<Functional>(obj);
}

bool IsMsClassObject(const py::object &obj) {
  constexpr auto ms_class_attr = "__ms_class__";
  return py::hasattr(obj, ms_class_attr) && py::cast<bool>(py::getattr(obj, ms_class_attr));
}

bool IsMetaFuncGraphObject(const py::object &obj) { return py::isinstance<MetaFuncGraph>(obj); }

static std::string ExtractFirstPart(const std::string &error_message) {
  const std::string separator = "----------------------------------------------------";
  size_t pos = error_message.find(separator);
  if (pos != std::string::npos) {
    return error_message.substr(0, pos);
  }
  return error_message;
}

std::pair<AbstractBasePtr, bool> EvalValue(const ValuePtr &value, const AbstractBasePtrList &inputs_abs_list) {
  if (value == nullptr) {
    return std::make_pair(nullptr, false);
  }
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    if (value->isa<Primitive>()) {
      auto prim = value->cast<PrimitivePtr>();
      auto eval_res = abstract::EvalOnePrim(prim, inputs_abs_list);
      if (eval_res != nullptr) {
        return std::make_pair(eval_res->abstract(), IsSideEffectPrimitive(prim));
      }
    } else if (value->ToAbstract()->isa<abstract::AbstractFunction>()) {
      auto analyze_res = pipeline::AbstractAnalyze(value, inputs_abs_list);
      if (analyze_res.eval_result != nullptr) {
        MS_LOG(DEBUG) << "Eval result, has_side_effect: "
                      << (analyze_res.eval_result->has_side_effect_node() ? "true" : "false");
        return std::make_pair(analyze_res.eval_result->abstract(), analyze_res.eval_result->has_side_effect_node());
      }
    }
    return std::make_pair(nullptr, false);
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to EvalValue for value: " << value->ToString() << ". The exception:\n" << e.what();
    PIJIT_DEBUG_LOG(LogCfg::kGraphBreak) << std::endl << "Eval failed reason: " << ExtractFirstPart(e.what());
    return std::make_pair(nullptr, false);
  }
}

void SetParameterName(const ParameterPtr &param) {
  MS_EXCEPTION_IF_NULL(param);
  if (param->name() != "") {
    return;
  }
  auto fg = param->func_graph();
  const auto &fg_params = fg->parameters();
  size_t index;
  for (index = 0; index < fg_params.size(); ++index) {
    if (param == fg_params[index]) {
      break;
    }
  }
  auto name = fg->ToString() + "_input_" + std::to_string(index);
  param->set_name(name);
}
}  // namespace

std::pair<AbstractBasePtr, bool> InferAndCheck(const ValuePtr &value, const AbstractBasePtrList &input_abs_list) {
  MS_EXCEPTION_IF_NULL(value);
  MS_LOG(DEBUG) << "Start to infer callable: " << value->ToString();
  const auto &res = EvalValue(value, input_abs_list);
  auto abs = res.first;
  if (abs == nullptr) {
    MS_LOG(DEBUG) << "Eval failed for value: " << value->ToString();
    return std::make_pair(nullptr, false);
  }
  if (value->isa<Primitive>() && !IsPrimitiveCallable(value->cast<PrimitivePtr>(), abs)) {
    MS_LOG(DEBUG) << "Check callable failed for value: " << value->ToString() << ", abs: " << abs->ToString();
    return std::make_pair(nullptr, false);
  }
  return res;
}

AbstractBasePtr BuildNodeAbstract(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  if (node->abstract() != nullptr) {
    return node->abstract();
  }
  if (node->isa<ValueNode>()) {
    return node->cast<ValueNodePtr>()->value()->ToAbstract();
  } else if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode->empty() || !cnode->input(0)->isa<ValueNode>()) {
      return nullptr;
    }
    ValuePtr value = cnode->input(0)->cast<ValueNodePtr>()->value();
    std::vector<AbstractBasePtr> abs_list;
    std::transform(cnode->inputs().begin() + 1, cnode->inputs().end(), std::back_inserter(abs_list),
                   [](const AnfNodePtr &node) {
                     if (node->abstract() == nullptr) {
                       node->set_abstract(BuildNodeAbstract(node));
                     }
                     return node->abstract();
                   });
    return EvalValue(value, abs_list).first;
  }
  MS_LOG(INFO) << "Unsupported Node type for GetAbstractOf() method, node: " << node->DebugString();
  return nullptr;
}

bool IsObjectCallable(const py::object &obj) {
  static constexpr auto check_list = {IsPrimitiveObject, IsPrimitiveFunctionalObject, IsMsClassObject,
                                      IsMetaFuncGraphObject};
  return std::any_of(check_list.begin(), check_list.end(), [&obj](const auto &func) { return func(obj); });
}

bool IsSideEffectPrimitive(const PrimitivePtr &prim) {
  return GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_IO) || GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM);
}

bool IsValidOutputAbstractScalar(const AbstractBasePtr &abs) {
  if (!abs->isa<abstract::AbstractScalar>()) {
    return false;
  }
  auto build_type = abs->BuildType();
  if (build_type->isa<String>()) {
    auto value = abs->BuildValue()->cast<StringImmPtr>();
    const auto &str = value->value();
    const std::string fake_prefix = "FakeNodeKey";
    return str.substr(0, fake_prefix.size()) != fake_prefix;
  }
  return build_type->isa<String>() || build_type->isa<Number>();
}

bool IsValidOutputAbstractTensor(const AbstractBasePtr &abs) {
  return abs->isa<abstract::AbstractTensor>() || abs->isa<abstract::AbstractRowTensor>() ||
         abs->isa<abstract::AbstractMapTensor>();
}

bool IsPrimitiveCallable(const PrimitivePtr &prim, const AbstractBasePtr &abs) {
  if (prim == nullptr || abs == nullptr || abs->isa<abstract::AbstractAny>()) {
    return false;
  }
  return !ShouldFallBackInRuntime(prim);
}

bool IsParameterSequence(const py::object &object) {
  if (object.ptr() == nullptr) {
    return false;
  }
  constexpr auto parameter_tuple_attr = "__parameter_tuple__";
  if (py::hasattr(object, parameter_tuple_attr)) {
    return true;
  }
  if (!py::isinstance<py::tuple>(object) && !py::isinstance<py::list>(object)) {
    return false;
  }
  auto object_tuple = object.cast<py::tuple>();
  if (object_tuple.size() == 0) {
    return false;
  }
  if (std::any_of(object_tuple.begin(), object_tuple.end(),
                  [](const auto &element) { return !parse::IsParameterObject(py::cast<py::object>(element)); })) {
    return false;
  }
  return true;
}

ParameterPtr AddParameter(const FuncGraphPtr &fg) {
  auto param = fg->add_parameter();
  SetParameterName(param);
  return param;
}

std::string GetParameterName(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(DEBUG) << "Node is null!";
    return "";
  }
  auto param = node->cast_ptr<Parameter>();
  if (param == nullptr) {
    MS_LOG(DEBUG) << "Node is not a Parameter, but is: " << node->DebugString();
    return "";
  }
  return param->name();
}

py::tuple GetMethodInfo(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  return python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_METHOD_INFO, obj);
}

std::string GetTensorMethodName(const py::object &obj) {
  constexpr auto tensor_func_list_mod_str = "mindspore.graph._pijit.tensor_func_list";
  constexpr auto get_tensor_method_name_str = "get_tensor_method_name";
  py::module mod = python_adapter::GetPyModule(tensor_func_list_mod_str);
  py::object name_obj = python_adapter::CallPyModFn(mod, get_tensor_method_name_str, FunctionId(obj));
  if (py::isinstance<py::none>(name_obj)) {
    return "";
  }
  return name_obj.cast<std::string>();
}

bool IsTensorMethod(const py::object &obj) {
  const auto &method_name = GetTensorMethodName(obj);
  return method_name != "";
}

bool IsTensorOverloadMethod(const py::object &obj) {
  const auto &method_name = GetTensorMethodName(obj);
  if (method_name == "") {
    return false;
  }
  return ops::tensor_method_overload_map.find(method_name) != ops::tensor_method_overload_map.end();
}

bool EnableTensorOverload() {
  auto ge_mode = (MsContext::GetInstance()->GetJitLevel() == kAttrJitLevelO2);
  return !ge_mode;
}

bool IsCellList(const py::object &obj) { return obj.ptr() != nullptr && py::hasattr(obj, PYTHON_CELL_AS_LIST); }

bool IsConvertToInterpretedObject(const py::object &obj) {
  // NOTE: py::function::check_ alias PyCallable_Check. Python class is callable
  // identify the function if need parse by ast
  return py::isinstance<Cell>(obj) || PyCFunction_Check(obj.ptr()) || PyMethod_Check(obj.ptr()) ||
         IsTensorOverloadMethod(obj);
}

ValuePtr ConvertPyObjToValue(const py::handle &handle, bool allow_interpreted_object) {
  MS_EXCEPTION_IF_NULL(handle.ptr());
  py::object obj = py::reinterpret_borrow<py::object>(handle);
  ValuePtr ret = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    PyRecursionScope rec_check(obj);

    if (py::list::check_(obj) || py::tuple::check_(obj) || pijit::IsCellList(obj)) {
      std::vector<ValuePtr> elements;
      for (const auto &i : obj) {
        auto v = ConvertPyObjToValue(i, allow_interpreted_object);
        if (v == nullptr) {
          return nullptr;
        }
        elements.push_back(v);
      }
      if (py::list::check_(obj)) {
        return std::make_shared<ValueList>(elements);
      } else {
        return std::make_shared<ValueTuple>(elements);
      }
    }
    if (py::dict::check_(obj)) {
      std::vector<std::pair<ValuePtr, ValuePtr>> elements;
      for (const auto &i : py::cast<py::dict>(obj)) {
        auto k = ConvertPyObjToValue(i.first, allow_interpreted_object);
        auto v = ConvertPyObjToValue(i.second, allow_interpreted_object);
        if (k == nullptr || v == nullptr) {
          return nullptr;
        }
        elements.push_back(std::make_pair(k, v));
      }
      return std::make_shared<ValueDictionary>(elements);
    }
    if (allow_interpreted_object && IsConvertToInterpretedObject(obj)) {
      return std::make_shared<parse::InterpretedObject>(obj);
    }
    if (parse::ConvertData(obj, &ret)) {
      return ret;
    }
  } catch (const std::exception &e) {
    MS_LOG(INFO) << e.what();
  }
  MS_LOG(INFO) << "Failed to convert python object." << py::str(handle);
  return nullptr;
}

ValuePtr ConvertPyCallableToValue(const py::handle &callable) { return ConvertPyObjToValue(callable, false); }

void PrintConstantAbstract(const AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return;
  }
  if (abstract->isa<abstract::AbstractFunction>()) {
    return;
  }
  if (abstract->isa<abstract::AbstractSequence>()) {
    const auto &elements = abstract->cast<abstract::AbstractSequencePtr>()->elements();
    std::for_each(elements.begin(), elements.end(), [](const auto &e) { PrintConstantAbstract(e); });
  }
  if (abstract->isa<abstract::AbstractDictionary>()) {
    const auto &elements = abstract->cast<abstract::AbstractDictionaryPtr>()->elements();
    std::for_each(elements.begin(), elements.end(), [](const auto &e) { PrintConstantAbstract(e.second); });
  }
  if (abstract->isa<abstract::AbstractTensor>()) {
    if (abstract->isa<abstract::AbstractRefTensor>()) {
      return;
    }
    MS_LOG(WARNING) << "Encounter constant Tensor node with abstract: " << abstract->ToString();
    return;
  }
  MS_LOG(INFO) << "Encounter constant value node with abstract: " << abstract->ToString();
}

void AttachCustomBPropToGraph(const FuncGraphPtr &graph, const py::object &obj) {
  if (graph == nullptr || obj.ptr() == nullptr) {
    return;
  }
  if (!py::hasattr(obj, parse::CUSTOM_BPROP_NAME)) {
    return;
  }
  bool enable_bprop_debug = py::cast<bool>(py::getattr(obj, "bprop_debug"));
  FuncGraphPtr bprop_graph = enable_bprop_debug
                               ? parse::ConvertToBpropCut(obj)
                               : parse::ConvertToFuncGraph(obj, {}, parse::PYTHON_MOD_GET_BPROP_METHOD);
  if (bprop_graph != nullptr) {
    (void)graph->transforms().emplace(parse::CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph));
    (void)bprop_graph->transforms().emplace("primal", FuncGraphTransform(graph));
    graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    graph->set_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP, true);
    MS_LOG(INFO) << "Add custom bprop to graph.";
  }
  return;
}
}  // namespace pijit
}  // namespace mindspore
