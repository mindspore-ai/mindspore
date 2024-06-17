/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/pi/graph_build/func_graph_builder.h"
#include <algorithm>
#include <utility>
#include <set>
#include <queue>
#include "frontend/operator/composite/do_signature.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/pi/pi_jit_config.h"
#include "ops/arithmetic_ops.h"
#include "ops/structure_ops.h"
#include "include/common/utils/convert_utils_py.h"
#include "ir/tensor.h"
#include "ir/anf.h"

namespace mindspore {
namespace {
constexpr auto kPiJitPyObjKey = "pi_jit_py_obj";
constexpr auto kTensorModule = "mindspore.common";
constexpr auto kAdapterFlag = "adapter_flag";
constexpr auto kInnerOpsModule = "mindspore.ops.operations._inner_ops";

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

bool IsValidScalar(const AbstractBasePtr &abs) {
  auto build_type = abs->BuildType();
  return build_type->isa<String>() || build_type->isa<Number>();
}

bool Mutable(const py::object &obj, const ValuePtr &value = nullptr) {
  // If a tensor has been set const arg, it should not be mutable.
  if (value != nullptr && value->isa<tensor::MetaTensor>()) {
    constexpr char const_arg_attr[] = "const_arg";
    if (py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr))) {
      return false;
    }
  }
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

bool IsConstant(const py::object &obj) {
  if (obj.ptr() == nullptr || Mutable(obj)) {
    return false;
  }
  if (py::isinstance<py::tuple>(obj)) {
    auto list_obj = py::cast<py::tuple>(obj);
    return std::all_of(list_obj.begin(), list_obj.end(),
                       [](const auto &obj) { return IsConstant(py::cast<py::object>(obj)); });
  }
  if (py::isinstance<py::list>(obj)) {
    auto list_obj = py::cast<py::list>(obj);
    return std::all_of(list_obj.begin(), list_obj.end(),
                       [](const auto &obj) { return IsConstant(py::cast<py::object>(obj)); });
  }
  if (py::isinstance<py::dict>(obj)) {
    auto dict_obj = py::cast<py::dict>(obj);
    return std::all_of(dict_obj.begin(), dict_obj.end(), [](const auto &pair) {
      return IsConstant(py::cast<py::object>(pair.first)) && IsConstant(py::cast<py::object>(pair.second));
    });
  }
  // Attention: should exclude BaseTensor in the future (when the BaseTensor PR is merged)
  return !py::isinstance<tensor::Tensor>(obj) && !IsStubTensor(obj);
}

bool TensorArgMutable(const py::object &obj, const ValuePtr &value) {
  if (!value->isa<tensor::MetaTensor>()) {
    return false;
  }
  constexpr char const_arg_attr[] = "const_arg";
  return !py::hasattr(obj, const_arg_attr) || !py::cast<bool>(py::getattr(obj, const_arg_attr));
}

bool NeedBroaden(const py::object &obj, const ValuePtr &value) {
  return TensorArgMutable(obj, value) || Mutable(obj, value) || value->isa<tensor::MetaSparseTensor>();
}

TypeId GetTypeIdFromClassName(const std::string &class_name) {
  static HashMap<std::string, TypeId> class_name_to_type_ids = {
    {"Tensor", kObjectTypeTensorType},  {"list", kObjectTypeList},
    {"tuple", kObjectTypeTuple},        {"int", kNumberTypeInt},
    {"float", kNumberTypeFloat},        {"CellList", kObjectTypeList},
    {"CellDict", kObjectTypeDictionary}};
  auto iter = class_name_to_type_ids.find(class_name);
  if (iter == class_name_to_type_ids.end()) {
    return kTypeUnknown;
  }
  return iter->second;
}

ValuePtr MaybeMakeEmptyTensor(const AbstractBasePtr &abs) {
  auto build_value = abs->BuildValue();
  if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    std::vector<ValuePtr> value_vec;
    for (auto &elem : abs_seq->elements()) {
      (void)value_vec.emplace_back(MaybeMakeEmptyTensor(elem));
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      return std::make_shared<ValueTuple>(value_vec);
    } else {
      return std::make_shared<ValueList>(value_vec);
    }
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    const auto &elements = abs_dict->elements();
    std::vector<std::pair<ValuePtr, ValuePtr>> val_dict;
    for (auto &element : elements) {
      auto key_value = MaybeMakeEmptyTensor(element.first);
      auto val_value = MaybeMakeEmptyTensor(element.second);
      (void)val_dict.emplace_back(std::pair<ValuePtr, ValuePtr>{key_value, val_value});
    }
    return std::make_shared<ValueDictionary>(val_dict);
  }
  if (build_value == kValueAny && abs->isa<abstract::AbstractTensor>()) {
    auto abs_tensor = abs->cast<abstract::AbstractTensorPtr>();
    TypePtr tensor_type_ptr = abs_tensor->element()->BuildType();
    ShapeVector tensor_shape = abs_tensor->shape()->shape();
    auto tensor = std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
    if (abs->isa<abstract::AbstractRefTensor>()) {
      auto abs_ref_tensor = abs->cast<abstract::AbstractRefPtr>();
      // We only need the parameter name, it was used to find the python Parameter object later
      auto param_info = std::make_shared<ParamInfo>();
      param_info->set_name(abs_ref_tensor->ref_key_value()->ToString());
      tensor->set_param_info(param_info);
    }
    return tensor;
  }
  return build_value;
}

bool FunctionShouldBeParseInAst(const py::object &obj) {
  static mindspore::HashSet<std::string> func_names{"cast_to_adapter_tensor", "cast_to_ms_tensor"};
  if (!py::hasattr(obj, "__name__")) {
    return false;
  }
  return func_names.find(py::cast<std::string>(obj.attr("__name__"))) != func_names.end();
}

py::object ConvertToPythonTensor(const py::object &obj,
                                 const FuncGraphBuilder::PyTensorConverter &tensor_convert_func) {
  constexpr auto ms_class_attr = "__ms_class__";
  if (py::hasattr(obj, ms_class_attr) && py::cast<bool>(py::getattr(obj, ms_class_attr))) {
    return obj;
  }
  if (py::isinstance<tensor::Tensor>(obj)) {
    return tensor_convert_func(obj);
  }
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    auto obj_tuple = py::cast<py::tuple>(obj);
    py::tuple ret(obj_tuple.size());
    for (size_t i = 0; i < obj_tuple.size(); ++i) {
      ret[i] = ConvertToPythonTensor(obj_tuple[i], tensor_convert_func);
    }
    if (py::isinstance<py::list>(obj)) {
      return ret.cast<py::list>();
    }
    return ret;
  }
  if (py::isinstance<py::dict>(obj)) {
    auto obj_dict = py::cast<py::dict>(obj);
    for (auto item : obj_dict) {
      obj_dict[item.first] = ConvertToPythonTensor(py::cast<py::object>(item.second), tensor_convert_func);
    }
    return obj_dict;
  }
  return obj;
}

py::object ConvertCppTensorToPyTensor(const py::object &cpp_tensor) {
  if (cpp_tensor.ptr() == nullptr || !py::isinstance<tensor::Tensor>(cpp_tensor)) {
    return py::object();
  }
  bool is_adapter_tensor =
    py::hasattr(cpp_tensor, kAdapterFlag) && py::cast<bool>(py::getattr(cpp_tensor, kAdapterFlag));
  py::module mod = python_adapter::GetPyModule(kTensorModule);
  auto py_tensor = python_adapter::CallPyModFn(mod, "Tensor", cpp_tensor, py::none(), py::none(), py::none(), true);
  if (is_adapter_tensor) {
    mod = python_adapter::GetPyModule(kInnerOpsModule);
    py_tensor = python_adapter::CallPyModFn(mod, "convert_to_adapter_tensor", py_tensor);
  }
  return py_tensor;
}
}  // namespace

ValuePtr FuncGraphBuilder::ConvertPyObjToValue(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    return nullptr;
  }
  ValuePtr ret = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    if (!parse::ConvertData(obj, &ret)) {
      return nullptr;
    }
  } catch (const std::exception &e) {
    MS_LOG(DEBUG) << "Failed to convert python object << " << py::str(obj) << " to value. The exception:\n" << e.what();
    return nullptr;
  }
  return ret;
}

py::object FuncGraphBuilder::ConvertToPyObj(const AbstractBasePtr &abs) {
  static auto convert_func = [](const py::object &tensor) { return ConvertCppTensorToPyTensor(tensor); };
  return FuncGraphBuilder::ConvertToPyObj(abs, convert_func);
}

py::object FuncGraphBuilder::ConvertToPyObj(const AbstractBasePtr &abs, const PyTensorConverter &tensor_convert_func) {
  if (abs->isa<abstract::AbstractNone>()) {
    return py::none();
  }

  auto build_value = MaybeMakeEmptyTensor(abs);
  auto py_obj = ValueToPyData(build_value, abs);
  // Return none means failed converting.
  if (py::isinstance<py::none>(py_obj)) {
    return py::object();
  }

  if (pijit::kPIJitConfigDefault.GetBoolConfig(pijit::GraphJitConfig::kTraceFlag)) {
    return ConvertToPythonTensor(py_obj, tensor_convert_func);
  }

  return py_obj;
}

AnfNodePtr FuncGraphBuilder::ConvertObjToNode(const py::object &input_obj) {
  if (py::hasattr(input_obj, "__parameter__") && py::isinstance<tensor::MetaTensor>(input_obj)) {
    // Add the fv parameter and set its abstract.
    return parse::ResolveParameterObj(graph_, input_obj);
  }
  auto val = ConvertPyObjToValue(input_obj);
  if (val == nullptr) {
    MS_LOG(INFO) << "The input object " << py::str(input_obj) << " convert to value failed.";
    return nullptr;
  }
  // Constant value input scene, the object should be converted to value node.
  auto node = NewValueNode(val);
  node->set_abstract(val->ToAbstract());
  return node;
}

AbstractBasePtr FuncGraphBuilder::EvalValue(const ValuePtr &value, const AbstractBasePtrList &inputs_abs_list) {
  if (value == nullptr) {
    return nullptr;
  }
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    if (value->isa<Primitive>()) {
      auto prim = value->cast<PrimitivePtr>();
      auto eval_res = abstract::EvalOnePrim(prim, inputs_abs_list);
      if (eval_res != nullptr) {
        return eval_res->abstract();
      }
    } else if (value->ToAbstract()->isa<abstract::AbstractFunction>()) {
      auto analyze_res = pipeline::AbstractAnalyze(value, inputs_abs_list);
      if (analyze_res.eval_result != nullptr) {
        return analyze_res.eval_result->abstract();
      }
    }
    return nullptr;
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to EvalValue for value: " << value->ToString();
    return nullptr;
  }
}

bool FuncGraphBuilder::CheckCallable(const ValuePtr &value, const AbstractBasePtr &abs) {
  if (value == nullptr || abs == nullptr || abs->isa<abstract::AbstractAny>()) {
    return false;
  }
  if (value->isa<Primitive>() && ShouldFallBackInRuntime(value->cast<PrimitivePtr>())) {
    return false;
  }
  return true;
}

bool FuncGraphBuilder::CheckGraphOutput(const AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return false;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto elements = abs->cast<abstract::AbstractSequencePtr>()->elements();
    return std::all_of(elements.begin(), elements.end(),
                       [](const AbstractBasePtr &elem) { return CheckGraphOutput(elem); });
  }
  if (abs->isa<abstract::AbstractScalar>()) {
    return IsValidScalar(abs);
  }
  return abs->isa<abstract::AbstractTensor>() || abs->isa<abstract::AbstractRowTensor>() ||
         abs->isa<abstract::AbstractMapTensor>();
}

bool FuncGraphBuilder::AddLocalVariable(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    MS_LOG(INFO) << "Failed to add local variable, py object is null";
    return false;
  }

  auto iter = py_obj_to_node_.find(obj.ptr());
  if (iter != py_obj_to_node_.end()) {
    MS_LOG(INFO) << "Py object already in map, no need to add. Associated node: "
                 << ((iter->second != nullptr) ? iter->second->DebugString() : "NULL");
    return true;
  }

  auto node = ConvertObjToNode(obj);
  if (node == nullptr) {
    MS_LOG(INFO) << "Failed to add local variable, convert python object to anf node failed";
    return false;
  }

  node->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(obj));
  (void)py_obj_to_node_.emplace(obj.ptr(), node);
  return true;
}

AnfNodePtr FuncGraphBuilder::ReadLocalVariable(const py::object &obj) {
  auto iter = py_obj_to_node_.find(obj.ptr());
  if (iter == py_obj_to_node_.end()) {
    return nullptr;
  }
  return iter->second;
}

AnfNodePtr FuncGraphBuilder::GetNodeByObject(const py::object &obj) {
  // Search the predecessors of the current builder for the local parameter with BFS.
  mindspore::HashSet<FuncGraphBuilder *> visited_builders;
  std::queue<FuncGraphBuilder *> builder_queue;
  builder_queue.push(this);
  while (!builder_queue.empty()) {
    const auto cur_builder = builder_queue.front();
    MS_EXCEPTION_IF_NULL(cur_builder);
    builder_queue.pop();
    (void)visited_builders.insert(cur_builder);
    auto node = cur_builder->ReadLocalVariable(obj);
    if (node != nullptr) {
      MS_LOG(INFO) << "Found node: " << node->DebugString() << " for python object: " << std::string(py::str(obj))
                   << "  " << obj.ptr();
      return node;
    }
    for (const auto &cur_pred_builder : cur_builder->prev_builders()) {
      if (visited_builders.count(cur_pred_builder) == 0) {
        builder_queue.push(cur_pred_builder);
      }
    }
  }
  return nullptr;
}

bool FuncGraphBuilder::AddTopGraphArgsInputs(const py::object &object) {
  // args object should always be list object.
  if (object.ptr() == nullptr || !py::isinstance<py::list>(object)) {
    MS_LOG(INFO) << "Get top graph args failed.";
    return false;
  }
  auto args = object.cast<py::list>();
  for (size_t i = 0; i < args.size(); ++i) {
    auto arg = args[i].cast<py::object>();
    if (arg.ptr() == nullptr) {
      return false;
    }
    auto value = ConvertPyObjToValue(arg);
    if (value == nullptr) {
      return false;
    }
    bool broaden = NeedBroaden(arg, value);
    AbstractBasePtr abs = abstract::ToAbstract(value, nullptr, nullptr);
    if (broaden) {
      abs = AbstractBroaden(abs);
    }
    if (abs == nullptr) {
      MS_LOG(INFO) << "Failed to add input for python object: " << std::string(py::str(arg)) << "  " << arg.ptr();
      return false;
    }
    auto para = graph_->add_parameter();
    para->set_abstract(abs);
    para->set_is_top_graph_param(true);
    MS_LOG(INFO) << "Add top arg input success, python object: " << py::str(arg) << ", node: " << para->DebugString()
                 << ", abstract: " << abs->ToString();
    (void)py_obj_to_node_.emplace(arg.ptr(), para);
    para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(arg));
  }
  return true;
}

bool FuncGraphBuilder::AddTopGraphVargsInputs(const py::object &vargs) {
  if (vargs.ptr() == nullptr) {
    MS_LOG(INFO) << "Top graph has no vargs input.";
    return true;
  }
  auto vargs_tuple = vargs.cast<py::tuple>();
  if (vargs_tuple.ptr() == nullptr) {
    MS_LOG(INFO) << "Vargs object should be tuple but got: " << py::str(vargs) << ", add top graph vargs failed.";
    return false;
  }
  auto value = ConvertPyObjToValue(vargs);
  if (value == nullptr || !value->isa<ValueTuple>()) {
    MS_LOG(INFO) << "Convert vargs to value failed, vargs: " << py::str(vargs);
    return false;
  }
  auto value_tuple = value->cast<ValueTuplePtr>();
  const auto &elements = value_tuple->value();
  if (elements.size() != vargs_tuple.size()) {
    MS_LOG(INFO) << "For top graph vargs, converted value element size is " << elements.size()
                 << ", python tuple element size is " << vargs_tuple.size() << ". Size not matched.";
    return false;
  }
  std::vector<AbstractBasePtr> new_elements;
  for (size_t i = 0; i < elements.size(); ++i) {
    auto cur_obj = vargs_tuple[i].cast<py::object>();
    auto cur_val = elements[i];
    bool broaden = NeedBroaden(cur_obj, cur_val);
    auto cur_abs = abstract::ToAbstract(cur_val, nullptr, nullptr);
    if (broaden) {
      cur_abs = AbstractBroaden(cur_abs);
    }
    if (cur_abs == nullptr) {
      MS_LOG(INFO) << "Fail to convert args element " << cur_val->ToString();
      return false;
    }
    new_elements.push_back(cur_abs);
  }
  auto new_vargs_abs = std::make_shared<abstract::AbstractTuple>(new_elements);
  auto para = graph_->add_parameter();
  para->set_abstract(new_vargs_abs);
  para->set_is_top_graph_param(true);
  MS_LOG(INFO) << "Add top vargs input success, python object: " << py::str(vargs) << ", node: " << para->DebugString()
               << ", abstract: " << new_vargs_abs->ToString();
  (void)py_obj_to_node_.emplace(vargs.ptr(), para);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(vargs));
  return true;
}

bool FuncGraphBuilder::AddTopGraphKwargsInputs(const py::object &kwargs) {
  if (kwargs.ptr() == nullptr) {
    MS_LOG(INFO) << "Top graph has no kwargs input.";
    return true;
  }
  auto kwargs_dict = kwargs.cast<py::dict>();
  if (kwargs_dict.ptr() == nullptr) {
    MS_LOG(INFO) << "Kwargs object should be tuple but got: " << py::str(kwargs) << ", add top graph kwargs failed.";
    return false;
  }
  auto value = ConvertPyObjToValue(kwargs);
  if (value == nullptr || !value->isa<ValueDictionary>()) {
    MS_LOG(INFO) << "Convert kwargs to value failed, kwargs: " << py::str(kwargs);
    return false;
  }
  auto value_dict = value->cast<ValueDictionaryPtr>();
  const auto &elements = value_dict->value();
  if (elements.size() != kwargs_dict.size()) {
    MS_LOG(INFO) << "Kwargs dict size is " << kwargs_dict.size() << " and corresponding value dict size is "
                 << elements.size() << ". Size not matched.";
  }
  std::vector<abstract::AbstractElementPair> new_key_values;
  for (size_t i = 0; i < elements.size(); ++i) {
    auto cur_key_val = elements[i].first;
    auto cur_val = elements[i].second;
    auto cur_key_obj = ValueToPyData(cur_key_val);
    if (!kwargs_dict.contains(cur_key_obj)) {
      return false;
    }
    auto cur_val_obj = kwargs_dict[cur_key_obj];
    auto cur_value_abs = abstract::ToAbstract(cur_val, nullptr, nullptr);
    bool broaden = NeedBroaden(cur_val_obj, cur_val);
    if (broaden) {
      cur_value_abs = AbstractBroaden(cur_value_abs);
    }
    if (cur_value_abs == nullptr) {
      MS_LOG(INFO) << "Fail to convert kwargs value element " << cur_val->ToString();
      return false;
    }
    auto cur_key_abs = abstract::ToAbstract(cur_key_val, nullptr, nullptr);
    new_key_values.push_back(abstract::AbstractElementPair{cur_key_abs, cur_value_abs});
  }
  auto new_kwargs_abs = std::make_shared<abstract::AbstractDictionary>(new_key_values);
  auto para = graph_->add_parameter();
  para->set_abstract(new_kwargs_abs);
  para->set_is_top_graph_param(true);
  MS_LOG(INFO) << "Add top kwargs input success, python object: " << py::str(kwargs)
               << ", node: " << para->DebugString() << ", abstract: " << new_kwargs_abs->ToString();
  (void)py_obj_to_node_.emplace(kwargs.ptr(), para);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(kwargs));
  return true;
}

bool FuncGraphBuilder::AddTopGraphInputs(std::vector<py::object> packed_inputs) {
  constexpr size_t args_index = 0;
  constexpr size_t vargs_index = 1;
  constexpr size_t kwargs_index = 2;
  constexpr size_t packed_inputs_size = 3;
  if (!prev_builders_.empty()) {
    MS_LOG(INFO) << "Current builder has prev builder, add top graph parameter failed.";
    return false;
  }
  if (packed_inputs.size() != packed_inputs_size) {
    MS_LOG(INFO) << "Top graph packed inputs size is not three but " << packed_inputs.size()
                 << ", add top graph parameter failed.";
    return false;
  }
  if (!AddTopGraphArgsInputs(packed_inputs[args_index])) {
    MS_LOG(INFO) << "Add top graph args inputs failed.";
    return false;
  }
  if (!AddTopGraphVargsInputs(packed_inputs[vargs_index])) {
    MS_LOG(INFO) << "Add top graph vargs inputs failed";
    return false;
  }
  if (!AddTopGraphKwargsInputs(packed_inputs[kwargs_index])) {
    MS_LOG(INFO) << "Add top graph kwargs inputs failed";
    return false;
  }
  MS_LOG(INFO) << "Add top graph inputs success.";
  return true;
}

py::object FuncGraphBuilder::AddSubGraphInput(const py::object &obj) {
  MS_LOG(INFO) << "Try add sub graph parameter for object: " << std::string(py::str(obj)) << "  " << obj.ptr();
  AbstractBasePtr abs = nullptr;
  auto node = GetNodeByObject(obj);
  if (node != nullptr) {
    abs = node->abstract();
  }
  // Handle constant subgraph input.
  if (abs == nullptr && IsConstant(obj)) {
    auto value = ConvertPyObjToValue(obj);
    if (value != nullptr) {
      abs = abstract::ToAbstract(value, nullptr, nullptr);
    }
  }
  if (abs == nullptr) {
    MS_LOG(INFO) << "Failed to add input for python object: " << std::string(py::str(obj)) << "  " << obj.ptr();
    return py::object();
  }
  auto para = graph_->add_parameter();
  para->set_abstract(abs);
  para->set_is_top_graph_param(false);
  MS_LOG(INFO) << "Add input success, node: " << para->DebugString() << " obj: " << py::str(obj) << "  " << obj.ptr();
  (void)py_obj_to_node_.emplace(obj.ptr(), para);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(obj));
  return obj;
}

py::object FuncGraphBuilder::AddNode(const py::object &callable_obj, const std::vector<py::object> &inputs_obj) {
  if (!CheckCallable(callable_obj)) {
    MS_LOG(INFO) << "The python obj " << py::str(callable_obj) << " is not callable.";
    return py::object();
  }
  auto callable_value = ConvertPyObjToValue(callable_obj);
  if (callable_value == nullptr) {
    MS_LOG(INFO) << "Convert python object " << py::str(callable_obj) << " to value failed.";
    return py::object();
  }
  if (FunctionShouldBeParseInAst(callable_obj)) {
    return TryToAddNode(callable_value, inputs_obj);
  }
  return AddNode(callable_value, inputs_obj);
}

bool FuncGraphBuilder::AddAttrPythonObject(const py::object &object) {
  if (object.ptr() == nullptr) {
    MS_LOG(INFO) << "Convert python object with empty object, convert failed.";
    return false;
  }
  // Attribute object is constant or Parameter, do not need to check constant.
  auto node = ConvertObjToNode(object);
  if (node == nullptr) {
    MS_LOG(INFO) << "Convert python object " << py::str(object) << " to anf node failed.";
    return false;
  }
  node->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(object));
  (void)py_obj_to_node_.emplace(object.ptr(), node);
  return true;
}

bool FuncGraphBuilder::GetInputNodesAndAbstracts(const ValuePtr &callable_value, const vector<py::object> &inputs_obj,
                                                 std::vector<AnfNodePtr> *input_node_list,
                                                 std::vector<AbstractBasePtr> *input_abs_list) {
  input_node_list->reserve(inputs_obj.size() + 1);
  input_abs_list->reserve(inputs_obj.size());

  (void)input_node_list->emplace_back(NewValueNode(callable_value));
  for (const auto &input_obj : inputs_obj) {
    if (input_obj.ptr() == nullptr) {
      MS_LOG(INFO) << "The input python object of " << callable_value->ToString() << ", is NULL";
      return false;
    }
    // Node with input of generator may cause change of generator, skip it in build node now.
    if (PyGen_CheckExact(input_obj.ptr())) {
      MS_LOG(INFO) << "The input python object is generator " << std::string(py::str(input_obj))
                   << ", do not build graph.";
      return false;
    }
    auto node = GetNodeByObject(input_obj);
    if (node == nullptr) {
      if (!IsConstant(input_obj)) {
        MS_LOG(INFO) << "Can not convert non-constant value to value node for obj: " << py::str(input_obj);
        return false;
      }
      auto new_node = ConvertObjToNode(input_obj);
      if (new_node == nullptr) {
        MS_LOG(INFO) << "Convert input python object " << py::str(input_obj) << " to anf node failed.";
        return false;
      }
      new_node->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(input_obj));
      (void)py_obj_to_node_.emplace(input_obj.ptr(), new_node);
      (void)input_node_list->emplace_back(new_node);
      (void)input_abs_list->emplace_back(new_node->abstract());
      MS_LOG(INFO) << "Add python input " << py::str(input_obj) << " with new node " << new_node->DebugString();
    } else {
      (void)input_node_list->emplace_back(node);
      (void)input_abs_list->emplace_back(node->abstract());
    }
  }
  return true;
}

CNodePtr FuncGraphBuilder::DoPrimitiveInferAndCheck(const PrimitivePtr &primitive,
                                                    const AnfNodePtrList &input_node_list,
                                                    const AbstractBasePtrList &args_abs_list) {
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    const CNodePtr &new_node = AddPrimitiveCNode(primitive, input_node_list, args_abs_list);
    if (new_node == nullptr) {
      MS_LOG(INFO) << "Failed to add CNode for Primitive: " << primitive->name();
      return nullptr;
    }

    const AbstractBasePtr &abs = GetAbstractOf(new_node);

    if (!CheckCallable(primitive, abs)) {
      MS_LOG(INFO) << "Check callable failed for Primitive: " << primitive->name();
      return nullptr;
    }
    new_node->set_abstract(abs);
    return new_node;
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to infer Primitive: " << primitive->name() << ". The exception:\n" << e.what();
    return nullptr;
  }
}

CNodePtr FuncGraphBuilder::AddPrimitiveCNode(const PrimitivePtr &primitive, const AnfNodePtrList &input_node_list,
                                             const AbstractBasePtrList &args_abs_list) {
  auto op_def = mindspore::ops::GetOpDef(primitive->name());

  if (op_def == nullptr) {
    if (primitive->has_signature()) {
      // Follow the implementations in DoSignatureEvaluator
      AnfNodePtrList args_node_list(input_node_list.cbegin() + 1, input_node_list.cend());
      AnfNodePtrList new_node_list =
        prim::GetNewInputsBySignatures(graph_, primitive->ToString(), primitive, args_abs_list, args_node_list);

      new_node_list.insert(new_node_list.begin(), input_node_list[0]);
      return graph_->NewCNodeInOrder(new_node_list);
    }
  } else if (primitive->isa<PrimitivePy>()) {
    // Follow the implementations in PrimitiveArgsToInputsEvaluator and DoTransPrimitiveFunctionEvaluator
    auto arg_signatures = op_def->signatures_;
    primitive->set_signatures(arg_signatures);
    primitive->set_has_signature(!arg_signatures.empty());

    const AnfNodePtrList &init_args = abstract::GetPrimitiveInitArgs(primitive->cast<PrimitivePyPtr>(), op_def);

    AnfNodePtrList call_args(input_node_list.cbegin() + 1, input_node_list.cend());
    AbstractBasePtrList call_abs_list;
    (void)std::transform(call_args.cbegin(), call_args.cend(), std::back_inserter(call_abs_list),
                         [](const AnfNodePtr &node) { return FuncGraphBuilder::GetAbstractOf(node); });
    const AnfNodePtrList &new_call_args =
      prim::GetNewInputsBySignatures(graph_, primitive->name(), primitive, call_abs_list, call_args);

    return abstract::GeneratePrimitiveCNode(
      primitive, op_def, graph_, init_args, new_call_args,
      [](const AnfNodePtr &node) { return FuncGraphBuilder::GetAbstractOf(node); });
  }
  MS_LOG(DEBUG) << "Primitive " << primitive->name() << " no need to process signatures and OpDef";
  return graph_->NewCNodeInOrder(input_node_list);
}

AbstractBasePtr FuncGraphBuilder::GetAbstractOf(const AnfNodePtr &node) {
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
                       node->set_abstract(FuncGraphBuilder::GetAbstractOf(node));
                     }
                     return node->abstract();
                   });
    return EvalValue(value, abs_list);
  }
  MS_LOG(INFO) << "Unsupported Node type for GetAbstractOf() method, node: " << node->DebugString();
  return nullptr;
}

AbstractBasePtr FuncGraphBuilder::DoInferAndCheck(const ValuePtr &callable_value,
                                                  const vector<AbstractBasePtr> &input_abs_list) {
  auto abs = EvalValue(callable_value, input_abs_list);
  if (abs == nullptr) {
    MS_LOG(DEBUG) << "Eval failed for value: " << callable_value->ToString();
    return nullptr;
  }
  if (!CheckCallable(callable_value, abs)) {
    MS_LOG(DEBUG) << "Check callable failed for value: " << callable_value->ToString() << ", abs: " << abs->ToString();
    return nullptr;
  }
  return abs;
}

py::object FuncGraphBuilder::TryToAddNode(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj) {
  // Collect the input nodes and input abstracts.
  std::vector<AnfNodePtr> input_node_list;
  std::vector<AbstractBasePtr> input_abs_list;
  if (!GetInputNodesAndAbstracts(callable_value, inputs_obj, &input_node_list, &input_abs_list)) {
    return py::object();
  }

  CNodePtr new_node;
  AbstractBasePtr abs;
  if (callable_value->isa<Primitive>()) {
    new_node = DoPrimitiveInferAndCheck(callable_value->cast<PrimitivePtr>(), input_node_list, input_abs_list);
    if (new_node != nullptr) {
      abs = new_node->abstract();
    }
  } else {
    // Do infer and check callable.
    abs = DoInferAndCheck(callable_value, input_abs_list);
    if (abs != nullptr) {
      new_node = graph_->NewCNodeInOrder(input_node_list);
    }
  }
  if (new_node == nullptr || abs == nullptr) {
    return py::object();
  }

  // Return the converted python object.
  py::object output_py_obj;
  if (abs->isa<abstract::FuncGraphAbstractClosure>()) {
    auto abs_func = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
    auto fg = abs_func->func_graph();
    if (fg == nullptr) {
      return py::object();
    }
    auto obj = fg->python_obj();
    if (obj == nullptr || !obj->isa<parse::PyObjectWrapper>()) {
      return py::object();
    }
    output_py_obj = obj->cast_ptr<parse::PyObjectWrapper>()->obj();
  } else {
    auto convert_func = [this](const py::object &tensor) { return ConvertToPyTensorOrParameter(tensor); };
    output_py_obj = ConvertToPyObj(abs, convert_func);
    if (output_py_obj.ptr() == nullptr) {
      MS_LOG(INFO) << "Convert abs " << abs->ToString() << " to python object failed.";
      return py::object();
    }
  }

  new_node->set_abstract(abs);
  MS_LOG(INFO) << "Add node: " << new_node->DebugString() << " for python object: " << py::str(output_py_obj);
  (void)py_obj_to_node_.emplace(output_py_obj.ptr(), new_node);
  new_node->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(output_py_obj));
  return output_py_obj;
}

py::object FuncGraphBuilder::ConvertToPyTensorOrParameter(const py::object &cpp_tensor) {
  if (cpp_tensor.ptr() == nullptr || !py::isinstance<tensor::Tensor>(cpp_tensor)) {
    return py::object();
  }
  auto tensor = py::cast<tensor::TensorPtr>(cpp_tensor);
  if (tensor->is_parameter()) {
    const std::string &name = tensor->param_info()->name();
    for (auto &it : py_obj_to_node_) {
      if (it.second == nullptr) {
        continue;
      }
      const AbstractBasePtr &abs = it.second->abstract();
      if (abs != nullptr && abs->isa<abstract::AbstractRefTensor>()) {
        auto abs_ref_tensor = abs->cast<abstract::AbstractRefPtr>();
        if (abs_ref_tensor->ref_key_value()->ToString() == name) {
          return py::reinterpret_borrow<py::object>(it.first);
        }
      }
    }
    MS_LOG(INFO) << "Python Parameter not found: " << name;
    return py::object();
  }

  return ConvertCppTensorToPyTensor(cpp_tensor);
}

py::object FuncGraphBuilder::AddNode(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj) {
  if (!callable_value->ToAbstract()->isa<abstract::AbstractFunction>()) {
    MS_LOG(INFO) << "The value " << callable_value->ToString() << " is not callable.";
    return py::object();
  }
  if (callable_value->isa<FuncGraph>()) {
    return AddFgCallNode(callable_value->cast<FuncGraphPtr>(), inputs_obj);
  }
  return TryToAddNode(callable_value, inputs_obj);
}

py::object FuncGraphBuilder::AddMultiNode(const std::string &name, const std::vector<py::object> &inputs_obj) {
  const std::string mod_str = "mindspore.ops.composite.multitype_ops";
  py::module mod = py::module::import(mod_str.c_str());
  if (!py::hasattr(mod, name.c_str())) {
    MS_LOG(INFO) << "Fail to find multitype function graph for name " << name;
    return py::object();
  }
  py::object fn = mod.attr(name.c_str());
  return AddNode(fn, inputs_obj);
}

bool FuncGraphBuilder::AddOutput(const py::object &output_obj, bool is_top_graph) {
  auto iter = py_obj_to_node_.find(output_obj.ptr());
  if (iter == py_obj_to_node_.end()) {
    MS_LOG(INFO) << "The output python object " << py::str(output_obj) << " should have been added to the graph.";
    return false;
  }
  auto node = iter->second;
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  // Only top graph has restriction on return value type.
  if (is_top_graph && !CheckGraphOutput(abs)) {
    MS_LOG(INFO) << "The output python object " << py::str(output_obj)
                 << " should not be the graph output, abstract: " << (abs == nullptr ? "null" : abs->ToString());
    return false;
  }
  (void)output_nodes_.emplace_back(node);
  return true;
}

FuncGraphPtr FuncGraphBuilder::graph() {
  if (has_set_output_) {
    return graph_;
  }
  if (output_nodes_.empty()) {
    MS_LOG(DEBUG) << "The graph " << graph_->ToString() << " has not been set output.";
    return nullptr;
  }
  bool all_value_node = std::any_of(output_nodes_.begin(), output_nodes_.end(),
                                    [](const AnfNodePtr &node) { return node->isa<ValueNode>(); });
  if (all_value_node) {
    MS_LOG(INFO) << "All graph output is value node, no need to run graph.";
    return nullptr;
  }
  // Single output case.
  if (output_nodes_.size() == 1) {
    // Use the python obj of the output node as the python obj of the func_graph output.
    auto node_output_py_obj = output_nodes_[0]->user_data<py::object>(kPiJitPyObjKey);
    if (node_output_py_obj == nullptr) {
      MS_LOG(DEBUG) << "Can not find the python object of the node " << output_nodes_[0]->DebugString();
      return nullptr;
    }
    graph_->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(*node_output_py_obj));
    graph_->set_output(output_nodes_[0]);
    has_set_output_ = true;
    return graph_;
  }
  // multiple output case.
  // Make the python tuple obj of the output nodes as the python obj of the func_graph output.
  py::tuple output_py_obj(output_nodes_.size());
  for (size_t i = 0; i < output_nodes_.size(); ++i) {
    auto node_output_py_obj = output_nodes_[i]->user_data<py::object>(kPiJitPyObjKey);
    if (node_output_py_obj == nullptr) {
      MS_LOG(DEBUG) << "Can not find the python object of the node " << output_nodes_[i]->DebugString();
      return nullptr;
    }
    output_py_obj[i] = *node_output_py_obj;
  }
  // Create make_tuple node and set its abstract.
  graph_->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(output_py_obj));
  output_nodes_.insert(output_nodes_.begin(), NewValueNode(prim::kPrimMakeTuple));
  AbstractBasePtrList abstract_list;
  (void)std::transform(output_nodes_.begin() + 1, output_nodes_.end(), std::back_inserter(abstract_list),
                       [](const AnfNodePtr &node) -> AbstractBasePtr { return node->abstract(); });
  auto output_node = graph_->NewCNodeInOrder(output_nodes_);
  auto fg_output_abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  output_node->set_abstract(fg_output_abs);

  graph_->set_output(output_node);
  has_set_output_ = true;
  return graph_;
}

void FuncGraphBuilder::ClearNodeAbstract() {
  if (!has_set_output_) {
    MS_LOG(INTERNAL_EXCEPTION) << "Graph not generated, can not clear abstract.";
  }
  // Clear all node abstract.
  auto mng = Manage(graph_, false);
  MS_EXCEPTION_IF_NULL(mng);
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  for (const auto &node : mng->all_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    const AbstractBasePtr &prev_inferred = node->abstract();
    auto is_func =
      node->isa<mindspore::ValueNode>() && prev_inferred != nullptr && prev_inferred->isa<abstract::AbstractFunction>();
    // Keep previous inferred value for parameter and ValueNode if the inferred value is not AbstractFunction.
    if (!node->isa<Parameter>() && !is_func) {
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

py::object FuncGraphBuilder::AddFgCallNode(const FuncGraphPtr &fg, const vector<py::object> &inputs_obj) {
  std::vector<AnfNodePtr> input_node_list;
  input_node_list.reserve(inputs_obj.size() + 1);

  (void)input_node_list.emplace_back(NewValueNode(fg));
  for (const auto &input_obj : inputs_obj) {
    auto node = GetNodeByObject(input_obj);
    if (node == nullptr) {
      if (!IsConstant(input_obj)) {
        MS_LOG(INFO) << "Can not convert non-constant value to value node for obj: " << py::str(input_obj);
        return py::object();
      }
      auto new_node = ConvertObjToNode(input_obj);
      if (new_node == nullptr) {
        MS_LOG(INFO) << "Convert input python object " << py::str(input_obj) << " to anf node failed.";
        return py::object();
      }
      new_node->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(input_obj));
      (void)py_obj_to_node_.emplace(input_obj.ptr(), new_node);
      (void)input_node_list.emplace_back(new_node);
      MS_LOG(DEBUG) << "Add constant python input " << py::str(input_obj) << " with node " << new_node->DebugString();
    } else {
      (void)input_node_list.emplace_back(node);
    }
  }

  auto new_node = graph_->NewCNodeInOrder(input_node_list);
  auto fg_output = fg->output();
  MS_EXCEPTION_IF_NULL(fg_output);
  auto fg_output_abs = fg_output->abstract();
  MS_EXCEPTION_IF_NULL(fg_output_abs);
  new_node->set_abstract(fg_output_abs);

  // Use the python obj of the func_graph output as the python obj of the output node.
  auto fg_output_obj_ptr = fg->user_data<py::object>(kPiJitPyObjKey);
  if (fg_output_obj_ptr == nullptr) {
    MS_LOG(DEBUG) << "Can not find the output python object of func_graph " << fg->ToString();
    return py::object();
  }
  auto fg_output_obj = *fg_output_obj_ptr;
  (void)py_obj_to_node_.emplace(fg_output_obj.ptr(), new_node);
  new_node->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(fg_output_obj));
  return fg_output_obj;
}

bool FuncGraphBuilder::CheckCallable(const py::object &obj) {
  constexpr auto ms_class_attr = "__ms_class__";
  return py::isinstance<MetaFuncGraph>(obj) ||
         (py::hasattr(obj, PYTHON_PRIMITIVE_FLAG) &&
          parse::data_converter::GetObjType(obj) != parse::RESOLVE_TYPE_CLASS_TYPE) ||
         FunctionShouldBeParseInAst(obj) ||
         (py::hasattr(obj, ms_class_attr) && py::cast<bool>(py::getattr(obj, ms_class_attr)));
}

py::object FuncGraphBuilder::ConvertMethod(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::tuple method_info = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_METHOD_INFO, obj);
  py::object class_name_obj = method_info[0];
  if (py::isinstance<py::none>(class_name_obj)) {
    MS_LOG(INFO) << "Can not get the method info of " << py::str(obj);
    return py::object();
  }
  auto class_name = class_name_obj.cast<std::string>();
  if (class_name == "Tensor" &&
      !py::cast<bool>(python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_IS_MS_TENSOR_METHOD, obj))) {
    return py::object();
  }
  auto type_id = GetTypeIdFromClassName(class_name);
  auto method_name = method_info[1].cast<std::string>();
  MS_LOG(DEBUG) << "type_id: " << type_id << ", method_name: " << method_name;
  Any require = pipeline::Resource::GetMethodPtr(type_id, method_name);
  if (require.empty()) {
    require = pipeline::Resource::GetAttrPtr(type_id, method_name);
  }

  if (require.empty()) {
    MS_LOG(DEBUG) << "Can not find the method registered.";
    return py::object();
  }

  if (require.is<std::string>()) {
    py::function fn = mindspore::python_adapter::GetPyFn(parse::kStandardMethodModelName, require.cast<std::string>());
    if (py::isinstance<py::none>(fn)) {
      MS_LOG(DEBUG) << "Can not find the method '" << require.cast<std::string>() << "' defined in standard_method.";
      return py::object();
    }
    return fn;
  } else if (require.is<PrimitivePtr>()) {
    auto ops_mod = python_adapter::GetPyModule("mindspore.ops");
    auto primitive_class = python_adapter::GetPyObjAttr(ops_mod, "Primitive");
    return primitive_class(require.cast<PrimitivePtr>()->name());
  }
  MS_LOG(DEBUG) << "The method or attr should be a string or a Primitive, but got " << require.ToString();
  return py::object();
}

void FuncGraphBuilder::RemoveOutput(const py::object &output_obj) {
  auto iter = py_obj_to_node_.find(output_obj.ptr());
  if (iter == py_obj_to_node_.end()) {
    MS_LOG(WARNING) << "The output python object " << py::str(output_obj) << " should have been added to the graph.";
    return;
  }
  auto output_nodes_iter = std::find(output_nodes_.begin(), output_nodes_.end(), iter->second);
  if (output_nodes_iter == output_nodes_.end()) {
    MS_LOG(WARNING) << "The node " << iter->second->DebugString() << " has not been added to the graph outputs.";
    return;
  }
  output_nodes_.erase(output_nodes_iter);
}

py::object FuncGraphBuilder::ConvertFunction(const py::object &obj) {
  auto dict = python_adapter::GetPyObjAttr(python_adapter::GetPyModule("mindspore._extends.parse.resources"),
                                           "convert_object_map");
  auto callable_obj_ptr = PyDict_GetItem(dict.ptr(), obj.ptr());
  return callable_obj_ptr == nullptr ? py::object() : py::cast<py::object>(callable_obj_ptr);
}

bool FuncGraphBuilder::CanConstantFoldFunc(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::object can_constant_fold = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_CAN_CONSTANT_FOLD, obj);
  return can_constant_fold.cast<bool>();
}

void FuncGraphBuilder::SetGraphName(const std::string &name) {
  if (name.empty()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph_->debug_info());
  graph_->debug_info()->set_name(name);
}

void FuncGraphBuilder::AddPrevBuilder(const FuncGraphBuilderPtr &builder) { prev_builders_.push_back(builder.get()); }

bool FuncGraphBuilder::ValidateCallableObject(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    return false;
  }
  // Check if object is invalid method for CellList/CellDict, which should not be converted to graph.
  if (CheckInvalidCellListDictMethod(obj)) {
    MS_LOG(INFO) << "The object " << py::str(obj) << " is a invalid CellList/CellDict method, "
                 << "can not convert to graph";
    return false;
  }
  return true;
}

bool FuncGraphBuilder::CheckInvalidCellListDictMethod(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::tuple method_info = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_METHOD_INFO, obj);
  constexpr size_t class_index = 0;
  constexpr size_t method_index = 1;
  py::object class_name_obj = method_info[class_index];
  if (class_name_obj.ptr() == nullptr || py::isinstance<py::none>(class_name_obj)) {
    return false;
  }
  auto class_name = class_name_obj.cast<std::string>();
  MS_LOG(INFO) << "class name: " << class_name;
  if (class_name != "CellList" && class_name != "CellDict") {
    return false;
  }
  auto method_name_obj = method_info[method_index];
  if (method_name_obj.ptr() == nullptr || py::isinstance<py::none>(method_name_obj)) {
    return false;
  }
  auto method_name = method_name_obj.cast<std::string>();
  static std::vector<std::string> inplace_method_name = {"clear", "update"};
  if (std::any_of(inplace_method_name.begin(), inplace_method_name.end(),
                  [&method_name](const std::string &name) { return name == method_name; })) {
    MS_LOG(INFO) << "CellDict/CellList inplace function " << method_name << " found";
    return true;
  }
  auto type_id = GetTypeIdFromClassName(class_name);
  Any require = pipeline::Resource::GetMethodPtr(type_id, method_name);
  return require.empty();
}
}  // namespace mindspore
