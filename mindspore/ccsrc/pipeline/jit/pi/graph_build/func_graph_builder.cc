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
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "ops/arithmetic_ops.h"
#include "ops/structure_ops.h"
#include "include/common/utils/convert_utils_py.h"
#include "ir/tensor.h"

namespace mindspore {
namespace {
bool ShouldFallBackInRuntime(const PrimitivePtr &prim) {
  static HashSet<std::string> prims_should_fallback_in_runtime = {kListInplaceExtendOpName,
                                                                  kListInplaceInsertOpName,
                                                                  kListInplacePopOpName,
                                                                  kListInplaceReverseOpName,
                                                                  kListInplaceClearOpName,
                                                                  kDictInplaceSetItemOpName,
                                                                  kRaiseOpName,
                                                                  kMakeSliceOpName,
                                                                  kJoinedStrOpName,
                                                                  kFormatOpName};
  return prims_should_fallback_in_runtime.find(prim->name()) != prims_should_fallback_in_runtime.end();
}

bool IsValidScalar(const AbstractBasePtr &abs) {
  auto build_value = abs->BuildValue();
  return build_value->isa<StringImm>() || build_value->isa<BoolImm>() || build_value->isa<IntegerImm>() ||
         build_value->isa<FloatImm>();
}

bool Mutable(const py::object &obj, const ValuePtr &value) {
  // If a tensor has been set const arg, it should not be mutable.
  if (value->isa<tensor::MetaTensor>()) {
    constexpr char const_arg_attr[] = "const_arg";
    if (py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr))) {
      return false;
    }
  }
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

bool TensorArgMutable(const py::object &obj, const ValuePtr &value) {
  if (!value->isa<tensor::MetaTensor>()) {
    return false;
  }
  constexpr char const_arg_attr[] = "const_arg";
  return !py::hasattr(obj, const_arg_attr) || !py::cast<bool>(py::getattr(obj, const_arg_attr));
}

TypeId GetTypeIdFromClassName(const std::string &class_name) {
  static HashMap<std::string, TypeId> class_name_to_type_ids = {
    {"Tensor", kObjectTypeTensorType}, {"list", kObjectTypeList}, {"tuple", kObjectTypeTuple}};
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
  if (build_value == kValueAny && abs->isa<abstract::AbstractTensor>()) {
    auto abs_tensor = abs->cast<abstract::AbstractTensorPtr>();
    TypePtr tensor_type_ptr = abs_tensor->element()->BuildType();
    ShapeVector tensor_shape = abs_tensor->shape()->shape();
    return std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
  }
  // To add dict.
  return build_value;
}
}  // namespace

ValuePtr FuncGraphBuilder::ConvertPyObjToValue(const py::object &obj) {
  ValuePtr ret = nullptr;
  try {
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
  if (abs->isa<abstract::AbstractNone>()) {
    return py::none();
  }

  auto build_value = MaybeMakeEmptyTensor(abs);
  auto py_obj = ValueToPyData(build_value, abs);
  // Return none means failed converting.
  if (py::isinstance<py::none>(py_obj)) {
    return py::object();
  }
  return py_obj;
}

AbstractBasePtr FuncGraphBuilder::EvalValue(const ValuePtr &value, const AbstractBasePtrList &inputs_abs_list) {
  if (value == nullptr) {
    return nullptr;
  }
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

py::object FuncGraphBuilder::AddInput(const py::object &obj) {
  auto value = ConvertPyObjToValue(obj);
  if (value == nullptr) {
    return py::object();
  }
  bool broaden = TensorArgMutable(obj, value) || Mutable(obj, value) || value->isa<tensor::MetaSparseTensor>();
  auto abs = abstract::ToAbstract(value, nullptr, nullptr);
  if (broaden) {
    abs = AbstractBroaden(abs);
  }
  auto para = graph_->add_parameter();
  para->set_abstract(abs);
  (void)converted_py_obj_.emplace(obj.ptr(), para);
  return obj;
}

py::object FuncGraphBuilder::AddNode(const py::object &callable_obj, const std::vector<py::object> &inputs_obj) {
  if (!CheckCallable(callable_obj)) {
    MS_LOG(ERROR) << "The python obj " << py::str(callable_obj) << " is not callable.";
  }
  auto callable_value = ConvertPyObjToValue(callable_obj);
  if (callable_value == nullptr) {
    MS_LOG(ERROR) << "Convert python object " << py::str(callable_obj) << " to value failed.";
    return py::object();
  }
  return AddNode(callable_value, inputs_obj);
}

bool FuncGraphBuilder::GetInputNodesAndAbstracts(const ValuePtr &callable_value, const vector<py::object> &inputs_obj,
                                                 std::vector<AnfNodePtr> *input_node_list,
                                                 std::vector<AbstractBasePtr> *input_abs_list) {
  input_node_list->reserve(inputs_obj.size() + 1);
  input_abs_list->reserve(inputs_obj.size());

  (void)input_node_list->emplace_back(NewValueNode(callable_value));
  for (const auto &input_obj : inputs_obj) {
    auto iter = converted_py_obj_.find(input_obj.ptr());
    if (iter == converted_py_obj_.end()) {
      MS_LOG(ERROR) << "The input python object " << py::str(input_obj) << " should have been add to the graph inputs.";
      return false;
    }
    (void)input_node_list->emplace_back(iter->second);
    (void)input_abs_list->emplace_back(iter->second->abstract());
  }
  return true;
}

AbstractBasePtr FuncGraphBuilder::DoInferAndCheck(const ValuePtr &callable_value,
                                                  const vector<AbstractBasePtr> &input_abs_list) {
  auto abs = EvalValue(callable_value, input_abs_list);
  if (abs == nullptr) {
    MS_LOG(ERROR) << "Eval failed for value: " << callable_value->ToString();
    return nullptr;
  }
  if (!CheckCallable(callable_value, abs)) {
    MS_LOG(ERROR) << "Check callable failed for value: " << callable_value->ToString() << ", abs: " << abs->ToString();
    return nullptr;
  }
  return abs;
}

py::object FuncGraphBuilder::AddNode(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj) {
  if (!callable_value->ToAbstract()->isa<abstract::AbstractFunction>()) {
    MS_LOG(ERROR) << "The value " << callable_value->ToString() << " is not callable.";
    return py::object();
  }
  if (callable_value->isa<FuncGraph>()) {
    return AddFgCallNode(callable_value->cast<FuncGraphPtr>(), inputs_obj);
  }
  // Collect the input nodes and input abstracts.
  std::vector<AnfNodePtr> input_node_list;
  std::vector<AbstractBasePtr> input_abs_list;
  if (!GetInputNodesAndAbstracts(callable_value, inputs_obj, &input_node_list, &input_abs_list)) {
    return py::object();
  }

  // Do infer and check callable.
  auto abs = DoInferAndCheck(callable_value, input_abs_list);
  if (abs == nullptr) {
    return py::object();
  }

  auto new_node = graph_->NewCNode(input_node_list);
  // Return the converted python object.
  py::object output_py_obj = ConvertToPyObj(abs);
  if (output_py_obj.ptr() == nullptr) {
    MS_LOG(ERROR) << "Convert abs " << abs->ToString() << " to python object failed.";
    return py::object();
  }

  new_node->set_abstract(abs);
  (void)converted_py_obj_.emplace(output_py_obj.ptr(), new_node);
  return output_py_obj;
}

bool FuncGraphBuilder::AddOutput(const py::object &output_obj) {
  auto iter = converted_py_obj_.find(output_obj.ptr());
  if (iter == converted_py_obj_.end()) {
    MS_LOG(ERROR) << "The output python object " << py::str(output_obj) << " should have been add to the graph.";
    return false;
  }
  auto node = iter->second;
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  if (!CheckGraphOutput(abs)) {
    MS_LOG(ERROR) << "The output python object " << py::str(output_obj)
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
  if (output_nodes_.size() == 1) {
    MS_LOG(ERROR) << "The graph " << graph_->ToString() << " has not been set output.";
    return nullptr;
  }
  AbstractBasePtrList abstract_list;
  (void)std::transform(output_nodes_.begin() + 1, output_nodes_.end(), std::back_inserter(abstract_list),
                       [](const AnfNodePtr &node) -> AbstractBasePtr { return node->abstract(); });
  auto output_node = graph_->NewCNode(output_nodes_);
  output_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  graph_->set_output(output_node);
  has_set_output_ = true;
  return graph_;
}

py::object FuncGraphBuilder::AddFgCallNode(const FuncGraphPtr &fg, const vector<py::object> &inputs_obj) {
  std::vector<AnfNodePtr> input_node_list;
  input_node_list.reserve(inputs_obj.size() + 1);

  (void)input_node_list.emplace_back(NewValueNode(fg));
  for (const auto &input_obj : inputs_obj) {
    auto iter = converted_py_obj_.find(input_obj.ptr());
    if (iter == converted_py_obj_.end()) {
      MS_LOG(ERROR) << "The input python object " << py::str(input_obj) << " should have been add to the graph inputs.";
      return py::object();
    }
    (void)input_node_list.emplace_back(iter->second);
  }

  auto new_node = graph_->NewCNode(input_node_list);
  auto fg_output = fg->output();
  MS_EXCEPTION_IF_NULL(fg_output);
  auto fg_output_abs = fg_output->abstract();
  MS_EXCEPTION_IF_NULL(fg_output_abs);
  // Return the converted python object.
  py::object output_py_obj = ConvertToPyObj(fg_output_abs);
  if (output_py_obj.ptr() == nullptr) {
    MS_LOG(ERROR) << "Convert abs " << fg_output_abs->ToString() << " to python object failed.";
    return py::object();
  }

  new_node->set_abstract(fg_output_abs);
  (void)converted_py_obj_.emplace(output_py_obj.ptr(), new_node);
  return output_py_obj;
}

bool FuncGraphBuilder::CheckCallable(const py::object &obj) {
  return py::hasattr(obj, PYTHON_PRIMITIVE_FLAG) || py::isinstance<MetaFuncGraph>(obj);
}

Any FuncGraphBuilder::ConvertMethod(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::tuple method_info = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_METHOD_INFO, obj);
  py::object class_name_obj = method_info[0];
  if (py::isinstance<py::none>(class_name_obj)) {
    return py::object();
  }
  auto type_id = GetTypeIdFromClassName(class_name_obj.cast<std::string>());
  auto method_name = method_info[0].cast<std::string>();
  Any require = pipeline::Resource::GetMethodPtr(type_id, method_name);
  if (require.empty()) {
    require = pipeline::Resource::GetAttrPtr(type_id, method_name);
  }
  return require;
}

py::object FuncGraphBuilder::GetStandardMethod(const string &func_name) {
  py::function fn = mindspore::python_adapter::GetPyFn(parse::kStandardMethodModelName, func_name);
  if (py::isinstance<py::none>(fn)) {
    return py::object();
  }
  return fn;
}
}  // namespace mindspore
