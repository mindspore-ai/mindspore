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
#include <unordered_set>
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "ops/arithmetic_ops.h"
#include "ops/structure_ops.h"
#include "include/common/utils/convert_utils_py.h"
#include "ir/tensor.h"

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
  static HashMap<std::string, TypeId> class_name_to_type_ids = {{"Tensor", kObjectTypeTensorType},
                                                                {"list", kObjectTypeList},
                                                                {"tuple", kObjectTypeTuple},
                                                                {"int", kNumberTypeInt},
                                                                {"float", kNumberTypeFloat}};
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
    return std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
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

py::object ConvertToPythonTensor(const py::object &obj) {
  if (py::isinstance<tensor::Tensor>(obj)) {
    bool is_adapter_tensor = py::hasattr(obj, kAdapterFlag) && py::cast<bool>(py::getattr(obj, kAdapterFlag));
    py::module mod = python_adapter::GetPyModule(kTensorModule);
    auto py_tensor = python_adapter::CallPyModFn(mod, "Tensor", obj, py::none(), py::none(), py::none(), true);
    if (is_adapter_tensor) {
      mod = python_adapter::GetPyModule(kInnerOpsModule);
      py_tensor = python_adapter::CallPyModFn(mod, "convert_to_adapter_tensor", py_tensor);
    }
    return py_tensor;
  }
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    auto obj_tuple = py::cast<py::tuple>(obj);
    py::tuple ret(obj_tuple.size());
    for (size_t i = 0; i < obj_tuple.size(); ++i) {
      ret[i] = ConvertToPythonTensor(obj_tuple[i]);
    }
    if (py::isinstance<py::list>(obj)) {
      return ret.cast<py::list>();
    }
    return ret;
  }
  return obj;
}
}  // namespace

ValuePtr FuncGraphBuilder::ConvertPyObjToValue(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    return nullptr;
  }
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

  return ConvertToPythonTensor(py_obj);
}

AbstractBasePtr FuncGraphBuilder::EvalValue(const ValuePtr &value, const AbstractBasePtrList &inputs_abs_list) {
  if (value == nullptr) {
    return nullptr;
  }
  try {
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
    MS_LOG(ERROR) << "Failed to EvalValue for value: " << value->ToString();
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
  (void)py_obj_to_node_.emplace(obj.ptr(), para);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(obj));
  return obj;
}

py::object FuncGraphBuilder::AddNode(const py::object &callable_obj, const std::vector<py::object> &inputs_obj) {
  if (!CheckCallable(callable_obj)) {
    MS_LOG(ERROR) << "The python obj " << py::str(callable_obj) << " is not callable.";
    return py::object();
  }
  auto callable_value = ConvertPyObjToValue(callable_obj);
  if (callable_value == nullptr) {
    MS_LOG(ERROR) << "Convert python object " << py::str(callable_obj) << " to value failed.";
    return py::object();
  }
  if (FunctionShouldBeParseInAst(callable_obj)) {
    return TryToAddNode(callable_value, inputs_obj);
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
    auto iter = py_obj_to_node_.find(input_obj.ptr());
    if (iter == py_obj_to_node_.end()) {
      auto val = ConvertPyObjToValue(input_obj);
      if (val == nullptr) {
        MS_LOG(ERROR) << "The input object " << py::str(input_obj) << " convert to value failed.";
        return false;
      }
      // Constant value input scene, the object should be converted to value node.
      auto node = NewValueNode(val);
      auto abs = val->ToAbstract();
      node->set_abstract(abs);
      node->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(input_obj));
      (void)py_obj_to_node_.emplace(input_obj.ptr(), node);
      (void)input_node_list->emplace_back(node);
      (void)input_abs_list->emplace_back(abs);
      MS_LOG(DEBUG) << "Add constant python input " << py::str(input_obj) << " with node " << node->DebugString();
    } else {
      (void)input_node_list->emplace_back(iter->second);
      (void)input_abs_list->emplace_back(iter->second->abstract());
    }
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

py::object FuncGraphBuilder::TryToAddNode(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj) {
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
  (void)py_obj_to_node_.emplace(output_py_obj.ptr(), new_node);
  new_node->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(output_py_obj));
  return output_py_obj;
}

py::object FuncGraphBuilder::AddNode(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj) {
  if (!callable_value->ToAbstract()->isa<abstract::AbstractFunction>()) {
    MS_LOG(ERROR) << "The value " << callable_value->ToString() << " is not callable.";
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
    MS_LOG(ERROR) << "Fail to find multitype function graph for name " << name;
    return py::object();
  }
  py::object fn = mod.attr(name.c_str());
  return AddNode(fn, inputs_obj);
}

bool FuncGraphBuilder::AddOutput(const py::object &output_obj, bool add_repeat) {
  auto iter = py_obj_to_node_.find(output_obj.ptr());
  if (iter == py_obj_to_node_.end()) {
    MS_LOG(ERROR) << "The output python object " << py::str(output_obj) << " should have been added to the graph.";
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
  if (!add_repeat) {
    auto iter = std::find(output_nodes_.begin(), output_nodes_.end(), node);
    if (iter != output_nodes_.end()) {
      MS_LOG(DEBUG) << "Output node " << node->DebugString() << " has already been set as output.";
      return true;
    }
  }
  (void)output_nodes_.emplace_back(node);
  return true;
}

FuncGraphPtr FuncGraphBuilder::graph() {
  if (has_set_output_) {
    return graph_;
  }
  if (output_nodes_.empty()) {
    MS_LOG(ERROR) << "The graph " << graph_->ToString() << " has not been set output.";
    return nullptr;
  }
  // Single output case.
  if (output_nodes_.size() == 1) {
    // Use the python obj of the output node as the python obj of the func_graph output.
    auto node_output_py_obj = output_nodes_[0]->user_data<py::object>(kPiJitPyObjKey);
    if (node_output_py_obj == nullptr) {
      MS_LOG(ERROR) << "Can not find the python object of the node " << output_nodes_[0]->DebugString();
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
      MS_LOG(ERROR) << "Can not find the python object of the node " << output_nodes_[i]->DebugString();
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
  auto output_node = graph_->NewCNode(output_nodes_);
  auto fg_output_abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  output_node->set_abstract(fg_output_abs);

  graph_->set_output(output_node);
  has_set_output_ = true;
  return graph_;
}

void FuncGraphBuilder::EraseUnusedParameter() {
  // Build output for graph.
  if (!has_set_output_) {
    (void)graph();
  }
  const auto &nodes = graph_->TopoSort(graph_->output());
  std::unordered_set<AnfNodePtr> used_params;
  for (auto node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    const auto &cnode_inputs = cnode->inputs();
    (void)std::copy_if(cnode_inputs.begin(), cnode_inputs.end(), std::inserter(used_params, used_params.begin()),
                       [](const AnfNodePtr &input) {
                         return input->isa<Parameter>();
                       });
  }
  std::vector<AnfNodePtr> new_params;
  const auto &origin_params = graph_->parameters();
  (void)std::copy_if(origin_params.begin(), origin_params.end(), std::back_inserter(new_params),
                     [&used_params](const AnfNodePtr param) {
                       return used_params.find(param) != used_params.end();
                     });
  graph_->set_parameters(new_params);
}

py::object FuncGraphBuilder::AddFgCallNode(const FuncGraphPtr &fg, const vector<py::object> &inputs_obj) {
  std::vector<AnfNodePtr> input_node_list;
  input_node_list.reserve(inputs_obj.size() + 1);

  (void)input_node_list.emplace_back(NewValueNode(fg));
  for (const auto &input_obj : inputs_obj) {
    auto iter = py_obj_to_node_.find(input_obj.ptr());
    if (iter == py_obj_to_node_.end()) {
      auto val = ConvertPyObjToValue(input_obj);
      if (val == nullptr) {
        MS_LOG(ERROR) << "The input object " << py::str(input_obj) << " convert to value failed.";
        return py::object();
      }
      // Constant value input scene, the object should be converted to value node.
      auto node = NewValueNode(val);
      auto abs = val->ToAbstract();
      node->set_abstract(abs);
      (void)py_obj_to_node_.emplace(input_obj.ptr(), node);
      (void)input_node_list.emplace_back(node);
      MS_LOG(DEBUG) << "Add constant python input " << py::str(input_obj) << " with node " << node->DebugString();
    } else {
      (void)input_node_list.emplace_back(iter->second);
    }
  }

  auto new_node = graph_->NewCNode(input_node_list);
  auto fg_output = fg->output();
  MS_EXCEPTION_IF_NULL(fg_output);
  auto fg_output_abs = fg_output->abstract();
  MS_EXCEPTION_IF_NULL(fg_output_abs);
  new_node->set_abstract(fg_output_abs);

  // Use the python obj of the func_graph output as the python obj of the output node.
  auto fg_output_obj_ptr = fg->user_data<py::object>(kPiJitPyObjKey);
  if (fg_output_obj_ptr == nullptr) {
    MS_LOG(ERROR) << "Can not find the output python object of func_graph " << fg->ToString();
    return py::object();
  }
  auto fg_output_obj = *fg_output_obj_ptr;
  (void)py_obj_to_node_.emplace(fg_output_obj.ptr(), new_node);
  new_node->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(fg_output_obj));
  return fg_output_obj;
}

bool FuncGraphBuilder::CheckCallable(const py::object &obj) {
  return py::isinstance<MetaFuncGraph>(obj) ||
         (py::hasattr(obj, PYTHON_PRIMITIVE_FLAG) &&
          parse::data_converter::GetObjType(obj) != parse::RESOLVE_TYPE_CLASS_TYPE) ||
         FunctionShouldBeParseInAst(obj);
}

py::object FuncGraphBuilder::ConvertMethod(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::tuple method_info = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_METHOD_INFO, obj);
  py::object class_name_obj = method_info[0];
  if (py::isinstance<py::none>(class_name_obj)) {
    MS_LOG(DEBUG) << "Can not get the method info of " << py::str(obj);
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
  MS_LOG(ERROR) << "The method or attr should be a string or a Primitive, but got " << require.ToString();
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
}  // namespace mindspore
