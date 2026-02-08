/**
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

#include "include/utils/operator/primitive_utils.h"

#include <string>
#include <memory>

#include "ir/primitive.h"
#include "ops/op_def.h"
#include "include/utils/python_adapter.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "utils/base_ref_py.h"
#include "include/utils/convert_utils_py.h"
#include "frontend/operator/primitive_py_utils.h"
#include "utils/operator/auto_generate/functional_signature_map.h"

namespace mindspore {
using OP_DTYPE = mindspore::ops::OP_DTYPE;

py::function GetBpropFunctionByObj(const py::object &obj, bool get_closure) {
  static const std::string get_bprop_fn = "get_bprop_fn";
  static const std::string ad_experimental_module = "mindspore.ops._grad_experimental";
  py::function fn = python_adapter::GetPyFn(ad_experimental_module, get_bprop_fn)(obj, get_closure);
  return fn;
}

py::function GetBpropFunction(const std::string &name) {
  auto fn = GetBpropFunctionByObj(py::str(name));
  return fn;
}

py::function GetComputeFunction(const std::string &name) {
  static const std::string module = "mindspore.graph._parse.builtin_operations";
  py::module mod = py::module::import(common::SafeCStr(module));
  if (!py::hasattr(mod, common::SafeCStr(name))) {
    PyErr_SetString(PyExc_NotImplementedError, common::SafeCStr(name));
    // If raise AttributeError, user can't understand. This case need raise NotImplementedError.
    throw(py::error_already_set());
  }
  py::object fn = mod.attr(common::SafeCStr(name));
  return fn;
}

py::tuple ConvertDatatoPyTuple(const VectorRef &args) {
  auto py_args = py::tuple(args.size());
  size_t i = 0;
  for (auto &arg : args) {
    py_args[i] = BaseRefToPyData(arg);
    MS_LOG(DEBUG) << "arg:" << i << ":" << arg.ToString();
    i++;
  }
  return py_args;
}

py::function GetComputeFunctionWithoutPyObj(const std::string &name) {
  static const std::string vm_module = "mindspore.ops.vm_impl_registry";
  static const std::string get_vm_impl_fn = "get_vm_impl_fn";
  py::function get_fn = python_adapter::GetPyFn(vm_module, get_vm_impl_fn);
  if (py::isinstance<py::none>(get_fn)) {
    MS_LOG(DEBUG) << "Failed to get the function 'get_vm_impl_fn'";
    return py::none();
  }
  py::function vm_fn = get_fn(py::str(name));
  return vm_fn;
}

BaseRef RunComputeFunctionWithoutPyObj(const PrimitivePtr &prim, const VectorRef &args) {
  auto func = GetComputeFunctionWithoutPyObj(prim->name());
  if (py::isinstance<py::none>(func)) {
    return nullptr;
  }
  auto py_args = ConvertDatatoPyTuple(args);
  py::object obj = func(*py_args);
  if (py::isinstance<py::none>(obj)) {
    return nullptr;
  }
  return std::make_shared<PyObjectRef>(obj);
}

BaseRef RunComputeFunction(const PrimitivePtr &prim, const VectorRef &args) {
  auto func = GetComputeFunction(prim->name());
  if (py::isinstance<py::none>(func)) {
    MS_LOG(EXCEPTION) << prim->name() << " 's compute function run failed, please check whether it is not implemented";
  }
  auto py_args = ConvertDatatoPyTuple(args);
  py::object obj = func(*py_args);
  return std::make_shared<PyObjectRef>(obj);
}

namespace prim {
std::string ErrorMessageForConvertRefDtype(const ValuePtr &func, const std::string &ref_type,
                                           const std::string &target_type, size_t index) {
  std::ostringstream buffer;
  if (func->isa<Primitive>()) {
    auto prim = func->cast<PrimitivePtr>();
    auto args_names_value = prim->GetAttr("input_names");
    if (args_names_value != nullptr) {
      auto args_names = GetValue<std::vector<std::string>>(args_names_value);
      if (index < args_names.size()) {
        buffer << " the argument[" << args_names[index] << "]'s data type of primitive[" << prim->name() << "] is ";
      }
    }
  }
  if (buffer.str().empty()) {
    buffer << " so data type ";
  }
  std::ostringstream ss;
  ss << "Data type conversion is not supported for a 'Parameter', nor for the input tensor of an in-place operator,"
     << buffer.str() << ref_type << ", which cannot be converted to data type " << target_type << " automatically.\n";
  return ss.str();
}

std::stringstream BuildApiInputInfo(const std::string &function_name, const std::vector<std::string> &arg_info_list) {
  std::stringstream ss;
  std::string result = std::accumulate(
    arg_info_list.begin(), arg_info_list.end(), std::string(),
    [](const std::string &a, const std::string &b) -> std::string { return a.empty() ? b : a + ", " + b; });
  ss << "Failed calling " << function_name << " with \"" << function_name << "(" << result << ")\".\n";
  ss << "The valid calling should be:\n";
  return ss;
}

std::map<std::string, std::vector<std::string>> GetFunctionalSignatureMap(bool is_method) {
  return is_method ? ops::tensor_method_overload_signature_map : ops::function_overload_signature_map;
}

std::string BuildFunctionalErrorMsg(const std::string &function_name, const std::vector<std::string> &arg_info_list,
                                    bool is_method) {
  std::stringstream ss = BuildApiInputInfo(function_name, arg_info_list);
  const auto &signature_map = GetFunctionalSignatureMap(is_method);
  auto it = signature_map.find(function_name);
  if (it != signature_map.end()) {
    const std::vector<std::string> &valid_arg_options = it->second;
    for (const std::string &arg_option : valid_arg_options) {
      ss << "\"" << arg_option << "\"\n";
    }
    ss << std::endl;
  } else {
    MS_LOG(EXCEPTION) << "Valid arg options are not correctly generated." << std::endl;
  }
  return ss.str();
}

std::string OpDTypeToString(OP_DTYPE dtype) {
  static const std::unordered_map<OP_DTYPE, std::string> kEnumToStringMap = {
    {OP_DTYPE::DT_BOOL, "bool"},
    {OP_DTYPE::DT_INT, "int"},
    {OP_DTYPE::DT_FLOAT, "float"},
    {OP_DTYPE::DT_NUMBER, "Number"},
    {OP_DTYPE::DT_TENSOR, "Tensor"},
    {OP_DTYPE::DT_STR, "string"},
    {OP_DTYPE::DT_ANY, "Any"},
    {OP_DTYPE::DT_TUPLE_BOOL, "tuple of bool"},
    {OP_DTYPE::DT_TUPLE_INT, "tuple of int"},
    {OP_DTYPE::DT_TUPLE_FLOAT, "tuple of float"},
    {OP_DTYPE::DT_TUPLE_NUMBER, "tuple of Number"},
    {OP_DTYPE::DT_TUPLE_TENSOR, "tuple of Tensor"},
    {OP_DTYPE::DT_TUPLE_STR, "tuple of string"},
    {OP_DTYPE::DT_TUPLE_ANY, "tuple of Any"},
    {OP_DTYPE::DT_LIST_BOOL, "list of bool"},
    {OP_DTYPE::DT_LIST_INT, "list of int"},
    {OP_DTYPE::DT_LIST_FLOAT, "list of float"},
    {OP_DTYPE::DT_LIST_NUMBER, "list of number"},
    {OP_DTYPE::DT_LIST_TENSOR, "list of tensor"},
    {OP_DTYPE::DT_LIST_STR, "list of string"},
    {OP_DTYPE::DT_LIST_ANY, "list of Any"},
    {OP_DTYPE::DT_TYPE, "mstype"},
    {OP_DTYPE::DT_STORAGE, "Storage"},
    {OP_DTYPE::DT_NONE, "None"},
  };

  auto it = kEnumToStringMap.find(dtype);
  if (it == kEnumToStringMap.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Failed to map Enum[" << dtype << "] to String.";
  }
  return it->second;
}

namespace {
static inline std::string GetRealTypeByHandler(const std::string &type, const std::string &handler) {
  if (handler.empty()) {
    return type;
  }
  static const std::unordered_map<std::string, std::string> handler_to_src_type{{"dtype_to_type_id", "mindspore.dtype"},
                                                                                {"str_to_enum", "string"}};
  const auto iter = handler_to_src_type.find(handler);
  return iter != handler_to_src_type.end() ? iter->second : type;
}

static inline std::string GetRealInputType(const ops::OpInputArg &op_arg) {
  return GetRealTypeByHandler(OpDTypeToString(op_arg.arg_dtype_), op_arg.arg_handler_);
}

static inline std::vector<std::string> GetRealTypes(const std::vector<std::string> &op_type_list,
                                                    const std::vector<ops::OpInputArg> &input_args) {
  if (input_args.size() != op_type_list.size()) {
    MS_LOG_EXCEPTION << "size of input_args and op_type_list should be equal, but got " << input_args.size() << " vs "
                     << op_type_list.size();
  }
  std::vector<std::string> real_types(op_type_list.size());
  for (size_t i = 0; i < op_type_list.size(); ++i) {
    real_types[i] = GetRealTypeByHandler(op_type_list[i], input_args[i].arg_handler_);
  }
  return real_types;
}
}  // namespace

std::string BuildOpErrorMsg(const ops::OpDefPtr &op_def, const std::vector<std::string> &op_type_list) {
  std::stringstream init_arg_ss;
  std::stringstream input_arg_ss;
  for (const auto &op_arg : op_def->args_) {
    if (op_arg.as_init_arg_) {
      init_arg_ss << op_arg.arg_name_ << "=<";
      for (const auto &dtype : op_arg.cast_dtype_) {
        init_arg_ss << OpDTypeToString(dtype) << ", ";
      }
      init_arg_ss << GetRealInputType(op_arg) << ">, ";
    } else {
      input_arg_ss << op_arg.arg_name_ << "=<";
      for (const auto &dtype : op_arg.cast_dtype_) {
        input_arg_ss << OpDTypeToString(dtype) << ", ";
      }
      input_arg_ss << GetRealInputType(op_arg) << ">, ";
    }
  }

  auto init_arg_str = init_arg_ss.str();
  auto input_arg_str = input_arg_ss.str();
  constexpr size_t truncate_offset = 2;
  init_arg_str =
    init_arg_str.empty() ? "" : init_arg_str.replace(init_arg_str.end() - truncate_offset, init_arg_str.end(), "");
  input_arg_str =
    input_arg_str.empty() ? "" : input_arg_str.replace(input_arg_str.end() - truncate_offset, input_arg_str.end(), "");

  std::stringstream real_init_arg_ss;
  std::stringstream real_input_arg_ss;
  auto real_op_type_list = GetRealTypes(op_type_list, op_def->args_);
  for (size_t i = 0; i < real_op_type_list.size(); i++) {
    const auto &op_arg = op_def->args_[i];
    if (op_arg.as_init_arg_) {
      real_init_arg_ss << op_arg.arg_name_ << "=" << real_op_type_list[i] << ", ";
    } else {
      real_input_arg_ss << op_arg.arg_name_ << "=" << real_op_type_list[i] << ", ";
    }
  }
  auto real_init_arg_str = real_init_arg_ss.str();
  auto real_input_arg_str = real_input_arg_ss.str();
  real_init_arg_str = real_init_arg_str.empty() ? ""
                                                : real_init_arg_str.replace(real_init_arg_str.end() - truncate_offset,
                                                                            real_init_arg_str.end(), "");
  real_input_arg_str =
    real_input_arg_str.empty()
      ? ""
      : real_input_arg_str.replace(real_input_arg_str.end() - truncate_offset, real_input_arg_str.end(), "");

  std::stringstream ss;
  ss << "Failed calling " << op_def->name_ << " with \"" << op_def->name_ << "(" << real_init_arg_str << ")("
     << real_input_arg_str << ")\"." << std::endl;
  ss << "The valid calling should be: " << std::endl;
  ss << "\"" << op_def->name_ << "(" << init_arg_str << ")(" << input_arg_str << ")\".";
  return ss.str();
}

std::string BuildOpInputsErrorMsg(const ops::OpDefPtr &op_def, const std::string &arg_name, const TypePtr &arg_type) {
  MS_EXCEPTION_IF_NULL(arg_type);
  std::stringstream inputs_ss;
  for (const auto &op_arg : op_def->args_) {
    if (op_arg.as_init_arg_) {
      continue;
    }
    inputs_ss << op_arg.arg_name_ << "=<";
    for (const auto &dtype : op_arg.cast_dtype_) {
      inputs_ss << OpDTypeToString(dtype) << ", ";
    }
    inputs_ss << GetRealInputType(op_arg) << ">, ";
  }
  constexpr size_t truncate_offset = 2;
  auto inputs_str = inputs_ss.str();
  inputs_str = inputs_str.empty() ? "" : inputs_str.replace(inputs_str.end() - truncate_offset, inputs_str.end(), "");
  std::stringstream ss;
  ss << "Failed calling " << op_def->name_ << " with \"" << arg_name << "=" << arg_type->ToString() << "\".";
  ss << "\nThe valid calling should be: \"" << inputs_str << "\".";
  return ss.str();
}
}  // namespace prim
}  // namespace mindspore
