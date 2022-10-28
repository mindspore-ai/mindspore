/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "include/common/utils/primitive_utils.h"

#include <memory>

#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "pybind_api/ir/base_ref_py.h"

namespace mindspore {
py::function GetBpropFunctionByObj(const py::object &obj, bool get_closure) {
  static const std::string get_bprop_fn = "get_bprop_fn";
  static const std::string ad_module = "mindspore.ops._grad";
  static const std::string ad_experimental_module = "mindspore.ops._grad_experimental";
  py::function fn = python_adapter::GetPyFn(ad_module, get_bprop_fn)(obj, get_closure);
  if (!fn || py::isinstance<py::none>(fn)) {
    fn = python_adapter::GetPyFn(ad_experimental_module, get_bprop_fn)(obj);
  }
  return fn;
}

py::function GetBpropFunction(const std::string &name) {
  auto fn = GetBpropFunctionByObj(py::str(name));
  return fn;
}

py::function GetTaylorRuleFunctionByObj(const py::object &obj) {
  static const std::string get_taylor_fprop_fn = "get_taylor_fprop_fn";
  static const std::string ad_module = "mindspore.ops._grad";
  py::function fn = python_adapter::GetPyFn(ad_module, get_taylor_fprop_fn)(obj);
  return fn;
}

py::function GetTaylorRuleFunction(const std::string &name) {
  auto fn = GetTaylorRuleFunctionByObj(py::str(name));
  return fn;
}

py::function GetComputeFunction(const std::string &name) {
  static const std::string module = "mindspore._extends.builtin_operations";
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

py::function GetVmapRuleFunctionByObj(const py::object &obj, int axis_size) {
  constexpr char get_vmap_rule_fn[] = "get_vmap_rule";
  constexpr char vmap_module[] = "mindspore.ops._vmap";
  py::function fn = python_adapter::GetPyFn(vmap_module, get_vmap_rule_fn)(obj, axis_size);
  return fn;
}

py::function GetVmapRuleFunction(const std::string &name, int axis_size) {
  auto fn = GetVmapRuleFunctionByObj(py::str(name), axis_size);
  return fn;
}
}  // namespace mindspore
