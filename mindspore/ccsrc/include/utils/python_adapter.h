/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYTHON_ADAPTER_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYTHON_ADAPTER_H_
#include <map>
#include <memory>
#include <string>

#include <Python.h>
#include "pybind11/embed.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include "utils/log_adapter.h"
#include "ir/tensor.h"
#include "base/base_ref.h"
#include "include/utils/visible.h"
#include "utils/callback_handler.h"

namespace py = pybind11;
namespace mindspore {
// A utility to call python interface
namespace python_adapter {
COMMON_EXPORT py::module GetPyModule(const std::string &module);
COMMON_EXPORT py::object GetPyObjAttr(const py::object &obj, const std::string &attr);
template <class... T>
py::object CallPyObjMethod(const py::object &obj, const std::string &method, T... args) {
  if (!method.empty() && !py::isinstance<py::none>(obj)) {
    return obj.attr(py::str(method))(args...);
  }
  return py::none();
}

// call python function of module
template <class... T>
py::object CallPyModFn(const py::module &mod, const std::string &function, T... args) {
  if (!function.empty() && !py::isinstance<py::none>(mod)) {
    return mod.attr(py::str(function))(args...);
  }
  return py::none();
}

// turn off the signature when ut use parser to construct a graph.
COMMON_EXPORT void set_use_signature_in_resolve(bool use_signature) noexcept;
COMMON_EXPORT bool UseSignatureInResolve();

COMMON_EXPORT std::shared_ptr<py::scoped_interpreter> set_python_scoped();
COMMON_EXPORT void ResetPythonScope();
COMMON_EXPORT bool IsPythonEnv();
COMMON_EXPORT void SetPythonPath(const std::string &path);
COMMON_EXPORT void set_python_env_flag(bool python_env) noexcept;
COMMON_EXPORT py::object GetPyFn(const std::string &module, const std::string &name);

// Call the python function
template <class... T>
py::object CallPyFn(const std::string &module, const std::string &name, T... args) {
  (void)set_python_scoped();
  if (!module.empty() && !name.empty()) {
    py::module mod = py::module::import(module.c_str());
    py::object fn = mod.attr(py::str(name))(args...);
    return fn;
  }
  return py::none();
}

// Cast shared_ptr to py::object.
template <typename T>
py::object CastToPyObj(const std::shared_ptr<T> &ptr) {
  // Use a pybind11 typecaster to create a PyObject from a shared_ptr<T> pointer.
  py::detail::type_caster<std::shared_ptr<T>> shared_ptr_caster;
  py::handle cast_handle = shared_ptr_caster.cast(ptr, py::return_value_policy::take_ownership, py::handle());
  py::object obj = py::cast<py::object>(cast_handle);
  MS_LOG(DEBUG) << ptr << ", obj: " << obj << ", handle ptr: " << cast_handle.ptr();
  return obj;
}

// Cast pointer to py::object.
template <typename T>
py::object CastToPyObj(const T *ptr) {
  // Use a pybind11 typecaster to create a PyObject from a T* pointer.
  py::detail::type_caster<T *> ptr_caster;
  py::handle cast_handle = ptr_caster.cast(ptr, py::return_value_policy::take_ownership, py::handle());
  py::object obj = py::cast<py::object>(cast_handle);
  MS_LOG(DEBUG) << ptr << ", obj: " << obj << ", handle ptr: " << cast_handle.ptr();
  return obj;
}

class COMMON_EXPORT PyAdapterCallback {
  HANDLER_DEFINE(ValuePtr, PyDataToValue, py::object);
  HANDLER_DEFINE(py::array, TensorToNumpy, tensor::Tensor);
  HANDLER_DEFINE(void, ProcessUnPairedCellHook, bool);
};
}  // namespace python_adapter
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYTHON_ADAPTER_H_
