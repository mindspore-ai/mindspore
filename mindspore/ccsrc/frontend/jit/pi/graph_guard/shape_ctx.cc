/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_guard/shape_ctx.h"
#include <algorithm>
#include <map>
#include <string>
#include "ir/tensor.h"
#include "frontend/jit/pi/python_adapter/pydef.h"
#include "include/utils/tensor_py.h"

namespace py = pybind11;

namespace mindspore {
namespace pijit {

#if IS_PYTHON_3_11_PLUS

ShapeContext::ShapeContext(PyFrameWrapper f, const py::object &enable_dynamic_dict) {}
ShapeContext::~ShapeContext() {}
void ShapeContext::ApplyEnableDynamic() {}
void ShapeContext::RevertEnableDynamic() {}
void ShapeContext::UpdateFastLocal(PyObject **fast_local, PyCodeObject *code, PyObject *arg, int index) {}

#else

ShapeContext::ShapeContext(PyFrameWrapper f, const py::object &enable_dynamic_dict)
    : frame_(f), enable_dynamic_dict_(enable_dynamic_dict.ptr()) {}

ShapeContext::~ShapeContext() {
  RevertEnableDynamic();
  Py_XDECREF(enable_dynamic_dict_);
}

bool CheckIsMethod(PyCodeObject *code) {
  // Check if function or method.
  if (!(code->co_flags & CO_OPTIMIZED)) {
    return false;
  }
  // Check the name of the first argument.
  if (code->co_argcount > 0) {
    PyObject *first_arg_name = PyTuple_GetItem(code->co_varnames, 0);
    const char *name = PyUnicode_AsUTF8(first_arg_name);
    if (name && (strcmp(name, "self") == 0 || strcmp(name, "cls") == 0)) {
      return true;
    }
  }
  return false;
}

void CheckArgName(PyObject *arg_name, PyCodeObject *code, int index) {
  PyObject *varnames = code->co_varnames;
  PyObject *expect_name = PyTuple_GetItem(varnames, index);
  if (PyUnicode_Compare(arg_name, expect_name)) {
    MS_LOG(INTERNAL_EXCEPTION) << "For enable_dynamic, arg_name " << std::string(py::str(arg_name))
                               << " does not match " << std::string(py::str(expect_name)) << " in varnames "
                               << std::string(py::str(varnames)) << ".";
  }
}

void ShapeContext::UpdateFastLocal(PyObject **fast_local, PyCodeObject *code, PyObject *arg, int index) {
  // Local variables.
  if (fast_local[index] != nullptr) {
    origin_[index] = fast_local[index];
    fast_local[index] = arg;
    MS_LOG(INFO) << "Replace local variable at index: " << index;
    return;
  }
  // Cell variables for closure.
  if (code->co_cell2arg == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Both fast_local[" << index << "] and code->co_cell2arg are nullptr.";
  }
  Py_ssize_t n_cells = PyTuple_GET_SIZE(code->co_cellvars);
  for (Py_ssize_t i = 0; i < n_cells; ++i) {
    // Check whether the closure variable is a function parameter.
    if (code->co_cell2arg[i] == index) {
      int index_cellvar = code->co_nlocals + i;
      PyObject *cellvar = fast_local[index_cellvar];
      origin_[index_cellvar] = cellvar;
      // Create new cellvar.
      PyObject *new_cellvar = PyCell_New(nullptr);
      PyCell_SET(new_cellvar, arg);
      fast_local[index_cellvar] = new_cellvar;
      MS_LOG(INFO) << "Replace cellvar at index: " << index_cellvar;
      return;
    }
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Failed to update fast_local.";
}

void ShapeContext::ApplyEnableDynamic() {
  if (enable_dynamic_dict_ == nullptr || applied_) {
    return;
  }
  PyCodeWrapper co_wrapper = frame_.GetCode();
  PyCodeObject *code = co_wrapper.ptr();
  bool is_method = CheckIsMethod(code);

  // In python3.11+, modify fast local maybe cause error
  PyObject **fast_local = const_cast<PyObject **>(frame_.FastLocal());
  PyObject *key = nullptr;
  PyObject *value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(enable_dynamic_dict_, &pos, &key, &value)) {
    if (!PyTuple_Check(value)) {
      MS_LOG(INTERNAL_EXCEPTION) << "In enable_dynamic_dict, value should be tuple type, but got "
                                 << std::string(py::str(value));
    }
    int index = PyLong_AsLong(key) + (is_method ? 1 : 0);
    PyObject *arg_name = PyTuple_GET_ITEM(value, 0);
    PyObject *arg = PyTuple_GET_ITEM(value, 1);
    CheckArgName(arg_name, code, index);
    MS_LOG(INFO) << "Apply enable_dynamic for " << std::string(py::str(arg_name)) << ". is_method: " << is_method
                 << ", key: " << std::string(py::str(key)) << ", index: " << index << ".";
    UpdateFastLocal(fast_local, code, arg, index);
  }
  applied_ = true;
}

void ShapeContext::RevertEnableDynamic() {
  if (!applied_) {
    return;
  }
  PyObject **fast_local = const_cast<PyObject **>(frame_.FastLocal());
  for (const auto &[index, arg] : origin_) {
    fast_local[index] = arg;
  }
  applied_ = false;
}

#endif

}  // namespace pijit
}  // namespace mindspore
