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

#include "include/pynative/forward/pyboost/fallback.h"
#include "utils/log_adapter.h"
#include "include/utils/python_attr.h"
#include "include/utils/tensor_utils.h"

namespace mindspore {
namespace pynative {
namespace {
PyObject *MergeSelfAndArgs(PyObject *self, PyObject *args) {
  if (!self) {
    PyErr_SetString(PyExc_RuntimeError, "self is null");
    return nullptr;
  }

  PyObject *result = nullptr;

  // Case 1: args is nullptr
  if (args == nullptr) {
    result = PyTuple_New(1);
    if (!result) {
      return nullptr;
    }

    Py_INCREF(self);  // SET_ITEM steals ref
    PyTuple_SET_ITEM(result, 0, self);
    return result;
  }

  // Case 2: args is a tuple
  if (PyTuple_Check(args)) {
    Py_ssize_t n = PyTuple_GET_SIZE(args);
    result = PyTuple_New(n + 1);
    if (!result) {
      return nullptr;
    }

    Py_INCREF(self);
    PyTuple_SET_ITEM(result, 0, self);

    for (Py_ssize_t i = 0; i < n; ++i) {
      PyObject *item = PyTuple_GET_ITEM(args, i);  // borrowed
      Py_INCREF(item);
      PyTuple_SET_ITEM(result, i + 1, item);
    }
    return result;
  }

  // Case 3: args is a single object
  result = PyTuple_New(2);
  if (!result) {
    return nullptr;
  }

  Py_INCREF(self);
  Py_INCREF(args);
  PyTuple_SET_ITEM(result, 0, self);
  PyTuple_SET_ITEM(result, 1, args);
  return result;
}

py::object GetFallbackFromObj(PyObject *py_args) {
  if (tensor::IsPyObjectTensorPy(py_args)) {
    return FastGetPyObjectAttr(py_args, GetFallbackStr().c_str());
  } else if (PyTuple_Check(py_args)) {
    Py_ssize_t size = PyTuple_Size(py_args);
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *item = PyTuple_GetItem(py_args, i);
      auto fallback = GetFallbackFromObj(item);
      if (fallback != nullptr) {
        MS_LOG(DEBUG) << "Get fallback from " << Py_TYPE(item)->tp_name;
        return fallback;
      }
    }
    return {};
  } else if (PyList_Check(py_args)) {
    Py_ssize_t size = PyList_Size(py_args);
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *item = PyList_GetItem(py_args, i);
      auto fallback = GetFallbackFromObj(item);
      if (fallback != nullptr) {
        MS_LOG(DEBUG) << "Get fallback from " << Py_TYPE(item)->tp_name;
        return fallback;
      }
    }
    return {};
  } else {
    MS_LOG(DEBUG) << "Cannot get __fallback__ attr from: " << Py_TYPE(py_args)->tp_name
                  << ". Supported type is <Tensor, tuple, list>";
    return {};
  }
}

py::object GetFallbackFromInput(PyObject *py_args) {
  auto fallback = GetFallbackFromObj(py_args);
  if (fallback == nullptr) {
    MS_LOG(EXCEPTION) << "Input has no __fallback__ attribute!";
  }
  return fallback;
}

void CheckCallResult(PyObject *ret) {
  if (ret == nullptr) {
    PyErr_Print();
    MS_LOG(EXCEPTION) << "Python fallback failed.";
  }
}
}  // namespace

class NoFallbackGuard {
 public:
  NoFallbackGuard() {}
  NoFallbackGuard &Enter() {
    fallback_enable_ = false;
    return *this;
  }
  bool Exit(py::object /*exc_type*/, py::object /*exc*/, py::object /*tb*/) {
    fallback_enable_ = true;
    return false;  // rethrow exception.
  }
  static bool fallback_enabled() { return fallback_enable_; }

 private:
  inline static bool fallback_enable_{true};
};

bool fallback_enabled() { return NoFallbackGuard::fallback_enabled(); }

const std::string &GetFallbackStr() {
  static const std::string kFallback = "__fallback__";
  return kFallback;
}

py::object HandleFallback(const py::args &args, const py::kwargs &kwargs, const py::object &callable) {
  MS_LOG(DEBUG) << "Start HandleFallback";
  py::object fallback = GetFallbackFromInput(args.ptr());
  if (fallback == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot found __fallback__ attr from inputs";
  }
  PyObject *ret = PyObject_CallFunctionObjArgs(fallback.ptr(), callable.ptr(), args.ptr(), kwargs.ptr(), NULL);
  CheckCallResult(ret);
  MS_LOG(DEBUG) << "End HandleFallback";
  return py::reinterpret_steal<py::object>(ret);
}

PyObject *HandleFallback(PyObject *py_args, const py::object &callable) {
  MS_LOG(DEBUG) << "Start HandleFallback";
  py::object fallback = GetFallbackFromInput(py_args);
  if (fallback == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot found __fallback__ attr from py_args";
  }
  // combine `self` and `py_args` to `new_args`
  PyObject *ret = PyObject_CallFunctionObjArgs(fallback.ptr(), callable.ptr(), py_args, NULL);
  CheckCallResult(ret);
  MS_LOG(DEBUG) << "End HandleFallback";
  return ret;
}

PyObject *HandleFallback(PyObject *self, PyObject *py_args, PyObject *py_kwargs, const py::object &callable) {
  MS_LOG(DEBUG) << "Start HandleFallback";
  auto fallback = FastGetPyObjectAttr(self, GetFallbackStr().c_str());
  if (fallback.ptr() == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot found __fallback__ attr from object " << Py_TYPE(self)->tp_name;
  }
  // combine `self` and `py_args` to `new_args`
  PyObject *new_args = MergeSelfAndArgs(self, py_args);
  PyObject *ret = PyObject_CallFunctionObjArgs(fallback.ptr(), callable.ptr(), new_args, py_kwargs, NULL);
  CheckCallResult(ret);
  MS_LOG(DEBUG) << "End HandleFallback";
  return ret;
}

void RegFallback(py::module *m) {
  py::class_<NoFallbackGuard>(*m, "NoFallbackGuard")
    .def(py::init<>())
    .def("__enter__", &NoFallbackGuard::Enter,
         py::return_value_policy::reference_internal)  // <-- important
    .def("__exit__", &NoFallbackGuard::Exit);
}
}  // namespace pynative
}  // namespace mindspore
