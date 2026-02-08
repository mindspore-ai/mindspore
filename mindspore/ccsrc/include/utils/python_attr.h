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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_UTILS_PYTHON_ATTR_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_UTILS_PYTHON_ATTR_H_

#include <Python.h>
#include <pybind11/pybind11.h>

namespace mindspore {
namespace py = pybind11;
/*
 * Higher performance version of PyObject_GetAttrString in numpy.
 */
inline py::object FastGetPyObjectAttr(PyObject *obj, const char *attr_name) {
  PyTypeObject *type = Py_TYPE(obj);
  PyObject *res = (PyObject *)NULL;

  if (type->tp_getattr != NULL) {
    res = (*type->tp_getattr)(obj, const_cast<char *>(attr_name));
    if (res == NULL) {
      PyErr_Clear();
    }
  } else if (type->tp_getattro != NULL) {
    auto key = py::reinterpret_steal<py::object>(PyUnicode_FromString(attr_name));
    if (key.ptr() == nullptr) {
      return py::object();
    }
    res = (*type->tp_getattro)(obj, key.ptr());
    if (res == NULL) {
      PyErr_Clear();
    }
  }
  return py::reinterpret_steal<py::object>(res);
}
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_UTILS_PYTHON_ATTR_H_
