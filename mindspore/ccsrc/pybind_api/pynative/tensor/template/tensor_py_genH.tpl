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

#ifndef TENSOR_PY_GEN_H
#define TENSOR_PY_GEN_H

#include "pybind_api/pynative/tensor/tensor_api/auto_generate/tensor_api.h"
#include "pybind11/pybind11.h"
#include "pybind_api/pynative/tensor/tensor_register/tensor_func_reg.h"

namespace py = pybind11;
namespace mindspore {
namespace tensor {

#define DEFINE_TENSOR_METHOD_CPYWRAPPER(NAME)                                                          \
  static PyObject *TensorMethod##NAME##_CPyWrapper(PyObject *self, PyObject *args, PyObject *kwargs) { \
    PyObject* result;                                                                                  \
    try {                                                                                              \
      result = TensorMethod##NAME(self, args, kwargs);                                                 \
    } catch (py::error_already_set &e) {                                                               \
      e.restore();                                                                                     \
      return NULL;                                                                                     \
    } catch (const std::runtime_error &e) {                                                            \
      if (dynamic_cast<const py::index_error *>(&e)) {                                                 \
        PyErr_SetString(PyExc_IndexError, e.what());                                                   \
      } else if (dynamic_cast<const py::value_error *>(&e)) {                                          \
        PyErr_SetString(PyExc_ValueError, e.what());                                                   \
      } else if (dynamic_cast<const py::type_error *>(&e)) {                                           \
        PyErr_SetString(PyExc_TypeError, e.what());                                                    \
      } else {                                                                                         \
        PyErr_SetString(PyExc_RuntimeError, e.what());                                                 \
      }                                                                                                \
      return NULL;                                                                                     \
    }                                                                                                  \
    return result;                                                                                     \
  }


#define DEFINE_TENSOR_METHODS_CPYWRAPPERS()         \
${CPyWrapper_defs}

extern PyMethodDef *TensorMethods;
}  // namespace tensor
}  // namespace mindspore
#endif  // TENSOR_PY_GEN_H