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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_EXCEPTION_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_EXCEPTION_H_
#include "pybind11/pybind11.h"

namespace py = pybind11;

#define HANDLE_MS_EXCEPTION try {
#define HANDLE_MS_EXCEPTION_END                             \
  }                                                         \
  catch (py::error_already_set & e) {                       \
    e.restore();                                            \
    return NULL;                                            \
  }                                                         \
  catch (const std::runtime_error &e) {                     \
    if (dynamic_cast<const py::index_error *>(&e)) {        \
      PyErr_SetString(PyExc_IndexError, e.what());          \
    } else if (dynamic_cast<const py::value_error *>(&e)) { \
      PyErr_SetString(PyExc_ValueError, e.what());          \
    } else if (dynamic_cast<const py::type_error *>(&e)) {  \
      PyErr_SetString(PyExc_TypeError, e.what());           \
    } else {                                                \
      PyErr_SetString(PyExc_RuntimeError, e.what());        \
    }                                                       \
    return NULL;                                            \
  }                                                         \
  catch (...) {                                             \
    PyErr_SetString(PyExc_RuntimeError, "Unknown Error!");  \
  }                                                         \
  return NULL;

#define HANDLE_MS_EXCEPTION_RET_FAIL_END                    \
  }                                                         \
  catch (py::error_already_set & e) {                       \
    e.restore();                                            \
    return -1;                                              \
  }                                                         \
  catch (const std::runtime_error &e) {                     \
    if (dynamic_cast<const py::index_error *>(&e)) {        \
      PyErr_SetString(PyExc_IndexError, e.what());          \
    } else if (dynamic_cast<const py::value_error *>(&e)) { \
      PyErr_SetString(PyExc_ValueError, e.what());          \
    } else if (dynamic_cast<const py::type_error *>(&e)) {  \
      PyErr_SetString(PyExc_TypeError, e.what());           \
    } else {                                                \
      PyErr_SetString(PyExc_RuntimeError, e.what());        \
    }                                                       \
    return -1;                                              \
  }                                                         \
  catch (...) {                                             \
    PyErr_SetString(PyExc_RuntimeError, "Unknown Error!");  \
  }                                                         \
  return -1;

#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_EXCEPTION_H_
