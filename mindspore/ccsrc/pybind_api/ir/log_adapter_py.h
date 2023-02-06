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

// NOTICE: This header file should only be included once in the whole project.
// We change the cpp file to header file, to avoid MSVC compiler problem.
#ifndef MINDSPORE_CCSRC_PYBINDAPI_IR_LOG_ADAPTER_PY_H_
#define MINDSPORE_CCSRC_PYBINDAPI_IR_LOG_ADAPTER_PY_H_

#include "utils/log_adapter.h"

#include <string>
#include "pybind11/pybind11.h"
#include "pybind_api/pybind_patch.h"

namespace py = pybind11;
namespace mindspore {
class PyExceptionInitializer {
 public:
  PyExceptionInitializer() {
    MS_LOG(INFO) << "Set exception handler";
    mindspore::LogWriter::SetExceptionHandler(HandleExceptionPy);
  }

  ~PyExceptionInitializer() = default;

 private:
  static void HandleExceptionPy(ExceptionType exception_type, const std::string &str) {
    if (exception_type == IndexError) {
      throw py::index_error(str);
    }
    if (exception_type == ValueError) {
      throw py::value_error(str);
    }
    if (exception_type == TypeError) {
      throw py::type_error(str);
    }
    if (exception_type == KeyError) {
      throw py::key_error(str);
    }
    if (exception_type == AttributeError) {
      throw py::attribute_error(str);
    }
    if (exception_type == NameError) {
      throw py::name_error(str);
    }
    if (exception_type == AssertionError) {
      throw py::assertion_error(str);
    }
    if (exception_type == BaseException) {
      throw py::base_exception(str);
    }
    if (exception_type == KeyboardInterrupt) {
      throw py::keyboard_interrupt(str);
    }
    if (exception_type == StopIteration) {
      throw py::stop_iteration(str);
    }
    if (exception_type == OverflowError) {
      throw py::overflow_error(str);
    }
    if (exception_type == ZeroDivisionError) {
      throw py::zero_division_error(str);
    }
    if (exception_type == EnvironmentError) {
      throw py::environment_error(str);
    }
    if (exception_type == IOError) {
      throw py::io_error(str);
    }
    if (exception_type == OSError) {
      throw py::os_error(str);
    }
    if (exception_type == MemoryError) {
      throw py::memory_error(str);
    }
    if (exception_type == UnboundLocalError) {
      throw py::unbound_local_error(str);
    }
    if (exception_type == NotImplementedError) {
      throw py::not_implemented_error(str);
    }
    if (exception_type == IndentationError) {
      throw py::indentation_error(str);
    }
    if (exception_type == RuntimeWarning) {
      throw py::runtime_warning(str);
    }
    py::pybind11_fail(str);
  }
};

static PyExceptionInitializer py_exception_initializer;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBINDAPI_IR_LOG_ADAPTER_PY_H_
