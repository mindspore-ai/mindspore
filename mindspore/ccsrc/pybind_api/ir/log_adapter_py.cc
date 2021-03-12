/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "utils/log_adapter.h"

#include <string>
#include "pybind11/pybind11.h"
#include "pybind_api/pybind_patch.h"

namespace py = pybind11;
namespace mindspore {
class PyExceptionInitializer {
 public:
  PyExceptionInitializer() { mindspore::LogWriter::set_exception_handler(HandleExceptionPy); }

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
    py::pybind11_fail(str);
  }
};

static PyExceptionInitializer py_exception_initializer;
}  // namespace mindspore
