/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/common/pybind_api/api_register.h"
#include "ir/variable.h"
#include "pipeline/jit/parse/data_converter.h"

namespace py = pybind11;
namespace mindspore {
REGISTER_PYBIND_DEFINE(Variable_, ([](const py::module *m) {
                         (void)py::class_<Variable, VariablePtr>(*m, "Variable_")
                           .def(py::init([](const py::object &py_value) {
                                  ValuePtr real_value = nullptr;
                                  if (!parse::ConvertData(py_value, &real_value)) {
                                    MS_EXCEPTION(TypeError)
                                      << "Convert python object failed, the object type is " << py_value.get_type()
                                      << ", value is '" << py::str(py_value) << "'.";
                                  }
                                  return std::make_shared<Variable>(real_value);
                                }),
                                py::arg("py_value"));
                       }));
}  // namespace mindspore
