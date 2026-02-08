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
#include "pybind_api/ops/ops_api.h"
#include "pybind11/pybind11.h"
#include "pybind_api/ops/custom/op_def_builder.h"
#include "include/frontend/jit/ps/pipeline_interface.h"
#ifdef _WIN32
#include "kernel/cpu/utils/cpu_utils.h"
#endif

namespace mindspore {
namespace py = pybind11;
void RegPyFuncOpBuilder(py::module *m) {
  (void)py::class_<mindspore::ops::PyFuncOpDefBuilder>(*m, "PyFuncOpDefBuilder")
    .def(pybind11::init<const std::string &>(), pybind11::arg("name"))
    .def(
      "arg",
      [](mindspore::ops::PyFuncOpDefBuilder &self, const std::string &arg_name, const std::string &role,
         const std::string &obj_type) -> mindspore::ops::PyFuncOpDefBuilder & {
        return self.Arg(arg_name, role, obj_type);
      },
      pybind11::arg("arg_name"), pybind11::arg("role"), pybind11::arg("obj_type"),
      pybind11::return_value_policy::reference_internal,
      "Add an argument. role in {'input','output'}. obj_type is canonical string.")
    .def("register_op", &mindspore::ops::PyFuncOpDefBuilder::Register, "Register the OpDef and its PyFunc kernel");
}

void RegOpsModule(py::module *m) {
  RegRandomSeededGenerator(m);
  RegGeneratorImpl(m);
  mindspore::initializer::RegRandomNormal(m);
  RegPyFuncOpBuilder(m);
}
}  // namespace mindspore
