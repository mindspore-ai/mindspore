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
#include "ir/param_value.h"
#include "pybind11/pybind11.h"
#include "pybind_api/api_register.h"

namespace mindspore {
namespace py = pybind11;

REGISTER_PYBIND_DEFINE(ParamValue, ([](const py::module *m) {
                         (void)py::class_<ParamValue, ParamValuePtr>(*m, "ParamValue")
                           .def(py::init())
                           .def("clone", &ParamValue::Clone)
                           .def_property("name", &ParamValue::name, &ParamValue::set_name)
                           .def_property("requires_grad", &ParamValue::requires_grad, &ParamValue::set_requires_grad)
                           .def_property("layerwise_parallel", &ParamValue::layerwise_parallel,
                                         &ParamValue::set_layerwise_parallel)
                           .def(py::pickle(
                             [](const ParamValue &p) {  // __getstate__
                               return py::make_tuple(p.name(), p.requires_grad(), p.layerwise_parallel());
                             },
                             [](const py::tuple &t) {  // __setstate__
                               if (t.size() != 6) {
                                 std::runtime_error("Invalid state for ParamValue!");
                               }
                               ParamValuePtr p = std::make_shared<ParamValue>();
                               p->set_name(t[1].cast<std::string>());
                               p->set_requires_grad(t[2].cast<bool>());
                               p->set_layerwise_parallel(t[3].cast<bool>());
                               return p;
                             }));
                       }));
}  // namespace mindspore
