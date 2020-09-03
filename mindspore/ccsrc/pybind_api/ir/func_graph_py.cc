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
#include <string>
#include "ir/meta_func_graph.h"
#include "ir/func_graph.h"

#include "pybind_api/api_register.h"

namespace mindspore {
REGISTER_PYBIND_DEFINE(FuncGraph, ([](const pybind11::module *m) {
                         // Define python "MetaFuncGraph_" class
                         (void)py::class_<MetaFuncGraph, std::shared_ptr<MetaFuncGraph>>(*m, "MetaFuncGraph_")
                           .def("set_signatures", &MetaFuncGraph::set_signatures, "Set primitive inputs signature.");
                         // Define python "FuncGraph" class
                         (void)py::class_<FuncGraph, FuncGraphPtr>(*m, "FuncGraph")
                           .def(py::init())
                           .def("str", &FuncGraph::ToString, "Get FuncGraph string representation.")
                           .def("get_return", &FuncGraph::get_return, "Get return node of FuncGraph");
                       }));
}  // namespace mindspore
