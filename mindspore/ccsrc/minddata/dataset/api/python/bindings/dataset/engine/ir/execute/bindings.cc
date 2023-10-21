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
#include "pybind11/pybind11.h"

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/include/dataset/execute.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(Execute, 0, ([](const py::module *m) {
                  (void)py::class_<PyExecute, std::shared_ptr<PyExecute>>(*m, "Execute")
                    .def(py::init([](py::object operation) {
                      // current only support one op in python layer
                      auto execute = std::make_shared<PyExecute>(toTensorOperation(operation));
                      return execute;
                    }))
                    .def("UpdateOperation",
                         [](PyExecute &self, py::object operation) {
                           // update the op from python layer
                           THROW_IF_ERROR(self.UpdateOperation(toTensorOperation(operation)));
                         })
                    .def("__call__",
                         [](PyExecute &self, const std::vector<std::shared_ptr<Tensor>> &input_tensor_list) {
                           // Python API only supports cpu for eager mode
                           std::vector<std::shared_ptr<dataset::Tensor>> de_output_tensor_list;
                           THROW_IF_ERROR(self(input_tensor_list, &de_output_tensor_list));
                           return de_output_tensor_list;
                         });
                }));

}  // namespace dataset
}  // namespace mindspore
