/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/execute.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(Execute, 0, ([](const py::module *m) {
                  (void)py::class_<Execute, std::shared_ptr<Execute>>(*m, "Execute")
                    .def(py::init([](py::object operation) {
                      auto execute = std::make_shared<Execute>(toTensorOperation(operation));
                      return execute;
                    }))
                    .def("__call__",
                         [](Execute &self, std::shared_ptr<Tensor> in) {
                           std::shared_ptr<Tensor> out = self(in);
                           if (out == nullptr) {
                             THROW_IF_ERROR([]() {
                               RETURN_STATUS_UNEXPECTED(
                                 "Failed to execute op in eager mode, please check ERROR log above.");
                             }());
                           }
                           return out;
                         })
                    .def("__call__", [](Execute &self, const std::vector<std::shared_ptr<Tensor>> &input_tensor_list) {
                      std::vector<std::shared_ptr<Tensor>> output_tensor_list;
                      THROW_IF_ERROR(self(input_tensor_list, &output_tensor_list));
                      if (output_tensor_list.empty()) {
                        THROW_IF_ERROR([]() {
                          RETURN_STATUS_UNEXPECTED("Failed to execute op in eager mode, please check ERROR log above.");
                        }());
                      }
                      return output_tensor_list;
                    });
                }));
}  // namespace dataset
}  // namespace mindspore
