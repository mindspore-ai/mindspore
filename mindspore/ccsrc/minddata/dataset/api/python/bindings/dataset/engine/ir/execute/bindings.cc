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
#include "minddata/dataset/core/type_id.h"
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
                         [](Execute &self, const std::shared_ptr<Tensor> &de_tensor) {
                           auto ms_tensor = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
                           THROW_IF_ERROR(self(ms_tensor, &ms_tensor));
                           std::shared_ptr<dataset::Tensor> de_output_tensor;
                           dataset::Tensor::CreateFromMemory(dataset::TensorShape(ms_tensor.Shape()),
                                                             MSTypeToDEType(static_cast<TypeId>(ms_tensor.DataType())),
                                                             (const uchar *)(ms_tensor.Data().get()),
                                                             ms_tensor.DataSize(), &de_output_tensor);
                           return de_output_tensor;
                         })
                    .def("__call__", [](Execute &self, const std::vector<std::shared_ptr<Tensor>> &input_tensor_list) {
                      std::vector<MSTensor> ms_input_tensor_list;
                      std::vector<MSTensor> ms_output_tensor_list;
                      for (auto &tensor : input_tensor_list) {
                        auto ms_tensor = mindspore::MSTensor(std::make_shared<DETensor>(tensor));
                        ms_input_tensor_list.emplace_back(std::move(ms_tensor));
                      }
                      THROW_IF_ERROR(self(ms_input_tensor_list, &ms_output_tensor_list));
                      std::vector<std::shared_ptr<dataset::Tensor>> de_output_tensor_list;
                      for (auto &tensor : ms_output_tensor_list) {
                        std::shared_ptr<dataset::Tensor> de_output_tensor;
                        dataset::Tensor::CreateFromMemory(
                          dataset::TensorShape(tensor.Shape()), MSTypeToDEType(static_cast<TypeId>(tensor.DataType())),
                          (const uchar *)(tensor.Data().get()), tensor.DataSize(), &de_output_tensor);
                        de_output_tensor_list.emplace_back(std::move(de_output_tensor));
                      }
                      return de_output_tensor_list;
                    });
                }));

}  // namespace dataset
}  // namespace mindspore
