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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/kernels/data/fill_op.h"
#include "minddata/dataset/kernels/data/to_float16_op.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(
  FillOp, 1, ([](const py::module *m) {
    (void)py::class_<FillOp, TensorOp, std::shared_ptr<FillOp>>(*m, "FillOp").def(py::init<std::shared_ptr<Tensor>>());
  }));

PYBIND_REGISTER(ToFloat16Op, 1, ([](const py::module *m) {
                  (void)py::class_<ToFloat16Op, TensorOp, std::shared_ptr<ToFloat16Op>>(*m, "ToFloat16Op",
                                                                                        py::dynamic_attr())
                    .def(py::init<>());
                }));

}  // namespace dataset
}  // namespace mindspore
