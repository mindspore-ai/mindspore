/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/include/vision_lite.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(
  RandomSelectSubpolicyOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomSelectSubpolicyOperation, TensorOperation,
                     std::shared_ptr<vision::RandomSelectSubpolicyOperation>>(*m, "RandomSelectSubpolicyOperation")
      .def(py::init([](const py::list &py_policy) {
        std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> cpp_policy;
        for (auto &py_sub : py_policy) {
          cpp_policy.push_back({});
          for (auto handle : py_sub.cast<py::list>()) {
            py::tuple tp = handle.cast<py::tuple>();
            if (tp.is_none() || tp.size() != 2) {
              THROW_IF_ERROR(Status(StatusCode::kUnexpectedError, "Each tuple in subpolicy should be (op, prob)."));
            }
            std::shared_ptr<TensorOperation> t_op;
            if (py::isinstance<TensorOperation>(tp[0])) {
              t_op = (tp[0]).cast<std::shared_ptr<TensorOperation>>();
            } else if (py::isinstance<TensorOp>(tp[0])) {
              t_op = std::make_shared<transforms::PreBuiltOperation>((tp[0]).cast<std::shared_ptr<TensorOp>>());
            } else if (py::isinstance<py::function>(tp[0])) {
              t_op = std::make_shared<transforms::PreBuiltOperation>(
                std::make_shared<PyFuncOp>((tp[0]).cast<py::function>()));
            } else {
              THROW_IF_ERROR(
                Status(StatusCode::kUnexpectedError, "op is neither a tensorOp, tensorOperation nor a pyfunc."));
            }
            double prob = (tp[1]).cast<py::float_>();
            if (prob < 0 || prob > 1) {
              THROW_IF_ERROR(Status(StatusCode::kUnexpectedError, "prob needs to be with [0,1]."));
            }
            cpp_policy.back().emplace_back(std::make_pair(t_op, prob));
          }
        }
        return std::make_shared<vision::RandomSelectSubpolicyOperation>(cpp_policy);
      }));
  }));
}  // namespace dataset
}  // namespace mindspore
