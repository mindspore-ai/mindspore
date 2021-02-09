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

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/core/global_context.h"

#include "minddata/dataset/kernels/py_func_op.h"
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"
#include "minddata/dataset/kernels/ir/vision/vision_ir.h"

namespace mindspore {
namespace dataset {

Status PyListToTensorOperations(const py::list &py_ops, std::vector<std::shared_ptr<TensorOperation>> *ops) {
  RETURN_UNEXPECTED_IF_NULL(ops);
  for (auto op : py_ops) {
    if (py::isinstance<TensorOp>(op)) {
      ops->emplace_back(std::make_shared<transforms::PreBuiltOperation>(op.cast<std::shared_ptr<TensorOp>>()));
    } else if (py::isinstance<py::function>(op)) {
      ops->emplace_back(
        std::make_shared<transforms::PreBuiltOperation>(std::make_shared<PyFuncOp>(op.cast<py::function>())));
    } else if (py::isinstance<TensorOperation>(op)) {
      ops->emplace_back(op.cast<std::shared_ptr<TensorOperation>>());
    } else {
      RETURN_STATUS_UNEXPECTED("element is neither a TensorOp, TensorOperation nor a pyfunc.");
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!ops->empty(), "TensorOp list is empty.");
  for (auto const &op : *ops) {
    RETURN_UNEXPECTED_IF_NULL(op);
  }
  return Status::OK();
}

PYBIND_REGISTER(TensorOperation, 0, ([](const py::module *m) {
                  (void)py::class_<TensorOperation, std::shared_ptr<TensorOperation>>(*m, "TensorOperation");
                  py::arg("TensorOperation");
                }));

PYBIND_REGISTER(
  ComposeOperation, 1, ([](const py::module *m) {
    (void)py::class_<transforms::ComposeOperation, TensorOperation, std::shared_ptr<transforms::ComposeOperation>>(
      *m, "ComposeOperation")
      .def(py::init([](const py::list &ops) {
        std::vector<std::shared_ptr<TensorOperation>> t_ops;
        THROW_IF_ERROR(PyListToTensorOperations(ops, &t_ops));
        auto compose = std::make_shared<transforms::ComposeOperation>(std::move(t_ops));
        THROW_IF_ERROR(compose->ValidateParams());
        return compose;
      }));
  }));

PYBIND_REGISTER(RandomChoiceOperation, 1, ([](const py::module *m) {
                  (void)py::class_<transforms::RandomChoiceOperation, TensorOperation,
                                   std::shared_ptr<transforms::RandomChoiceOperation>>(*m, "RandomChoiceOperation")
                    .def(py::init([](const py::list &ops) {
                      std::vector<std::shared_ptr<TensorOperation>> t_ops;
                      THROW_IF_ERROR(PyListToTensorOperations(ops, &t_ops));
                      auto random_choice = std::make_shared<transforms::RandomChoiceOperation>(std::move(t_ops));
                      THROW_IF_ERROR(random_choice->ValidateParams());
                      return random_choice;
                    }));
                }));

PYBIND_REGISTER(RandomApplyOperation, 1, ([](const py::module *m) {
                  (void)py::class_<transforms::RandomApplyOperation, TensorOperation,
                                   std::shared_ptr<transforms::RandomApplyOperation>>(*m, "RandomApplyOperation")
                    .def(py::init([](double prob, const py::list &ops) {
                      std::vector<std::shared_ptr<TensorOperation>> t_ops;
                      THROW_IF_ERROR(PyListToTensorOperations(ops, &t_ops));
                      auto random_apply = std::make_shared<transforms::RandomApplyOperation>(std::move(t_ops), prob);
                      THROW_IF_ERROR(random_apply->ValidateParams());
                      return random_apply;
                    }));
                }));
}  // namespace dataset
}  // namespace mindspore
