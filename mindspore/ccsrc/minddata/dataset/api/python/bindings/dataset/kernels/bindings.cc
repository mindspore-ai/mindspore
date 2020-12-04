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

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/core/global_context.h"

#include "mindspore/ccsrc/minddata/dataset/kernels/data/compose_op.h"
#include "mindspore/ccsrc/minddata/dataset/kernels/data/no_op.h"
#include "minddata/dataset/kernels/py_func_op.h"
#include "mindspore/ccsrc/minddata/dataset/kernels/data/random_apply_op.h"
#include "mindspore/ccsrc/minddata/dataset/kernels/data/random_choice_op.h"

namespace mindspore {
namespace dataset {

Status PyListToTensorOps(const py::list &py_ops, std::vector<std::shared_ptr<TensorOp>> *ops) {
  RETURN_UNEXPECTED_IF_NULL(ops);
  for (auto op : py_ops) {
    if (py::isinstance<TensorOp>(op)) {
      ops->emplace_back(op.cast<std::shared_ptr<TensorOp>>());
    } else if (py::isinstance<py::function>(op)) {
      ops->emplace_back(std::make_shared<PyFuncOp>(op.cast<py::function>()));
    } else {
      RETURN_STATUS_UNEXPECTED("element is neither a TensorOp nor a pyfunc.");
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!ops->empty(), "TensorOp list is empty.");
  for (auto const &op : *ops) {
    RETURN_UNEXPECTED_IF_NULL(op);
  }
  return Status::OK();
}

PYBIND_REGISTER(TensorOp, 0, ([](const py::module *m) {
                  (void)py::class_<TensorOp, std::shared_ptr<TensorOp>>(*m, "TensorOp")
                    .def("__deepcopy__", [](py::object &t, py::dict memo) { return t; });
                }));
}  // namespace dataset
}  // namespace mindspore
