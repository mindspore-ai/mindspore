/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "dataset/kernels/py_func_op.h"

#include <memory>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/make_unique.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status PyFuncOp::Compute(const std::vector<std::shared_ptr<Tensor>> &input,
                         std::vector<std::shared_ptr<Tensor>> *output) {
  IO_CHECK_VECTOR(input, output);
  Status ret = Status(StatusCode::kOK, "PyFunc Call Succeed");
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      ret = Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
      goto ComputeReturn;
    }
    try {
      // Transform input tensor vector into numpy array vector
      py::tuple input_args(input.size());
      for (size_t i = 0; i < input.size(); i++) {
        py::array new_data;
        RETURN_IF_NOT_OK(input.at(i)->GetDataAsNumpy(&new_data));
        // possible memcpy here
        input_args[i] = new_data;
      }
      // Invoke python function
      py::object ret_py_obj = this->py_func_ptr_(*input_args);
      // Process the return value
      if (py::isinstance<py::array>(ret_py_obj)) {
        // In case of a n-1 mapping, the return value will be a numpy array
        std::shared_ptr<Tensor> out;
        RETURN_IF_NOT_OK(Tensor::CreateTensor(&out, ret_py_obj.cast<py::array>()));
        output->push_back(out);
      } else if (py::isinstance<py::tuple>(ret_py_obj)) {
        // In case of a n-m mapping, the return value will be a tuple of numpy arrays
        py::tuple ret_py_tuple = ret_py_obj.cast<py::tuple>();
        // Iterate over two containers simultaneously for memory copy
        for (size_t i = 0; i < ret_py_tuple.size(); i++) {
          py::object ret_py_ele = ret_py_tuple[i];
          if (!py::isinstance<py::array>(ret_py_ele)) {
            goto ShapeMisMatch;
          }
          std::shared_ptr<Tensor> out;
          RETURN_IF_NOT_OK(Tensor::CreateTensor(&out, ret_py_ele.cast<py::array>()));
          output->push_back(out);
        }
      } else {
        goto ShapeMisMatch;
      }
    } catch (const py::error_already_set &e) {
      ret = Status(StatusCode::kPyFuncException, e.what());
    }
  }

ComputeReturn:
  return ret;

ShapeMisMatch:
  ret = Status(StatusCode::kShapeMisMatch, "PyFunc should return a numpy array or a numpy array tuple");
  goto ComputeReturn;
}
}  // namespace dataset
}  // namespace mindspore
