/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/py_func_op.h"

#include <memory>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status PyFuncOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  Status ret = Status(StatusCode::kSuccess, "PyFunc Call Succeed");
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      ret = Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized");
      goto ComputeReturn;
    }
    try {
      // Transform input tensor vector into numpy array vector
      py::tuple input_args(input.size());
      py::object ret_py_obj;
      if (input.size() > 0) {
        for (size_t i = 0; i < input.size(); i++) {
          py::array new_data;
          RETURN_IF_NOT_OK(input.at(i)->GetDataAsNumpy(&new_data));
          // possible memcpy here
          input_args[i] = new_data;
        }
        // Invoke python function
        ret_py_obj = this->py_func_ptr_(*input_args);
      } else {
        ret_py_obj = this->py_func_ptr_();
      }
      if (output_type_ != DataType::DE_UNKNOWN) {
        RETURN_IF_NOT_OK(CastOutput(ret_py_obj, output));
      } else {
        if (py::isinstance<py::tuple>(ret_py_obj)) {
          // In case of a n-m mapping, the return value will be a tuple of numpy arrays
          py::tuple ret_py_tuple = ret_py_obj.cast<py::tuple>();
          // Iterate over two containers simultaneously for memory copy
          for (size_t i = 0; i < ret_py_tuple.size(); i++) {
            py::object ret_py_ele = ret_py_tuple[i];
            // Object is none if pyfunc timeout
            if (ret_py_ele.is_none()) {
              MS_LOG(INFO) << "PyFunc execute time out";
              goto TimeoutError;
            }
            if (!py::isinstance<py::array>(ret_py_ele)) {
              goto ShapeMisMatch;
            }
            std::shared_ptr<Tensor> out;
            RETURN_IF_NOT_OK(Tensor::CreateFromNpArray(ret_py_ele.cast<py::array>(), &out));
            output->push_back(out);
          }
        } else if (py::isinstance<py::array>(ret_py_obj)) {
          // In case of a n-1 mapping, the return value will be a numpy array
          std::shared_ptr<Tensor> out;
          RETURN_IF_NOT_OK(Tensor::CreateFromNpArray(ret_py_obj.cast<py::array>(), &out));
          output->push_back(out);
        } else {
          goto ShapeMisMatch;
        }
      }
    } catch (const py::error_already_set &e) {
      ret = Status(StatusCode::kMDPyFuncException, e.what());
    }
  }

ComputeReturn:
  return ret;

ShapeMisMatch:
  ret = Status(StatusCode::kMDShapeMisMatch, __LINE__, __FILE__,
               "PyFunc should return a numpy array or a numpy array tuple");
  goto ComputeReturn;

TimeoutError:
  ret = Status(StatusCode::kMDTimeOut, __LINE__, __FILE__, "PyFunc execute time out");
  goto ComputeReturn;
}

Status PyFuncOp::CastOutput(const py::object &ret_py_obj, TensorRow *output) {
  try {
    std::shared_ptr<Tensor> out;
    switch (output_type_) {
      case DataType::DE_INT32:
        RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({1}), DataType(DataType::DE_INT32), &out));
        RETURN_IF_NOT_OK(out->SetItemAt({0}, ret_py_obj.cast<int32_t>()));
        break;
      case DataType::DE_BOOL:
        RETURN_IF_NOT_OK(Tensor::CreateScalar(ret_py_obj.cast<bool>(), &out));
        break;
      default:
        RETURN_STATUS_UNEXPECTED("No cast for the specified DataType was found.");
    }
    output->push_back(out);
  } catch (const std::exception &e) {
    return Status(StatusCode::kMDUnexpectedError, e.what());
  }
  return Status::OK();
}

Status PyFuncOp::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["tensor_op_name"] = py_func_ptr_.attr("__class__").attr("__name__").cast<std::string>();
  args["is_python_front_end_op"] = true;
  *out_json = args;
  return Status::OK();
}

bool PyFuncOp::IsRandom() {
  bool random = true;
  if (py::hasattr(py_func_ptr_, "random") && py::reinterpret_borrow<py::bool_>(py_func_ptr_.attr("random")) == false)
    random = false;
  return random;
}
}  // namespace dataset
}  // namespace mindspore
