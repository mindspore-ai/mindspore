/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
Status ConvertNumpyToTensor(const py::object &py_obj, TensorRow *output) {
  std::shared_ptr<Tensor> out;
  // Python object like bool, int, float, list or tuple can also be converted
  // to a NumPy array by the following cast, but the data type will be unknown
  // if it is not a valid NumPy object
  RETURN_IF_NOT_OK(Tensor::CreateFromNpArray(py_obj.cast<py::array>(), &out));
  output->push_back(out);
  return Status::OK();
}

Status ConvertPythonToTensor(py::object py_obj, TensorRow *output) {
  // Python objects such as dictionary are converted to a tensor
  // Note that the tensor will hold a reference to the python object while
  // the python object will be kept alive in Python layer.
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateFromPythonObject(py_obj, &out));
  output->push_back(out);
  return Status::OK();
}

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
      py::object ret_py_obj;
      if (input.size() > 0) {
        py::tuple input_args(input.size());
        for (size_t i = 0; i < input.size(); i++) {
          if (input.at(i)->type().IsPython()) {
            py::dict new_data;
            RETURN_IF_NOT_OK(input.at(i)->GetDataAsPythonObject(&new_data));
            input_args[i] = new_data;
          } else {
            py::array new_data;
            RETURN_IF_NOT_OK(input.at(i)->GetDataAsNumpy(&new_data));
            // possible memcpy here
            input_args[i] = new_data;
          }
        }
        // Invoke python function
        ret_py_obj = this->py_func_ptr_(*input_args);
      } else {
        ret_py_obj = this->py_func_ptr_();
      }
      if (output_type_ != DataType::DE_UNKNOWN) {
        RETURN_IF_NOT_OK(CastOutput(ret_py_obj, output));
      } else {
        // scenario 1: map multi-processing, subprocess stop first and will get none
        // scenario 2: thread mode, user pyfunc return none
        if (ret_py_obj.is_none()) {
          MS_LOG(INFO) << "Maybe the multi workers of map operation had stopped, so got None from the pyfunc.";
          goto TimeoutError;
        } else if (py::isinstance<py::tuple>(ret_py_obj)) {
          // In case of a n-m mapping, the return value will be a tuple of numpy arrays
          auto ret_py_tuple = ret_py_obj.cast<py::tuple>();
          // Iterate over two containers simultaneously for memory copy
          for (size_t i = 0; i < ret_py_tuple.size(); i++) {
            py::object ret_py_ele = ret_py_tuple[i];
            // Object is none if pyfunc timeout
            if (ret_py_ele.is_none()) {
              MS_LOG(INFO) << "Expected pyfunc to return NumPy array(s) or Python dict(s), but got None. "
                              "If python_multiprocessing is True, it may be due to pyfunc execution timeout.";
              goto TimeoutError;
            } else if (py::isinstance<py::dict>(ret_py_ele)) {
              RETURN_IF_NOT_OK(ConvertPythonToTensor(ret_py_ele, output));
            } else {
              RETURN_IF_NOT_OK(ConvertNumpyToTensor(ret_py_ele, output));
            }
          }
        } else {
          // In case of a n-1 mapping, the return value will be a numpy array or a python object
          // Note that for Python dictionaries, only a reference will be stored in tensor.
          if (py::isinstance<py::dict>(ret_py_obj)) {
            RETURN_IF_NOT_OK(ConvertPythonToTensor(ret_py_obj, output));
          } else {
            RETURN_IF_NOT_OK(ConvertNumpyToTensor(ret_py_obj, output));
          }
        }
      }
    } catch (const py::error_already_set &e) {
      ret = Status(StatusCode::kMDPyFuncException, e.what());
    }
  }

ComputeReturn:
  return ret;

TimeoutError:
  ret = STATUS_ERROR(StatusCode::kMDTimeOut,
                     "Expect pyfunc to return numpy array(s), but got None. If python_multiprocessing is "
                     "True, it maybe due to pyfunc execution timeout.");
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
  {
    py::gil_scoped_acquire gil_acquire;
    if (py_func_ptr_.attr("to_json")) {
      args = nlohmann::json::parse(py_func_ptr_.attr("to_json")().cast<std::string>());
    }
  }
  *out_json = args;
  return Status::OK();
}

Status PyFuncOp::from_json(nlohmann::json json_obj, std::vector<std::shared_ptr<TensorOperation>> *result) {
  std::vector<std::shared_ptr<TensorOperation>> output;
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "tensor_op_name", kPyFuncOp));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "tensor_op_params", kPyFuncOp));
  std::string op_name = json_obj["tensor_op_name"];
  nlohmann::json op_params = json_obj["tensor_op_params"];
  std::string python_module = json_obj["python_module"];
  std::shared_ptr<TensorOperation> operation = nullptr;
  py::function py_func =
    py::module::import(python_module.c_str()).attr(op_name.c_str()).attr("from_json")(op_params.dump());
  operation = std::make_shared<transforms::PreBuiltOperation>(std::make_shared<PyFuncOp>(py_func));
  output.push_back(operation);
  *result = output;
  return Status::OK();
}

bool PyFuncOp::IsRandom() {
  bool random = true;
  if (py::hasattr(py_func_ptr_, "random") &&
      static_cast<bool>(py::reinterpret_borrow<py::bool_>(py_func_ptr_.attr("random"))) == false) {
    random = false;
  }
  return random;
}
}  // namespace dataset
}  // namespace mindspore
