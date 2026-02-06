/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/utils/tensor_py_wrapper.h"
#include "include/utils/tensor_py.h"

namespace mindspore {
namespace tensor {

TensorPyWrapper::TensorPyWrapper(const py::object &input) {
  SetTensorPyObj(input);
  PyObject *py_obj = input.ptr();
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)py_obj;
  SetTensorWrapper(tensor->value.GetTensor());
}

TensorPyWrapperPtr ConvertToTensorPyWrapper(const py::handle &obj) {
  TensorPyWrapperPtr result = std::make_shared<TensorPyWrapper>(py::reinterpret_borrow<py::object>(obj));
  return result;
}

py::object GetTensorFromTensorPyWrapper(const TensorPyWrapperPtr &self) {
  MS_EXCEPTION_IF_NULL(self);
  return self->GetTensorWrapper();
}

abstract::AbstractBasePtr TensorPyWrapper::ToAbstract() {
  py::object baseTensor = GetTensorWrapper();
  PyObject *py_obj = baseTensor.ptr();
  PyType<TensorPy> *tensor = (PyType<TensorPy> *)py_obj;
  return tensor->value.GetTensor()->ToAbstract();
}

py::object GetTensorPyFromValue(const ValuePtr &value) {
  if (value == nullptr) {
    return py::none();
  }

  if (value->isa<Tensor>()) {
    auto tensor = value->cast<TensorPtr>();
    return PackTensorToPyObject(tensor);
  }
  TensorPyWrapperPtr result = value->cast<TensorPyWrapperPtr>();
  MS_EXCEPTION_IF_NULL(result);
  py::object back = GetTensorFromTensorPyWrapper(result);
  if (back == py::none()) {
    MS_LOG(ERROR) << "GetTensorPyFromValue back is none";
  }
  return back;
}

const MetaTensorPtr GetMetaTensorFromValue(const ValuePtr &value) {
  if (value == nullptr) {
    return nullptr;
  }

  if (value->isa<TensorPyWrapper>()) {
    auto tensorpy = value->cast<TensorPyWrapperPtr>();
    auto tensorPyWrapper = tensorpy->GetTensorWrapper();
    PyObject *py_obj = tensorPyWrapper.ptr();
    PyType<TensorPy> *tensor = reinterpret_cast<PyType<TensorPy> *>(py_obj);
    return (tensor->value.GetTensor())->cast<MetaTensorPtr>();
  }

  if (value->isa<Tensor>()) {
    auto tensor = value->cast<TensorPtr>();
    return tensor->cast<MetaTensorPtr>();
  }

  return value->cast<MetaTensorPtr>();
}

}  // namespace tensor
}  // namespace mindspore
