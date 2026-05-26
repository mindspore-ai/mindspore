/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_MS_DISPATCH_PY_OBJ_H_
#define MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_MS_DISPATCH_PY_OBJ_H_

#include <memory>
#include <string>

#include "pybind11/pybind11.h"
#include "include/utils/tensor_py.h"
#include "utils/log_adapter.h"

namespace py = pybind11;
namespace mindspore {
namespace pynative {

inline const std::string &GetSourcePyObjStr() {
  static const std::string kSourcePyObj = "source_py_obj";
  return kSourcePyObj;
}

inline bool IsSourcePyObjKey(const std::string &key) { return key == GetSourcePyObjStr(); }

inline bool HasSourcePyObj(const tensor::TensorPtr &tensor) {
  return tensor != nullptr && tensor->user_data<tensor::TensorPyUserData>(GetSourcePyObjStr()) != nullptr;
}

inline py::object GetSourcePyObj(const tensor::TensorPtr &tensor) {
  if (tensor == nullptr) {
    return py::none();
  }
  const auto user_data = tensor->user_data<tensor::TensorPyUserData>(GetSourcePyObjStr());
  MS_EXCEPTION_IF_NULL(user_data);
  return user_data->obj;
}

inline py::object PackSourcePyObj(const tensor::TensorPtr &tensor) { return GetSourcePyObj(tensor); }

inline PyObject *PackSourcePyObjPtr(const tensor::TensorPtr &tensor) {
  auto source_obj = GetSourcePyObj(tensor);
  source_obj.inc_ref();
  return source_obj.ptr();
}

inline void ClearSourcePyObj(const tensor::TensorPtr &tensor) {
  if (tensor == nullptr) {
    return;
  }
  tensor->set_user_data<tensor::TensorPyUserData>(GetSourcePyObjStr(), nullptr);
}

inline tensor::TensorPtr WrapTensorWithSourcePyObj(const tensor::TensorPtr &original_tensor,
                                                   const py::object &source_obj) {
  if (original_tensor == nullptr) {
    return nullptr;
  }

  auto new_tensor = std::make_shared<tensor::Tensor>(*original_tensor);
  if (original_tensor->is_parameter()) {
    new_tensor->set_param_info(original_tensor->param_info());
  }

  auto py_obj_data = std::make_shared<tensor::TensorPyUserData>();
  py_obj_data->obj = source_obj;
  new_tensor->set_user_data<tensor::TensorPyUserData>(GetSourcePyObjStr(), py_obj_data);

  return new_tensor;
}

inline tensor::TensorPtr ConvertPyObjectToTensorWithSourcePyObj(const py::object &obj) {
  auto tensor = tensor::ConvertToTensor(obj);
  if (tensor != nullptr && tensor->has_ms_dispatch()) {
    return WrapTensorWithSourcePyObj(tensor, obj);
  }
  return tensor;
}

inline tensor::TensorPtr EnsureSourcePyObjForMsDispatch(const tensor::TensorPtr &tensor, const py::object &source_obj) {
  if (tensor == nullptr || !tensor->has_ms_dispatch() || !tensor::IsTensorPy(source_obj)) {
    return tensor;
  }
  if (HasSourcePyObj(tensor)) {
    return tensor;
  }
  return WrapTensorWithSourcePyObj(tensor, source_obj);
}

}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_MS_DISPATCH_PY_OBJ_H_
