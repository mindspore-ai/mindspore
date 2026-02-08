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

#include "include/utils/pyobj_manager.h"

namespace mindspore {
PyObjManager &PyObjManager::Get() {
  static PyObjManager manager;
  return manager;
}

void PyObjManager::Clear() {
  Py_XDECREF(tensor_module_);
  Py_XDECREF(abc_module_);
  Py_XDECREF(tensor_python_class_);
  Py_XDECREF(untyped_storage_class_);
}

PyObject *PyObjManager::GetTensorModule() {
  if (tensor_module_ == nullptr) {
    tensor_module_ = PyImport_ImportModule("mindspore.common.tensor");
  }
  return tensor_module_;
}

PyObject *PyObjManager::GetAbcModule() {
  if (abc_module_ == nullptr) {
    abc_module_ = PyImport_ImportModule("abc");
  }
  return abc_module_;
}

PyObject *PyObjManager::GetHookUtilsClass() {
  if (hook_utils_class_ == nullptr) {
    const auto hook_handle_module = PyImport_ImportModule("mindspore.common.hook_handle");
    hook_utils_class_ = PyObject_GetAttrString(hook_handle_module, "_HookUtils");
  }
  return hook_utils_class_;
}

PyObject *PyObjManager::GetTensorPythonClass() {
  if (tensor_python_class_ == nullptr) {
    PyObject *tensor_module = GetTensorModule();
    if (tensor_module == nullptr) {
      return nullptr;
    }

    tensor_python_class_ = PyObject_GetAttrString(tensor_module, "Tensor");
    if (tensor_python_class_ == nullptr) {
      return nullptr;
    }
  }
  return tensor_python_class_;
}

PyTypeObject *PyObjManager::GetTensorPythonType() {
  if (python_tensor_type_ == nullptr) {
    python_tensor_type_ = reinterpret_cast<PyTypeObject *>(GetTensorPythonClass());
    return python_tensor_type_;
  }
  return python_tensor_type_;
}

PyTypeObject *PyObjManager::GetUntypedStorageClass() {
  if (untyped_storage_class_ == nullptr) {
    const auto mindspore_module = PyImport_ImportModule("mindspore");
    untyped_storage_class_ = (PyTypeObject *)PyObject_GetAttrString(mindspore_module, "UntypedStorage");
  }
  return untyped_storage_class_;
}
}  // namespace mindspore
