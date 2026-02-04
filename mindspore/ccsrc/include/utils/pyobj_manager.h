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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYOBJ_MANAGER_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYOBJ_MANAGER_H_

#include <Python.h>
#include "include/utils/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {
class COMMON_EXPORT PyObjManager {
 public:
  static PyObjManager &Get();

  void Clear();
  PyObject *GetTensorModule();
  PyObject *GetAbcModule();
  PyObject *GetHookUtilsClass();
  PyObject *GetTensorPythonClass();
  PyTypeObject *GetTensorPythonType();
  PyTypeObject *GetUntypedStorageClass();

 private:
  PyObjManager() = default;
  ~PyObjManager() = default;
  DISABLE_COPY_AND_ASSIGN(PyObjManager);

  PyObject *tensor_module_{nullptr};
  PyObject *abc_module_{nullptr};
  PyObject *hook_utils_class_{nullptr};
  PyObject *tensor_python_class_{nullptr};
  PyTypeObject *python_tensor_type_{nullptr};
  PyTypeObject *untyped_storage_class_{nullptr};
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYOBJ_MANAGER_H_
