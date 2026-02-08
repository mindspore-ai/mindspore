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

#ifndef MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_STORAGE_STORAGE_PY_H
#define MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_STORAGE_STORAGE_PY_H

#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind_api/pynative/tensor/storage/storage.h"

namespace mindspore {

struct StoragePy {
  PyObject_HEAD Storage cdata;
};

inline const Storage &StoragePy_Unpack(StoragePy *storage) { return storage->cdata; }

inline const Storage &StoragePy_Unpack(PyObject *obj) { return StoragePy_Unpack(reinterpret_cast<StoragePy *>(obj)); }

PyObject *CreateStorageObj(const Storage &storage);

PyObject *CreateStoragePyObj(const Storage &storage);

}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_STORAGE_STORAGE_PY_H
